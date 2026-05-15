"""Build `pose_manifest.csv` and `pose_lmdb/` directly from docked SDF files.

This builder treats the processed SDF molecules as the single source of truth:
- each valid ligand molecule becomes one LMDB record
- each LMDB key is mirrored into one `pose_manifest.csv` row
- pose ranks are assigned per `(grid, ligand)` by docking score

Ties are broken deterministically by source file path and molecule index so
`pose_id = {pdb_key}_{ligand}_{pose_rank}` stays unique without relying on
Glide's internal ligand numbering.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import lmdb
import pandas as pd
import torch
from rdkit.Chem import rdmolfiles
from torch_geometric.utils.smiles import from_rdmol

try:
    from constants import (
        POSE_DUPLICATES_CSV,
        POSE_LMDB_DIR,
        POSE_MANIFEST_COLUMNS,
        POSE_MANIFEST_CSV,
        RAW_OUTPUTS_DIR,
    )
    from ids import canonicalize_smiles, make_pdb_key, make_pose_id, parse_grid_string
    from manifest_io import assert_unique, ensure_parent_dir, read_csv_checked, write_manifest
except ImportError:  # pragma: no cover
    from .constants import (
        POSE_DUPLICATES_CSV,
        POSE_LMDB_DIR,
        POSE_MANIFEST_COLUMNS,
        POSE_MANIFEST_CSV,
        RAW_OUTPUTS_DIR,
    )
    from .ids import canonicalize_smiles, make_pdb_key, make_pose_id, parse_grid_string
    from .manifest_io import assert_unique, ensure_parent_dir, read_csv_checked, write_manifest


VARIANT_SUFFIX_RE = re.compile(r"^(?P<ligand>.+)-(?P<variant_idx>\d+)$")


def _iter_docked_sdfs(outputs_dir: str | Path) -> list[Path]:
    """Collect all docked SDF files under the outputs directory."""
    outputs_dir = Path(outputs_dir)
    return sorted(outputs_dir.glob("*_docked/batch_*_docked.sdf"))


def _parse_variant_ligand(variant: str) -> str:
    """Strip the random suffix from `s_lp_Variant` and canonicalize the ligand."""
    variant = str(variant).strip()
    match = VARIANT_SUFFIX_RE.match(variant)
    if not match:
        raise ValueError(
            "Expected s_lp_Variant to look like '{smiles}-{int}', "
            f"got: {variant}"
        )
    return canonicalize_smiles(match.group("ligand"))


def _iter_ligand_mols(sdf_path: Path):
    """Yield `(mol_index, mol)` for valid ligand molecules in a docked SDF."""
    supplier = rdmolfiles.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol_index, mol in enumerate(supplier):
        if mol is None:
            continue
        if not mol.HasProp("s_lp_Variant"):
            continue
        if not mol.HasProp("r_i_docking_score"):
            continue
        if not mol.HasProp("s_i_glide_gridfile"):
            continue
        yield mol_index, mol


def _read_sdf_rows(sdf_path: Path) -> list[dict[str, object]]:
    """Load docked SDF molecules with normalized metadata."""
    rows = []
    resolved_sdf = str(sdf_path.resolve())
    for mol_index, mol in _iter_ligand_mols(sdf_path):
        rows.append(
            {
                "mol_index": mol_index,
                "mol": mol,
                "ligand": _parse_variant_ligand(mol.GetProp("s_lp_Variant")),
                "grid": mol.GetProp("s_i_glide_gridfile").strip(),
                "glide_score": float(mol.GetProp("r_i_docking_score")),
                "source_sdf": resolved_sdf,
            }
        )
    return rows


def rdmol_to_pyg_with_pos(mol):
    """Convert an RDKit molecule into a PyG graph with 3D coordinates."""
    data = from_rdmol(mol)
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    data.pos = torch.tensor(pos, dtype=torch.float)
    data.is_protein_atom = torch.zeros(data.num_nodes, dtype=torch.bool)
    return data


def _annotate_pose_ranks(sdf_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Assign deterministic pose ranks within each `(grid, ligand)` group."""
    if not sdf_rows:
        return []

    sdf_df = pd.DataFrame(sdf_rows)
    sdf_df = sdf_df.sort_values(
        ["grid", "ligand", "glide_score", "source_sdf", "mol_index"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    sdf_df["pose_rank"] = sdf_df.groupby(["grid", "ligand"]).cumcount() + 1
    sdf_df["is_top_rank"] = sdf_df["pose_rank"] == 1
    return sdf_df.to_dict("records")


def _pose_manifest_row(record: dict[str, object]) -> dict[str, object]:
    """Create a manifest row and final LMDB key for one docked pose."""
    ligand = str(record["ligand"]).strip()
    grid = str(record["grid"]).strip()
    pdb_id, uniprot_id = parse_grid_string(grid)
    pdb_key = make_pdb_key(pdb_id, uniprot_id)
    pose_id = make_pose_id(
        pdb_key=pdb_key,
        ligand=ligand,
        pose_number=int(record["pose_rank"]),
    )
    return {
        "pose_id": pose_id,
        "pdb_key": pdb_key,
        "ligand": ligand,
        "grid": grid,
        "uniprot_id": uniprot_id,
        "pdb_id": pdb_id,
        "glide_score": float(record["glide_score"]),
        "pose_rank": int(record["pose_rank"]),
        "is_top_rank": bool(record["is_top_rank"]),
        "source_sdf": str(record["source_sdf"]),
    }


def _read_existing_manifest(path: str | Path) -> pd.DataFrame:
    """Load an existing pose manifest if present, otherwise return an empty one."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(POSE_MANIFEST_COLUMNS))
    return read_csv_checked(path, POSE_MANIFEST_COLUMNS)


def _append_csv_rows(df: pd.DataFrame, path: str | Path) -> Path:
    """Append rows to a CSV, creating it with headers if needed."""
    path = ensure_parent_dir(path)
    header = not Path(path).exists()
    df.to_csv(path, mode="a", header=header, index=False)
    return path


def build_pose_manifest_and_lmdb(
    outputs_dir: str | Path,
    pose_manifest_csv: str | Path,
    pose_lmdb_dir: str | Path,
    duplicate_pose_csv: str | Path = POSE_DUPLICATES_CSV,
    map_size: int = int(50e9),
) -> pd.DataFrame:
    """Build pose manifest rows and write matching PyG objects to LMDB.

    This function is incremental:
    - existing manifest rows are preserved
    - existing LMDB keys are not overwritten
    - duplicate pose keys are recorded in a separate CSV and skipped
    """
    sdf_paths = _iter_docked_sdfs(outputs_dir)
    if not sdf_paths:
        raise FileNotFoundError(f"No batch_*_docked.sdf files found under {outputs_dir}")

    existing_manifest_df = _read_existing_manifest(pose_manifest_csv)
    existing_pose_ids = set(existing_manifest_df["pose_id"].tolist())

    sdf_rows: list[dict[str, object]] = []
    for sdf_path in sdf_paths:
        sdf_rows.extend(_read_sdf_rows(sdf_path))
    ranked_rows = _annotate_pose_ranks(sdf_rows)

    pose_lmdb_dir = ensure_parent_dir(Path(pose_lmdb_dir) / "data.mdb").parent
    env = lmdb.open(
        str(pose_lmdb_dir),
        subdir=True,
        map_size=map_size,
        metasync=False,
        sync=False,
        meminit=False,
        writemap=True,
    )

    new_manifest_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    try:
        with env.begin(write=True) as txn:
            for record in ranked_rows:
                manifest_row = _pose_manifest_row(record)
                pose_key_bytes = manifest_row["pose_id"].encode("utf-8")
                exists_in_manifest = manifest_row["pose_id"] in existing_pose_ids
                exists_in_lmdb = txn.get(pose_key_bytes) is not None
                if exists_in_manifest or exists_in_lmdb:
                    duplicate_rows.append(
                        {
                            "pose_id": manifest_row["pose_id"],
                            "source_sdf": manifest_row["source_sdf"],
                            "pdb_key": manifest_row["pdb_key"],
                            "ligand": manifest_row["ligand"],
                            "glide_score": manifest_row["glide_score"],
                            "pose_rank": manifest_row["pose_rank"],
                            "exists_in_manifest": exists_in_manifest,
                            "exists_in_lmdb": exists_in_lmdb,
                        }
                    )
                    continue

                graph = rdmol_to_pyg_with_pos(record["mol"])
                graph.pose_id = manifest_row["pose_id"]
                graph.pdb_key = manifest_row["pdb_key"]
                graph.ligand = manifest_row["ligand"]
                graph.glide_score = manifest_row["glide_score"]
                inserted = txn.put(
                    pose_key_bytes,
                    pickle.dumps(graph),
                    overwrite=False,
                )
                if not inserted:
                    duplicate_rows.append(
                        {
                            "pose_id": manifest_row["pose_id"],
                            "source_sdf": manifest_row["source_sdf"],
                            "pdb_key": manifest_row["pdb_key"],
                            "ligand": manifest_row["ligand"],
                            "glide_score": manifest_row["glide_score"],
                            "pose_rank": manifest_row["pose_rank"],
                            "exists_in_manifest": exists_in_manifest,
                            "exists_in_lmdb": True,
                        }
                    )
                    continue

                new_manifest_rows.append(manifest_row)
                existing_pose_ids.add(manifest_row["pose_id"])
    finally:
        env.close()

    new_pose_df = pd.DataFrame(new_manifest_rows, columns=list(POSE_MANIFEST_COLUMNS))
    duplicate_df = pd.DataFrame(
        duplicate_rows,
        columns=[
            "pose_id",
            "source_sdf",
            "pdb_key",
            "ligand",
            "glide_score",
            "pose_rank",
            "exists_in_manifest",
            "exists_in_lmdb",
        ],
    )

    if not new_pose_df.empty:
        new_pose_df = new_pose_df.sort_values(["pdb_key", "source_sdf", "ligand", "pose_rank"]).reset_index(drop=True)
        combined_pose_df = pd.concat([existing_manifest_df, new_pose_df], ignore_index=True)
    else:
        combined_pose_df = existing_manifest_df.copy()

    if not combined_pose_df.empty:
        combined_pose_df = combined_pose_df.sort_values(["pdb_key", "source_sdf", "ligand", "pose_rank"]).reset_index(drop=True)
        assert_unique(combined_pose_df, ["pose_id"], "pose_manifest")

    if not new_pose_df.empty:
        _append_csv_rows(new_pose_df, pose_manifest_csv)
    elif not Path(pose_manifest_csv).exists():
        write_manifest(combined_pose_df, pose_manifest_csv)

    if not duplicate_df.empty:
        duplicate_df = duplicate_df.sort_values(["pose_id", "source_sdf"]).reset_index(drop=True)
        _append_csv_rows(duplicate_df, duplicate_pose_csv)

    return combined_pose_df


def parse_args() -> argparse.Namespace:
    """CLI for generating both pose manifest and pose LMDB."""
    parser = argparse.ArgumentParser(
        description="Build pose_manifest.csv and pose_lmdb/ directly from docked SDF outputs."
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(RAW_OUTPUTS_DIR),
        help="Directory containing *_docked subdirectories with docked SDF files.",
    )
    parser.add_argument(
        "--pose-manifest-csv",
        default=str(POSE_MANIFEST_CSV),
        help="Output path for pose_manifest.csv.",
    )
    parser.add_argument(
        "--pose-lmdb-dir",
        default=str(POSE_LMDB_DIR),
        help="Output directory for pose_lmdb.",
    )
    parser.add_argument(
        "--duplicate-pose-csv",
        default=str(POSE_DUPLICATES_CSV),
        help="Output path for duplicate pose-key records.",
    )
    parser.add_argument(
        "--map-size",
        type=int,
        default=int(50e9),
        help="LMDB map size in bytes.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    build_pose_manifest_and_lmdb(
        outputs_dir=args.outputs_dir,
        pose_manifest_csv=args.pose_manifest_csv,
        pose_lmdb_dir=args.pose_lmdb_dir,
        duplicate_pose_csv=args.duplicate_pose_csv,
        map_size=args.map_size,
    )


if __name__ == "__main__":
    main()
