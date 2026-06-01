"""Build `pose_manifest.csv` and `pose_lmdb/` directly from docked SDF files.

This builder treats the processed SDF molecules as the single source of truth:
- each valid ligand molecule becomes one LMDB record
- each LMDB key is mirrored into one `pose_manifest.csv` row
- pose IDs are based on the protein context, ligand identity, and pose hash
- pose ranks are assigned per `(pdb_key, ligand)` by docking score

Pose hashes use the PyG graph node features and 3D coordinates rounded to
three decimals. Rank is metadata, not part of the persistent pose identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import lmdb
import pandas as pd
import torch
from rdkit.Chem import rdmolfiles
from torch_geometric.utils.smiles import from_rdmol

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from data_processing.common.constants import (
        POSE_DUPLICATES_CSV,
        POSE_LMDB_DIR,
        POSE_MANIFEST_COLUMNS,
        POSE_MANIFEST_CSV,
        RAW_OUTPUTS_DIR,
    )
    from data_processing.common.ids import canonicalize_smiles, make_pdb_key, parse_grid_string
    from data_processing.common.manifest_io import assert_unique, ensure_parent_dir, write_manifest
else:
    from ..common.constants import (
        POSE_DUPLICATES_CSV,
        POSE_LMDB_DIR,
        POSE_MANIFEST_COLUMNS,
        POSE_MANIFEST_CSV,
        RAW_OUTPUTS_DIR,
    )
    from ..common.ids import canonicalize_smiles, make_pdb_key, parse_grid_string
    from ..common.manifest_io import assert_unique, ensure_parent_dir, write_manifest


VARIANT_SUFFIX_RE = re.compile(r"^(?P<ligand>.+)-(?P<variant_idx>\d+)$")
POSE_HASH_DECIMALS = 3
POSE_HASH_HEXDIGITS = 16


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


def _iter_sdf_records(sdf_path: Path) -> Iterator[dict[str, object]]:
    """Yield docked SDF molecules with normalized metadata one at a time."""
    resolved_sdf = str(sdf_path.resolve())
    for _, mol in _iter_ligand_mols(sdf_path):
        yield {
            "mol": mol,
            "ligand": _parse_variant_ligand(mol.GetProp("s_lp_Variant")),
            "grid": mol.GetProp("s_i_glide_gridfile").strip(),
            "glide_score": float(mol.GetProp("r_i_docking_score")),
            "source_sdf": resolved_sdf,
        }


def rdmol_to_pyg_with_pos(mol):
    """Convert an RDKit molecule into a PyG graph with 3D coordinates."""
    data = from_rdmol(mol)
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    data.pos = torch.tensor(pos, dtype=torch.float)
    data.is_protein_atom = torch.zeros(data.num_nodes, dtype=torch.bool)
    return data


def _format_position_value(value: float, decimals: int = POSE_HASH_DECIMALS) -> str:
    """Format coordinates deterministically for content hashing."""
    rounded = round(float(value), decimals)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


def pose_graph_hash(graph) -> str:
    """Hash PyG node features and 3D positions rounded to three decimals."""
    x_values = graph.x.detach().cpu().tolist()
    pos_values = [
        [_format_position_value(value) for value in row]
        for row in graph.pos.detach().cpu().tolist()
    ]
    payload = {"pos": pos_values, "x": x_values}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:POSE_HASH_HEXDIGITS]


def _make_pose_id(pdb_key: str, ligand: str, pose_hash: str) -> str:
    """Build a stable pose ID that does not depend on pose rank."""
    ligand = canonicalize_smiles(ligand)
    return f"{pdb_key}_{ligand}_{pose_hash}"


def _annotate_pose_ranks(pose_df: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic pose ranks within each `(pdb_key, ligand)` group."""
    if pose_df.empty:
        return pose_df.copy()

    ranked_df = pose_df.copy()
    ranked_df["glide_score"] = pd.to_numeric(ranked_df["glide_score"], errors="raise")
    ranked_df = ranked_df.sort_values(
        ["pdb_key", "ligand", "glide_score", "source_sdf", "pose_id"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked_df["pose_rank"] = ranked_df.groupby(["pdb_key", "ligand"]).cumcount() + 1
    ranked_df["is_top_rank"] = ranked_df["pose_rank"] == 1
    return ranked_df


def _pose_manifest_row(record: dict[str, object], pose_hash: str) -> dict[str, object]:
    """Create a manifest row and final LMDB key for one docked pose."""
    ligand = str(record["ligand"]).strip()
    grid = str(record["grid"]).strip()
    pdb_id, uniprot_id = parse_grid_string(grid)
    pdb_key = make_pdb_key(pdb_id, uniprot_id)
    pose_id = _make_pose_id(pdb_key=pdb_key, ligand=ligand, pose_hash=pose_hash)
    return {
        "pose_id": pose_id,
        "pose_hash": pose_hash,
        "pdb_key": pdb_key,
        "ligand": ligand,
        "grid": grid,
        "uniprot_id": uniprot_id,
        "pdb_id": pdb_id,
        "glide_score": float(record["glide_score"]),
        "pose_rank": 0,
        "is_top_rank": False,
        "source_sdf": str(record["source_sdf"]),
    }


def _read_existing_manifest(path: str | Path) -> pd.DataFrame:
    """Load an existing pose manifest if present, otherwise return an empty one."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(POSE_MANIFEST_COLUMNS))

    df = pd.read_csv(path)
    missing_columns = [column for column in POSE_MANIFEST_COLUMNS if column not in df.columns]
    if missing_columns == ["pose_hash"]:
        df["pose_hash"] = ""
    elif missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )
    return df.loc[:, list(POSE_MANIFEST_COLUMNS)]


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
            for sdf_path in sdf_paths:
                for record in _iter_sdf_records(sdf_path):
                    graph = rdmol_to_pyg_with_pos(record["mol"])
                    pose_hash = pose_graph_hash(graph)
                    manifest_row = _pose_manifest_row(record, pose_hash)
                    pose_key_bytes = manifest_row["pose_id"].encode("utf-8")
                    exists_in_manifest = manifest_row["pose_id"] in existing_pose_ids
                    exists_in_lmdb = txn.get(pose_key_bytes) is not None
                    if exists_in_manifest or exists_in_lmdb:
                        duplicate_rows.append(
                            {
                                "pose_id": manifest_row["pose_id"],
                                "pose_hash": manifest_row["pose_hash"],
                                "source_sdf": manifest_row["source_sdf"],
                                "pdb_key": manifest_row["pdb_key"],
                                "ligand": manifest_row["ligand"],
                                "glide_score": manifest_row["glide_score"],
                                "pose_rank": None,
                                "exists_in_manifest": exists_in_manifest,
                                "exists_in_lmdb": exists_in_lmdb,
                            }
                        )
                        continue

                    graph.pose_id = manifest_row["pose_id"]
                    graph.pose_hash = manifest_row["pose_hash"]
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
                                "pose_hash": manifest_row["pose_hash"],
                                "source_sdf": manifest_row["source_sdf"],
                                "pdb_key": manifest_row["pdb_key"],
                                "ligand": manifest_row["ligand"],
                                "glide_score": manifest_row["glide_score"],
                                "pose_rank": None,
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
            "pose_hash",
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
        combined_pose_df = pd.concat([existing_manifest_df, new_pose_df], ignore_index=True)
    else:
        combined_pose_df = existing_manifest_df.copy()

    if not combined_pose_df.empty:
        combined_pose_df = _annotate_pose_ranks(combined_pose_df)
        assert_unique(combined_pose_df, ["pose_id"], "pose_manifest")

    write_manifest(combined_pose_df, pose_manifest_csv)

    if not duplicate_df.empty:
        pose_ranks = combined_pose_df.set_index("pose_id")["pose_rank"].to_dict()
        duplicate_df["pose_rank"] = duplicate_df["pose_id"].map(pose_ranks)
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
