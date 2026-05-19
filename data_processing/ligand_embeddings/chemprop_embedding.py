#!/usr/bin/env python3
"""Build indexed frozen Chemprop ligand embeddings as a dense .npy matrix.

Rows in the output matrix correspond exactly to `ligand_idx` values in the manifest:

    embedding_matrix[i] == Chemprop fingerprint for the unique canonical SMILES
                           whose ligand_idx is i

This is intended for downstream model training where the affinity manifest already
contains a stable integer `ligand_idx` column.

Example:
    python build_chemprop_npy.py \
        --manifest affinity_split_manifest.csv \
        --model-paths checkpoints/best.ckpt \
        --embeddings-out chemprop_embeddings.float32.npy \
        --smiles-out chemprop_smiles_by_idx.json \
        --ffn-block-index 0 \
        --overwrite

Downstream loading:
    import numpy as np
    embs = np.load("chemprop_embeddings.float32.npy", mmap_mode="r")
    ligand_emb = embs[ligand_idx]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen Chemprop embeddings as an indexed .npy matrix."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("affinity_split_manifest.csv"),
        help="CSV manifest containing canonical SMILES and ligand_idx.",
    )
    parser.add_argument(
        "--smiles-column",
        default="ligand",
        help="Column containing canonical SMILES. Default: ligand",
    )
    parser.add_argument(
        "--idx-column",
        default="ligand_idx",
        help="Column containing integer ligand indices. Default: ligand_idx",
    )
    parser.add_argument(
        "--model-paths",
        nargs="+",
        required=True,
        help="Chemprop checkpoint/model path(s), passed to `chemprop fingerprint --model-paths`.",
    )
    parser.add_argument(
        "--embeddings-out",
        type=Path,
        default=Path("chemprop_embeddings.float32.npy"),
        help="Output dense embedding matrix path. Default: chemprop_embeddings.float32.npy",
    )
    parser.add_argument(
        "--smiles-out",
        type=Path,
        default=Path("chemprop_smiles_by_idx.json"),
        help="Output JSON list where smiles_by_idx[i] is the SMILES for ligand_idx i.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("chemprop_embeddings_metadata.json"),
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Optional working directory for Chemprop input/output files.",
    )
    parser.add_argument(
        "--keep-tmp",
        action="store_true",
        help="Keep temporary Chemprop input/output files.",
    )
    parser.add_argument(
        "--reuse-fingerprints",
        type=Path,
        default=None,
        help="Skip Chemprop and reuse an existing .npz/.npy/.csv fingerprint file.",
    )
    parser.add_argument(
        "--ffn-block-index",
        type=int,
        default=0,
        help="Chemprop encoding layer. 0 is post-aggregation graph representation. Default: 0",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size passed to Chemprop. Default: 256",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers passed to Chemprop. Default: 0",
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="Lightning accelerator passed to Chemprop, e.g. auto/cpu/gpu. Default: auto",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Lightning devices passed to Chemprop. Default: auto",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
        help="Output embedding dtype. Default: float32",
    )
    parser.add_argument(
        "--ensemble-mode",
        choices=["mean", "concat", "error"],
        default="mean",
        help=(
            "If Chemprop writes one fingerprint file per model, combine them by mean, concat, "
            "or error. Default: mean."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output files.",
    )
    parser.add_argument(
        "--extra-chemprop-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args appended to `chemprop fingerprint`. Place this last.",
    )
    return parser.parse_args()


def build_smiles_by_idx(manifest: Path, smiles_column: str, idx_column: str) -> list[str]:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    df = pd.read_csv(manifest)
    missing = [col for col in [smiles_column, idx_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available columns: {list(df.columns)}")

    slim = df[[smiles_column, idx_column]].copy()
    slim[smiles_column] = slim[smiles_column].astype(str).str.strip()
    slim = slim[slim[smiles_column] != ""]
    slim = slim.dropna(subset=[smiles_column, idx_column])

    if slim.empty:
        raise ValueError("No valid ligand rows found after dropping empty SMILES/indices.")

    # Validate integer-like ligand_idx values.
    idx_numeric = pd.to_numeric(slim[idx_column], errors="raise")
    if not np.all(np.equal(idx_numeric, np.floor(idx_numeric))):
        bad = slim.loc[~np.equal(idx_numeric, np.floor(idx_numeric)), idx_column].head().tolist()
        raise ValueError(f"Non-integer ligand_idx values found, examples: {bad}")
    slim[idx_column] = idx_numeric.astype(int)

    if (slim[idx_column] < 0).any():
        bad = slim.loc[slim[idx_column] < 0, idx_column].head().tolist()
        raise ValueError(f"Negative ligand_idx values found, examples: {bad}")

    # Each ligand_idx must map to exactly one canonical SMILES.
    n_smiles_per_idx = slim.groupby(idx_column)[smiles_column].nunique(dropna=True)
    bad_idx = n_smiles_per_idx[n_smiles_per_idx > 1]
    if not bad_idx.empty:
        examples = bad_idx.head(10).index.tolist()
        raise ValueError(f"Some ligand_idx values map to multiple SMILES, examples: {examples}")

    # Each canonical SMILES should map to exactly one ligand_idx.
    n_idx_per_smiles = slim.groupby(smiles_column)[idx_column].nunique(dropna=True)
    bad_smiles = n_idx_per_smiles[n_idx_per_smiles > 1]
    if not bad_smiles.empty:
        examples = bad_smiles.head(10).index.tolist()
        raise ValueError(f"Some SMILES map to multiple ligand_idx values, examples: {examples}")

    idx_to_smiles = (
        slim.drop_duplicates(subset=[idx_column])
        .sort_values(idx_column)
        .set_index(idx_column)[smiles_column]
        .to_dict()
    )

    max_idx = max(idx_to_smiles)
    expected = set(range(max_idx + 1))
    missing_indices = sorted(expected - set(idx_to_smiles))
    if missing_indices:
        preview = missing_indices[:20]
        raise ValueError(
            "ligand_idx values must be contiguous from 0..max_idx so they can index a dense .npy. "
            f"Missing examples: {preview}"
        )

    return [idx_to_smiles[i] for i in range(max_idx + 1)]


def write_smiles_csv(smiles_by_idx: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for smiles in smiles_by_idx:
            writer.writerow([smiles])


def run_chemprop_fingerprint(
    smiles_csv: Path,
    output_npz: Path,
    model_paths: list[str],
    ffn_block_index: int,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: str,
    extra_args: list[str],
) -> list[Path]:
    if shutil.which("chemprop") is None:
        raise RuntimeError(
            "Could not find `chemprop` on PATH. Activate/install Chemprop first."
        )

    cmd = [
        "chemprop",
        "fingerprint",
        "--test-path",
        str(smiles_csv),
        "--smiles-columns",
        "smiles",
        "--model-paths",
        *model_paths,
        "--output",
        str(output_npz),
        "--ffn-block-index",
        str(ffn_block_index),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--accelerator",
        accelerator,
        "--devices",
        devices,
        *extra_args,
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    if output_npz.exists():
        return [output_npz]

    # Chemprop may append model indices to the stem when multiple models are used.
    candidates = sorted(output_npz.parent.glob(f"{output_npz.stem}*.npz"))
    if not candidates:
        # Also allow CSV fallback if Chemprop ignored .npz for some reason.
        candidates = sorted(output_npz.parent.glob(f"{output_npz.stem}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Chemprop finished but no fingerprint output found near {output_npz}")

    return candidates


def load_fingerprint_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            keys = list(data.keys())
            if len(keys) == 1:
                arr = data[keys[0]]
            elif "fps" in keys:
                arr = data["fps"]
            elif "fingerprints" in keys:
                arr = data["fingerprints"]
            else:
                # Pick the first 2D array. This handles small Chemprop format changes.
                arrays = [(k, data[k]) for k in keys if getattr(data[k], "ndim", None) == 2]
                if not arrays:
                    raise ValueError(f"No 2D fingerprint array found in {path}. Keys: {keys}")
                arr = arrays[0][1]
    elif suffix == ".csv":
        df = pd.read_csv(path)
        # Chemprop CSV fingerprint outputs can include SMILES/name columns; keep numeric cols only.
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError(f"No numeric fingerprint columns found in CSV: {path}")
        arr = numeric_df.to_numpy()
    else:
        raise ValueError(f"Unsupported fingerprint file extension: {path}")

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D fingerprint array from {path}, got shape {arr.shape}")
    return arr


def combine_fingerprint_arrays(arrays: list[np.ndarray], mode: str) -> np.ndarray:
    if len(arrays) == 1:
        return arrays[0]
    if mode == "error":
        shapes = [arr.shape for arr in arrays]
        raise ValueError(f"Multiple fingerprint arrays found with shapes {shapes}; set --ensemble-mode mean or concat.")
    if mode == "mean":
        first_shape = arrays[0].shape
        if any(arr.shape != first_shape for arr in arrays):
            shapes = [arr.shape for arr in arrays]
            raise ValueError(f"Cannot mean-combine arrays with different shapes: {shapes}")
        return np.mean(np.stack(arrays, axis=0), axis=0)
    if mode == "concat":
        n_rows = arrays[0].shape[0]
        if any(arr.shape[0] != n_rows for arr in arrays):
            shapes = [arr.shape for arr in arrays]
            raise ValueError(f"Cannot concat arrays with different row counts: {shapes}")
        return np.concatenate(arrays, axis=1)
    raise ValueError(f"Unknown ensemble mode: {mode}")


def check_can_write(paths: list[Path], overwrite: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output file(s) already exist: "
            + ", ".join(str(p) for p in existing)
            + ". Pass --overwrite to replace them."
        )


def main() -> None:
    args = parse_args()
    check_can_write([args.embeddings_out, args.smiles_out, args.metadata_out], args.overwrite)

    smiles_by_idx = build_smiles_by_idx(args.manifest, args.smiles_column, args.idx_column)
    print(f"Found {len(smiles_by_idx):,} unique ligand indices.", flush=True)

    tmp_ctx = None
    if args.tmp_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="chemprop_npy_")
        work_dir = Path(tmp_ctx.name)
    else:
        work_dir = args.tmp_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.reuse_fingerprints is None:
            smiles_csv = work_dir / "chemprop_smiles_by_ligand_idx.csv"
            fingerprint_out = work_dir / "chemprop_fingerprints.npz"
            write_smiles_csv(smiles_by_idx, smiles_csv)
            fingerprint_paths = run_chemprop_fingerprint(
                smiles_csv=smiles_csv,
                output_npz=fingerprint_out,
                model_paths=args.model_paths,
                ffn_block_index=args.ffn_block_index,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                accelerator=args.accelerator,
                devices=args.devices,
                extra_args=args.extra_chemprop_args,
            )
        else:
            path = args.reuse_fingerprints
            if path.is_dir():
                fingerprint_paths = sorted(path.glob("*.npz")) or sorted(path.glob("*.npy")) or sorted(path.glob("*.csv"))
                if not fingerprint_paths:
                    raise FileNotFoundError(f"No .npz/.npy/.csv fingerprint files found in {path}")
            else:
                fingerprint_paths = [path]

        arrays = [load_fingerprint_file(path) for path in fingerprint_paths]
        embeddings = combine_fingerprint_arrays(arrays, args.ensemble_mode).astype(args.dtype, copy=False)

        if embeddings.shape[0] != len(smiles_by_idx):
            raise ValueError(
                f"Fingerprint row count {embeddings.shape[0]:,} does not match number of ligand indices "
                f"{len(smiles_by_idx):,}. The Chemprop input order and manifest index order may be mismatched."
            )

        args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embeddings_out, embeddings)

        args.smiles_out.parent.mkdir(parents=True, exist_ok=True)
        args.smiles_out.write_text(json.dumps(smiles_by_idx, indent=2) + "\n")

        metadata = {
            "manifest": str(args.manifest),
            "smiles_column": args.smiles_column,
            "idx_column": args.idx_column,
            "model_paths": args.model_paths,
            "ffn_block_index": args.ffn_block_index,
            "dtype": str(embeddings.dtype),
            "shape": list(embeddings.shape),
            "ensemble_mode": args.ensemble_mode,
            "fingerprint_files": [str(path) for path in fingerprint_paths],
            "row_contract": "embedding_matrix[ligand_idx] corresponds to smiles_by_idx[ligand_idx]",
        }
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(json.dumps(metadata, indent=2) + "\n")

        print(f"Wrote embeddings: {args.embeddings_out} shape={embeddings.shape} dtype={embeddings.dtype}")
        print(f"Wrote SMILES index: {args.smiles_out}")
        print(f"Wrote metadata: {args.metadata_out}")
    finally:
        if tmp_ctx is not None and args.keep_tmp:
            print(f"Temporary files kept at: {work_dir}")
            tmp_ctx = None  # intentionally leak so TemporaryDirectory cleanup does not run
        elif tmp_ctx is not None:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
