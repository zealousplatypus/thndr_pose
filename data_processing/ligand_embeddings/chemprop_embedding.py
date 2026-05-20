#!/usr/bin/env python3
"""Build indexed frozen Chemprop ligand embeddings as a dense .npy matrix.

Rows in the output matrix correspond exactly to `ligand_idx` values in the manifest:

    embedding_matrix[i] == Chemprop fingerprint for the unique canonical SMILES
                           whose ligand_idx is i

This is intended for downstream model training where the split manifest already
contains a stable integer `ligand_idx` column.

Example:
    python -m data_processing.ligand_embeddings.chemprop_embedding \
        --model-paths checkpoints/best.ckpt \
        --ffn-block-index 0 \
        --overwrite

Downstream loading:
    import numpy as np
    embs = np.load("processed/chemprop_embeddings.float32.npy", mmap_mode="r")
    ligand_emb = embs[ligand_idx]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from data_processing.common.constants import (
        CHEMPROP_EMBEDDINGS_METADATA_JSON,
        CHEMPROP_EMBEDDINGS_NPY,
        CHEMPROP_SMILES_BY_IDX_JSON,
        SPLIT_MANIFEST_CSV,
    )
    from data_processing.common.manifest_io import ensure_parent_dir, read_csv_checked
else:
    from ..common.constants import (
        CHEMPROP_EMBEDDINGS_METADATA_JSON,
        CHEMPROP_EMBEDDINGS_NPY,
        CHEMPROP_SMILES_BY_IDX_JSON,
        SPLIT_MANIFEST_CSV,
    )
    from ..common.manifest_io import ensure_parent_dir, read_csv_checked

LOGGER = logging.getLogger(__name__)

DEFAULT_SMILES_COLUMN = "ligand"
DEFAULT_IDX_COLUMN = "ligand_idx"
DEFAULT_FFN_BLOCK_INDEX = 0
DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 0
DEFAULT_ACCELERATOR = "auto"
DEFAULT_DEVICES = "auto"
DEFAULT_DTYPE = "float32"
DEFAULT_ENSEMBLE_MODE = "mean"
CHEMPROP_TMP_PREFIX = "chemprop_npy_"
FINGERPRINT_EXTENSIONS = (".npz", ".npy", ".csv")


def build_smiles_by_idx(
    manifest: str | Path = SPLIT_MANIFEST_CSV,
    smiles_column: str = DEFAULT_SMILES_COLUMN,
    idx_column: str = DEFAULT_IDX_COLUMN,
) -> list[str]:
    """Return canonical SMILES ordered by contiguous `ligand_idx`."""
    df = read_csv_checked(manifest, [smiles_column, idx_column])
    slim = df[[smiles_column, idx_column]].copy()
    slim = slim.dropna(subset=[smiles_column, idx_column])
    slim[smiles_column] = slim[smiles_column].astype(str).str.strip()
    slim = slim[slim[smiles_column] != ""]

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
    path = ensure_parent_dir(path)
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

    LOGGER.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if output_npz.exists():
        return [output_npz]

    # Chemprop may append model indices to the stem when multiple models are used.
    candidates: list[Path] = []
    for extension in FINGERPRINT_EXTENSIONS:
        candidates.extend(sorted(output_npz.parent.glob(f"{output_npz.stem}*{extension}")))
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


def check_can_write(paths: list[str | Path], overwrite: bool) -> None:
    existing = [Path(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output file(s) already exist: "
            + ", ".join(str(p) for p in existing)
            + ". Pass --overwrite to replace them."
        )


def _resolve_reused_fingerprint_paths(path: str | Path) -> list[Path]:
    """Return fingerprint files from a supplied file or directory."""
    path = Path(path)
    if path.is_dir():
        fingerprint_paths: list[Path] = []
        for extension in FINGERPRINT_EXTENSIONS:
            fingerprint_paths.extend(sorted(path.glob(f"*{extension}")))
        if not fingerprint_paths:
            raise FileNotFoundError(f"No .npz/.npy/.csv fingerprint files found in {path}")
        return fingerprint_paths
    return [path]


def build_chemprop_embeddings(
    manifest: str | Path = SPLIT_MANIFEST_CSV,
    model_paths: list[str] | tuple[str, ...] | None = None,
    embeddings_out: str | Path = CHEMPROP_EMBEDDINGS_NPY,
    smiles_out: str | Path = CHEMPROP_SMILES_BY_IDX_JSON,
    metadata_out: str | Path = CHEMPROP_EMBEDDINGS_METADATA_JSON,
    smiles_column: str = DEFAULT_SMILES_COLUMN,
    idx_column: str = DEFAULT_IDX_COLUMN,
    tmp_dir: str | Path | None = None,
    keep_tmp: bool = False,
    reuse_fingerprints: str | Path | None = None,
    ffn_block_index: int = DEFAULT_FFN_BLOCK_INDEX,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    accelerator: str = DEFAULT_ACCELERATOR,
    devices: str = DEFAULT_DEVICES,
    dtype: str = DEFAULT_DTYPE,
    ensemble_mode: str = DEFAULT_ENSEMBLE_MODE,
    overwrite: bool = False,
    extra_chemprop_args: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    """Build and persist a dense Chemprop embedding matrix indexed by ligand_idx."""
    model_paths = list(model_paths or [])
    extra_chemprop_args = list(extra_chemprop_args or [])
    manifest = Path(manifest)
    embeddings_out = Path(embeddings_out)
    smiles_out = Path(smiles_out)
    metadata_out = Path(metadata_out)

    if reuse_fingerprints is None and not model_paths:
        raise ValueError("model_paths is required unless reuse_fingerprints is provided.")

    check_can_write([embeddings_out, smiles_out, metadata_out], overwrite)

    smiles_by_idx = build_smiles_by_idx(
        manifest=manifest,
        smiles_column=smiles_column,
        idx_column=idx_column,
    )
    LOGGER.info("Found %d unique ligand indices.", len(smiles_by_idx))

    tmp_ctx = None
    if tmp_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix=CHEMPROP_TMP_PREFIX)
        work_dir = Path(tmp_ctx.name)
    else:
        work_dir = Path(tmp_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        if reuse_fingerprints is None:
            smiles_csv = work_dir / "chemprop_smiles_by_ligand_idx.csv"
            fingerprint_out = work_dir / "chemprop_fingerprints.npz"
            write_smiles_csv(smiles_by_idx, smiles_csv)
            fingerprint_paths = run_chemprop_fingerprint(
                smiles_csv=smiles_csv,
                output_npz=fingerprint_out,
                model_paths=model_paths,
                ffn_block_index=ffn_block_index,
                batch_size=batch_size,
                num_workers=num_workers,
                accelerator=accelerator,
                devices=devices,
                extra_args=extra_chemprop_args,
            )
        else:
            fingerprint_paths = _resolve_reused_fingerprint_paths(reuse_fingerprints)

        arrays = [load_fingerprint_file(path) for path in fingerprint_paths]
        embeddings = combine_fingerprint_arrays(arrays, ensemble_mode).astype(dtype, copy=False)

        if embeddings.shape[0] != len(smiles_by_idx):
            raise ValueError(
                f"Fingerprint row count {embeddings.shape[0]:,} does not match number of ligand indices "
                f"{len(smiles_by_idx):,}. The Chemprop input order and manifest index order may be mismatched."
            )

        np.save(ensure_parent_dir(embeddings_out), embeddings)

        ensure_parent_dir(smiles_out).write_text(json.dumps(smiles_by_idx, indent=2) + "\n")

        metadata = {
            "manifest": str(manifest),
            "smiles_column": smiles_column,
            "idx_column": idx_column,
            "model_paths": model_paths,
            "ffn_block_index": ffn_block_index,
            "dtype": str(embeddings.dtype),
            "shape": list(embeddings.shape),
            "ensemble_mode": ensemble_mode,
            "fingerprint_files": [str(path) for path in fingerprint_paths],
            "row_contract": "embedding_matrix[ligand_idx] corresponds to smiles_by_idx[ligand_idx]",
        }
        ensure_parent_dir(metadata_out).write_text(json.dumps(metadata, indent=2) + "\n")

        LOGGER.info(
            "Wrote embeddings: %s shape=%s dtype=%s",
            embeddings_out,
            embeddings.shape,
            embeddings.dtype,
        )
        LOGGER.info("Wrote SMILES index: %s", smiles_out)
        LOGGER.info("Wrote metadata: %s", metadata_out)
        return embeddings
    finally:
        if tmp_ctx is not None and keep_tmp:
            LOGGER.info("Temporary files kept at: %s", work_dir)
            tmp_ctx = None  # intentionally leak so TemporaryDirectory cleanup does not run
        elif tmp_ctx is not None:
            tmp_ctx.cleanup()


def parse_args() -> argparse.Namespace:
    """CLI for generating frozen Chemprop embeddings."""
    parser = argparse.ArgumentParser(
        description="Generate frozen Chemprop embeddings as an indexed .npy matrix."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SPLIT_MANIFEST_CSV,
        help="CSV manifest containing canonical SMILES and ligand_idx.",
    )
    parser.add_argument(
        "--smiles-column",
        default=DEFAULT_SMILES_COLUMN,
        help=f"Column containing canonical SMILES. Default: {DEFAULT_SMILES_COLUMN}",
    )
    parser.add_argument(
        "--idx-column",
        default=DEFAULT_IDX_COLUMN,
        help=f"Column containing integer ligand indices. Default: {DEFAULT_IDX_COLUMN}",
    )
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[],
        help="Chemprop checkpoint/model path(s), passed to `chemprop fingerprint --model-paths`.",
    )
    parser.add_argument(
        "--embeddings-out",
        type=Path,
        default=CHEMPROP_EMBEDDINGS_NPY,
        help="Output dense embedding matrix path.",
    )
    parser.add_argument(
        "--smiles-out",
        type=Path,
        default=CHEMPROP_SMILES_BY_IDX_JSON,
        help="Output JSON list where smiles_by_idx[i] is the SMILES for ligand_idx i.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=CHEMPROP_EMBEDDINGS_METADATA_JSON,
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
        default=DEFAULT_FFN_BLOCK_INDEX,
        help="Chemprop encoding layer. 0 is post-aggregation graph representation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size passed to Chemprop.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of dataloader workers passed to Chemprop.",
    )
    parser.add_argument(
        "--accelerator",
        default=DEFAULT_ACCELERATOR,
        help="Lightning accelerator passed to Chemprop, e.g. auto/cpu/gpu.",
    )
    parser.add_argument(
        "--devices",
        default=DEFAULT_DEVICES,
        help="Lightning devices passed to Chemprop.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "float64"],
        default=DEFAULT_DTYPE,
        help="Output embedding dtype.",
    )
    parser.add_argument(
        "--ensemble-mode",
        choices=["mean", "concat", "error"],
        default=DEFAULT_ENSEMBLE_MODE,
        help=(
            "If Chemprop writes one fingerprint file per model, combine them by mean, concat, "
            "or error."
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


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    build_chemprop_embeddings(
        manifest=args.manifest,
        model_paths=args.model_paths,
        embeddings_out=args.embeddings_out,
        smiles_out=args.smiles_out,
        metadata_out=args.metadata_out,
        smiles_column=args.smiles_column,
        idx_column=args.idx_column,
        tmp_dir=args.tmp_dir,
        keep_tmp=args.keep_tmp,
        reuse_fingerprints=args.reuse_fingerprints,
        ffn_block_index=args.ffn_block_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        devices=args.devices,
        dtype=args.dtype,
        ensemble_mode=args.ensemble_mode,
        overwrite=args.overwrite,
        extra_chemprop_args=args.extra_chemprop_args,
    )


if __name__ == "__main__":
    main()
