#!/usr/bin/env python3
"""Build indexed frozen ESM protein embeddings as a dense .npy matrix.

Rows in the output matrix correspond exactly to `protein_idx` values in the manifest:

    embedding_matrix[i] == mean-pooled ESM embedding for the sequence
                           whose protein_idx is i

This is intended for downstream model training where the protein sequence
manifest already contains a stable integer `protein_idx` column.

Example:
    python -m data_processing.protein_embeddings.esm_embedding

Downstream loading:
    import numpy as np
    embs = np.load("processed/esm_embeddings.float32.npy", mmap_mode="r")
    protein_emb = embs[protein_idx]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    candidates = [
        Path.cwd(),
        *Path.cwd().parents,
        Path(__file__).resolve().parent,
        *Path(__file__).resolve().parents,
    ]
    for candidate in candidates:
        if (candidate / "data_processing" / "common" / "constants.py").exists():
            sys.path.append(str(candidate))
            break

from data_processing.common.constants import (
    DEFAULT_ESM_BATCH_SIZE,
    DEFAULT_ESM_DTYPE,
    DEFAULT_ESM_MODEL_NAME,
    DEFAULT_ESM_REPR_LAYER,
    DEFAULT_ESM_TRUNCATE_TO,
    ESM_EMBEDDINGS_NPY,
    PROTEIN_SEQUENCE_MANIFEST_COLUMNS,
    PROTEIN_SEQUENCE_MANIFEST_CSV,
)
from data_processing.common.manifest_io import ensure_parent_dir, read_csv_checked


LOGGER = logging.getLogger(__name__)
DEFAULT_IDX_COLUMN = PROTEIN_SEQUENCE_MANIFEST_COLUMNS[0]
DEFAULT_SEQUENCE_COLUMN = PROTEIN_SEQUENCE_MANIFEST_COLUMNS[1]


def validate_sequence_manifest(
    df: pd.DataFrame,
    sequence_column: str = DEFAULT_SEQUENCE_COLUMN,
    idx_column: str = DEFAULT_IDX_COLUMN,
) -> pd.DataFrame:
    """Return validated protein rows sorted by `protein_idx`."""
    slim = df[[idx_column, sequence_column]].copy()
    slim = slim.dropna(subset=[idx_column, sequence_column])
    slim[sequence_column] = slim[sequence_column].astype(str).str.strip()
    slim = slim[slim[sequence_column] != ""]

    if slim.empty:
        raise ValueError("No valid protein rows found after dropping empty sequences/indices.")

    idx_numeric = pd.to_numeric(slim[idx_column], errors="raise")
    if not np.all(np.equal(idx_numeric, np.floor(idx_numeric))):
        bad = slim.loc[~np.equal(idx_numeric, np.floor(idx_numeric)), idx_column].head().tolist()
        raise ValueError(f"Non-integer protein_idx values found, examples: {bad}")
    slim[idx_column] = idx_numeric.astype(int)

    if (slim[idx_column] < 0).any():
        bad = slim.loc[slim[idx_column] < 0, idx_column].head().tolist()
        raise ValueError(f"Negative protein_idx values found, examples: {bad}")

    n_sequences_per_idx = slim.groupby(idx_column)[sequence_column].nunique(dropna=True)
    bad_idx = n_sequences_per_idx[n_sequences_per_idx > 1]
    if not bad_idx.empty:
        examples = bad_idx.head(10).index.tolist()
        raise ValueError(f"Some protein_idx values map to multiple sequences, examples: {examples}")

    return (
        slim.drop_duplicates(subset=[idx_column])
        .sort_values(idx_column)
        .reset_index(drop=True)
    )


def load_esm_model(model_name: str):
    """Load an ESM model by name from the fair-esm package."""
    try:
        import esm
    except ImportError as exc:
        raise ImportError(
            "Missing dependency `esm`. Install with: pip install fair-esm"
        ) from exc

    if not hasattr(esm.pretrained, model_name):
        available = sorted(name for name in dir(esm.pretrained) if name.startswith("esm"))
        raise ValueError(
            f"Unknown ESM model {model_name!r}. Available pretrained functions include: {available}"
        )

    loader: Callable = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    return model, alphabet


def normalize_sequence(sequence: str) -> str:
    """Normalize sequence characters for ESM."""
    sequence = str(sequence).strip().upper()
    # ESM handles standard amino acids well. Map rare/ambiguous characters to X.
    valid = set("ACDEFGHIKLMNPQRSTVWYX")
    return "".join(char if char in valid else "X" for char in sequence)


def build_esm_embeddings(
    manifest: str | Path = PROTEIN_SEQUENCE_MANIFEST_CSV,
    embeddings_out: str | Path = ESM_EMBEDDINGS_NPY,
    sequence_column: str = DEFAULT_SEQUENCE_COLUMN,
    idx_column: str = DEFAULT_IDX_COLUMN,
    model_name: str = DEFAULT_ESM_MODEL_NAME,
    repr_layer: int = DEFAULT_ESM_REPR_LAYER,
    batch_size: int = DEFAULT_ESM_BATCH_SIZE,
    device: str | None = None,
    dtype: str = DEFAULT_ESM_DTYPE,
    truncate_to: int | None = DEFAULT_ESM_TRUNCATE_TO,
) -> np.ndarray:
    """Generate mean-pooled ESM embeddings indexed by protein_idx."""
    manifest = Path(manifest)
    embeddings_out = Path(embeddings_out)

    df = read_csv_checked(manifest, [idx_column, sequence_column])
    df = validate_sequence_manifest(
        df,
        sequence_column=sequence_column,
        idx_column=idx_column,
    )
    df[sequence_column] = df[sequence_column].map(normalize_sequence)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, alphabet = load_esm_model(model_name)
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    max_idx = int(df[idx_column].max())
    embedding_dim = int(model.embed_dim)
    embeddings = np.full((max_idx + 1, embedding_dim), np.nan, dtype=np.dtype(dtype))

    rows = list(df[[idx_column, sequence_column]].itertuples(index=False, name=None))

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            labels: list[str] = []
            sequences: list[str] = []
            lengths: list[int] = []
            protein_indices: list[int] = []

            for protein_idx_raw, sequence_raw in batch_rows:
                protein_idx = int(protein_idx_raw)
                sequence = str(sequence_raw)
                if truncate_to is not None and len(sequence) > truncate_to:
                    LOGGER.warning(
                        "Truncating protein_idx=%d from length %d to %d residues",
                        protein_idx,
                        len(sequence),
                        truncate_to,
                    )
                    sequence = sequence[:truncate_to]

                labels.append(str(protein_idx))
                sequences.append(sequence)
                lengths.append(len(sequence))
                protein_indices.append(protein_idx)

            _, _, tokens = batch_converter(list(zip(labels, sequences)))
            tokens = tokens.to(device)

            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = results["representations"][repr_layer]

            for batch_idx, protein_idx in enumerate(protein_indices):
                seq_len = lengths[batch_idx]
                # Skip BOS token at 0 and EOS token after sequence.
                pooled = token_representations[batch_idx, 1 : seq_len + 1].mean(dim=0)
                embeddings[protein_idx] = pooled.detach().cpu().numpy().astype(dtype, copy=False)

            LOGGER.info("Embedded %d/%d proteins", min(start + batch_size, len(rows)), len(rows))

    missing = np.isnan(embeddings).all(axis=1)
    if missing.any():
        missing_indices = np.where(missing)[0].tolist()
        LOGGER.warning(
            "Output contains all-NaN rows for missing protein_idx values: %s",
            missing_indices[:20],
        )

    np.save(ensure_parent_dir(embeddings_out), embeddings)
    LOGGER.info("Wrote embeddings with shape %s to %s", embeddings.shape, embeddings_out)
    return embeddings


def parse_args() -> argparse.Namespace:
    """CLI for generating frozen ESM embeddings."""
    parser = argparse.ArgumentParser(
        description="Generate frozen ESM protein embeddings as an indexed .npy matrix."
    )
    parser.add_argument(
        "--manifest",
        "--input",
        dest="manifest",
        type=Path,
        default=PROTEIN_SEQUENCE_MANIFEST_CSV,
        help="CSV manifest containing protein_idx and sequence columns.",
    )
    parser.add_argument(
        "--sequence-column",
        default=DEFAULT_SEQUENCE_COLUMN,
        help=f"Column containing protein sequences. Default: {DEFAULT_SEQUENCE_COLUMN}",
    )
    parser.add_argument(
        "--idx-column",
        default=DEFAULT_IDX_COLUMN,
        help=f"Column containing integer protein indices. Default: {DEFAULT_IDX_COLUMN}",
    )
    parser.add_argument(
        "--embeddings-out",
        "--output",
        dest="embeddings_out",
        type=Path,
        default=ESM_EMBEDDINGS_NPY,
        help="Output dense embedding matrix path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ESM_MODEL_NAME,
        help="ESM pretrained model loader name.",
    )
    parser.add_argument(
        "--repr-layer",
        type=int,
        default=DEFAULT_ESM_REPR_LAYER,
        help="ESM representation layer.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_ESM_BATCH_SIZE)
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit for auto")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default=DEFAULT_ESM_DTYPE,
    )
    parser.add_argument(
        "--truncate-to",
        type=int,
        default=DEFAULT_ESM_TRUNCATE_TO,
        help="Max residues per sequence for ESM-2 positional limit; use 0 to disable",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )
    truncate_to = None if args.truncate_to == 0 else args.truncate_to
    build_esm_embeddings(
        manifest=args.manifest,
        embeddings_out=args.embeddings_out,
        sequence_column=args.sequence_column,
        idx_column=args.idx_column,
        model_name=args.model,
        repr_layer=args.repr_layer,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        truncate_to=truncate_to,
    )


if __name__ == "__main__":
    main()
