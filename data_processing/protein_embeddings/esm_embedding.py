#!/usr/bin/env python3
"""Build a baseline ESMC embedding matrix from a CSV of protein sequences.

The input CSV must contain:
    - protein_idx: integer row index in the output matrix
    - sequence: protein sequence

The script enforces that protein_idx values are exactly 0..N-1, so:
    embeddings[i] == ESMC embedding for the CSV row whose protein_idx == i

Example:
    python -m data_processing.protein_embeddings.esm_embedding \
        --input processed/protein_sequence_manifest.csv \
        --output processed/esmc_embeddings.float32.npy
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from ..common.constants import (
    DEFAULT_ESM_BATCH_SIZE,
    DEFAULT_ESM_DTYPE,
    DEFAULT_ESM_MODEL_NAME,
    DEFAULT_ESM_TRUNCATE_TO,
    ESM_EMBEDDINGS_NPY,
    PROTEIN_SEQUENCE_MANIFEST_COLUMNS,
    PROTEIN_SEQUENCE_MANIFEST_CSV,
)
from ..common.manifest_io import ensure_parent_dir, read_csv_checked


LOGGER = logging.getLogger(__name__)

DEFAULT_IDX_COLUMN = PROTEIN_SEQUENCE_MANIFEST_COLUMNS[0]
DEFAULT_SEQUENCE_COLUMN = PROTEIN_SEQUENCE_MANIFEST_COLUMNS[1]


def read_sequence_csv(path: str | Path) -> pd.DataFrame:
    """Read sequences and return rows sorted by protein_idx."""
    df = read_csv_checked(path, PROTEIN_SEQUENCE_MANIFEST_COLUMNS)
    df = df.loc[:, PROTEIN_SEQUENCE_MANIFEST_COLUMNS].copy()

    if df.isna().any().any():
        raise ValueError("protein_idx and sequence columns cannot contain missing values")

    protein_idx = pd.to_numeric(df[DEFAULT_IDX_COLUMN], errors="raise")

    if not np.array_equal(protein_idx, protein_idx.astype(int)):
        raise ValueError("protein_idx values must be integers")

    df[DEFAULT_IDX_COLUMN] = protein_idx.astype(int)
    df[DEFAULT_SEQUENCE_COLUMN] = (
        df[DEFAULT_SEQUENCE_COLUMN]
        .astype(str)
        .str.strip()
    )

    if df[DEFAULT_SEQUENCE_COLUMN].eq("").any():
        raise ValueError("sequence column contains empty strings")

    if df[DEFAULT_IDX_COLUMN].duplicated().any():
        raise ValueError("protein_idx values must be unique")

    df = df.sort_values(DEFAULT_IDX_COLUMN, ignore_index=True)

    expected_idx = np.arange(len(df))

    if not np.array_equal(
        df[DEFAULT_IDX_COLUMN].to_numpy(),
        expected_idx,
    ):
        raise ValueError(
            "protein_idx must be contiguous and zero-based: 0..N-1"
        )

    return df


def load_esmc_model(model_name: str, device: str) -> ESMC:
    """Load an ESMC model by name."""
    model = ESMC.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model


def normalize_sequence(sequence: str) -> str:
    """Normalize sequence characters for ESMC."""
    sequence = str(sequence).strip().upper()

    # Map ambiguous/rare residues to X.
    valid = set("ACDEFGHIKLMNPQRSTVWYX")

    return "".join(
        char if char in valid else "X"
        for char in sequence
    )


def build_esm_embeddings(
    input_csv: str | Path = PROTEIN_SEQUENCE_MANIFEST_CSV,
    output_npy: str | Path = ESM_EMBEDDINGS_NPY,
    model_name: str = DEFAULT_ESM_MODEL_NAME,
    batch_size: int = DEFAULT_ESM_BATCH_SIZE,
    device: str | None = None,
    dtype: str = DEFAULT_ESM_DTYPE,
    truncate_to: int | None = DEFAULT_ESM_TRUNCATE_TO,
) -> np.ndarray:
    """Generate mean-pooled ESMC embeddings indexed by protein_idx."""

    # NOTE:
    # ESMC SDK currently does not expose the same efficient
    # batched tokenization API as old ESM2, so this implementation
    # processes proteins sequentially for simplicity and correctness.

    df = read_sequence_csv(input_csv)

    df[DEFAULT_SEQUENCE_COLUMN] = (
        df[DEFAULT_SEQUENCE_COLUMN]
        .map(normalize_sequence)
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_esmc_model(model_name, device)

    embeddings: np.ndarray | None = None

    rows = list(
        df[
            [DEFAULT_IDX_COLUMN, DEFAULT_SEQUENCE_COLUMN]
        ].itertuples(index=False, name=None)
    )

    with torch.no_grad():

        for row_idx, (protein_idx_raw, sequence_raw) in enumerate(rows):

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

            protein = ESMProtein(sequence=sequence)

            protein_tensor = model.encode(protein)

            output = model.logits(
                protein_tensor,
                LogitsConfig(
                    sequence=True,
                    return_embeddings=True,
                ),
            )

            token_embeddings = output.embeddings

            if not isinstance(token_embeddings, torch.Tensor):
                token_embeddings = torch.tensor(token_embeddings)

            # ESMC may return either:
            #   (seq_len, hidden_dim)
            # or
            #   (1, seq_len, hidden_dim)
            if token_embeddings.ndim == 3:
                token_embeddings = token_embeddings.squeeze(0)

            if token_embeddings.ndim != 2:
                raise ValueError(
                    f"Expected ESMC embeddings to have shape "
                    f"(seq_len, hidden_dim) or (1, seq_len, hidden_dim), "
                    f"got {tuple(token_embeddings.shape)}"
                )

            # Mean pool across residues/tokens.
            pooled = token_embeddings.mean(dim=0)

            pooled_np = (
                pooled.detach()
                .cpu()
                .numpy()
                .astype(dtype, copy=False)
            )

            # Lazy initialize embedding matrix once embedding_dim is known.
            if embeddings is None:

                embedding_dim = int(pooled_np.shape[0])

                embeddings = np.empty(
                    (len(df), embedding_dim),
                    dtype=np.dtype(dtype),
                )

            embeddings[protein_idx] = pooled_np

            LOGGER.info(
                "Embedded %d/%d proteins",
                row_idx + 1,
                len(rows),
            )

    assert embeddings is not None

    output_npy = ensure_parent_dir(output_npy)

    np.save(output_npy, embeddings)

    LOGGER.info(
        "Wrote embeddings with shape %s to %s",
        embeddings.shape,
        output_npy,
    )

    return embeddings


def parse_args() -> argparse.Namespace:
    """CLI for generating frozen ESMC embeddings."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen ESMC protein embeddings "
            "as an indexed .npy matrix."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PROTEIN_SEQUENCE_MANIFEST_CSV,
        help="CSV manifest containing protein_idx and sequence columns.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ESM_EMBEDDINGS_NPY,
        help="Output dense embedding matrix path.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_ESM_MODEL_NAME,
        help="ESMC pretrained model name.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_ESM_BATCH_SIZE,
        help="Unused currently; reserved for future batching support.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="cuda, cpu, or omit for auto",
    )

    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default=DEFAULT_ESM_DTYPE,
    )

    parser.add_argument(
        "--truncate-to",
        type=int,
        default=DEFAULT_ESM_TRUNCATE_TO,
        help=(
            "Maximum residues per sequence; "
            "use 0 to disable truncation."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )

    truncate_to = (
        None if args.truncate_to == 0
        else args.truncate_to
    )

    build_esm_embeddings(
        input_csv=args.input,
        output_npy=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        truncate_to=truncate_to,
    )


if __name__ == "__main__":
    main()