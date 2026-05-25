"""Generate frozen ESM protein embeddings from a protein sequence manifest.

Input CSV must contain:
    - protein_idx
    - sequence

Output:
    - .npy array whose row i is the embedding for protein_idx == i

By default this uses mean-pooled per-residue representations from ESM-2.
For a faster/lighter baseline, use:
    --model esm2_t12_35M_UR50D --repr-layer 12

Example:
    python esm_embedding.py \
        --input protein_sequence_manifest.csv \
        --output esm_embeddings.npy \
        --model esm2_t33_650M_UR50D \
        --repr-layer 33 \
        --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch


LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = ("protein_idx", "sequence")
DEFAULT_MODEL_NAME = "esm2_t33_650M_UR50D"
DEFAULT_REPR_LAYER = 33

# TODO: make this standardized with the other manifest validation functions
def validate_sequence_manifest(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input manifest is missing required columns: {missing_columns}")

    if df["protein_idx"].isna().any():
        raise ValueError("Input manifest contains null protein_idx values")

    if df["protein_idx"].duplicated().any():
        duplicates = df.loc[df["protein_idx"].duplicated(), "protein_idx"].tolist()
        raise ValueError(f"protein_idx values must be unique. Duplicates: {duplicates[:10]}")

    if df["sequence"].isna().any() or (df["sequence"].astype(str).str.len() == 0).any():
        raise ValueError("Input manifest contains missing/empty sequence values")

    if (df["protein_idx"].astype(int) < 0).any():
        raise ValueError("protein_idx must be nonnegative")


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

# TODO: Does this handle ESM correctly?
def embed_sequences(
    input_csv: str | Path,
    output_npy: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    repr_layer: int = DEFAULT_REPR_LAYER,
    batch_size: int = 4,
    device: str | None = None,
    dtype: str = "float32",
    truncate_to: int | None = 1022,
) -> np.ndarray:
    """Generate mean-pooled ESM embeddings indexed by protein_idx."""
    input_path = Path(input_csv)
    output_path = Path(output_npy)

    df = pd.read_csv(input_path)
    validate_sequence_manifest(df)

    df = df.copy()
    df["protein_idx"] = df["protein_idx"].astype(int)
    df["sequence"] = df["sequence"].map(normalize_sequence)
    df = df.sort_values("protein_idx")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, alphabet = load_esm_model(model_name)
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_idx = int(df["protein_idx"].max())
    embedding_dim = int(model.embed_dim)
    embeddings = np.full((max_idx + 1, embedding_dim), np.nan, dtype=np.dtype(dtype))

    rows = list(df.itertuples(index=False))

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            labels: list[str] = []
            sequences: list[str] = []
            lengths: list[int] = []
            protein_indices: list[int] = []

            for row in batch_rows:
                protein_idx = int(row.protein_idx)
                sequence = str(row.sequence)
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

            # TODO: really I just need to scan each of these lines and make sure they make sense
            _, _, tokens = batch_converter(list(zip(labels, sequences)))
            tokens = tokens.to(device)

            results = model(tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = results["representations"][repr_layer]

            for batch_idx, protein_idx in enumerate(protein_indices):
                seq_len = lengths[batch_idx]
                # Skip BOS token at 0 and EOS token after sequence.
                pooled = token_representations[batch_idx, 1 : seq_len + 1].mean(dim=0)
                embeddings[protein_idx] = pooled.detach().cpu().numpy().astype(dtype, copy=False)
            # TODO: everything below this is standard

            LOGGER.info("Embedded %d/%d proteins", min(start + batch_size, len(rows)), len(rows))

    missing = np.isnan(embeddings).all(axis=1)
    if missing.any():
        missing_indices = np.where(missing)[0].tolist()
        LOGGER.warning(
            "Output contains all-NaN rows for missing protein_idx values: %s",
            missing_indices[:20],
        )

    np.save(output_path, embeddings)
    LOGGER.info("Wrote embeddings with shape %s to %s", embeddings.shape, output_path)
    return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="protein_sequence_manifest.csv",
        help="Input CSV with protein_idx and sequence columns",
    )
    parser.add_argument("--output", default="esm_embeddings.npy", help="Output .npy path")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="ESM pretrained model loader name")
    parser.add_argument("--repr-layer", type=int, default=DEFAULT_REPR_LAYER, help="ESM representation layer")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit for auto")
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"), default="float32")
    parser.add_argument(
        "--truncate-to",
        type=int,
        default=1022,
        help="Max residues per sequence for ESM-2 positional limit; use 0 to disable",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    truncate_to = None if args.truncate_to == 0 else args.truncate_to
    embed_sequences(
        input_csv=args.input,
        output_npy=args.output,
        model_name=args.model,
        repr_layer=args.repr_layer,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        truncate_to=truncate_to,
    )


if __name__ == "__main__":
    main()
