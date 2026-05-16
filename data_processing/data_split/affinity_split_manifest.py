"""Attach ligand-level split labels to affinity examples."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..common.constants import (
    ACTIVE_SPLIT_NAMES,
    AFFINITY_MANIFEST_CSV,
    AFFINITY_SPLIT_MANIFEST_CSV,
    SPLIT_MANIFEST_CSV,
    AFFINITY_MANIFEST_COLUMNS,
    SPLIT_MANIFEST_COLUMNS
)
from ..common.manifest_io import assert_unique, read_csv_checked, write_manifest


LOGGER = logging.getLogger(__name__)


def build_affinity_split_manifest(
    affinity_csv: str | Path = AFFINITY_MANIFEST_CSV,
    split_csv: str | Path = SPLIT_MANIFEST_CSV,
) -> pd.DataFrame:
    """Join active split labels back onto affinity examples."""
    affinity_df = read_csv_checked(affinity_csv, AFFINITY_MANIFEST_COLUMNS)
    split_df = read_csv_checked(split_csv, SPLIT_MANIFEST_COLUMNS)
    split_df = split_df[split_df["split"].isin(ACTIVE_SPLIT_NAMES)].copy()

    affinity_split_df = affinity_df.merge(
        split_df[["ligand", "split"]],
        on="ligand",
        how="inner",
        validate="many_to_one",
    )
    affinity_split_df = affinity_split_df.sort_values(
        ["split", "uniprot_id", "ligand"],
        ignore_index=True,
    )
    assert_unique(affinity_split_df, ["uniprot_id", "ligand"], "affinity_split_manifest")
    return affinity_split_df


def parse_args() -> argparse.Namespace:
    """CLI for generating `affinity_split_manifest.csv`."""
    parser = argparse.ArgumentParser(
        description="Build affinity_split_manifest.csv from affinity and split manifests."
    )
    parser.add_argument(
        "--affinity-csv",
        default=str(AFFINITY_MANIFEST_CSV),
        help="Path to affinity_manifest.csv.",
    )
    parser.add_argument(
        "--split-csv",
        default=str(SPLIT_MANIFEST_CSV),
        help="Path to split_manifest.csv.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(AFFINITY_SPLIT_MANIFEST_CSV),
        help="Output path for affinity_split_manifest.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    affinity_split_df = build_affinity_split_manifest(
        affinity_csv=args.affinity_csv,
        split_csv=args.split_csv,
    )
    output_path = write_manifest(affinity_split_df, args.output_csv)
    LOGGER.info(
        "Wrote affinity split manifest: %s (%d rows)",
        output_path,
        len(affinity_split_df),
    )


if __name__ == "__main__":
    main()
