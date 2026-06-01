"""Attach ligand-level split labels to affinity examples."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..common.constants import (
    ACTIVE_SPLIT_NAMES,
    AFFINITY_MANIFEST_CSV,
    AFFINITY_MANIFEST_COLUMNS,
    AFFINITY_SPLIT_MANIFEST_COLUMNS,
    AFFINITY_SPLIT_MANIFEST_CSV,
    PROTEIN_MANIFEST_CSV,
    PROTEIN_MANIFEST_COLUMNS,
    SPLIT_MANIFEST_CSV,
    SPLIT_MANIFEST_COLUMNS,
)
from ..common.manifest_io import assert_unique, read_csv_checked, write_manifest
from .protein_subset import (
    assert_outputs_writable,
    make_protein_subset_slug,
    normalize_uniprot_ids,
    resolve_subset_paths,
)


LOGGER = logging.getLogger(__name__)


def build_affinity_split_manifest(
    affinity_csv: str | Path = AFFINITY_MANIFEST_CSV,
    split_csv: str | Path = SPLIT_MANIFEST_CSV,
    protein_csv: str | Path = PROTEIN_MANIFEST_CSV,
    include_uniprot_ids: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Join active split labels and index columns back onto affinity examples."""
    affinity_df = read_csv_checked(affinity_csv, AFFINITY_MANIFEST_COLUMNS)
    split_df = read_csv_checked(split_csv, SPLIT_MANIFEST_COLUMNS)
    protein_df = read_csv_checked(protein_csv, PROTEIN_MANIFEST_COLUMNS)
    split_df = split_df[split_df["split"].isin(ACTIVE_SPLIT_NAMES)].copy()

    normalized_uniprot_ids = normalize_uniprot_ids(include_uniprot_ids)
    if normalized_uniprot_ids is not None:
        subset = set(normalized_uniprot_ids)
        affinity_df = affinity_df[affinity_df["uniprot_id"].astype(str).isin(subset)].copy()

    affinity_split_df = affinity_df.merge(
        split_df[["ligand", "ligand_idx", "split"]],
        on="ligand",
        how="inner",
        validate="many_to_one",
    )
    affinity_split_df = affinity_split_df.merge(
        protein_df[["uniprot_id", "protein_idx"]],
        on="uniprot_id",
        how="left",
        validate="many_to_one",
    )
    if affinity_split_df["protein_idx"].isna().any():
        missing = sorted(affinity_split_df.loc[affinity_split_df["protein_idx"].isna(), "uniprot_id"].unique())
        raise ValueError(f"Affinity examples reference proteins missing from protein manifest: {missing[:10]}")

    affinity_split_df = affinity_split_df.loc[:, AFFINITY_SPLIT_MANIFEST_COLUMNS]
    affinity_split_df = affinity_split_df.sort_values(
        ["split", "protein_idx", "ligand_idx"],
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
        "--uniprot-ids",
        nargs="+",
        default=None,
        help=(
            "Restrict rows to these UniProt IDs and use slug-prefixed default paths "
            "for split/protein/affinity outputs."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    parser.add_argument(
        "--split-csv",
        default=None,
        help="Path to split_manifest.csv (default depends on --uniprot-ids).",
    )
    parser.add_argument(
        "--protein-csv",
        default=None,
        help="Path to protein_manifest.csv (default: global protein_manifest.csv).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output path for affinity_split_manifest.csv (default depends on --uniprot-ids).",
    )
    args = parser.parse_args()

    normalized_uniprot_ids = normalize_uniprot_ids(args.uniprot_ids)
    slug = make_protein_subset_slug(normalized_uniprot_ids) if normalized_uniprot_ids else None
    default_paths = resolve_subset_paths(slug=slug)

    if args.split_csv is None:
        args.split_csv = str(default_paths.split_manifest_csv)
    if args.protein_csv is None:
        args.protein_csv = str(PROTEIN_MANIFEST_CSV)
    if args.output_csv is None:
        args.output_csv = str(default_paths.affinity_split_manifest_csv)

    args.include_uniprot_ids = normalized_uniprot_ids
    return args


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    assert_outputs_writable([Path(args.output_csv)], overwrite=args.overwrite)

    affinity_split_df = build_affinity_split_manifest(
        affinity_csv=args.affinity_csv,
        split_csv=args.split_csv,
        protein_csv=args.protein_csv,
        include_uniprot_ids=args.include_uniprot_ids,
    )
    output_path = write_manifest(affinity_split_df, args.output_csv)
    LOGGER.info(
        "Wrote affinity split manifest: %s (%d rows)",
        output_path,
        len(affinity_split_df),
    )


if __name__ == "__main__":
    main()
