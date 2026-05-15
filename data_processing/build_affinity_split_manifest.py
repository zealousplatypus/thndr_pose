"""Attach ligand-level split labels onto the affinity manifest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from constants import (
        ACTIVE_SPLIT_NAMES,
        AFFINITY_MANIFEST_CSV,
        AFFINITY_SPLIT_MANIFEST_CSV,
        PDB_MANIFEST_CSV,
        POSE_MANIFEST_CSV,
        SPLIT_MANIFEST_CSV,
    )
    from manifest_io import assert_unique, read_csv_checked, write_manifest
    from selector_utils import collapse_pose_membership_to_examples, resolve_pose_membership
except ImportError:  # pragma: no cover
    from .constants import (
        ACTIVE_SPLIT_NAMES,
        AFFINITY_MANIFEST_CSV,
        AFFINITY_SPLIT_MANIFEST_CSV,
        PDB_MANIFEST_CSV,
        POSE_MANIFEST_CSV,
        SPLIT_MANIFEST_CSV,
    )
    from .manifest_io import assert_unique, read_csv_checked, write_manifest
    from .selector_utils import collapse_pose_membership_to_examples, resolve_pose_membership


LOGGER = logging.getLogger(__name__)


def build_affinity_split_manifest(
    affinity_csv: str | Path,
    split_csv: str | Path,
    pdb_csv: str | Path = PDB_MANIFEST_CSV,
    pose_csv: str | Path = POSE_MANIFEST_CSV,
    uniprot_to_pdb_csv: str | Path | None = None,
    ligand_to_pose_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Join the affinity labels with the ligand-based train/val/test split."""
    affinity_df = read_csv_checked(affinity_csv, ["uniprot_id", "ligand", "affinity"])
    split_df = read_csv_checked(split_csv, ["ligand", "split"])
    split_df = split_df[split_df["split"].isin(ACTIVE_SPLIT_NAMES)].copy()
    if uniprot_to_pdb_csv is not None or ligand_to_pose_csv is not None:
        pdb_df = read_csv_checked(pdb_csv, ["uniprot_id", "pdb_key", "pdb_id"])
        pose_df = read_csv_checked(
            pose_csv,
            [
                "pose_id",
                "pdb_key",
                "ligand",
                "uniprot_id",
                "pdb_id",
                "glide_score",
                "pose_rank",
            ],
        )
        affinity_split_df = collapse_pose_membership_to_examples(
            resolve_pose_membership(
                affinity_df=affinity_df,
                pdb_df=pdb_df,
                pose_df=pose_df,
                uniprot_to_pdb_csv=uniprot_to_pdb_csv,
                ligand_to_pose_csv=ligand_to_pose_csv,
                split_df=split_df,
                active_splits_only=True,
            )
        )
    else:
        affinity_split_df = affinity_df.merge(
            split_df[["ligand", "split"]],
            on="ligand",
            how="inner",
            validate="many_to_one",
        )
    affinity_split_df = (
        affinity_split_df.sort_values(["split", "uniprot_id", "ligand"])
        .reset_index(drop=True)
    )
    assert_unique(affinity_split_df, ["uniprot_id", "ligand"], "affinity_split_manifest")
    return affinity_split_df


def parse_args() -> argparse.Namespace:
    """CLI for generating affinity_split_manifest.csv."""
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
        "--pdb-csv",
        default=str(PDB_MANIFEST_CSV),
        help="Path to pdb_manifest.csv, used for experiment-specific selector filtering.",
    )
    parser.add_argument(
        "--pose-csv",
        default=str(POSE_MANIFEST_CSV),
        help="Path to pose_manifest.csv, used for experiment-specific selector filtering.",
    )
    parser.add_argument(
        "--uniprot-to-pdb-csv",
        default=None,
        help="Optional selector CSV restricting which PDBs participate in this experiment.",
    )
    parser.add_argument(
        "--ligand-to-pose-csv",
        default=None,
        help=(
            "Optional selector CSV restricting which pose groups participate in this "
            "experiment. Supports ligand+pdb_id, ligand+pdb_key, or legacy ligand+pose_id."
        ),
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
        pdb_csv=args.pdb_csv,
        pose_csv=args.pose_csv,
        uniprot_to_pdb_csv=args.uniprot_to_pdb_csv,
        ligand_to_pose_csv=args.ligand_to_pose_csv,
    )
    output_path = write_manifest(affinity_split_df, args.output_csv)
    LOGGER.info(
        "Wrote affinity split manifest: %s (%d rows)",
        output_path,
        len(affinity_split_df),
    )


if __name__ == "__main__":
    main()
