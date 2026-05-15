"""Shared training-time experiment resolution for manifest-driven MVP models."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


MVP_ROOT = Path(__file__).resolve().parent.parent
if str(MVP_ROOT) not in sys.path:
    sys.path.append(str(MVP_ROOT))

from data_processing.constants import (  # noqa: E402
    ACTIVE_SPLIT_NAMES,
    AFFINITY_MANIFEST_CSV,
    AFFINITY_SPLIT_MANIFEST_CSV,
    PDB_MANIFEST_CSV,
    POSE_MANIFEST_CSV,
    SPLIT_MANIFEST_CSV,
)
from data_processing.manifest_io import read_csv_checked  # noqa: E402
from data_processing.selector_utils import (  # noqa: E402
    collapse_pose_membership_to_examples,
    resolve_pose_membership,
    resolve_selected_pdbs,
    resolve_selected_poses,
)


@dataclass(frozen=True)
class ResolvedExperiment:
    """Manifest-backed experiment definition shared by all training code."""

    affinity_examples: pd.DataFrame
    pose_membership: pd.DataFrame
    selected_pdbs: pd.DataFrame
    selected_poses: pd.DataFrame


def _load_affinity_examples(
    affinity_csv: str | Path,
    split_csv: str | Path | None = None,
    affinity_split_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Load affinity rows and attach split labels when requested."""
    if affinity_split_csv is not None:
        affinity_df = read_csv_checked(
            affinity_split_csv,
            ["uniprot_id", "ligand", "affinity", "split"],
        )
        affinity_df = affinity_df[affinity_df["split"].isin(ACTIVE_SPLIT_NAMES)].copy()
        return affinity_df.sort_values(["split", "uniprot_id", "ligand"]).reset_index(drop=True)

    affinity_df = read_csv_checked(affinity_csv, ["uniprot_id", "ligand", "affinity"])
    if split_csv is None:
        return affinity_df.sort_values(["uniprot_id", "ligand"]).reset_index(drop=True)

    split_df = read_csv_checked(split_csv, ["ligand", "split"])
    split_df = split_df[split_df["split"].isin(ACTIVE_SPLIT_NAMES)].copy()
    affinity_df = affinity_df.merge(
        split_df[["ligand", "split"]],
        on="ligand",
        how="inner",
        validate="many_to_one",
    )
    return affinity_df.sort_values(["split", "uniprot_id", "ligand"]).reset_index(drop=True)


def resolve_experiment_inputs(
    affinity_csv: str | Path = AFFINITY_MANIFEST_CSV,
    pdb_csv: str | Path = PDB_MANIFEST_CSV,
    pose_csv: str | Path = POSE_MANIFEST_CSV,
    split_csv: str | Path | None = None,
    affinity_split_csv: str | Path | None = AFFINITY_SPLIT_MANIFEST_CSV,
    uniprot_to_pdb_csv: str | Path | None = None,
    ligand_to_pose_csv: str | Path | None = None,
) -> ResolvedExperiment:
    """Resolve selector CSVs into one manifest-backed experiment definition.

    `ligand_to_pose.csv` may select either exact `pose_id`s or PDB-scoped pose
    groups using `(ligand, pdb_id)` or `(ligand, pdb_key)`.
    """
    affinity_df = _load_affinity_examples(
        affinity_csv=affinity_csv,
        split_csv=split_csv,
        affinity_split_csv=affinity_split_csv,
    )
    pdb_df = read_csv_checked(pdb_csv, ["uniprot_id", "pdb_key", "pdb_id"])
    pose_df = read_csv_checked(
        pose_csv,
        [
            "pose_id",
            "pdb_key",
            "ligand",
            "grid",
            "uniprot_id",
            "pdb_id",
            "glide_score",
            "pose_rank",
            "is_top_rank",
            "source_sdf",
        ],
    )

    selected_pdbs = resolve_selected_pdbs(pdb_df, uniprot_to_pdb_csv=uniprot_to_pdb_csv)
    selected_poses = resolve_selected_poses(pose_df, ligand_to_pose_csv=ligand_to_pose_csv)
    pose_membership = resolve_pose_membership(
        affinity_df=affinity_df,
        pdb_df=pdb_df,
        pose_df=pose_df,
        uniprot_to_pdb_csv=uniprot_to_pdb_csv,
        ligand_to_pose_csv=ligand_to_pose_csv,
    )
    affinity_examples = collapse_pose_membership_to_examples(pose_membership)
    return ResolvedExperiment(
        affinity_examples=affinity_examples,
        pose_membership=pose_membership,
        selected_pdbs=selected_pdbs,
        selected_poses=selected_poses,
    )


__all__ = [
    "ResolvedExperiment",
    "resolve_experiment_inputs",
]
