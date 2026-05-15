"""Shared selector-resolution helpers for experiment-specific MVP runs.

These utilities turn the user-provided selector CSVs into validated subsets of
the global manifests so both data-processing scripts and training code can use
the same experiment definition logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from constants import ACTIVE_SPLIT_NAMES
    from manifest_io import assert_unique, read_csv_checked
except ImportError:  # pragma: no cover
    from .constants import ACTIVE_SPLIT_NAMES
    from .manifest_io import assert_unique, read_csv_checked


def resolve_selected_pdbs(
    pdb_df: pd.DataFrame,
    uniprot_to_pdb_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Return the experiment-selected PDB rows from `pdb_manifest.csv`.

    The selector may use either `(uniprot_id, pdb_key)` or `(uniprot_id, pdb_id)`.
    """
    selected_pdbs = pdb_df.loc[:, ["uniprot_id", "pdb_key", "pdb_id"]].drop_duplicates().copy()
    assert_unique(selected_pdbs, ["pdb_key"], "pdb_manifest")
    if uniprot_to_pdb_csv is None:
        return selected_pdbs.sort_values(["uniprot_id", "pdb_id"]).reset_index(drop=True)

    selector_df = read_csv_checked(uniprot_to_pdb_csv, ["uniprot_id"])
    if "pdb_key" in selector_df.columns:
        join_columns = ["uniprot_id", "pdb_key"]
    elif "pdb_id" in selector_df.columns:
        join_columns = ["uniprot_id", "pdb_id"]
    else:
        raise ValueError(
            f"{uniprot_to_pdb_csv} must contain either ['uniprot_id', 'pdb_key'] "
            "or ['uniprot_id', 'pdb_id']."
        )

    selector_df = selector_df.loc[:, join_columns].drop_duplicates().copy()
    assert_unique(selector_df, join_columns, "uniprot_to_pdb selector")
    available_pdb_keys = set(map(tuple, selected_pdbs.loc[:, join_columns].drop_duplicates().to_records(index=False)))
    requested_pdb_keys = set(map(tuple, selector_df.loc[:, join_columns].to_records(index=False)))
    missing_pdb_keys = sorted(requested_pdb_keys - available_pdb_keys)[:10]
    if missing_pdb_keys:
        raise ValueError(f"uniprot_to_pdb selector contains unresolved rows: {missing_pdb_keys}")

    selected_pdbs = selected_pdbs.merge(
        selector_df,
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    return selected_pdbs.sort_values(["uniprot_id", "pdb_id"]).reset_index(drop=True)


def resolve_selected_poses(
    pose_df: pd.DataFrame,
    ligand_to_pose_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Return the experiment-selected pose rows from `pose_manifest.csv`.

    Supported selector schemas are:
    - `ligand`, `pdb_key`: select all poses for that ligand docked against that PDB
    - `ligand`, `pdb_id`: same as above using the shorter structure ID
    - `ligand`, `pose_id`: legacy exact-pose selection
    """
    selected_poses = pose_df.copy()
    assert_unique(selected_poses, ["pose_id"], "pose_manifest")
    if ligand_to_pose_csv is None:
        return selected_poses.sort_values(["uniprot_id", "pdb_key", "ligand", "pose_rank"]).reset_index(
            drop=True
        )

    selector_df = read_csv_checked(ligand_to_pose_csv, ["ligand"])
    if "pdb_key" in selector_df.columns:
        join_columns = ["ligand", "pdb_key"]
        validation = "many_to_one"
    elif "pdb_id" in selector_df.columns:
        join_columns = ["ligand", "pdb_id"]
        validation = "many_to_one"
    elif "pose_id" in selector_df.columns:
        join_columns = ["ligand", "pose_id"]
        validation = "many_to_one"
    else:
        raise ValueError(
            f"{ligand_to_pose_csv} must contain `ligand` plus one of `pdb_key`, `pdb_id`, or `pose_id`."
        )

    selector_df = selector_df.loc[:, join_columns].drop_duplicates().copy()
    assert_unique(selector_df, join_columns, "ligand_to_pose selector")

    available_pose_groups = selected_poses.loc[:, join_columns].drop_duplicates().copy()
    available_group_keys = set(map(tuple, available_pose_groups.to_records(index=False)))
    requested_group_keys = set(map(tuple, selector_df.to_records(index=False)))
    missing_group_keys = sorted(requested_group_keys - available_group_keys)[:10]
    if missing_group_keys:
        raise ValueError(f"ligand_to_pose selector contains unresolved rows: {missing_group_keys}")

    selected_poses = selected_poses.merge(
        selector_df,
        on=join_columns,
        how="inner",
        validate=validation,
    )
    return selected_poses.sort_values(["uniprot_id", "pdb_key", "ligand", "pose_rank"]).reset_index(
        drop=True
    )


def resolve_pose_membership(
    affinity_df: pd.DataFrame,
    pdb_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    uniprot_to_pdb_csv: str | Path | None = None,
    ligand_to_pose_csv: str | Path | None = None,
    split_df: pd.DataFrame | None = None,
    active_splits_only: bool = True,
) -> pd.DataFrame:
    """Resolve experiment-specific eligible pose rows for each affinity example."""
    assert_unique(affinity_df, ["uniprot_id", "ligand"], "affinity manifest")
    selected_pdbs = resolve_selected_pdbs(pdb_df, uniprot_to_pdb_csv=uniprot_to_pdb_csv)
    selected_poses = resolve_selected_poses(pose_df, ligand_to_pose_csv=ligand_to_pose_csv)

    eligible_poses = selected_poses.merge(
        selected_pdbs[["uniprot_id", "pdb_key", "pdb_id"]],
        on=["uniprot_id", "pdb_key", "pdb_id"],
        how="inner",
        validate="many_to_one",
    )
    resolved_df = affinity_df.merge(
        eligible_poses,
        on=["uniprot_id", "ligand"],
        how="inner",
        validate="one_to_many",
    )

    if split_df is not None:
        split_to_use = split_df.copy()
        if active_splits_only:
            split_to_use = split_to_use[split_to_use["split"].isin(ACTIVE_SPLIT_NAMES)].copy()
        resolved_df = resolved_df.merge(
            split_to_use[["ligand", "split"]],
            on="ligand",
            how="inner",
            validate="many_to_one",
        )

    return resolved_df.sort_values(
        ["uniprot_id", "ligand", "pdb_key", "pose_rank", "pose_id"]
    ).reset_index(drop=True)


def collapse_pose_membership_to_examples(pose_membership_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse pose membership rows back to one row per supervised example."""
    example_columns = ["uniprot_id", "ligand", "affinity"]
    if "split" in pose_membership_df.columns:
        example_columns.append("split")
    example_df = pose_membership_df.loc[:, example_columns].drop_duplicates().reset_index(drop=True)
    sort_columns = ["uniprot_id", "ligand"]
    if "split" in example_df.columns:
        sort_columns = ["split"] + sort_columns
    example_df = example_df.sort_values(sort_columns).reset_index(drop=True)
    assert_unique(example_df, ["uniprot_id", "ligand"], "resolved affinity examples")
    return example_df
