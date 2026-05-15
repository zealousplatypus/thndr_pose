"""Glide-score baseline predictor for selector-resolved MVP experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MVP_ROOT = Path(__file__).resolve().parent.parent
if str(MVP_ROOT) not in sys.path:
    sys.path.append(str(MVP_ROOT))

from training.input_resolution import ResolvedExperiment  # noqa: E402


MODEL_NAME = "glide_score_baseline"


def build_feature_frame(resolved: ResolvedExperiment) -> pd.DataFrame:
    """Compute one selected Glide-score feature per affinity example."""
    pose_membership_df = resolved.pose_membership
    top_pose_df = pose_membership_df[pose_membership_df["pose_rank"] == 1].copy()
    if top_pose_df.empty:
        raise ValueError("No eligible top-ranked poses remain after applying the experiment selectors.")

    group_columns = ["uniprot_id", "ligand", "affinity"]
    if "split" in top_pose_df.columns:
        group_columns.append("split")

    feature_df = (
        top_pose_df.groupby(group_columns, as_index=False)
        .agg(
            selected_glide_score=("glide_score", "min"),
            num_selected_pdbs=("pdb_key", "nunique"),
            num_selected_top_poses=("pose_id", "nunique"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return feature_df


def fit_linear_regression(train_df: pd.DataFrame) -> dict[str, float]:
    """Fit a one-feature linear regression using least squares."""
    if train_df.empty:
        raise ValueError("Training split is empty; cannot fit the Glide-score baseline.")

    x = train_df["selected_glide_score"].to_numpy(dtype=float)
    y = train_df["affinity"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "intercept": float(intercept),
        "slope": float(slope),
    }


def predict_affinity(df: pd.DataFrame, model: dict[str, float]) -> pd.Series:
    """Run the fitted baseline on a feature DataFrame."""
    return model["intercept"] + model["slope"] * df["selected_glide_score"]


def fit_and_predict(
    resolved: ResolvedExperiment,
    train_split_name: str = "train",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit the Glide-score baseline and return predictions plus model metadata."""
    feature_df = build_feature_frame(resolved)
    if "split" in feature_df.columns:
        train_df = feature_df[feature_df["split"] == train_split_name].copy()
    else:
        train_df = feature_df.copy()
    model = fit_linear_regression(train_df)

    prediction_df = feature_df.copy()
    prediction_df["pred_affinity"] = predict_affinity(prediction_df, model)
    sort_columns = ["uniprot_id", "ligand"]
    if "split" in prediction_df.columns:
        sort_columns = ["split"] + sort_columns
    prediction_df = prediction_df.sort_values(sort_columns).reset_index(drop=True)
    model_metadata = {
        "name": MODEL_NAME,
        "train_split_name": train_split_name,
        "intercept": model["intercept"],
        "slope": model["slope"],
    }
    return prediction_df, model_metadata


__all__ = [
    "MODEL_NAME",
    "build_feature_frame",
    "fit_and_predict",
    "fit_linear_regression",
    "predict_affinity",
]
