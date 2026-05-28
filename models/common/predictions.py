"""Prediction output helpers shared by affinity baselines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_processing.common.manifest_io import ensure_parent_dir, write_manifest

from .metrics import affinity_metrics
from .plots import write_scatter_plot


PREDICTION_COLUMNS = (
    "uniprot_id",
    "protein_idx",
    "ligand",
    "ligand_idx",
    "split",
    "true_affinity",
    "predicted_affinity",
    "abs_error",
    "squared_error",
)


def add_error_columns(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute and squared error columns."""
    predictions_df = predictions_df.copy()
    residual = predictions_df["predicted_affinity"] - predictions_df["true_affinity"]
    predictions_df["abs_error"] = residual.abs()
    predictions_df["squared_error"] = residual.pow(2)
    return predictions_df


def write_results(
    split: str,
    metrics: dict[str, float],
    num_examples: int,
    output_path: str | Path,
) -> Path:
    """Write the text results summary for one split."""
    output_path = ensure_parent_dir(output_path)
    output_path.write_text(
        f"split: {split}\n"
        f"num_examples: {num_examples}\n"
        f"pearson_r: {metrics['pearson_r']}\n"
        f"rmse: {metrics['rmse']}\n"
        f"mae: {metrics['mae']}\n",
        encoding="utf-8",
    )
    return output_path


def write_split_outputs(
    predictions_df: pd.DataFrame,
    output_dir: str | Path,
    splits: tuple[str, ...] | list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Write per-split prediction CSVs, scatter plots, and result summaries."""
    output_dir = Path(output_dir)
    predictions_df = add_error_columns(predictions_df)
    if splits is None:
        splits = list(predictions_df["split"].dropna().unique())

    outputs: dict[str, pd.DataFrame] = {}
    for split in splits:
        split_df = predictions_df[predictions_df["split"] == split].copy()
        available_columns = [
            column for column in PREDICTION_COLUMNS
            if column in split_df.columns
        ]
        split_df = split_df.loc[:, available_columns].reset_index(drop=True)
        outputs[str(split)] = split_df

        metrics = affinity_metrics(split_df)
        write_manifest(split_df, output_dir / f"predict_vs_true_affinity_{split}.csv")
        write_scatter_plot(split_df, str(split), metrics, output_dir / f"scatter_{split}.png")
        write_results(str(split), metrics, len(split_df), output_dir / f"results_{split}.txt")

    return outputs

