"""Reusable evaluation metrics and plotting helpers for MVP experiments."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Compute Pearson correlation when at least two points are present."""
    if len(y_true) < 2:
        return None
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def summarize_prediction_metrics(
    df: pd.DataFrame,
    requested_splits: tuple[str, ...] = ("train", "val", "test"),
    truth_column: str = "affinity",
    prediction_column: str = "pred_affinity",
    split_column: str = "split",
) -> tuple[dict[str, dict[str, float | int | None]], dict[str, list[str]]]:
    """Summarize regression metrics for the requested splits."""
    metrics: dict[str, dict[str, float | int | None]] = {}
    split_notes: dict[str, list[str]] = {}
    for split_name in requested_splits:
        split_df = df[df[split_column] == split_name].copy() if split_column in df.columns else pd.DataFrame()
        notes: list[str] = []
        if split_df.empty:
            metrics[split_name] = {
                "count": 0,
                "mae": None,
                "rmse": None,
                "r2": None,
                "pearson": None,
            }
            notes.append(f"No examples available for split '{split_name}'.")
            split_notes[split_name] = notes
            continue

        y_true = split_df[truth_column].to_numpy(dtype=float)
        y_pred = split_df[prediction_column].to_numpy(dtype=float)
        residual = y_pred - y_true
        mae = float(np.mean(np.abs(residual)))
        rmse = float(math.sqrt(np.mean(np.square(residual))))
        total_var = float(np.sum(np.square(y_true - y_true.mean())))
        r2 = None if len(y_true) < 2 or np.isclose(total_var, 0.0) else float(
            1.0 - np.sum(np.square(residual)) / total_var
        )
        pearson = _pearson_corr(y_true, y_pred)
        if len(y_true) < 2:
            notes.append("Split has fewer than two examples; correlation-style metrics are undefined.")
        elif np.allclose(y_true, y_true[0]):
            notes.append("True affinities are constant in this split; R^2/Pearson may be undefined.")
        elif np.allclose(y_pred, y_pred[0]):
            notes.append("Predictions are constant in this split; Pearson is undefined.")
        metrics[split_name] = {
            "count": int(len(split_df)),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson": pearson,
        }
        split_notes[split_name] = notes
    return metrics, split_notes


def write_prediction_scatter_plot(
    df: pd.DataFrame,
    split_name: str,
    output_path: str | Path,
    run_name: str,
    truth_column: str = "affinity",
    prediction_column: str = "pred_affinity",
    split_column: str = "split",
) -> str:
    """Write a predicted-vs-true scatter plot and return a status note."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df = df[df[split_column] == split_name].copy() if split_column in df.columns else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if split_df.empty:
        ax.text(0.5, 0.5, f"No {split_name} examples", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        note = f"No examples available for split '{split_name}'; wrote placeholder plot."
    else:
        x = split_df[truth_column].to_numpy(dtype=float)
        y = split_df[prediction_column].to_numpy(dtype=float)
        min_val = float(min(x.min(), y.min()))
        max_val = float(max(x.max(), y.max()))
        if math.isclose(min_val, max_val):
            padding = 0.5
        else:
            padding = 0.05 * (max_val - min_val)
        low = min_val - padding
        high = max_val + padding
        ax.scatter(x, y, alpha=0.7, edgecolors="none")
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("True affinity")
        ax.set_ylabel("Predicted affinity")
        ax.set_title(f"{run_name}: {split_name} predicted vs true")
        note = f"Wrote {split_name} scatter plot with {len(split_df)} examples."

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return note


__all__ = [
    "summarize_prediction_metrics",
    "write_prediction_scatter_plot",
]
