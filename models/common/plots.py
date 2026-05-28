"""Plotting helpers for affinity prediction baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from data_processing.common.manifest_io import ensure_parent_dir


def write_scatter_plot(
    df: pd.DataFrame,
    split: str,
    metrics: dict[str, float],
    output_path: str | Path,
) -> Path:
    """Write a predicted-vs-true affinity scatter plot for one split."""
    output_path = ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["true_affinity"], df["predicted_affinity"], alpha=0.7)
    ax.set_xlabel("True affinity")
    ax.set_ylabel("Predicted affinity")
    ax.set_title(
        f"{split} predicted vs true affinity "
        f"(Pearson r={metrics.get('pearson_r', float('nan')):.4g})"
    )

    if not df.empty:
        values = pd.concat([df["true_affinity"], df["predicted_affinity"]])
        axis_min = values.min()
        axis_max = values.max()
        ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", color="gray")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def write_train_val_loss_plot(
    history_df: pd.DataFrame,
    output_path: str | Path,
    *,
    epoch_column: str = "epoch",
    train_column: str = "train_loss",
    val_column: str = "val_loss",
    title: str = "Train and validation loss",
) -> Path:
    """Plot train/validation loss curves over epochs."""
    output_path = ensure_parent_dir(output_path)
    if history_df.empty:
        raise ValueError("Loss history is empty; cannot write loss curve plot.")

    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = history_df[epoch_column]

    if train_column in history_df.columns and history_df[train_column].notna().any():
        ax.plot(
            epochs,
            history_df[train_column],
            marker="o",
            label="train",
            linewidth=2,
        )

    if val_column in history_df.columns and history_df[val_column].notna().any():
        ax.plot(
            epochs,
            history_df[val_column],
            marker="o",
            label="val",
            linewidth=2,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

