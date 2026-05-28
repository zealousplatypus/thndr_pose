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

