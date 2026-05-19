"""Evaluate a ligand-level linear baseline from minimum Glide scores.

The model fits a line on train examples only:

    affinity = slope * min_glide_score + intercept

Each ligand is represented by the best (minimum) Glide score observed in
`pose_manifest.csv`, then predictions are reported against the true affinity
for each active split in `affinity_split_manifest.csv`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.common.constants import (
    ACTIVE_SPLIT_NAMES,
    AFFINITY_SPLIT_MANIFEST_CSV,
    POSE_MANIFEST_CSV,
    RUNS_DIR,
)
from data_processing.common.manifest_io import ensure_parent_dir, read_csv_checked, write_manifest


LOGGER = logging.getLogger(__name__)

OUTPUT_COLUMNS = (
    "uniprot_id",
    "ligand",
    "split",
    "min_glide_score",
    "true_affinity",
    "predicted_affinity",
)
DEFAULT_OUTPUT_DIR = RUNS_DIR / "linear_glide_score"


def _minimum_glide_scores(pose_csv: str | Path) -> pd.DataFrame:
    """Return one minimum Glide score per ligand."""
    pose_df = read_csv_checked(pose_csv, ["ligand", "glide_score"])
    pose_df = pose_df[["ligand", "glide_score"]].copy()
    pose_df["glide_score"] = pd.to_numeric(pose_df["glide_score"], errors="coerce")
    pose_df = pose_df.dropna(subset=["ligand", "glide_score"])
    if pose_df.empty:
        raise ValueError("Pose manifest contains no rows with ligand and glide_score values.")

    return (
        pose_df.groupby("ligand", as_index=False)["glide_score"]
        .min()
        .rename(columns={"glide_score": "min_glide_score"})
    )


def _load_examples(
    pose_csv: str | Path,
    affinity_split_csv: str | Path,
) -> pd.DataFrame:
    """Join ligand-level minimum Glide scores onto split affinity examples."""
    min_glide_df = _minimum_glide_scores(pose_csv)
    affinity_df = read_csv_checked(
        affinity_split_csv,
        ["uniprot_id", "ligand", "affinity", "split"],
    )
    affinity_df = affinity_df[affinity_df["split"].isin(ACTIVE_SPLIT_NAMES)].copy()
    affinity_df["affinity"] = pd.to_numeric(affinity_df["affinity"], errors="coerce")
    affinity_df = affinity_df.dropna(subset=["ligand", "affinity", "split"])

    examples_df = affinity_df.merge(
        min_glide_df,
        on="ligand",
        how="inner",
        validate="many_to_one",
    )
    dropped_rows = len(affinity_df) - len(examples_df)
    if dropped_rows:
        LOGGER.warning(
            "Dropped %d affinity rows without a matching ligand in the pose manifest.",
            dropped_rows,
        )
    if examples_df.empty:
        raise ValueError("No affinity examples matched ligands in the pose manifest.")
    return examples_df


def _fit_train_line(train_df: pd.DataFrame) -> tuple[float, float]:
    """Fit affinity as a linear function of minimum Glide score."""
    if len(train_df) < 2:
        raise ValueError("At least two train examples are required to fit a line.")
    if train_df["min_glide_score"].nunique() < 2:
        raise ValueError("Train examples must contain at least two unique Glide scores.")

    slope, intercept = np.polyfit(
        train_df["min_glide_score"].to_numpy(dtype=float),
        train_df["affinity"].to_numpy(dtype=float),
        deg=1,
    )
    return float(slope), float(intercept)


def _pearson_r(df: pd.DataFrame) -> float:
    """Compute Pearson r for predicted vs true affinity."""
    if len(df) < 2:
        return float("nan")
    return float(df["predicted_affinity"].corr(df["true_affinity"], method="pearson"))


def _write_scatter_plot(
    df: pd.DataFrame,
    split: str,
    pearson_r: float,
    output_path: str | Path,
) -> Path:
    """Write a predicted-vs-true affinity scatter plot for one split."""
    output_path = ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["true_affinity"], df["predicted_affinity"], alpha=0.7)
    ax.set_xlabel("True affinity")
    ax.set_ylabel("Predicted affinity")
    ax.set_title(f"{split} predicted vs true affinity (Pearson r={pearson_r:.4g})")

    if not df.empty:
        values = pd.concat([df["true_affinity"], df["predicted_affinity"]])
        axis_min = values.min()
        axis_max = values.max()
        ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", color="gray")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _write_results(
    split: str,
    pearson_r: float,
    num_examples: int,
    output_path: str | Path,
) -> Path:
    """Write the text results summary for one split."""
    output_path = ensure_parent_dir(output_path)
    output_path.write_text(
        f"split: {split}\n"
        f"num_examples: {num_examples}\n"
        f"pearson_r: {pearson_r}\n",
        encoding="utf-8",
    )
    return output_path


def evaluate_linear_glide_score(
    pose_csv: str | Path = POSE_MANIFEST_CSV,
    affinity_split_csv: str | Path = AFFINITY_SPLIT_MANIFEST_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Fit the train-only line and write predictions, plots, and result files."""
    output_dir = Path(output_dir)
    examples_df = _load_examples(pose_csv, affinity_split_csv)
    train_df = examples_df[examples_df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("No train examples found after joining pose and affinity manifests.")

    slope, intercept = _fit_train_line(train_df)
    LOGGER.info("Fitted affinity = %.6f * min_glide_score + %.6f", slope, intercept)

    examples_df["predicted_affinity"] = slope * examples_df["min_glide_score"] + intercept
    examples_df = examples_df.rename(columns={"affinity": "true_affinity"})

    outputs: dict[str, pd.DataFrame] = {}
    for split in ACTIVE_SPLIT_NAMES:
        split_df = (
            examples_df[examples_df["split"] == split]
            .loc[:, OUTPUT_COLUMNS]
            .sort_values(["uniprot_id", "ligand"], ignore_index=True)
        )
        outputs[split] = split_df

        csv_path = output_dir / f"predict_vs_true_affinity_{split}.csv"
        plot_path = output_dir / f"scatter_{split}.png"
        results_path = output_dir / f"results_{split}.txt"

        write_manifest(split_df, csv_path)
        pearson_r = _pearson_r(split_df)
        _write_scatter_plot(split_df, split, pearson_r, plot_path)
        _write_results(split, pearson_r, len(split_df), results_path)
        LOGGER.info("Wrote %s outputs with %d examples", split, len(split_df))

    return outputs


def parse_args() -> argparse.Namespace:
    """CLI for running the linear Glide score baseline."""
    parser = argparse.ArgumentParser(
        description="Fit a train-only linear baseline from minimum Glide score to affinity."
    )
    parser.add_argument(
        "--pose-csv",
        default=str(POSE_MANIFEST_CSV),
        help="Path to pose_manifest.csv.",
    )
    parser.add_argument(
        "--affinity-split-csv",
        default=str(AFFINITY_SPLIT_MANIFEST_CSV),
        help="Path to affinity_split_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for prediction CSVs, scatter plots, and result text files.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    evaluate_linear_glide_score(
        pose_csv=args.pose_csv,
        affinity_split_csv=args.affinity_split_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
