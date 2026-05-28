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

import numpy as np
import pandas as pd

from data_processing.common.constants import (
    ACTIVE_SPLIT_NAMES,
    AFFINITY_SPLIT_MANIFEST_CSV,
    POSE_MANIFEST_CSV,
    RUNS_DIR,
)
from data_processing.common.manifest_io import read_csv_checked, write_manifest
from models.common.run_io import (
    copy_experiment_config,
    make_run_dir,
    write_run_metadata,
)

from .config import ExperimentConfig, load_experiment_config


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
    """Return one minimum Glide score per (uniprot_id, ligand)."""
    pose_df = read_csv_checked(pose_csv, ["uniprot_id", "ligand", "glide_score"])
    pose_df = pose_df[["uniprot_id", "ligand", "glide_score"]].copy()
    pose_df["glide_score"] = pd.to_numeric(pose_df["glide_score"], errors="coerce")
    pose_df = pose_df.dropna(subset=["uniprot_id", "ligand", "glide_score"])
    if pose_df.empty:
        raise ValueError(
            "Pose manifest contains no rows with uniprot_id, ligand, and glide_score values."
        )

    return (
        pose_df.groupby(["uniprot_id", "ligand"], as_index=False)["glide_score"]
        .min()
        .rename(columns={"glide_score": "min_glide_score"})
    )


def _load_examples(
    pose_csv: str | Path,
    affinity_split_csv: str | Path,
    data_config: ExperimentConfig | None = None,
) -> pd.DataFrame:
    """Join ligand-level minimum Glide scores onto split affinity examples."""
    min_glide_df = _minimum_glide_scores(pose_csv)

    if data_config is None:
        required_columns = ["uniprot_id", "ligand", "affinity", "split"]
        active_splits = ACTIVE_SPLIT_NAMES
        uniprot_col = "uniprot_id"
        ligand_col = "ligand"
        target_col = "affinity"
        split_col = "split"
        include_uniprot_ids: tuple[str, ...] = ()
    else:
        required_columns = [
            data_config.data.uniprot_id_column,
            data_config.data.ligand_column,
            data_config.data.target_column,
            data_config.data.split_column,
        ]
        active_splits = data_config.data.active_splits
        uniprot_col = data_config.data.uniprot_id_column
        ligand_col = data_config.data.ligand_column
        target_col = data_config.data.target_column
        split_col = data_config.data.split_column
        include_uniprot_ids = data_config.data.include_uniprot_ids

    affinity_df = read_csv_checked(affinity_split_csv, required_columns)
    affinity_df = affinity_df[affinity_df[split_col].isin(active_splits)].copy()
    if include_uniprot_ids:
        include = set(include_uniprot_ids)
        affinity_df = affinity_df[affinity_df[uniprot_col].astype(str).isin(include)].copy()

    affinity_df[target_col] = pd.to_numeric(affinity_df[target_col], errors="coerce")
    affinity_df = affinity_df.dropna(subset=[ligand_col, target_col, split_col])

    examples_df = affinity_df.merge(
        min_glide_df,
        left_on=[uniprot_col, ligand_col],
        right_on=["uniprot_id", "ligand"],
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
    if (examples_df[split_col] == "train").sum() == 0:
        raise ValueError("No train examples remain after filtering.")

    rename_map = {
        uniprot_col: "uniprot_id",
        ligand_col: "ligand",
        target_col: "affinity",
        split_col: "split",
    }
    return examples_df.rename(columns=rename_map)


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
) -> None:
    """Write a predicted-vs-true affinity scatter plot for one split."""
    from models.common.plots import write_scatter_plot

    write_scatter_plot(df, split, {"pearson_r": pearson_r}, output_path)


def _write_results(
    split: str,
    pearson_r: float,
    num_examples: int,
    output_path: str | Path,
    slope: float | None = None,
    intercept: float | None = None,
) -> None:
    """Write the text results summary for one split."""
    from data_processing.common.manifest_io import ensure_parent_dir

    output_path = ensure_parent_dir(output_path)
    lines = [
        f"split: {split}",
        f"num_examples: {num_examples}",
        f"pearson_r: {pearson_r}",
    ]
    if slope is not None and intercept is not None:
        lines.extend(
            [
                f"slope: {slope}",
                f"intercept: {intercept}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_linear_glide_score(
    config: ExperimentConfig | None = None,
    pose_csv: str | Path | None = None,
    affinity_split_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    experiment_config_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit the train-only line and write predictions, plots, and result files."""
    if experiment_config_path is not None:
        config = load_experiment_config(experiment_config_path)

    if config is not None:
        run_dir = make_run_dir(
            config.paths.runs_dir,
            config.experiment_name,
            overwrite=config.outputs.overwrite_existing_run,
        )
        if experiment_config_path is not None:
            copy_experiment_config(experiment_config_path, run_dir)
        output_dir = run_dir
        pose_csv = config.paths.pose_csv
        affinity_split_csv = config.paths.affinity_split_csv
        active_splits = config.data.active_splits
    else:
        pose_csv = pose_csv or POSE_MANIFEST_CSV
        affinity_split_csv = affinity_split_csv or AFFINITY_SPLIT_MANIFEST_CSV
        output_dir = output_dir or DEFAULT_OUTPUT_DIR
        active_splits = ACTIVE_SPLIT_NAMES

    output_dir = Path(output_dir)
    examples_df = _load_examples(pose_csv, affinity_split_csv, data_config=config)
    train_df = examples_df[examples_df["split"] == "train"].copy()

    slope, intercept = _fit_train_line(train_df)
    LOGGER.info("Fitted affinity = %.6f * min_glide_score + %.6f", slope, intercept)

    examples_df["predicted_affinity"] = slope * examples_df["min_glide_score"] + intercept
    examples_df = examples_df.rename(columns={"affinity": "true_affinity"})

    outputs: dict[str, pd.DataFrame] = {}
    split_counts: dict[str, int] = {}
    for split in active_splits:
        split_df = (
            examples_df[examples_df["split"] == split]
            .loc[:, OUTPUT_COLUMNS]
            .sort_values(["uniprot_id", "ligand"], ignore_index=True)
        )
        outputs[split] = split_df
        split_counts[split] = len(split_df)

        if split_df.empty:
            continue

        csv_path = output_dir / f"predict_vs_true_affinity_{split}.csv"
        plot_path = output_dir / f"scatter_{split}.png"
        results_path = output_dir / f"results_{split}.txt"

        write_manifest(split_df, csv_path)
        pearson_r = _pearson_r(split_df)
        _write_scatter_plot(split_df, split, pearson_r, plot_path)
        _write_results(
            split,
            pearson_r,
            len(split_df),
            results_path,
            slope=slope,
            intercept=intercept,
        )
        LOGGER.info("Wrote %s outputs with %d examples", split, len(split_df))

    if config is not None:
        write_run_metadata(
            {
                "experiment_name": config.experiment_name,
                "num_train": split_counts.get("train", 0),
                "num_val": split_counts.get("val", 0),
                "num_test": split_counts.get("test", 0),
                "include_uniprot_ids": list(config.data.include_uniprot_ids),
                "active_splits": list(config.data.active_splits),
                "slope": slope,
                "intercept": intercept,
            },
            output_dir,
        )

    return outputs


def parse_args() -> argparse.Namespace:
    """CLI for running the linear Glide score baseline."""
    parser = argparse.ArgumentParser(
        description="Fit a train-only linear baseline from minimum Glide score to affinity."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Path to experiment JSON (e.g. configs/linear_glide_score_test.json).",
    )
    parser.add_argument(
        "--pose-csv",
        default=None,
        help="Path to pose_manifest.csv (ignored when --experiment is set).",
    )
    parser.add_argument(
        "--affinity-split-csv",
        default=None,
        help="Path to affinity_split_manifest.csv (ignored when --experiment is set).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs (ignored when --experiment is set).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.experiment is not None:
        evaluate_linear_glide_score(experiment_config_path=args.experiment)
    else:
        evaluate_linear_glide_score(
            pose_csv=args.pose_csv,
            affinity_split_csv=args.affinity_split_csv,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
