"""Run the Glide-score baseline through the standardized MVP experiment flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


MVP_ROOT = Path(__file__).resolve().parent.parent
if str(MVP_ROOT) not in sys.path:
    sys.path.append(str(MVP_ROOT))

from data_processing.constants import (  # noqa: E402
    AFFINITY_MANIFEST_CSV,
    AFFINITY_SPLIT_MANIFEST_CSV,
    DEFAULT_FINGERPRINT_RADIUS,
    DEFAULT_FINGERPRINT_SIZE,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TANIMOTO_THRESHOLD,
    DEFAULT_TEST_FRACTION,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VAL_FRACTION,
    PDB_MANIFEST_CSV,
    POSE_MANIFEST_CSV,
    RAW_BINDING_CSV,
    RUNS_DIR,
    SPLIT_MANIFEST_CSV,
)
from training.input_resolution import resolve_experiment_inputs  # noqa: E402
from training.metrics import summarize_prediction_metrics  # noqa: E402
from training.run_experiment import SplitConfig, run_model_experiment  # noqa: E402
from data_processing.build_split_manifest import DEFAULT_NUM_RESTARTS  # noqa: E402
from models.glide_score_baseline import MODEL_NAME, build_feature_frame, fit_and_predict  # noqa: E402


DEFAULT_RUN_NAME = MODEL_NAME


def build_baseline_frame(
    affinity_csv: str | Path = AFFINITY_MANIFEST_CSV,
    pdb_csv: str | Path = PDB_MANIFEST_CSV,
    pose_csv: str | Path = POSE_MANIFEST_CSV,
    split_csv: str | Path | None = SPLIT_MANIFEST_CSV,
    affinity_split_csv: str | Path | None = AFFINITY_SPLIT_MANIFEST_CSV,
    uniprot_to_pdb_csv: str | Path | None = None,
    ligand_to_pose_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Resolve selectors and compute one Glide-score feature per affinity row."""
    resolved = resolve_experiment_inputs(
        affinity_csv=affinity_csv,
        pdb_csv=pdb_csv,
        pose_csv=pose_csv,
        split_csv=split_csv,
        affinity_split_csv=affinity_split_csv,
        uniprot_to_pdb_csv=uniprot_to_pdb_csv,
        ligand_to_pose_csv=ligand_to_pose_csv,
    )
    return build_feature_frame(resolved)


def train_and_evaluate(
    affinity_csv: str | Path = AFFINITY_MANIFEST_CSV,
    pdb_csv: str | Path = PDB_MANIFEST_CSV,
    pose_csv: str | Path = POSE_MANIFEST_CSV,
    split_csv: str | Path | None = SPLIT_MANIFEST_CSV,
    affinity_split_csv: str | Path | None = AFFINITY_SPLIT_MANIFEST_CSV,
    uniprot_to_pdb_csv: str | Path | None = None,
    ligand_to_pose_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train the baseline and return predictions plus summary metadata."""
    resolved = resolve_experiment_inputs(
        affinity_csv=affinity_csv,
        pdb_csv=pdb_csv,
        pose_csv=pose_csv,
        split_csv=split_csv,
        affinity_split_csv=affinity_split_csv,
        uniprot_to_pdb_csv=uniprot_to_pdb_csv,
        ligand_to_pose_csv=ligand_to_pose_csv,
    )
    prediction_df, model_metadata = fit_and_predict(resolved)
    split_metrics, split_notes = summarize_prediction_metrics(
        prediction_df,
        requested_splits=("train", "val", "test"),
    )

    metrics = {
        "model": model_metadata,
        "metrics": split_metrics,
        "metric_notes": split_notes,
        "num_examples": int(len(prediction_df)),
    }
    return prediction_df, metrics


def parse_args() -> argparse.Namespace:
    """CLI for the manifest-only Glide-score baseline."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the standardized MVP experiment flow using the Glide-score baseline."
        )
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run directory name under mvp/runs.")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help="Base directory that stores experiment runs.")
    parser.add_argument("--raw-binding-csv", default=str(RAW_BINDING_CSV), help="Path to raw mol_binding_data.csv.")
    parser.add_argument("--affinity-csv", default=str(AFFINITY_MANIFEST_CSV), help="Path to affinity_manifest.csv.")
    parser.add_argument("--pdb-csv", default=str(PDB_MANIFEST_CSV), help="Path to pdb_manifest.csv.")
    parser.add_argument("--pose-csv", default=str(POSE_MANIFEST_CSV), help="Path to pose_manifest.csv.")
    parser.add_argument(
        "--forbid-test-from-split-csv",
        default=None,
        help="Optional previous split manifest whose test ligands may not reappear in test.",
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
        "--tanimoto-threshold",
        type=float,
        default=DEFAULT_TANIMOTO_THRESHOLD,
        help="Ligands with similarity >= this value cannot span active splits.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_FINGERPRINT_RADIUS,
        help="Morgan fingerprint radius.",
    )
    parser.add_argument(
        "--fp-size",
        type=int,
        default=DEFAULT_FINGERPRINT_SIZE,
        help="Morgan fingerprint size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Base random seed for split generation.",
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=DEFAULT_NUM_RESTARTS,
        help="Number of randomized restarts to evaluate before choosing the best split.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Desired train fraction.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Desired validation fraction.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Desired test fraction.",
    )
    parser.add_argument(
        "--report-restart-diversity",
        action="store_true",
        help="Log overlap statistics comparing alternate restarts with the winning split.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    split_config = SplitConfig(
        tanimoto_threshold=args.tanimoto_threshold,
        radius=args.radius,
        fp_size=args.fp_size,
        seed=args.seed,
        num_restarts=args.num_restarts,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        forbid_test_from_split_csv=args.forbid_test_from_split_csv,
        report_restart_diversity=args.report_restart_diversity,
    )
    _, metrics, artifacts = run_model_experiment(
        run_name=args.run_name,
        model_name=MODEL_NAME,
        predictor=fit_and_predict,
        raw_binding_csv=args.raw_binding_csv,
        affinity_csv=args.affinity_csv,
        pdb_csv=args.pdb_csv,
        pose_csv=args.pose_csv,
        runs_dir=args.runs_dir,
        uniprot_to_pdb_csv=args.uniprot_to_pdb_csv,
        ligand_to_pose_csv=args.ligand_to_pose_csv,
        split_config=split_config,
    )
    print(json.dumps({"metrics": metrics, "run_dir": str(artifacts.run_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
