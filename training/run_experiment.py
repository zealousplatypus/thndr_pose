"""Model-agnostic experiment runner for manifest-driven MVP evaluations."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


MVP_ROOT = Path(__file__).resolve().parent.parent
if str(MVP_ROOT) not in sys.path:
    sys.path.append(str(MVP_ROOT))

from data_processing.build_affinity_split_manifest import build_affinity_split_manifest  # noqa: E402
from data_processing.build_split_manifest import DEFAULT_NUM_RESTARTS, build_split_manifest  # noqa: E402
from data_processing.constants import (  # noqa: E402
    ALL_SPLIT_NAMES,
    AFFINITY_MANIFEST_CSV,
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
)
from data_processing.manifest_io import ensure_parent_dir, read_csv_checked  # noqa: E402
from training.input_resolution import resolve_experiment_inputs  # noqa: E402
from training.metrics import summarize_prediction_metrics, write_prediction_scatter_plot  # noqa: E402


PredictorFn = Callable[[object], tuple[pd.DataFrame, dict[str, object]]]


@dataclass(frozen=True)
class SplitConfig:
    """Serializable split configuration for one experiment run."""

    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD
    radius: int = DEFAULT_FINGERPRINT_RADIUS
    fp_size: int = DEFAULT_FINGERPRINT_SIZE
    seed: int = DEFAULT_SPLIT_SEED
    num_restarts: int = DEFAULT_NUM_RESTARTS
    train_fraction: float = DEFAULT_TRAIN_FRACTION
    val_fraction: float = DEFAULT_VAL_FRACTION
    test_fraction: float = DEFAULT_TEST_FRACTION
    forbid_test_from_split_csv: str | None = None
    report_restart_diversity: bool = False


@dataclass(frozen=True)
class RunArtifacts:
    """Standardized file layout for one experiment run."""

    run_dir: Path
    split_manifest_csv: Path
    affinity_split_manifest_csv: Path
    predictions_csv: Path
    metrics_json: Path
    run_note_path: Path
    train_plot_path: Path
    test_plot_path: Path
    split_conflict_graph_cache: Path
    uniprot_to_pdb_csv: Path | None
    ligand_to_pose_csv: Path | None


def _copy_selector_csv(source_path: str | Path | None, destination_path: Path | None) -> Path | None:
    """Copy a selector CSV into the run directory."""
    if source_path is None or destination_path is None:
        return None
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Selector CSV not found: {source}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination_path)
    return destination_path


def prepare_run_artifacts(
    run_name: str,
    runs_dir: str | Path = RUNS_DIR,
    uniprot_to_pdb_csv: str | Path | None = None,
    ligand_to_pose_csv: str | Path | None = None,
) -> RunArtifacts:
    """Create the run directory layout and local copies of selector CSVs."""
    run_dir = ensure_parent_dir(Path(runs_dir) / run_name / "placeholder.txt").parent
    selector_uniprot = _copy_selector_csv(
        uniprot_to_pdb_csv,
        run_dir / "uniprot_to_pdb.csv" if uniprot_to_pdb_csv is not None else None,
    )
    selector_pose = _copy_selector_csv(
        ligand_to_pose_csv,
        run_dir / "ligand_to_pose.csv" if ligand_to_pose_csv is not None else None,
    )
    return RunArtifacts(
        run_dir=run_dir,
        split_manifest_csv=run_dir / "split_manifest.csv",
        affinity_split_manifest_csv=run_dir / "affinity_split_manifest.csv",
        predictions_csv=run_dir / "glide_score_baseline_predictions.csv",
        metrics_json=run_dir / "glide_score_baseline_metrics.json",
        run_note_path=run_dir / "README.md",
        train_plot_path=run_dir / "train_pred_vs_true.png",
        test_plot_path=run_dir / "test_pred_vs_true.png",
        split_conflict_graph_cache=run_dir / "split_conflict_graph.pkl",
        uniprot_to_pdb_csv=selector_uniprot,
        ligand_to_pose_csv=selector_pose,
    )


def _compute_source_counts(
    raw_binding_csv: str | Path,
    affinity_csv: str | Path,
    selected_uniprot_ids: list[str],
) -> dict[str, object]:
    """Compute raw and deduplicated affinity counts for the selected targets."""
    raw_binding_df = read_csv_checked(raw_binding_csv, ["uniprot_id"])
    affinity_df = read_csv_checked(affinity_csv, ["uniprot_id", "ligand", "affinity"])
    selected_uniprot_ids = sorted(set(selected_uniprot_ids))

    raw_counts_by_uniprot = {
        uniprot_id: int((raw_binding_df["uniprot_id"] == uniprot_id).sum())
        for uniprot_id in selected_uniprot_ids
    }
    affinity_counts_by_uniprot = {
        uniprot_id: int((affinity_df["uniprot_id"] == uniprot_id).sum())
        for uniprot_id in selected_uniprot_ids
    }
    summary: dict[str, object] = {
        "raw_binding_rows_by_uniprot": raw_counts_by_uniprot,
        "affinity_manifest_rows_by_uniprot": affinity_counts_by_uniprot,
        "num_binding_rows_for_selected_uniprots_in_source": int(sum(raw_counts_by_uniprot.values())),
        "num_binding_rows_for_selected_uniprots_after_affinity_dedup": int(sum(affinity_counts_by_uniprot.values())),
    }
    if len(selected_uniprot_ids) == 1:
        uniprot_id = selected_uniprot_ids[0]
        summary["selected_uniprot_id"] = uniprot_id
        summary["num_binding_rows_for_selected_uniprot_in_source"] = raw_counts_by_uniprot[uniprot_id]
        summary["num_binding_rows_for_selected_uniprot_after_affinity_dedup"] = affinity_counts_by_uniprot[uniprot_id]
    return summary


def _build_run_note(
    run_name: str,
    model_name: str,
    selected_pdbs: pd.DataFrame,
    split_config: SplitConfig,
    split_sizes: dict[str, int],
) -> str:
    """Build a short Markdown note for the run directory."""
    selected_uniprots = sorted(selected_pdbs["uniprot_id"].drop_duplicates().tolist())
    selected_pdb_keys = sorted(selected_pdbs["pdb_key"].drop_duplicates().tolist())
    return "\n".join(
        [
            f"# {run_name}",
            "",
            f"- Model: `{model_name}`",
            f"- Selected UniProt IDs: `{', '.join(selected_uniprots) if selected_uniprots else 'none'}`",
            f"- Selected PDB keys: `{', '.join(selected_pdb_keys) if selected_pdb_keys else 'none'}`",
            "- Split determinism: splits are deterministic for this experiment definition.",
            "- Determinism inputs: selector CSV contents, split seed, Tanimoto threshold, fingerprint parameters, and split fractions.",
            "- Important: when selector CSVs are applied, the splits are not determined by `mol_binding_data.csv` alone.",
            "",
            "## Split Parameters",
            f"- `seed`: `{split_config.seed}`",
            f"- `tanimoto_threshold`: `{split_config.tanimoto_threshold}`",
            f"- `radius`: `{split_config.radius}`",
            f"- `fp_size`: `{split_config.fp_size}`",
            f"- `num_restarts`: `{split_config.num_restarts}`",
            f"- `train_fraction`: `{split_config.train_fraction}`",
            f"- `val_fraction`: `{split_config.val_fraction}`",
            f"- `test_fraction`: `{split_config.test_fraction}`",
            "",
            "## Final Split Sizes",
            *(f"- `{split_name}`: `{split_sizes.get(split_name, 0)}`" for split_name in ALL_SPLIT_NAMES),
        ]
    )


def run_model_experiment(
    *,
    run_name: str,
    model_name: str,
    predictor: PredictorFn,
    raw_binding_csv: str | Path = RAW_BINDING_CSV,
    affinity_csv: str | Path = AFFINITY_MANIFEST_CSV,
    pdb_csv: str | Path = PDB_MANIFEST_CSV,
    pose_csv: str | Path = POSE_MANIFEST_CSV,
    runs_dir: str | Path = RUNS_DIR,
    uniprot_to_pdb_csv: str | Path | None = None,
    ligand_to_pose_csv: str | Path | None = None,
    split_config: SplitConfig = SplitConfig(),
) -> tuple[pd.DataFrame, dict[str, object], RunArtifacts]:
    """Run one full experiment and write standardized artifacts to disk."""
    artifacts = prepare_run_artifacts(
        run_name=run_name,
        runs_dir=runs_dir,
        uniprot_to_pdb_csv=uniprot_to_pdb_csv,
        ligand_to_pose_csv=ligand_to_pose_csv,
    )

    split_df = build_split_manifest(
        affinity_csv=affinity_csv,
        pdb_csv=pdb_csv,
        pose_csv=pose_csv,
        tanimoto_threshold=split_config.tanimoto_threshold,
        radius=split_config.radius,
        fp_size=split_config.fp_size,
        seed=split_config.seed,
        num_restarts=split_config.num_restarts,
        train_fraction=split_config.train_fraction,
        val_fraction=split_config.val_fraction,
        test_fraction=split_config.test_fraction,
        conflict_graph_cache=artifacts.split_conflict_graph_cache,
        forbid_test_from_split_csv=split_config.forbid_test_from_split_csv,
        uniprot_to_pdb_csv=artifacts.uniprot_to_pdb_csv,
        ligand_to_pose_csv=artifacts.ligand_to_pose_csv,
        report_restart_diversity=split_config.report_restart_diversity,
    )
    split_df.to_csv(artifacts.split_manifest_csv, index=False)

    affinity_split_df = build_affinity_split_manifest(
        affinity_csv=affinity_csv,
        split_csv=artifacts.split_manifest_csv,
        pdb_csv=pdb_csv,
        pose_csv=pose_csv,
        uniprot_to_pdb_csv=artifacts.uniprot_to_pdb_csv,
        ligand_to_pose_csv=artifacts.ligand_to_pose_csv,
    )
    affinity_split_df.to_csv(artifacts.affinity_split_manifest_csv, index=False)

    selector_only_resolved = resolve_experiment_inputs(
        affinity_csv=affinity_csv,
        pdb_csv=pdb_csv,
        pose_csv=pose_csv,
        split_csv=None,
        affinity_split_csv=None,
        uniprot_to_pdb_csv=artifacts.uniprot_to_pdb_csv,
        ligand_to_pose_csv=artifacts.ligand_to_pose_csv,
    )
    resolved = resolve_experiment_inputs(
        affinity_csv=affinity_csv,
        pdb_csv=pdb_csv,
        pose_csv=pose_csv,
        split_csv=artifacts.split_manifest_csv,
        affinity_split_csv=artifacts.affinity_split_manifest_csv,
        uniprot_to_pdb_csv=artifacts.uniprot_to_pdb_csv,
        ligand_to_pose_csv=artifacts.ligand_to_pose_csv,
    )
    prediction_df, model_metadata = predictor(resolved)
    prediction_df.to_csv(artifacts.predictions_csv, index=False)

    split_metrics, split_notes = summarize_prediction_metrics(
        prediction_df,
        requested_splits=("train", "val", "test"),
    )
    train_plot_note = write_prediction_scatter_plot(
        prediction_df,
        split_name="train",
        output_path=artifacts.train_plot_path,
        run_name=run_name,
    )
    test_plot_note = write_prediction_scatter_plot(
        prediction_df,
        split_name="test",
        output_path=artifacts.test_plot_path,
        run_name=run_name,
    )

    selected_uniprot_ids = sorted(resolved.selected_pdbs["uniprot_id"].drop_duplicates().tolist())
    selected_pdb_ids = sorted(resolved.selected_pdbs["pdb_id"].drop_duplicates().tolist())
    selected_pdb_keys = sorted(resolved.selected_pdbs["pdb_key"].drop_duplicates().tolist())
    split_sizes = {split_name: int((split_df["split"] == split_name).sum()) for split_name in ALL_SPLIT_NAMES}
    source_counts = _compute_source_counts(
        raw_binding_csv=raw_binding_csv,
        affinity_csv=affinity_csv,
        selected_uniprot_ids=selected_uniprot_ids,
    )
    metrics_payload: dict[str, object] = {
        "run_name": run_name,
        "model": model_metadata,
        "metrics": split_metrics,
        "metric_notes": split_notes,
        "num_examples": int(len(prediction_df)),
        "num_examples_scored_in_experiment": int(len(prediction_df)),
        "num_examples_eligible_after_selector_filtering": int(len(selector_only_resolved.affinity_examples)),
        "source_counts": source_counts,
        "experiment": {
            "selected_uniprot_ids": selected_uniprot_ids,
            "selected_pdb_ids": selected_pdb_ids,
            "selected_pdb_keys": selected_pdb_keys,
        },
        "split_parameters": asdict(split_config),
        "split_sizes": split_sizes,
        "artifacts": {
            "run_dir": str(artifacts.run_dir),
            "uniprot_to_pdb_csv": str(artifacts.uniprot_to_pdb_csv) if artifacts.uniprot_to_pdb_csv else None,
            "ligand_to_pose_csv": str(artifacts.ligand_to_pose_csv) if artifacts.ligand_to_pose_csv else None,
            "split_manifest_csv": str(artifacts.split_manifest_csv),
            "affinity_split_manifest_csv": str(artifacts.affinity_split_manifest_csv),
            "predictions_csv": str(artifacts.predictions_csv),
            "metrics_json": str(artifacts.metrics_json),
            "train_plot_path": str(artifacts.train_plot_path),
            "test_plot_path": str(artifacts.test_plot_path),
            "run_note_path": str(artifacts.run_note_path),
        },
        "plot_notes": {
            "train": train_plot_note,
            "test": test_plot_note,
        },
        "determinism_note": (
            "Splits are deterministic for the experiment definition and split parameters. "
            "When selector CSVs are applied, they are not determined by mol_binding_data.csv alone."
        ),
    }

    artifacts.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with artifacts.metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, sort_keys=True)

    artifacts.run_note_path.write_text(
        _build_run_note(
            run_name=run_name,
            model_name=model_name,
            selected_pdbs=resolved.selected_pdbs,
            split_config=split_config,
            split_sizes=split_sizes,
        ),
        encoding="utf-8",
    )
    return prediction_df, metrics_payload, artifacts


__all__ = [
    "RunArtifacts",
    "SplitConfig",
    "prepare_run_artifacts",
    "run_model_experiment",
]
