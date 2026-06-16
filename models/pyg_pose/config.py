"""
Configuration for the PyG Pose model.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_processing.common.constants import (
    AFFINITY_SPLIT_MANIFEST_CSV,
    MVP_ROOT,
    POSE_LMDB_DIR,
    POSE_MANIFEST_CSV,
    RUNS_DIR,
)


@dataclass(frozen=True)
class PathsConfig:
    """Paths for the PyG Pose model."""

    affinity_split_csv: Path = AFFINITY_SPLIT_MANIFEST_CSV
    pose_manifest_csv: Path = POSE_MANIFEST_CSV
    pose_lmdb_dir: Path = POSE_LMDB_DIR
    runs_dir: Path = RUNS_DIR


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration for the PyG Pose model."""

    hidden_dim: int = 128


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration for the PyG Pose model."""

    seed: int = 0
    batch_size: int = 64
    num_workers: int = 0
    max_epochs: int = 200
    learning_rate: float = 0.001


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment configuration for the PyG Pose model."""

    experiment_name: str
    uniprot_id: str
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    overwrite: bool = False

    @property
    def run_dir(self) -> Path:
        """Return the configured run directory."""
        return self.paths.runs_dir / self.experiment_name


def _resolve_path(value: str | Path | None) -> Path | None:
    """Resolve relative config paths from the repository root."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return MVP_ROOT / path


def _paths_config(data: dict[str, Any] | None) -> PathsConfig:
    return PathsConfig(
        pose_manifest_csv=_resolve_path(data.get("pose_manifest_csv", POSE_MANIFEST_CSV)),
        affinity_split_csv=_resolve_path(data.get("affinity_split_csv", AFFINITY_SPLIT_MANIFEST_CSV)),
        pose_lmdb_dir=_resolve_path(data.get("pose_lmdb_dir", POSE_LMDB_DIR)),
        runs_dir=_resolve_path(data.get("runs_dir", RUNS_DIR)),
    )


def _model_config(data: dict[str, Any] | None) -> ModelConfig:
    return ModelConfig(
        hidden_dim=int(data.get("hidden_dim", 128)),
    )


def _training_config(data: dict[str, Any] | None) -> TrainingConfig:
    return TrainingConfig(
        seed=int(data.get("seed", 0)),
        batch_size=int(data.get("batch_size", 64)),
        num_workers=int(data.get("num_workers", 0)),
        max_epochs=int(data.get("max_epochs", 200)),
        learning_rate=float(data.get("learning_rate", 0.001)),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment JSON file."""
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_name = str(raw.get("experiment_name", "")).strip()
    if not experiment_name:
        raise ValueError("experiment_name is required and must be non-empty.")
    if "uniprot_id" not in raw:
        raise ValueError("uniprot_id is required.")
    uniprot_id = str(raw["uniprot_id"]).strip()
    if not uniprot_id:
        raise ValueError("uniprot_id is required and must be non-empty.")
    overwrite = bool(raw.get("overwrite_existing_run", False))
    return ExperimentConfig(
        experiment_name=experiment_name,
        uniprot_id=uniprot_id,
        paths=_paths_config(raw.get("paths")),
        model=_model_config(raw.get("model")),
        training=_training_config(raw.get("training")),
        overwrite=overwrite,
    )
