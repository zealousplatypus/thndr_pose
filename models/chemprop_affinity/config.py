import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_processing.common.constants import (
    ACTIVE_SPLIT_NAMES, 
    AFFINITY_SPLIT_MANIFEST_CSV, 
    MVP_ROOT,
    RUNS_DIR,
)

@dataclass(frozen=True)
class PathConfig:
    """Filesystem inputs and run root."""

    affinity_split_csv: Path = AFFINITY_SPLIT_MANIFEST_CSV
    runs_dir: Path = RUNS_DIR

@dataclass(frozen=True)
class ModelConfig:
    """Chemprop model settings."""

    message_hidden_dim: int = 300
    message_depth: int = 3
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.1
    batchnorm: bool = True

@dataclass(frozen=True)
class TrainingConfig: 
    """Trainer and dataloader settings."""

    seed: int = 0
    batch_size: int = 64
    max_epochs: int = 50
    learning_rate: float = 0.001

@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level Chemprop affinity experiment config."""

    experiment_name: str
    uniprot_id: str
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

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

def _path_config(data: dict[str, Any] | None) -> PathConfig:
    return PathConfig(
        affinity_split_csv=_resolve_path(
            data.get("affinity_split_csv", AFFINITY_SPLIT_MANIFEST_CSV)
        ),
        runs_dir=_resolve_path(
            data.get("output_dir", RUNS_DIR)
        )
    )

def _training_config(data: dict[str, Any] | None) -> TrainingConfig:
    return TrainingConfig(
        seed=int(data.get("seed", 0)),
        batch_size=int(data.get("batch_size", 64)),
        max_epochs=int(data.get("max_epochs", 50)),
        learning_rate=float(data.get("learning_rate", 0.001))
    )

def _model_config(data: dict[str, Any] | None) -> ModelConfig:
    return ModelConfig(
        message_depth=int(data.get("message_depth", 3)),
        message_hidden_dim=int(data.get("message_hidden_dim", 300)),
        ffn_hidden_dim=int(data.get("ffn_hidden_dim", 300)),
        ffn_num_layers=int(data.get("ffn_num_layers", 2)),
        dropout=float(data.get("dropout", 0.1)),
        batchnorm=bool(data.get("batchnorm", True))
    )

def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment JSON file."""
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_name = str(raw.get("experiment_name", "")).strip()
    uniprot_id = str(raw["uniprot_id"]).strip() #TODO make this fail safely
    config = ExperimentConfig(
        experiment_name=experiment_name,
        uniprot_id=uniprot_id,
        paths=_path_config(raw.get("paths")),
        training=_training_config(raw.get("training")),
        model=_model_config(raw.get("model"))
    )
    return config