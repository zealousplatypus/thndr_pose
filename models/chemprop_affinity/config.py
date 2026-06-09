from dataclasses import dataclass, field
from pathlib import Path

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

@dataclass(frozen=True)
class TrainingConfig: 
    """Trainer and dataloader settings."""

    seed: int = 0
    batch_size: int = 64
    max_epochs: int = 50

@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level Chemprop affinity experiment config."""

    experiment_name: str
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def run_dir(self) -> Path:
        """Return the configured run directory."""
        return self.paths.runs_dir / self.experiment_name