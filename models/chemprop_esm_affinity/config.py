"""Experiment config loading and validation for Chemprop + ESM affinity runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_processing.common.constants import (
    ACTIVE_SPLIT_NAMES,
    AFFINITY_SPLIT_MANIFEST_CSV,
    ESM_EMBEDDINGS_NPY,
    MVP_ROOT,
    PROTEIN_MANIFEST_CSV,
    RUNS_DIR,
)


_SAFE_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class PathConfig:
    """Filesystem inputs and run root."""

    affinity_split_csv: Path = AFFINITY_SPLIT_MANIFEST_CSV
    protein_manifest_csv: Path | None = PROTEIN_MANIFEST_CSV
    esm_embeddings_npy: Path = ESM_EMBEDDINGS_NPY
    runs_dir: Path = RUNS_DIR


@dataclass(frozen=True)
class DataConfig:
    """Manifest columns and row filters."""

    smiles_column: str = "ligand"
    target_column: str = "affinity"
    split_column: str = "split"
    protein_idx_column: str = "protein_idx"
    uniprot_id_column: str = "uniprot_id"
    ligand_idx_column: str = "ligand_idx"
    active_splits: tuple[str, ...] = ACTIVE_SPLIT_NAMES
    include_uniprot_ids: tuple[str, ...] = ()
    drop_invalid_smiles: bool = True


@dataclass(frozen=True)
class MessagePassingConfig:
    """Chemprop message passing settings."""

    type: str = "bond"
    hidden_dim: int = 300
    depth: int = 3
    dropout: float = 0.1
    undirected: bool = False
    activation: str = "relu"


@dataclass(frozen=True)
class FFNConfig:
    """Chemprop regression feed-forward-network settings."""

    hidden_dim: int = 300
    num_layers: int = 2
    dropout: float = 0.1


@dataclass(frozen=True)
class ModelConfig:
    """Chemprop model settings."""

    message_passing: MessagePassingConfig = field(default_factory=MessagePassingConfig)
    aggregation: str = "mean"
    ffn: FFNConfig = field(default_factory=FFNConfig)
    batch_norm: bool = True
    normalize_targets: bool = True
    normalize_protein_descriptors: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Trainer and dataloader settings."""

    seed: int = 0
    batch_size: int = 64
    max_epochs: int = 50
    accelerator: str = "auto"
    devices: str | int | list[int] = "auto"
    num_workers: int = 0
    learning_rate: float = 0.001
    patience: int = 10
    monitor_metric: str = "val_loss"


@dataclass(frozen=True)
class OutputConfig:
    """Output and checkpoint settings."""

    save_predictions_all: bool = False
    save_train_val_predictions: bool = True
    save_checkpoints: bool = True
    save_model_state_dict: bool = True
    evaluate_test_during_training: bool = False
    overwrite_existing_run: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level Chemprop + ESM experiment config."""

    experiment_name: str
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

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
    data = data or {}
    return PathConfig(
        affinity_split_csv=_resolve_path(
            data.get("affinity_split_csv", AFFINITY_SPLIT_MANIFEST_CSV)
        )
        or AFFINITY_SPLIT_MANIFEST_CSV,
        protein_manifest_csv=_resolve_path(
            data.get("protein_manifest_csv", PROTEIN_MANIFEST_CSV)
        ),
        esm_embeddings_npy=_resolve_path(
            data.get("esm_embeddings_npy", ESM_EMBEDDINGS_NPY)
        )
        or ESM_EMBEDDINGS_NPY,
        runs_dir=_resolve_path(data.get("runs_dir", RUNS_DIR)) or RUNS_DIR,
    )


def _data_config(data: dict[str, Any] | None) -> DataConfig:
    data = data or {}
    include_uniprot_ids = data.get("include_uniprot_ids") or ()
    active_splits = data.get("active_splits", ACTIVE_SPLIT_NAMES)
    return DataConfig(
        smiles_column=data.get("smiles_column", "ligand"),
        target_column=data.get("target_column", "affinity"),
        split_column=data.get("split_column", "split"),
        protein_idx_column=data.get("protein_idx_column", "protein_idx"),
        uniprot_id_column=data.get("uniprot_id_column", "uniprot_id"),
        ligand_idx_column=data.get("ligand_idx_column", "ligand_idx"),
        active_splits=tuple(str(split) for split in active_splits),
        include_uniprot_ids=tuple(str(uniprot_id) for uniprot_id in include_uniprot_ids),
        drop_invalid_smiles=bool(data.get("drop_invalid_smiles", True)),
    )


def _model_config(data: dict[str, Any] | None) -> ModelConfig:
    data = data or {}
    mp_data = data.get("message_passing", {})
    ffn_data = data.get("ffn", {})
    return ModelConfig(
        message_passing=MessagePassingConfig(
            type=mp_data.get("type", "bond"),
            hidden_dim=int(mp_data.get("hidden_dim", 300)),
            depth=int(mp_data.get("depth", 3)),
            dropout=float(mp_data.get("dropout", 0.1)),
            undirected=bool(mp_data.get("undirected", False)),
            activation=mp_data.get("activation", "relu"),
        ),
        aggregation=data.get("aggregation", "mean"),
        ffn=FFNConfig(
            hidden_dim=int(ffn_data.get("hidden_dim", 300)),
            num_layers=int(ffn_data.get("num_layers", 2)),
            dropout=float(ffn_data.get("dropout", 0.1)),
        ),
        batch_norm=bool(data.get("batch_norm", True)),
        normalize_targets=bool(data.get("normalize_targets", True)),
        normalize_protein_descriptors=bool(
            data.get("normalize_protein_descriptors", True)
        ),
    )


def _training_config(data: dict[str, Any] | None) -> TrainingConfig:
    data = data or {}
    return TrainingConfig(
        seed=int(data.get("seed", 0)),
        batch_size=int(data.get("batch_size", 64)),
        max_epochs=int(data.get("max_epochs", 50)),
        accelerator=data.get("accelerator", "auto"),
        devices=data.get("devices", "auto"),
        num_workers=int(data.get("num_workers", 0)),
        learning_rate=float(data.get("learning_rate", 0.001)),
        patience=int(data.get("patience", 10)),
        monitor_metric=data.get("monitor_metric", "val_loss"),
    )


def _output_config(data: dict[str, Any] | None) -> OutputConfig:
    data = data or {}
    return OutputConfig(
        save_predictions_all=bool(data.get("save_predictions_all", False)),
        save_train_val_predictions=bool(data.get("save_train_val_predictions", True)),
        save_checkpoints=bool(data.get("save_checkpoints", True)),
        save_model_state_dict=bool(data.get("save_model_state_dict", True)),
        evaluate_test_during_training=bool(
            data.get("evaluate_test_during_training", False)
        ),
        overwrite_existing_run=bool(data.get("overwrite_existing_run", False)),
    )


def load_experiment_config(
    path: str | Path,
    validate_paths: bool = True,
    validate_run_dir: bool = True,
) -> ExperimentConfig:
    """Load and validate an experiment JSON file."""
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_name = str(raw.get("experiment_name", "")).strip()
    config = ExperimentConfig(
        experiment_name=experiment_name,
        paths=_path_config(raw.get("paths")),
        data=_data_config(raw.get("data")),
        model=_model_config(raw.get("model")),
        training=_training_config(raw.get("training")),
        outputs=_output_config(raw.get("outputs")),
    )
    validate_experiment_config(
        config,
        validate_paths=validate_paths,
        validate_run_dir=validate_run_dir,
    )
    return config


def validate_experiment_config(
    config: ExperimentConfig,
    validate_paths: bool = True,
    validate_run_dir: bool = True,
) -> None:
    """Fail fast on invalid experiment configuration."""
    if not config.experiment_name:
        raise ValueError("experiment_name must be nonempty")
    if not _SAFE_EXPERIMENT_RE.match(config.experiment_name):
        raise ValueError(
            "experiment_name may only contain letters, numbers, '_', '-', and '.'"
        )
    if "train" not in config.data.active_splits:
        raise ValueError("active_splits must include 'train'")
    if config.model.message_passing.type != "bond":
        raise ValueError("Only bond message passing is supported for the first baseline")
    if config.model.aggregation != "mean":
        raise ValueError("Only mean aggregation is supported for the first baseline")
    if config.training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if config.training.max_epochs <= 0:
        raise ValueError("training.max_epochs must be positive")

    if validate_paths:
        required_paths = [
            config.paths.affinity_split_csv,
            config.paths.esm_embeddings_npy,
        ]
        if config.paths.protein_manifest_csv is not None:
            required_paths.append(config.paths.protein_manifest_csv)
        missing = [path for path in required_paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                "Experiment config references missing files: "
                + ", ".join(str(path) for path in missing)
            )

    if (
        validate_run_dir
        and config.run_dir.exists()
        and not config.outputs.overwrite_existing_run
    ):
        raise FileExistsError(
            f"Run directory already exists: {config.run_dir}. "
            "Set overwrite_existing_run=true to reuse it."
        )

