"""Experiment config loading for the linear Glide score baseline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_processing.common.constants import (
    ACTIVE_SPLIT_NAMES,
    AFFINITY_SPLIT_MANIFEST_CSV,
    MVP_ROOT,
    POSE_MANIFEST_CSV,
    RUNS_DIR,
)


_SAFE_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class PathConfig:
    """Filesystem inputs and run root."""

    pose_csv: Path = POSE_MANIFEST_CSV
    affinity_split_csv: Path = AFFINITY_SPLIT_MANIFEST_CSV
    runs_dir: Path = RUNS_DIR


@dataclass(frozen=True)
class DataConfig:
    """Manifest columns and row filters."""

    uniprot_id_column: str = "uniprot_id"
    ligand_column: str = "ligand"
    target_column: str = "affinity"
    split_column: str = "split"
    active_splits: tuple[str, ...] = ACTIVE_SPLIT_NAMES
    include_uniprot_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputConfig:
    """Output settings."""

    overwrite_existing_run: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level linear Glide experiment config."""

    experiment_name: str
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    @property
    def run_dir(self) -> Path:
        """Return the configured run directory."""
        return self.paths.runs_dir / self.experiment_name


def _resolve_path(value: str | Path | None, default: Path) -> Path:
    """Resolve relative config paths from the repository root."""
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return MVP_ROOT / path


def load_experiment_config(
    path: str | Path,
    validate_paths: bool = True,
    validate_run_dir: bool = True,
) -> ExperimentConfig:
    """Load and validate a linear Glide experiment JSON file."""
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    paths_raw = raw.get("paths", {})
    data_raw = raw.get("data", {})
    outputs_raw = raw.get("outputs", {})
    include_uniprot_ids = data_raw.get("include_uniprot_ids") or ()
    active_splits = data_raw.get("active_splits", ACTIVE_SPLIT_NAMES)

    config = ExperimentConfig(
        experiment_name=str(raw.get("experiment_name", "")).strip(),
        paths=PathConfig(
            pose_csv=_resolve_path(paths_raw.get("pose_csv"), POSE_MANIFEST_CSV),
            affinity_split_csv=_resolve_path(
                paths_raw.get("affinity_split_csv"),
                AFFINITY_SPLIT_MANIFEST_CSV,
            ),
            runs_dir=_resolve_path(paths_raw.get("runs_dir"), RUNS_DIR),
        ),
        data=DataConfig(
            uniprot_id_column=data_raw.get("uniprot_id_column", "uniprot_id"),
            ligand_column=data_raw.get("ligand_column", "ligand"),
            target_column=data_raw.get("target_column", "affinity"),
            split_column=data_raw.get("split_column", "split"),
            active_splits=tuple(str(split) for split in active_splits),
            include_uniprot_ids=tuple(
                str(uniprot_id) for uniprot_id in include_uniprot_ids
            ),
        ),
        outputs=OutputConfig(
            overwrite_existing_run=bool(
                outputs_raw.get("overwrite_existing_run", False)
            ),
        ),
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

    if validate_paths:
        required_paths = [
            config.paths.pose_csv,
            config.paths.affinity_split_csv,
        ]
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
