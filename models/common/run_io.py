"""Run-directory helpers for model baselines."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from data_processing.common.manifest_io import ensure_parent_dir


def make_run_dir(
    runs_dir: str | Path,
    experiment_name: str,
    overwrite: bool = False,
) -> Path:
    """Create and return `runs/<experiment_name>`."""
    run_dir = Path(runs_dir) / experiment_name
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Set overwrite_existing_run=true to reuse it."
            )
        if not run_dir.is_dir():
            raise FileExistsError(f"Run path exists and is not a directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "model").mkdir(exist_ok=True)
    return run_dir


def copy_experiment_config(source_path: str | Path, run_dir: str | Path) -> Path:
    """Copy the source experiment JSON into the run directory."""
    destination = Path(run_dir) / "experiment.json"
    ensure_parent_dir(destination)
    shutil.copy2(source_path, destination)
    return destination


def _json_default(value: Any) -> Any:
    """Serialize pathlib paths and dataclasses for metadata JSON."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a pretty JSON file."""
    path = ensure_parent_dir(path)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def write_run_metadata(metadata: dict[str, Any], run_dir: str | Path) -> Path:
    """Write `run_metadata.json` inside a run directory."""
    return write_json(metadata, Path(run_dir) / "run_metadata.json")

