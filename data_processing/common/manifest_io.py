"""Small I/O utilities shared across MVP manifest builders.

These helpers keep the scripts short and make validation failures easier to
read than if each script reimplemented the same checks differently.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for an output file if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_checked(path: str | Path, required_columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Read a CSV and fail fast if expected columns are missing."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    return df


def write_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a manifest CSV after ensuring the output directory exists."""
    path = ensure_parent_dir(path)
    df.to_csv(path, index=False)
    return path


def assert_unique(df: pd.DataFrame, columns: list[str] | tuple[str, ...], name: str) -> None:
    """Check that a set of manifest columns defines unique rows."""
    duplicates = df[df.duplicated(subset=list(columns), keep=False)]
    if not duplicates.empty:
        sample = duplicates.head(10)
        raise ValueError(
            f"{name} is not unique on columns {list(columns)}. "
            f"First duplicate rows:\n{sample}"
        )
