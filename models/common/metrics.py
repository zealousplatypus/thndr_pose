"""Metrics for affinity prediction baselines."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def pearson_r(df: pd.DataFrame) -> float:
    """Compute Pearson r for predicted vs true affinity."""
    if len(df) < 2:
        return float("nan")
    value = df["predicted_affinity"].corr(df["true_affinity"], method="pearson")
    return float(value) if value is not None else float("nan")


def rmse(df: pd.DataFrame) -> float:
    """Compute root mean squared error."""
    if df.empty:
        return float("nan")
    residuals = (
        df["predicted_affinity"].to_numpy(dtype=float)
        - df["true_affinity"].to_numpy(dtype=float)
    )
    return float(math.sqrt(np.mean(np.square(residuals))))


def mae(df: pd.DataFrame) -> float:
    """Compute mean absolute error."""
    if df.empty:
        return float("nan")
    residuals = (
        df["predicted_affinity"].to_numpy(dtype=float)
        - df["true_affinity"].to_numpy(dtype=float)
    )
    return float(np.mean(np.abs(residuals)))


def affinity_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Return the standard affinity regression metrics."""
    return {
        "pearson_r": pearson_r(df),
        "rmse": rmse(df),
        "mae": mae(df),
    }

