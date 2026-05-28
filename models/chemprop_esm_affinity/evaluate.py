"""Prediction helpers for Chemprop + ESM affinity models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import AffinityDataBundle


def _flatten_predictions(raw_predictions: Any) -> np.ndarray:
    """Convert Lightning/Chemprop prediction output into a flat NumPy array."""
    arrays: list[np.ndarray] = []
    batches = raw_predictions if isinstance(raw_predictions, list) else [raw_predictions]
    for batch in batches:
        if isinstance(batch, (list, tuple)) and len(batch) == 1:
            batch = batch[0]
        if hasattr(batch, "detach"):
            batch = batch.detach().cpu().numpy()
        arrays.append(np.asarray(batch, dtype=float).reshape(-1))
    if not arrays:
        return np.empty((0,), dtype=float)
    return np.concatenate(arrays)


def predict_split(
    trainer: Any,
    model: Any,
    dataloader: Any,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict one split and attach predictions to metadata rows."""
    raw_predictions = trainer.predict(model, dataloaders=dataloader)
    predictions = _flatten_predictions(raw_predictions)
    if len(predictions) != len(metadata_df):
        raise ValueError(
            "Prediction count does not match metadata rows: "
            f"{len(predictions)} predictions vs {len(metadata_df)} metadata rows"
        )
    output_df = metadata_df.copy()
    output_df["predicted_affinity"] = predictions
    return output_df


def predict_splits(
    trainer: Any,
    model: Any,
    data_bundle: AffinityDataBundle,
    dataloaders: dict[str, Any],
    splits: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Predict multiple splits and concatenate metadata-aligned outputs."""
    split_predictions = []
    for split in splits:
        split_predictions.append(
            predict_split(
                trainer=trainer,
                model=model,
                dataloader=dataloaders[split],
                metadata_df=data_bundle.metadata_by_split[split],
            )
        )
    if not split_predictions:
        return pd.DataFrame()
    return pd.concat(split_predictions, ignore_index=True)

