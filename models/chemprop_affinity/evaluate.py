import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl

from .data import ChempropDataBundle


def predict_splits(model, chemprop_data_bundle: ChempropDataBundle) -> pd.DataFrame:
    """Return a dataframe with true and predicted affinities for all splits."""
    df = chemprop_data_bundle.df_filtered
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="cpu",
        devices=1,
    )

    split_dfs: list[pd.DataFrame] = []
    with torch.inference_mode():
        for split, data_loader in chemprop_data_bundle.data_loaders.items():
            raw_predictions = trainer.predict(model, data_loader)
            predictions = np.concatenate(raw_predictions, axis=0).flatten()
            split_df = df[df["split"] == split].copy()
            if len(predictions) != len(split_df):
                raise ValueError(
                    f"Prediction count does not match {split} rows: "
                    f"{len(predictions)} predictions vs {len(split_df)} rows"
                )
            split_df = split_df.rename(columns={"affinity": "true_affinity"})
            split_df["predicted_affinity"] = predictions
            split_dfs.append(split_df)

    return pd.concat(split_dfs, ignore_index=True)
