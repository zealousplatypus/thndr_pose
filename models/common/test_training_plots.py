"""Tests for shared training loss plotting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from models.common.lightning_callbacks import EpochLossHistoryCallback
from models.common.plots import write_train_val_loss_plot


def test_write_train_val_loss_plot_creates_png(tmp_path: Path) -> None:
    history_df = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train_loss": [1.2, 0.8, 0.5],
            "val_loss": [1.5, 1.0, 0.9],
        }
    )
    output_path = write_train_val_loss_plot(history_df, tmp_path / "loss_curve.png")
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_epoch_loss_history_callback_to_dataframe() -> None:
    callback = EpochLossHistoryCallback(
        records=[
            {"epoch": 0, "train_loss": 1.0, "val_loss": 1.5},
            {"epoch": 1, "train_loss": 0.7, "val_loss": 1.1},
        ]
    )
    history_df = callback.to_dataframe()
    assert list(history_df["epoch"]) == [0, 1]
    assert history_df.loc[1, "val_loss"] == 1.1
