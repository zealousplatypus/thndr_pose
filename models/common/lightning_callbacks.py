"""PyTorch Lightning callbacks shared by model training baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def _metric_float(metrics: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Return the first available metric as a float."""
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    return None


@dataclass
class EpochLossHistoryCallback:
    """Collect per-epoch train and validation loss values during Lightning training."""

    train_metric_keys: tuple[str, ...] = ("train_loss_epoch", "train_loss")
    val_metric_keys: tuple[str, ...] = ("val_loss",)
    records: list[dict[str, float | int | None]] = field(default_factory=list)

    def _on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        metrics = dict(trainer.callback_metrics)
        self.records.append(
            {
                "epoch": int(trainer.current_epoch),
                "train_loss": _metric_float(metrics, self.train_metric_keys),
                "val_loss": _metric_float(metrics, self.val_metric_keys),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return collected epoch losses as a dataframe."""
        if not self.records:
            return pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
        return pd.DataFrame(self.records).sort_values("epoch", ignore_index=True)

    def bind_lightning_callback(self) -> Any:
        """Return a Lightning Callback instance wired to this collector."""
        history = self

        try:
            from lightning.pytorch.callbacks import Callback
        except ModuleNotFoundError:
            from pytorch_lightning.callbacks import Callback

        class _Collector(Callback):
            def on_validation_epoch_end(
                self,
                trainer: Any,
                pl_module: Any,
            ) -> None:
                history._on_validation_epoch_end(trainer, pl_module)

        return _Collector()
