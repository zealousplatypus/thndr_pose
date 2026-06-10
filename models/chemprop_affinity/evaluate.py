import pandas as pd
import numpy as np
import torch
from lightning import pytorch as pl
from pathlib import Path

# from chemprop import data, featurizers, models


def predict(model, data_loader):
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="cpu",
            devices=1
        )
        prediction = trainer.predict(model, data_loader)
    return prediction