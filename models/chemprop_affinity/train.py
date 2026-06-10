import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from chemprop import models

from data_processing.common.manifest_io import write_manifest

from models.common.lightning_callbacks import EpochLossHistoryCallback
from models.common.plots import write_train_val_loss_plot
from models.common.run_io import copy_experiment_config, make_run_dir

from .config import ExperimentConfig, load_experiment_config
from .data import build_chemprop_data, ChempropDataBundle
from .model import build_chemprop_model
from .evaluate import predict

LOGGER = logging.getLogger(__name__)

def train_chemprop_affinity(config_path: str | Path) -> pd.DataFrame:

    config = load_experiment_config(config_path)

    # set up 
    chemprop_data_bundle = build_chemprop_data(config)
    mpnn = build_chemprop_model(config, chemprop_data_bundle.scaler)
    
    # Configure model checkpointing
    run_dir = make_run_dir(
        config.paths.runs_dir,
        config.experiment_name,
        config.overwrite
    )
    copy_experiment_config(config_path, run_dir)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir,  # Directory where model checkpoints will be saved
        "best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        "val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=True,  # Always save the most recent checkpoint, even if it's not the best
    )
    loss_history = EpochLossHistoryCallback()
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True, # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="auto",
        max_epochs=config.training.max_epochs, # number of epochs to train for
        callbacks=[
            checkpoint_callback,
            loss_history.bind_lightning_callback()
        ] # Use the configured checkpoint callback
    )

    # Training
    trainer.fit(mpnn, chemprop_data_bundle.train_dataloader, chemprop_data_bundle.dev_dataloader)

    # Saving stuff
    loss_history_df = loss_history.to_dataframe()
    if not loss_history_df.empty:
        write_manifest(loss_history_df, run_dir / "training_loss_history.csv")
        write_train_val_loss_plot(
            loss_history_df,
            run_dir / "loss_curve.png",
            title=f"{config.experiment_name} train/val loss",
        )
        LOGGER.info("Wrote training loss history and loss curve plot")



    best_model = checkpoint_callback.best_model_path
    mpnn = models.MPNN.load_from_checkpoint(best_model)

    torch.save(mpnn.state_dict(), run_dir / "model" / "best_model_state_dict.pt")

    train_predictions = predict(mpnn, chemprop_data_bundle.train_dataloader)
    dev_predictions = predict(mpnn, chemprop_data_bundle.dev_dataloader)
    test_predictions = predict(mpnn, chemprop_data_bundle.test_dataloader)

    print(chemprop_data_bundle.test_dataloader)
    print(test_predictions)

    # # Writing train and val predictions
    # prediction_splits = ("train", "val")
    # predictions_df = predict_splits(
    #     trainer=trainer,
    #     model=mpnn,
    #     data_bundle=chemprop_data_bundle,
    #     dataloaders=chemprop_data.dataloaders,
    #     splits=prediction_splits,
    # )
    # if config.outputs.save_train_val_predictions:
    #     from models.common.predictions import write_split_outputs

    #     write_split_outputs(predictions_df, run_dir, splits=prediction_splits)
    #     write_manifest(predictions_df, run_dir / "predictions_train_val.csv")

def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Train a Chemprop ligand encoder as a baseline."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Path to experiment JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()

def main(): 
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )
    train_chemprop_affinity(args.experiment)


if __name__ == "__main__":
    main()