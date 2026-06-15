from pathlib import Path
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

# general model utilities
from models.common.lightning_callbacks import EpochLossHistoryCallback
from models.common.plots import write_train_val_loss_plot
from models.common.run_io import copy_experiment_config, make_run_dir
from models.pyg_pose.config import load_experiment_config

# model specific imports
from models.pyg_pose.data import build_pyg_pose_data
from models.pyg_pose.model import LitPoseGNN


def train(config_path: str | Path) -> pd.DataFrame:
    config = load_experiment_config(config_path)
    L.seed_everything(config.training.seed, workers=True)

    run_dir = make_run_dir(
        config.paths.runs_dir,
        config.experiment_name,
        config.overwrite,
    )
    copy_experiment_config(config_path, run_dir)
    checkpoint_dir = run_dir / "checkpoints"

    data_bundle = build_pyg_pose_data(config)
    train_loader = data_bundle.data_loaders["train"]
    val_loader = data_bundle.data_loaders.get("val")

    num_atom_features = data_bundle.datasets["train"][0].x.shape[1]
    model = LitPoseGNN(
        num_atom_features=num_atom_features,
        hidden_dim=config.model.hidden_dim,
        learning_rate=config.training.learning_rate,
    )

    monitor = "val_loss" if val_loader is not None else "train_loss"
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="best",
        monitor=monitor,
        mode="min",
        save_last=True,
        enable_version_counter=False,
    )

    loss_history = EpochLossHistoryCallback()

    trainer = L.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        max_epochs=config.training.max_epochs,
        default_root_dir=run_dir,
        callbacks=[
            checkpoint_callback,
            loss_history.bind_lightning_callback(),
        ],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    history_df = loss_history.to_dataframe()
    history_df.to_csv(run_dir / "loss_history.csv", index=False)
    if not history_df.empty:
        write_train_val_loss_plot(history_df, run_dir / "loss_history.png")

    return history_df