"""CLI for training the Chemprop + ESM affinity baseline."""

from __future__ import annotations

import argparse
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from data_processing.common.manifest_io import ensure_parent_dir, write_manifest
from models.common.lightning_callbacks import EpochLossHistoryCallback
from models.common.plots import write_train_val_loss_plot
from models.common.run_io import copy_experiment_config, make_run_dir, write_run_metadata

from .config import ExperimentConfig, load_experiment_config
from .data import (
    build_chemprop_data,
    format_dry_run_summary,
    load_affinity_data,
    summarize_affinity_data,
)
from .evaluate import predict_splits
from .model import build_chemprop_model


LOGGER = logging.getLogger(__name__)


def _lightning_modules() -> tuple[Any, Any, Any, Any]:
    """Import Lightning modules with compatibility for package names."""
    try:
        import lightning.pytorch as lightning
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    except ModuleNotFoundError:
        try:
            import pytorch_lightning as lightning
            from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch Lightning is required for Chemprop training. "
                "Install lightning or pytorch-lightning in this environment."
            ) from exc
    return lightning, lightning.Trainer, ModelCheckpoint, EarlyStopping


def _torch_module() -> Any:
    """Import torch lazily."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for Chemprop training. Install torch in this environment."
        ) from exc
    return torch


def _save_preprocessing_state(
    config: ExperimentConfig,
    chemprop_data: Any,
    esm_dim: int,
    output_path: str | Path,
) -> Path:
    """Persist scalers and model input metadata."""
    output_path = ensure_parent_dir(output_path)
    state = {
        "target_scaler": chemprop_data.target_scaler,
        "protein_descriptor_scaler": chemprop_data.descriptor_scaler,
        "columns": {
            "smiles_column": config.data.smiles_column,
            "target_column": config.data.target_column,
            "split_column": config.data.split_column,
            "protein_idx_column": config.data.protein_idx_column,
            "uniprot_id_column": config.data.uniprot_id_column,
            "ligand_idx_column": config.data.ligand_idx_column,
        },
        "esm_dim": int(esm_dim),
        "active_splits": tuple(config.data.active_splits),
        "model_input_dims": {
            "esm_dim": int(esm_dim),
            "message_passing_hidden_dim": config.model.message_passing.hidden_dim,
        },
    }
    with output_path.open("wb") as handle:
        pickle.dump(state, handle)
    return output_path


def _load_checkpoint_state(model: Any, checkpoint_path: str | Path) -> None:
    """Load Lightning checkpoint weights into an existing model."""
    torch = _torch_module()
    checkpoint_path = Path(checkpoint_path)
    load_kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in torch.load.__code__.co_varnames:
        # Local Lightning .ckpt files embed Chemprop transforms (e.g. ScaleTransform).
        load_kwargs["weights_only"] = False

    checkpoint = torch.load(checkpoint_path, **load_kwargs)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)


def _copy_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Copy a checkpoint if source and destination differ."""
    source = Path(source)
    destination = ensure_parent_dir(destination)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def dry_run(config_path: str | Path) -> dict[str, Any]:
    """Validate config/data and print dry-run counts."""
    config = load_experiment_config(config_path, validate_run_dir=False)
    data_bundle = load_affinity_data(config, require_val=False)
    summary = summarize_affinity_data(config, data_bundle)
    print(format_dry_run_summary(summary))
    return summary


def train_chemprop_esm_affinity(config_path: str | Path) -> pd.DataFrame:
    """Train the Chemprop + ESM baseline and write train/val outputs."""
    lightning, Trainer, ModelCheckpoint, EarlyStopping = _lightning_modules()
    torch = _torch_module()

    config = load_experiment_config(config_path)
    if config.outputs.evaluate_test_during_training:
        raise ValueError(
            "evaluate_test_during_training must remain false. "
            "Use evaluate_test.py for explicit test evaluation."
        )

    lightning.seed_everything(config.training.seed, workers=True)
    run_dir = make_run_dir(
        config.paths.runs_dir,
        config.experiment_name,
        overwrite=config.outputs.overwrite_existing_run,
    )
    copy_experiment_config(config_path, run_dir)

    data_bundle = load_affinity_data(config, require_val=True)
    chemprop_data = build_chemprop_data(config, data_bundle, fit_scalers=True)
    model = build_chemprop_model(
        config,
        esm_dim=data_bundle.esm_dim,
        target_scaler=chemprop_data.target_scaler,
        descriptor_scaler=chemprop_data.descriptor_scaler,
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor=config.training.monitor_metric,
        mode="min",
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor=config.training.monitor_metric,
        mode="min",
        patience=config.training.patience,
    )
    loss_history = EpochLossHistoryCallback()
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        default_root_dir=run_dir,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            loss_history.bind_lightning_callback(),
        ],
    )
    trainer.fit(
        model,
        train_dataloaders=chemprop_data.dataloaders["train"],
        val_dataloaders=chemprop_data.dataloaders["val"],
    )

    best_checkpoint = Path(checkpoint_callback.best_model_path)
    if not best_checkpoint.exists():
        best_checkpoint = checkpoint_dir / "last.ckpt"
    model_best_checkpoint = _copy_checkpoint(best_checkpoint, run_dir / "model" / "best_model.ckpt")
    if (checkpoint_dir / "last.ckpt").exists():
        _copy_checkpoint(checkpoint_dir / "last.ckpt", run_dir / "checkpoints" / "last.ckpt")

    loss_history_df = loss_history.to_dataframe()
    if not loss_history_df.empty:
        write_manifest(loss_history_df, run_dir / "training_loss_history.csv")
        write_train_val_loss_plot(
            loss_history_df,
            run_dir / "loss_curve.png",
            title=f"{config.experiment_name} train/val loss",
        )
        LOGGER.info("Wrote training loss history and loss curve plot")

    _load_checkpoint_state(model, model_best_checkpoint)
    if config.outputs.save_model_state_dict:
        torch.save(model.state_dict(), run_dir / "model" / "best_model_state_dict.pt")
    _save_preprocessing_state(
        config,
        chemprop_data,
        data_bundle.esm_dim,
        run_dir / "model" / "preprocessing_state.pkl",
    )

    prediction_splits = ("train", "val")
    predictions_df = predict_splits(
        trainer=trainer,
        model=model,
        data_bundle=data_bundle,
        dataloaders=chemprop_data.dataloaders,
        splits=prediction_splits,
    )
    if config.outputs.save_train_val_predictions:
        from models.common.predictions import write_split_outputs

        write_split_outputs(predictions_df, run_dir, splits=prediction_splits)
        write_manifest(predictions_df, run_dir / "predictions_train_val.csv")

    split_counts = {
        split: int(len(data_bundle.split_dfs.get(split, [])))
        for split in config.data.active_splits
    }
    write_run_metadata(
        {
            "experiment_name": config.experiment_name,
            "num_train": split_counts.get("train", 0),
            "num_val": split_counts.get("val", 0),
            "num_test": split_counts.get("test", 0),
            "esm_embedding_shape": list(data_bundle.esm_shape),
            "num_unique_ligands": int(data_bundle.examples_df[config.data.smiles_column].nunique()),
            "num_unique_proteins": int(
                data_bundle.examples_df[config.data.uniprot_id_column].nunique()
            ),
            "best_checkpoint": str(Path("checkpoints") / best_checkpoint.name),
            "num_epochs_logged": int(len(loss_history_df)),
        },
        run_dir,
    )
    LOGGER.info("Wrote Chemprop + ESM run outputs to %s", run_dir)
    return predictions_df


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Train a Chemprop ligand encoder with frozen ESM protein descriptors."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Path to experiment JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data and print counts without training.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(levelname)s: %(message)s",
    )
    if args.dry_run:
        dry_run(args.experiment)
    else:
        train_chemprop_esm_affinity(args.experiment)


if __name__ == "__main__":
    main()

