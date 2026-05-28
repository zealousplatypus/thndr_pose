"""Explicit held-out test evaluation for Chemprop + ESM affinity runs."""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from data_processing.common.manifest_io import write_manifest
from models.common.run_io import write_json

from .config import load_experiment_config
from .data import build_chemprop_data, load_affinity_data
from .evaluate import predict_splits
from .model import build_chemprop_model
from .train import _lightning_modules, _load_checkpoint_state


LOGGER = logging.getLogger(__name__)


def _load_preprocessing_state(run_dir: Path) -> dict[str, Any]:
    """Load saved scaler and model input metadata."""
    path = run_dir / "model" / "preprocessing_state.pkl"
    with path.open("rb") as handle:
        return pickle.load(handle)


def evaluate_test(config_path: str | Path, checkpoint_path: str | Path) -> Path:
    """Load a saved Chemprop model and write timestamped test outputs."""
    _, Trainer, _, _ = _lightning_modules()
    config = load_experiment_config(config_path, validate_run_dir=False)
    run_dir = config.run_dir
    preprocessing_state = _load_preprocessing_state(run_dir)

    data_bundle = load_affinity_data(config, require_val=False)
    chemprop_data = build_chemprop_data(
        config,
        data_bundle,
        target_scaler=preprocessing_state.get("target_scaler"),
        descriptor_scaler=preprocessing_state.get("protein_descriptor_scaler"),
        fit_scalers=False,
    )
    model = build_chemprop_model(
        config,
        esm_dim=int(preprocessing_state.get("esm_dim", data_bundle.esm_dim)),
        target_scaler=preprocessing_state.get("target_scaler"),
        descriptor_scaler=preprocessing_state.get("protein_descriptor_scaler"),
    )
    _load_checkpoint_state(model, checkpoint_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir / f"test_evaluation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    trainer = Trainer(
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        default_root_dir=output_dir,
    )
    predictions_df = predict_splits(
        trainer=trainer,
        model=model,
        data_bundle=data_bundle,
        dataloaders=chemprop_data.dataloaders,
        splits=("test",),
    )
    from models.common.predictions import write_split_outputs

    write_split_outputs(predictions_df, output_dir, splits=("test",))
    write_manifest(predictions_df, output_dir / "predictions_test.csv")
    write_json(
        {
            "experiment_name": config.experiment_name,
            "checkpoint": str(checkpoint_path),
            "num_test": int(len(predictions_df)),
            "output_dir": str(output_dir),
        },
        output_dir / "test_eval_metadata.json",
    )
    LOGGER.info("Wrote explicit test evaluation to %s", output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Chemprop + ESM checkpoint on the test split."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Path to the copied run experiment.json.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a saved Chemprop/PyTorch Lightning checkpoint.",
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
    evaluate_test(args.experiment, args.checkpoint)


if __name__ == "__main__":
    main()

