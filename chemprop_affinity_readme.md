# Chemprop + ESM Affinity Model

This package trains a Chemprop message-passing neural network (MPNN) on ligand SMILES, conditioned on frozen ESM protein embeddings passed as Chemprop `X_d` descriptors. Training uses PyTorch Lightning on train/val splits; held-out test evaluation is a separate explicit step.

Run all commands from the repository root.

## Overview

```
affinity_split_manifest.csv  ──┐
protein_manifest.csv         ──┼──► data.py ──► Chemprop datasets/dataloaders
esm_embeddings.float32.npy   ──┘         │
                                         ▼
experiment.json ──► config.py ──► model.py ──► Chemprop MPNN
                                         │
                                         ▼
                                   train.py (Lightning fit)
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            evaluate.py (predictions)              evaluate_test.py (held-out test)
```

## File reference

### `__init__.py`

Package docstring only. Marks this directory as the **Chemprop + ESM affinity** model package: a trainable Chemprop ligand encoder conditioned on frozen ESM protein descriptors.

---

### `config.py`

Experiment configuration loading and validation.

**Dataclasses**

| Class | Purpose |
|-------|---------|
| `PathConfig` | Input paths (affinity split CSV, protein manifest, ESM `.npy`, runs directory) |
| `DataConfig` | Manifest column names, active splits, optional UniProt filters, SMILES validation |
| `MessagePassingConfig` | Chemprop bond message-passing hyperparameters |
| `FFNConfig` | Regression feed-forward network hyperparameters |
| `ModelConfig` | Full model settings (message passing, aggregation, FFN, normalization flags) |
| `TrainingConfig` | Lightning trainer settings (seed, batch size, epochs, LR, early stopping) |
| `OutputConfig` | What to save (predictions, checkpoints, overwrite behavior) |
| `ExperimentConfig` | Top-level config combining all sections; exposes `run_dir` property |

**Functions**

- `load_experiment_config(path, validate_paths=True, validate_run_dir=True)` — Parse a JSON experiment file into an `ExperimentConfig` and validate it.
- `validate_experiment_config(config, ...)` — Fail-fast checks: nonempty safe experiment name, required splits, supported model types (bond MP + mean aggregation only for this baseline), positive batch size/epochs, input file existence, and run-directory collision unless `overwrite_existing_run` is set.

Relative paths in JSON are resolved from the repository root (`MVP_ROOT`).

---

### `data.py`

Data loading, validation, and Chemprop dataset construction.

**Dataclasses**

| Class | Purpose |
|-------|---------|
| `AffinityDataBundle` | Validated manifest rows split by train/val/test, per-split metadata DataFrames, per-split ESM descriptor arrays, and the full ESM embedding matrix |
| `ChempropDataBundle` | Chemprop `MoleculeDataset` objects and dataloaders per split, plus fitted target and descriptor scalers |

**Key functions**

- `load_esm_embeddings(path)` — Load a 2D NumPy ESM embedding matrix.
- `load_affinity_data(config, require_val=True)` — Read the affinity split manifest, filter by active splits and optional UniProt IDs, validate numeric targets and protein indices, attach ESM rows via `protein_idx`, and validate against the protein manifest when configured.
- `summarize_affinity_data(config, bundle)` — Build a dry-run summary dict (row counts, split counts, unique proteins/ligands, ESM shape).
- `format_dry_run_summary(summary)` — Render the summary as human-readable text.
- `build_chemprop_data(config, bundle, target_scaler=None, descriptor_scaler=None, fit_scalers=True)` — Convert each split into Chemprop `MoleculeDatapoint` objects (SMILES + affinity target + ESM `X_d`), optionally drop invalid SMILES, fit or apply target/descriptor normalization on train, and build shuffled train / non-shuffled val/test dataloaders.

Internal helpers handle Chemprop version compatibility (lazy imports, kwargs filtering for differing Chemprop v2 APIs).

---

### `model.py`

Chemprop MPNN construction for the ESM-conditioned baseline.

**Key function**

- `build_chemprop_model(config, esm_dim, target_scaler=None, descriptor_scaler=None)` — Assemble a Chemprop `MPNN`:
  - **Message passing:** `BondMessagePassing` from config
  - **Aggregation:** `MeanAggregation`
  - **Predictor:** `RegressionFFN` with input dimension = message-passing output + `esm_dim` (ligand graph embedding concatenated with protein descriptor)
  - **Transforms:** optional `UnscaleTransform` on targets and `ScaleTransform` on `X_d` from fitted scalers

Only bond message passing and mean aggregation are supported in this baseline.

---

### `train.py`

CLI and training orchestration.

**Key functions**

- `dry_run(config_path)` — Load config (without run-dir check), load data, print split/protein/ligand counts.
- `train_chemprop_esm_affinity(config_path)` — Full training pipeline:
  1. Load and validate config; reject `evaluate_test_during_training` (test must use `evaluate_test.py`)
  2. Create run directory under `runs/<experiment_name>/`
  3. Load data, build Chemprop datasets/dataloaders, build model
  4. Train with PyTorch Lightning (early stopping on `val_loss`, checkpointing)
  5. Save best checkpoint, optional state dict, preprocessing scalers (`preprocessing_state.pkl`), loss history CSV/plot
  6. Predict train and val splits; write per-split outputs and combined CSV
  7. Write `run_metadata.json`

**CLI**

```bash
python -m models.chemprop_esm_affinity.train --experiment configs/chemprop_esm_affinity_example.json
python -m models.chemprop_esm_affinity.train --experiment configs/chemprop_esm_affinity_example.json --dry-run
```

---

### `evaluate.py`

Prediction helpers used during and after training.

**Functions**

- `_flatten_predictions(raw_predictions)` — Normalize Lightning/Chemprop batch outputs into a flat NumPy array.
- `predict_split(trainer, model, dataloader, metadata_df)` — Run `trainer.predict` on one split and attach `predicted_affinity` to metadata rows (row count must match).
- `predict_splits(trainer, model, data_bundle, dataloaders, splits)` — Predict multiple splits and concatenate into one DataFrame.

---

### `evaluate_test.py`

Explicit held-out test evaluation (test is intentionally excluded from the training loop).

**Key function**

- `evaluate_test(config_path, checkpoint_path)` — Load saved preprocessing state and checkpoint from a completed run, rebuild data/model with train-fitted scalers (no refitting), predict the test split only, and write timestamped outputs under `runs/<experiment_name>/test_evaluation_<timestamp>/`.

**CLI**

```bash
python -m models.chemprop_esm_affinity.evaluate_test \
  --experiment runs/<experiment_name>/experiment.json \
  --checkpoint runs/<experiment_name>/checkpoints/best.ckpt
```

---

### `test_config_data.py`

Unit tests for config loading and data pipeline (pytest).

Covers:

- Protein filtering via `include_uniprot_ids`
- Correct ESM descriptor lookup by `protein_idx`
- Dry-run summary formatting and split counts
- Rejection of out-of-range `protein_idx` values

Run with:

```bash
pytest models/chemprop_esm_affinity/test_config_data.py
```

---

## Typical workflow

1. Build manifests and ESM embeddings (see root `README.MD` data processing section).
2. Copy or edit an experiment JSON in `configs/` (example: `configs/chemprop_esm_affinity_example.json`).
3. Dry-run to verify data:

   ```bash
   python -m models.chemprop_esm_affinity.train --experiment configs/chemprop_esm_affinity_example.json --dry-run
   ```

4. Train:

   ```bash
   python -m models.chemprop_esm_affinity.train --experiment configs/chemprop_esm_affinity_example.json
   ```

5. Evaluate test when ready:

   ```bash
   python -m models.chemprop_esm_affinity.evaluate_test \
     --experiment runs/<experiment_name>/experiment.json \
     --checkpoint runs/<experiment_name>/checkpoints/best.ckpt
   ```

## Run outputs

Under `runs/<experiment_name>/`:

| Artifact | Description |
|----------|-------------|
| `experiment.json` | Copy of the experiment config |
| `checkpoints/best.ckpt`, `checkpoints/last.ckpt` | Lightning checkpoints |
| `model/best_model.ckpt` | Copied best checkpoint |
| `model/best_model_state_dict.pt` | Raw PyTorch weights (optional) |
| `model/preprocessing_state.pkl` | Target/descriptor scalers and column metadata |
| `training_loss_history.csv`, `loss_curve.png` | Epoch train/val loss |
| `predictions_train_val.csv`, per-split prediction files | Train/val predictions |
| `run_metadata.json` | Run summary statistics |
| `test_evaluation_<timestamp>/` | Test predictions (from `evaluate_test.py`) |

## Dependencies

- **Chemprop v2** — datapoints, datasets, MPNN model
- **PyTorch** — training backend
- **PyTorch Lightning** (`lightning` or `pytorch-lightning`) — trainer, checkpointing, early stopping
- **pandas**, **numpy** — manifest and embedding I/O

Shared utilities from `models/common/` (callbacks, plots, run I/O, prediction writers) and `data_processing/common/` (constants, manifest I/O) are used but live outside this directory.
