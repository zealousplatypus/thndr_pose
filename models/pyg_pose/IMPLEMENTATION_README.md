# PyG Pose Implementation Spec

Minimal goal: predict affinity rows from `affinity_split_manifest.csv` by mapping
`ligand + protein -> top pose -> streaming PyG dataloaders -> Lightning model`.

This package should not rebuild pose graphs. It should consume:

- `affinity_split_manifest.csv`: labels, ligand/protein IDs, and train/val/test split.
- `pose_manifest.csv`: pose metadata and `pose_id` keys.
- `pose_lmdb/`: pickled PyG `Data` objects keyed by `pose_id`.

## Component 1: Config

File: `models/pyg_pose/config.py`

Required config fields:

- `experiment_name`
- `uniprot_id`
- `paths.affinity_split_csv`
- `paths.pose_manifest_csv`
- `paths.pose_lmdb_dir`
- `paths.runs_dir`
- `model.hidden_dim`
- `training.seed`
- `training.batch_size`
- `training.num_workers`
- `training.max_epochs`
- `training.learning_rate`
- `overwrite_existing_run`

Behavior:

- Resolve relative paths from the repo root.
- Expose `config.run_dir == config.paths.runs_dir / config.experiment_name`.
- Keep defaults aligned with `data_processing.common.constants`.

## Component 2: Pose Training DataFrame

File: `models/pyg_pose/data.py`

Function:

```python
def build_pose_training_df(config: ExperimentConfig) -> pd.DataFrame:
    ...
```

Inputs:

- `config.paths.affinity_split_csv`
- `config.paths.pose_manifest_csv`
- `config.uniprot_id`

Output dataframe must include at minimum:

- `uniprot_id`
- `protein_idx`
- `ligand`
- `ligand_idx`
- `affinity`
- `split`
- `pose_id`
- `pdb_key`
- `pdb_id`
- `glide_score`
- `pose_rank`

Behavior:

- Read `affinity_split_manifest.csv`.
- Filter to `config.uniprot_id`.
- Read `pose_manifest.csv`.
- Normalize `is_top_rank` to bool if needed.
- Keep only top-pose rows where `is_top_rank == True`.
- Join affinity rows to pose rows on `["uniprot_id", "ligand"]`.
- Use an inner join for now, dropping affinity rows without a top pose.
- Raise `ValueError` if no train examples remain.
- Preserve `split` from the affinity split manifest.

Non-goals:

- Do not load LMDB objects here.
- Do not use all poses yet.
- Do not recompute split labels.

## Component 3: Streaming Dataset

File: `models/pyg_pose/data.py`

Class:

```python
class PoseAffinityDataset(torch_geometric.data.Dataset):
    def __init__(self, df: pd.DataFrame, lmdb_path: str | Path, transform=None):
        ...
```

Required dataframe columns:

- `pose_id`
- `affinity`

Behavior:

- Store the dataframe reset to a clean row index.
- Lazily open LMDB inside the worker process, not during dataframe construction.
- In `get(idx)`:
  - read `pose_id` from `self.df`
  - fetch the pickled PyG object from LMDB
  - unpickle it
  - attach `data.y = torch.tensor([affinity], dtype=torch.float32)`
  - attach or preserve `data.pose_id`
  - return the PyG `Data`
- Raise `KeyError` if a `pose_id` is missing from LMDB.

Non-goals:

- Do not preload graphs.
- Do not normalize targets in the dataset.
- Do not mutate the source dataframe.

## Component 4: Dataloader Bundle

File: `models/pyg_pose/data.py`

Dataclass:

```python
@dataclass(frozen=True)
class PoseDataBundle:
    df: pd.DataFrame
    datasets: dict[str, PoseAffinityDataset]
    data_loaders: dict[str, DataLoader]
```

Function:

```python
def build_pyg_pose_data(config: ExperimentConfig) -> PoseDataBundle:
    ...
```

Behavior:

- Call `build_pose_training_df(config)`.
- For each split in `("train", "val", "test")` present in the dataframe:
  - create a split dataframe
  - create a `PoseAffinityDataset`
  - create a PyG `DataLoader`
- Return `data_loaders` as a dictionary keyed by split.
- Shuffle only the train dataloader.
- Use `config.training.batch_size` and `config.training.num_workers`.

Required contract:

```python
bundle = build_pyg_pose_data(config)
train_loader = bundle.data_loaders["train"]
val_loader = bundle.data_loaders.get("val")
test_loader = bundle.data_loaders.get("test")
```

## Component 5: Model

File: `models/pyg_pose/model.py`

Class:

```python
class LitPoseGNN(L.LightningModule):
    ...
```

Minimum behavior:

- Accept `num_atom_features`, `hidden_dim`, and `learning_rate`.
- Concatenate `batch.x.float()` with `batch.pos.float()`.
- Run graph convolutions and global mean pooling.
- Predict one scalar affinity per graph.
- Use MSE loss for train and validation.
- Log `train_loss` and `val_loss`.
- Configure Adam with `learning_rate`.

Small fix:

- When logging loss, pass the real graph batch size to avoid Lightning inferring atom count as batch size.

## Component 6: Training Entrypoint

File: `models/pyg_pose/train.py`

Function:

```python
def train(config_path: str | Path) -> pd.DataFrame:
    ...
```

Behavior:

- Load config.
- Set the Lightning seed from config.
- Build the data bundle.
- Infer `num_atom_features` from the first train example.
- Create `LitPoseGNN`.
- Create run directory at `config.run_dir`.
- Save checkpoints under `config.run_dir / "checkpoints"`.
- Fit with:
  - `train_dataloaders=bundle.data_loaders["train"]`
  - `val_dataloaders=bundle.data_loaders.get("val")`
- Return the epoch loss history dataframe.
- Write loss history and loss plot if existing common helpers make that simple.

Required imports:

- Use `models.common.run_io.make_run_dir`, not `models.common.utils`.
- Use `models.common.lightning_callbacks.EpochLossHistoryCallback`.

Non-goals:

- Prediction CSVs can wait until training is stable.
- No CLI is required for the first working version.

## Component 7: Minimal Tests

Suggested file: `models/pyg_pose/test_data.py`

Test fixtures:

- Tiny affinity split CSV with train/val/test rows.
- Tiny pose manifest CSV with matching top poses.
- Tiny LMDB with pickled PyG `Data` objects.

Required tests:

- `build_pose_training_df` returns joined rows with `pose_id`, `affinity`, and `split`.
- Missing top-pose affinity rows are dropped.
- `build_pyg_pose_data` returns `{"train": ..., "val": ..., "test": ...}` when all splits exist.
- One batch from the train loader contains `x`, `pos`, `edge_index`, `y`, and `pose_id`.

Optional smoke test:

- Run `trainer.fit(..., fast_dev_run=True)` on the toy loaders.

## Implementation Order

1. Fix `config.py`.
2. Implement `build_pose_training_df`.
3. Refactor `PoseAffinityDataset` to accept a dataframe.
4. Implement `PoseDataBundle` and `build_pyg_pose_data`.
5. Wire `train.py`.
6. Add focused data tests.
7. Add prediction/evaluation only after the core training loop works.
