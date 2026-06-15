from dataclasses import dataclass
from pathlib import Path
import lmdb
import pickle
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from data_processing.common.constants import (
    AFFINITY_SPLIT_MANIFEST_COLUMNS,
    POSE_MANIFEST_COLUMNS,
)
from data_processing.common.manifest_io import read_csv_checked
from models.pyg_pose.config import ExperimentConfig


POSE_TRAINING_COLUMNS = (
    "uniprot_id",
    "protein_idx",
    "ligand",
    "ligand_idx",
    "affinity",
    "split",
    "pose_id",
    "pdb_key",
    "pdb_id",
    "glide_score",
    "pose_rank",
)


def build_pose_training_df(config: ExperimentConfig) -> pd.DataFrame:
    """Join affinity split rows to top-pose rows for one protein."""
    affinity_df = read_csv_checked(
        config.paths.affinity_split_csv,
        AFFINITY_SPLIT_MANIFEST_COLUMNS,
    )
    affinity_df = affinity_df[
        affinity_df["uniprot_id"].astype(str) == str(config.uniprot_id)
    ].copy()

    pose_df = read_csv_checked(config.paths.pose_manifest_csv, POSE_MANIFEST_COLUMNS)
    pose_df["is_top_rank"] = (
        pose_df["is_top_rank"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
    )
    pose_df["glide_score"] = pd.to_numeric(pose_df["glide_score"], errors="raise")

    top_pose_df = pose_df[
        (pose_df["uniprot_id"].astype(str) == str(config.uniprot_id))
        & (pose_df["is_top_rank"])
    ].copy()

    top_pose_df = (
        top_pose_df.sort_values(
            ["uniprot_id", "ligand", "glide_score", "pose_rank", "pose_id"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        )
        .drop_duplicates(["uniprot_id", "ligand"], keep="first")
    )
    top_pose_df = top_pose_df[
        ["uniprot_id", "ligand", "pose_id", "pdb_key", "pdb_id", "glide_score", "pose_rank"]
    ]

    pose_training_df = affinity_df.merge(
        top_pose_df,
        on=["uniprot_id", "ligand"],
        how="inner",
        validate="many_to_one",
    )

    dropped_rows = len(affinity_df) - len(pose_training_df)
    if dropped_rows:
        print(f"Dropping {dropped_rows} affinity rows without top poses")

    if (pose_training_df["split"] == "train").sum() == 0:
        raise ValueError("No train examples remain after filtering to top poses.")

    return pose_training_df.loc[:, POSE_TRAINING_COLUMNS].reset_index(drop=True)


class PoseAffinityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lmdb_path: str | Path, transform=None):
        super().__init__(root=None, transform=transform)

        self.df = df.copy().reset_index(drop=True)
        self.lmdb_path = str(lmdb_path)
        self.env = None

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        pose_id = str(row["pose_id"])

        env = self._get_env()
        with env.begin(write=False) as txn:
            value = txn.get(pose_id.encode("utf-8"))

        if value is None:
            raise KeyError(f"Missing pose_id in LMDB: {pose_id}")

        data = pickle.loads(value)
        data.y = torch.tensor([row["affinity"]], dtype=torch.float32)
        data.pose_id = pose_id
        return data

    def _get_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=True,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        return self.env


@dataclass(frozen=True)
class PoseDataBundle:
    df: pd.DataFrame
    datasets: dict[str, PoseAffinityDataset]
    data_loaders: dict[str, DataLoader]


def build_pyg_pose_data(config: ExperimentConfig) -> PoseDataBundle:
    """Build split datasets and streaming PyG dataloaders."""
    df = build_pose_training_df(config)
    datasets = {}
    data_loaders = {}

    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            continue

        dataset = PoseAffinityDataset(split_df, config.paths.pose_lmdb_dir)
        datasets[split] = dataset
        data_loaders[split] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=(split == "train"),
            num_workers=config.training.num_workers,
        )

    return PoseDataBundle(df=df, datasets=datasets, data_loaders=data_loaders)