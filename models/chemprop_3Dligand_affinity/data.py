from dataclasses import dataclass

import pandas as pd
import numpy as np
import lmdb
import pickle
from chemprop import data, featurizers
from .config import ExperimentConfig
from typing import Any
from pathlib import Path

# TODO I'll implement this cleaner version later
@dataclass(frozen=True)
class ChempropDataBundle:
    """Chemprop datasets, dataloaders, and fitted preprocessing state."""

    df_filtered: pd.DataFrame
    data_loaders: Any
    scaler: Any

def get_top_pose_coords(smi: str, pose_df: pd.DataFrame, env: lmdb.Environment) -> np.ndarray:
    """Get the coordinates of the top pose for a given SMILES string."""

    # find the pose_id for the top pose for the given smiles string
    pose_id = pose_df[pose_df['ligand'] == smi]['pose_id'].values[0]
    with env.begin() as txn:
        value = txn.get(pose_id.encode('utf-8'))
    pyg_obj = pickle.loads(value)
    return pyg_obj.pos
    
def load_data_splits(config: ExperimentConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return filtered affinity rows and train/val/test row indices."""
    affinity_df = pd.read_csv(config.paths.affinity_split_csv)
    uniprot_id = config.uniprot_id
    
    df_filtered = affinity_df[affinity_df['uniprot_id'] == uniprot_id].reset_index(drop=True)
    train_indices = df_filtered.index[df_filtered['split'] == 'train'].values 
    val_indices = df_filtered.index[df_filtered['split'] == 'val'].values 
    test_indices = df_filtered.index[df_filtered['split'] == 'test'].values 

    return df_filtered, train_indices, val_indices, test_indices

def build_chemprop_data(config: ExperimentConfig):
    """Output: ChempropDataBundle: datasets, dataloaders, scaler."""
    df_filtered, train_indices, val_indices, test_indices = load_data_splits(config)
    smis = df_filtered.loc[:, 'ligand'].values
    ys = df_filtered.loc[:, 'affinity'].values

    # make top pose lookup table for given smiles string
    pose_df = pd.read_csv(config.paths.pose_manifest_csv)
    pose_df = pose_df[pose_df['uniprot_id'] == config.uniprot_id]
    pose_df = pose_df[pose_df['is_top_rank']]
    # print(config.paths.pose_lmdb_dir)
    env = lmdb.open(
        str(config.paths.pose_lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        )
    
    # build all data points and add coordinates
    all_data = []
    for smi, y in zip(smis, ys):
        dp = data.MoleculeDatapoint.from_smi(smi, np.array([y], dtype=np.float32))
        coords = get_top_pose_coords(smi, pose_df, env)
        # center coordinates
        coords = coords - coords.mean(axis=0)

        # check that the num atoms match
        n_atoms = dp.mol.GetNumAtoms()
        if coords is None or n_atoms != coords.shape[0]:
            raise ValueError(f"Molecule {smi} has {n_atoms} atoms, but {coords.shape[0]} coordinates were provided.")

        dp.V_f = coords
        all_data.append(dp)

    train_data, val_data, test_data = data.split_data_by_indices(
    all_data, [train_indices], [val_indices], [test_indices]
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dset = data.MoleculeDataset(train_data[0], featurizer)
    scaler = train_dset.normalize_targets()
    val_dset = data.MoleculeDataset(val_data[0], featurizer)
    val_dset.normalize_targets(scaler)
    test_dset = data.MoleculeDataset(test_data[0], featurizer)

    train_loader = data.build_dataloader(train_dset)
    val_loader = data.build_dataloader(val_dset, shuffle=False)
    test_loader = data.build_dataloader(test_dset, shuffle=False)

    return ChempropDataBundle(df_filtered, 
                            {'train': train_loader, 'val': val_loader, 'test': test_loader},
                            scaler
                            )