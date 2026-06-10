from dataclasses import dataclass

import pandas as pd
import numpy as np

from chemprop import data, featurizers
from models.chemprop_affinity.config import ExperimentConfig
from typing import Any

# TODO I'll implement this cleaner version later
@dataclass(frozen=True)
class ChempropDataBundle:
    """Chemprop datasets, dataloaders, and fitted preprocessing state."""

    df_filtered: pd.DataFrame
    data_loaders: Any
    scaler: Any

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
    all_data = [data.MoleculeDatapoint.from_smi(smi, np.array([y], dtype=np.float32)) 
        for smi, y in zip(smis, ys)]

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