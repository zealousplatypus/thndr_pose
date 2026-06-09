from random import shuffle
from data_processing.common.constants import ACTIVE_SPLIT_NAMES
import pandas as pd

from chemprop import data, featurizers
from models.chemprop_affinity.config import ExperimentConfig
from typing import Any

# TODO I'll implement this cleaner version later
# @dataclass(frozen=True)
class ChempropDataBundle:
    """Chemprop datasets, dataloaders, and fitted preprocessing state."""

    train_dataloader: Any
    dev_dataloader: Any
    test_dataloader: Any
    scaler: Any

def load_data_splits(config: ExperimentConfig) -> dict[str: Any]:
    """Outputs three dfs:
    train_indices, val_indices, test_indices"""
    affinity_df = pd.read(config.paths.affinity_split_csv)
    uniprot_id = config.uniprot_id
    
    df_filtered = affinity_df[affinity_df['uniprot_id'] == uniprot_id]
    train_indices = df_filtered.index[df_filtered['split'] == 'train'].values 
    val_indices = df_filtered.index[df_filtered['split'] == 'val'].values 
    test_indices = df_filtered.index[df_filtered['split'] == 'test'].values 

    return df_filtered, train_indices, val_indices, test_indices

def build_chemprop_data(config: ExperimentConfig):
    """Output: ChempropDataBundle: datasets, dataloaders, scaler."""
    df_filtered, train_indices, val_indices, test_indices = load_data_splits(config) 
    smis = df_filtered.loc[:, 'smiles'].values
    ys = df_filtered.loc[:, 'affinity'].values
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    train_data, val_data, test_data = data.split_data_by_indices(
    all_data, train_indices, val_indices, test_indices
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_data, featurizer)
    scaler = train_dset.normalize_targets()
    val_dset = data.MoleculeDataset(val_data, featurizer)
    val_dset.normalize_targets(scaler)
    test_dset = data.MoleculeDataset(test_data, featurizer)

    train_loader = data.build_dataloader(train_dset)
    val_loader = data.build_dataloader(val_dset, shuffle=False)
    test_loader = data.build_dataloader(test_dset, shuffle=False)

    return ChempropDataBundle(train_loader, val_loader, test_loader, scaler)