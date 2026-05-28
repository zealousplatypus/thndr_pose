"""Data loading for Chemprop + frozen ESM affinity modeling."""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_processing.common.constants import (
    AFFINITY_SPLIT_MANIFEST_COLUMNS,
    PROTEIN_MANIFEST_COLUMNS,
)
from data_processing.common.manifest_io import read_csv_checked

from .config import ExperimentConfig


LOGGER = logging.getLogger(__name__)

METADATA_COLUMNS = (
    "uniprot_id",
    "protein_idx",
    "ligand",
    "ligand_idx",
    "split",
    "true_affinity",
)


@dataclass(frozen=True)
class AffinityDataBundle:
    """Validated manifest rows plus frozen ESM descriptors."""

    examples_df: pd.DataFrame
    split_dfs: dict[str, pd.DataFrame]
    metadata_by_split: dict[str, pd.DataFrame]
    descriptors_by_split: dict[str, np.ndarray]
    esm_embeddings: np.ndarray

    @property
    def esm_shape(self) -> tuple[int, int]:
        """Return the ESM embedding matrix shape."""
        return tuple(int(value) for value in self.esm_embeddings.shape)

    @property
    def esm_dim(self) -> int:
        """Return the frozen ESM descriptor dimension."""
        return int(self.esm_embeddings.shape[1])


@dataclass(frozen=True)
class ChempropDataBundle:
    """Chemprop datasets, dataloaders, and fitted preprocessing state."""

    datasets: dict[str, Any]
    dataloaders: dict[str, Any]
    target_scaler: Any | None
    descriptor_scaler: Any | None


def _required_columns(config: ExperimentConfig) -> tuple[str, ...]:
    """Return required manifest columns for the configured column names."""
    return (
        config.data.uniprot_id_column,
        config.data.protein_idx_column,
        config.data.smiles_column,
        config.data.ligand_idx_column,
        config.data.target_column,
        config.data.split_column,
    )


def _read_affinity_manifest(config: ExperimentConfig) -> pd.DataFrame:
    """Read the affinity split manifest with required-column validation."""
    required = _required_columns(config)
    default_required = tuple(AFFINITY_SPLIT_MANIFEST_COLUMNS)
    if required == default_required:
        return read_csv_checked(config.paths.affinity_split_csv, AFFINITY_SPLIT_MANIFEST_COLUMNS)
    return read_csv_checked(config.paths.affinity_split_csv, required)


def _validate_protein_manifest(config: ExperimentConfig, examples_df: pd.DataFrame) -> None:
    """Validate `uniprot_id -> protein_idx` against the protein manifest."""
    protein_manifest_csv = config.paths.protein_manifest_csv
    if protein_manifest_csv is None:
        return
    protein_df = read_csv_checked(protein_manifest_csv, PROTEIN_MANIFEST_COLUMNS)
    protein_df = protein_df.loc[:, list(PROTEIN_MANIFEST_COLUMNS)].copy()
    protein_df["protein_idx"] = pd.to_numeric(
        protein_df["protein_idx"],
        errors="raise",
    ).astype(int)
    mapping = dict(zip(protein_df["uniprot_id"].astype(str), protein_df["protein_idx"]))

    uniprot_col = config.data.uniprot_id_column
    protein_idx_col = config.data.protein_idx_column
    missing = sorted(set(examples_df[uniprot_col].astype(str)) - set(mapping))
    if missing:
        raise ValueError(
            "Affinity examples reference proteins missing from protein manifest: "
            f"{missing[:10]}"
        )

    mismatched = []
    for uniprot_id, protein_idx in examples_df[[uniprot_col, protein_idx_col]].itertuples(
        index=False,
        name=None,
    ):
        expected_idx = mapping[str(uniprot_id)]
        observed_idx = int(protein_idx)
        if expected_idx != observed_idx:
            mismatched.append((str(uniprot_id), observed_idx, expected_idx))
    if mismatched:
        raise ValueError(
            "Affinity manifest protein_idx values disagree with protein manifest. "
            f"First mismatches: {mismatched[:10]}"
        )


def load_esm_embeddings(path: str | Path) -> np.ndarray:
    """Load and validate a frozen ESM embedding matrix."""
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected ESM embeddings to be a 2D matrix, got shape {embeddings.shape}"
        )
    return embeddings


def load_affinity_data(
    config: ExperimentConfig,
    require_val: bool = True,
) -> AffinityDataBundle:
    """Load manifest rows, filter proteins/splits, and attach ESM descriptors."""
    df = _read_affinity_manifest(config)
    data_config = config.data
    columns = _required_columns(config)
    df = df.loc[:, list(columns)].copy()

    df[data_config.split_column] = df[data_config.split_column].astype(str)
    df = df[df[data_config.split_column].isin(data_config.active_splits)].copy()
    if data_config.include_uniprot_ids:
        include = set(data_config.include_uniprot_ids)
        df = df[df[data_config.uniprot_id_column].astype(str).isin(include)].copy()

    df[data_config.target_column] = pd.to_numeric(
        df[data_config.target_column],
        errors="coerce",
    )
    protein_idx_numeric = pd.to_numeric(
        df[data_config.protein_idx_column],
        errors="coerce",
    )
    valid_integer_idx = protein_idx_numeric.notna() & np.equal(
        protein_idx_numeric,
        np.floor(protein_idx_numeric),
    )
    df = df[valid_integer_idx].copy()
    df[data_config.protein_idx_column] = protein_idx_numeric[valid_integer_idx].astype(int)

    required_nonnull = [
        data_config.smiles_column,
        data_config.target_column,
        data_config.split_column,
        data_config.protein_idx_column,
    ]
    df = df.dropna(subset=required_nonnull).copy()
    df[data_config.smiles_column] = df[data_config.smiles_column].astype(str).str.strip()
    df = df[df[data_config.smiles_column] != ""].copy()
    df = df[df[data_config.protein_idx_column] >= 0].copy()

    if df.empty:
        raise ValueError("No affinity examples remain after filtering.")
    if (df[data_config.split_column] == "train").sum() == 0:
        raise ValueError("No train examples remain after filtering.")
    if require_val and "val" in data_config.active_splits:
        if (df[data_config.split_column] == "val").sum() == 0:
            raise ValueError("No val examples remain after filtering.")

    embeddings = load_esm_embeddings(config.paths.esm_embeddings_npy)
    protein_idx = df[data_config.protein_idx_column].to_numpy(dtype=int)
    if protein_idx.max(initial=-1) >= embeddings.shape[0]:
        raise ValueError(
            "Affinity examples contain protein_idx outside the ESM matrix range. "
            f"Max protein_idx={protein_idx.max()}, ESM rows={embeddings.shape[0]}"
        )

    _validate_protein_manifest(config, df)

    df = df.reset_index(drop=True)
    df["protein_x_d_row"] = list(embeddings[df[data_config.protein_idx_column].to_numpy(dtype=int)])

    split_dfs: dict[str, pd.DataFrame] = {}
    metadata_by_split: dict[str, pd.DataFrame] = {}
    descriptors_by_split: dict[str, np.ndarray] = {}
    rename_map = {
        data_config.uniprot_id_column: "uniprot_id",
        data_config.protein_idx_column: "protein_idx",
        data_config.smiles_column: "ligand",
        data_config.ligand_idx_column: "ligand_idx",
        data_config.split_column: "split",
        data_config.target_column: "true_affinity",
    }
    for split in data_config.active_splits:
        split_df = df[df[data_config.split_column] == split].copy().reset_index(drop=True)
        split_dfs[split] = split_df
        if split_df.empty:
            metadata_by_split[split] = pd.DataFrame(columns=METADATA_COLUMNS)
            descriptors_by_split[split] = np.empty((0, embeddings.shape[1]), dtype=np.float32)
            continue

        metadata = (
            split_df[
                [
                    data_config.uniprot_id_column,
                    data_config.protein_idx_column,
                    data_config.smiles_column,
                    data_config.ligand_idx_column,
                    data_config.split_column,
                    data_config.target_column,
                ]
            ]
            .rename(columns=rename_map)
            .loc[:, METADATA_COLUMNS]
            .reset_index(drop=True)
        )
        metadata_by_split[split] = metadata
        descriptors_by_split[split] = np.stack(
            split_df["protein_x_d_row"].to_numpy(),
        ).astype(np.float32, copy=False)

    return AffinityDataBundle(
        examples_df=df,
        split_dfs=split_dfs,
        metadata_by_split=metadata_by_split,
        descriptors_by_split=descriptors_by_split,
        esm_embeddings=embeddings,
    )


def summarize_affinity_data(
    config: ExperimentConfig,
    bundle: AffinityDataBundle,
) -> dict[str, Any]:
    """Return dry-run summary fields."""
    split_counts = {
        split: int(len(bundle.split_dfs.get(split, [])))
        for split in config.data.active_splits
    }
    return {
        "experiment_name": config.experiment_name,
        "output_dir": str(config.run_dir),
        "rows_after_filtering": int(len(bundle.examples_df)),
        "split_counts": split_counts,
        "unique_proteins": int(bundle.examples_df[config.data.uniprot_id_column].nunique()),
        "unique_ligands": int(bundle.examples_df[config.data.smiles_column].nunique()),
        "esm_shape": list(bundle.esm_shape),
        "chemprop_x_d_dim": bundle.esm_dim,
    }


def format_dry_run_summary(summary: dict[str, Any]) -> str:
    """Format the dry-run summary requested by the implementation plan."""
    split_counts = summary["split_counts"]
    return (
        f"Experiment: {summary['experiment_name']}\n"
        f"Output dir: {summary['output_dir']}\n"
        f"Rows after filtering: {summary['rows_after_filtering']}\n"
        "Train/val/test: "
        f"{split_counts.get('train', 0)} / "
        f"{split_counts.get('val', 0)} / "
        f"{split_counts.get('test', 0)}\n"
        f"Unique proteins: {summary['unique_proteins']}\n"
        f"Unique ligands: {summary['unique_ligands']}\n"
        f"ESM shape: {summary['esm_shape']}\n"
        f"Chemprop X_d dim: {summary['chemprop_x_d_dim']}"
    )


def _chemprop_modules() -> tuple[Any, Any]:
    """Import Chemprop modules lazily with an actionable error."""
    try:
        chemprop_data = importlib.import_module("chemprop.data")
        chemprop_utils = importlib.import_module("chemprop.utils")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Chemprop is required for datapoint construction and training. "
            "Install Chemprop v2 in this environment before running training."
        ) from exc
    return chemprop_data, chemprop_utils


def _call_with_supported_kwargs(callable_obj: Any, **kwargs: Any) -> Any:
    """Call a Chemprop function/class with kwargs supported by this version."""
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return callable_obj(**kwargs)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return callable_obj(**supported)


def _make_datapoint(smiles: str, target: float, x_d: np.ndarray) -> Any:
    """Build one Chemprop MoleculeDatapoint."""
    chemprop_data, chemprop_utils = _chemprop_modules()
    molecule_datapoint = chemprop_data.MoleculeDatapoint
    y = np.array([target], dtype=np.float32)
    x_d = x_d.astype(np.float32, copy=False)

    if hasattr(molecule_datapoint, "from_smi"):
        return _call_with_supported_kwargs(
            molecule_datapoint.from_smi,
            smi=smiles,
            smiles=smiles,
            y=y,
            targets=y,
            x_d=x_d,
        )

    mol = chemprop_utils.make_mol(smiles, keep_h=False, add_h=False)
    return _call_with_supported_kwargs(
        molecule_datapoint,
        mol=mol,
        y=y,
        targets=y,
        x_d=x_d,
    )


def _build_dataset(datapoints: list[Any]) -> Any:
    """Build a Chemprop MoleculeDataset."""
    chemprop_data, _ = _chemprop_modules()
    return chemprop_data.MoleculeDataset(datapoints)


def _normalize_targets(dataset: Any, scaler: Any | None = None) -> Any | None:
    """Normalize Chemprop targets with version-compatible calls."""
    if not hasattr(dataset, "normalize_targets"):
        return scaler
    if scaler is None:
        return dataset.normalize_targets()
    dataset.normalize_targets(scaler)
    return scaler


def _normalize_descriptors(dataset: Any, scaler: Any | None = None) -> Any | None:
    """Normalize Chemprop X_d descriptors with version-compatible calls."""
    if not hasattr(dataset, "normalize_inputs"):
        return scaler
    if scaler is None:
        return dataset.normalize_inputs("X_d")
    dataset.normalize_inputs("X_d", scaler)
    return scaler


def _build_dataloader(
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> Any:
    """Build a Chemprop dataloader."""
    chemprop_data, _ = _chemprop_modules()
    return _call_with_supported_kwargs(
        chemprop_data.build_dataloader,
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def build_chemprop_data(
    config: ExperimentConfig,
    bundle: AffinityDataBundle,
    target_scaler: Any | None = None,
    descriptor_scaler: Any | None = None,
    fit_scalers: bool = True,
) -> ChempropDataBundle:
    """Build Chemprop datasets and dataloaders from a validated data bundle."""
    datasets: dict[str, Any] = {}
    dataloaders: dict[str, Any] = {}
    for split in config.data.active_splits:
        split_df = bundle.split_dfs[split]
        datapoints = []
        valid_row_positions = []
        for row_idx, row in split_df.iterrows():
            descriptor = bundle.descriptors_by_split[split][row_idx]
            try:
                datapoint = _make_datapoint(
                    smiles=str(row[config.data.smiles_column]),
                    target=float(row[config.data.target_column]),
                    x_d=descriptor,
                )
            except Exception:
                if config.data.drop_invalid_smiles:
                    LOGGER.warning(
                        "Dropping invalid SMILES in split=%s row=%d: %s",
                        split,
                        row_idx,
                        row[config.data.smiles_column],
                    )
                    continue
                raise
            datapoints.append(datapoint)
            valid_row_positions.append(row_idx)

        if len(valid_row_positions) != len(split_df):
            bundle.metadata_by_split[split] = (
                bundle.metadata_by_split[split]
                .iloc[valid_row_positions]
                .reset_index(drop=True)
            )
            bundle.descriptors_by_split[split] = bundle.descriptors_by_split[split][
                valid_row_positions
            ]
        if split == "train" and not datapoints:
            raise ValueError("No train datapoints remain after SMILES validation.")

        dataset = _build_dataset(datapoints)
        datasets[split] = dataset

    if config.model.normalize_targets:
        if fit_scalers:
            target_scaler = _normalize_targets(datasets["train"])
        for split, dataset in datasets.items():
            if split != "train":
                _normalize_targets(dataset, target_scaler)

    if config.model.normalize_protein_descriptors:
        if fit_scalers:
            descriptor_scaler = _normalize_descriptors(datasets["train"])
        for split, dataset in datasets.items():
            if split != "train":
                _normalize_descriptors(dataset, descriptor_scaler)

    for split, dataset in datasets.items():
        dataloaders[split] = _build_dataloader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=(split == "train"),
            num_workers=config.training.num_workers,
        )

    return ChempropDataBundle(
        datasets=datasets,
        dataloaders=dataloaders,
        target_scaler=target_scaler,
        descriptor_scaler=descriptor_scaler,
    )

