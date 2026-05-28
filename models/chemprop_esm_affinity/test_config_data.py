"""Tests for Chemprop + ESM affinity config and dry-run data loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.chemprop_esm_affinity.config import load_experiment_config
from models.chemprop_esm_affinity.data import (
    format_dry_run_summary,
    load_affinity_data,
    summarize_affinity_data,
)


def _write_experiment_json(tmp_path: Path, **overrides: object) -> Path:
    config = {
        "experiment_name": "unit_test_chemprop_esm",
        "paths": {
            "affinity_split_csv": str(tmp_path / "affinity_split_manifest.csv"),
            "protein_manifest_csv": str(tmp_path / "protein_manifest.csv"),
            "esm_embeddings_npy": str(tmp_path / "esm_embeddings.float32.npy"),
            "runs_dir": str(tmp_path / "runs"),
        },
        "data": {
            "include_uniprot_ids": ["P22222"],
            "active_splits": ["train", "val", "test"],
        },
        "training": {
            "max_epochs": 1,
        },
    }
    config.update(overrides)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _write_inputs(tmp_path: Path) -> None:
    affinity_df = pd.DataFrame(
        [
            ["P11111", 0, "CCO", 0, 6.0, "train"],
            ["P22222", 1, "CCC", 1, 7.0, "train"],
            ["P22222", 1, "CCN", 2, 8.0, "val"],
            ["P22222", 1, "CNC", 3, 9.0, "test"],
        ],
        columns=[
            "uniprot_id",
            "protein_idx",
            "ligand",
            "ligand_idx",
            "affinity",
            "split",
        ],
    )
    affinity_df.to_csv(tmp_path / "affinity_split_manifest.csv", index=False)
    protein_df = pd.DataFrame(
        [["P11111", 0], ["P22222", 1]],
        columns=["uniprot_id", "protein_idx"],
    )
    protein_df.to_csv(tmp_path / "protein_manifest.csv", index=False)
    np.save(
        tmp_path / "esm_embeddings.float32.npy",
        np.array([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]], dtype=np.float32),
    )


def test_load_affinity_data_filters_proteins_and_uses_protein_idx(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    config = load_experiment_config(
        _write_experiment_json(tmp_path),
        validate_run_dir=False,
    )

    bundle = load_affinity_data(config)

    assert set(bundle.examples_df["uniprot_id"]) == {"P22222"}
    assert bundle.esm_shape == (2, 3)
    np.testing.assert_array_equal(
        bundle.descriptors_by_split["train"][0],
        np.array([9.0, 8.0, 7.0], dtype=np.float32),
    )
    assert bundle.metadata_by_split["train"].loc[0, "ligand"] == "CCC"


def test_dry_run_summary_contains_split_counts(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    config = load_experiment_config(
        _write_experiment_json(tmp_path),
        validate_run_dir=False,
    )
    bundle = load_affinity_data(config, require_val=False)

    summary = summarize_affinity_data(config, bundle)
    rendered = format_dry_run_summary(summary)

    assert summary["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert "Train/val/test: 1 / 1 / 1" in rendered
    assert "Chemprop X_d dim: 3" in rendered


def test_load_affinity_data_rejects_out_of_range_protein_idx(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    affinity_df = pd.read_csv(tmp_path / "affinity_split_manifest.csv")
    affinity_df.loc[affinity_df["uniprot_id"] == "P22222", "protein_idx"] = 9
    affinity_df.to_csv(tmp_path / "affinity_split_manifest.csv", index=False)
    config = load_experiment_config(
        _write_experiment_json(tmp_path),
        validate_run_dir=False,
    )

    with pytest.raises(ValueError, match="outside the ESM matrix range"):
        load_affinity_data(config)

