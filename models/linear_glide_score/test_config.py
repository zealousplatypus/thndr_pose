"""Tests for linear Glide experiment config and protein filtering."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from models.linear_glide_score.config import load_experiment_config
from models.linear_glide_score.linear_evaluation import _load_examples


def _write_inputs(tmp_path: Path) -> None:
    pose_df = pd.DataFrame(
        [
            ["CCO", -5.0],
            ["CCC", -6.0],
            ["CCN", -7.0],
        ],
        columns=["ligand", "glide_score"],
    )
    pose_df.to_csv(tmp_path / "pose_manifest.csv", index=False)

    affinity_df = pd.DataFrame(
        [
            ["P11111", "CCO", 6.0, "train"],
            ["P22222", "CCC", 7.0, "train"],
            ["P22222", "CCN", 8.0, "val"],
            ["P11111", "CCO", 6.5, "test"],
        ],
        columns=["uniprot_id", "ligand", "affinity", "split"],
    )
    affinity_df.to_csv(tmp_path / "affinity_split_manifest.csv", index=False)


def test_load_examples_filters_include_uniprot_ids(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "linear_test",
                "paths": {
                    "pose_csv": str(tmp_path / "pose_manifest.csv"),
                    "affinity_split_csv": str(tmp_path / "affinity_split_manifest.csv"),
                    "runs_dir": str(tmp_path / "runs"),
                },
                "data": {
                    "include_uniprot_ids": ["P22222"],
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_experiment_config(config_path, validate_run_dir=False)
    examples_df = _load_examples(
        config.paths.pose_csv,
        config.paths.affinity_split_csv,
        data_config=config,
    )
    assert set(examples_df["uniprot_id"]) == {"P22222"}
    assert len(examples_df) == 2
