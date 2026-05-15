"""Build the affinity manifest from the raw binding CSV.

Outputs:
    - affinity_manifest.csv

The raw CSV is the only source of truth for experimental affinity labels at
this stage. In the local MVP pipeline we use canonical SMILES directly as the
readable ligand identifier, which removes the extra hash translation layer
while still giving us stable ligand identity across equivalent SMILES strings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from constants import (
        AFFINITY_MANIFEST_COLUMNS,
        AFFINITY_MANIFEST_CSV,
        RAW_BINDING_CSV,
        RAW_BINDING_REQUIRED_COLUMNS,
    )
    from ids import canonicalize_smiles, make_ligand_id
    from manifest_io import assert_unique, read_csv_checked, write_manifest
except ImportError:  # pragma: no cover - enables `python -m ...` usage later.
    from .constants import (
        AFFINITY_MANIFEST_COLUMNS,
        AFFINITY_MANIFEST_CSV,
        RAW_BINDING_CSV,
        RAW_BINDING_REQUIRED_COLUMNS,
    )
    from .ids import canonicalize_smiles, make_ligand_id
    from .manifest_io import assert_unique, read_csv_checked, write_manifest


LOGGER = logging.getLogger(__name__)


def _annotate_ligand_columns(binding_df: pd.DataFrame) -> pd.DataFrame:
    """Add canonical SMILES and normalized affinity columns to the binding rows.

    We copy the dataframe so callers keep their original object unchanged.
    """
    binding_df = binding_df.copy()

    # Keep the original CSV string for provenance, but use canonical SMILES as
    # the ligand identifier so downstream files remain readable and stable.
    binding_df["raw_smiles"] = binding_df["smiles"]
    binding_df["canonical_smiles"] = binding_df["smiles"].map(canonicalize_smiles)
    binding_df["ligand"] = binding_df["canonical_smiles"].map(make_ligand_id)
    binding_df["affinity"] = pd.to_numeric(binding_df["pIC50"], errors="raise")
    return binding_df


def build_affinity_manifest(binding_df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per `(uniprot_id, ligand)` affinity observation."""
    affinity_df = (
        binding_df.loc[:, list(AFFINITY_MANIFEST_COLUMNS)]
        .sort_values(["uniprot_id", "ligand"])
        .reset_index(drop=True)
    )
    assert_unique(affinity_df, ["uniprot_id", "ligand"], "affinity_manifest")
    return affinity_df


def build_manifest(raw_binding_csv: str | Path) -> pd.DataFrame:
    """Read the raw affinity CSV and return the affinity manifest."""
    binding_df = read_csv_checked(raw_binding_csv, RAW_BINDING_REQUIRED_COLUMNS)
    binding_df = _annotate_ligand_columns(binding_df)
    return build_affinity_manifest(binding_df)


def parse_args() -> argparse.Namespace:
    """CLI for building the first MVP manifests."""
    parser = argparse.ArgumentParser(
        description="Build affinity_manifest.csv from mol_binding_data.csv."
    )
    parser.add_argument(
        "--input-csv",
        default=str(RAW_BINDING_CSV),
        help="Path to the raw binding CSV.",
    )
    parser.add_argument(
        "--affinity-out",
        default=str(AFFINITY_MANIFEST_CSV),
        help="Output path for affinity_manifest.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    affinity_df = build_manifest(args.input_csv)
    affinity_path = write_manifest(affinity_df, args.affinity_out)
    LOGGER.info(
        "Wrote affinity manifest: %s (%d rows)",
        affinity_path,
        len(affinity_df),
    )


if __name__ == "__main__":
    main()
