"""Build `pdb_manifest.csv` from the docked Glide output CSVs.

This manifest is an inventory of the structure-level contexts that actually
exist in the docking outputs. We intentionally derive it from the docked CSVs
instead of directory names so it reflects successful downstream artifacts.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from constants import PDB_MANIFEST_COLUMNS, PDB_MANIFEST_CSV, RAW_OUTPUTS_DIR
    from ids import make_pdb_key, parse_grid_string
    from manifest_io import assert_unique, write_manifest
except ImportError:  # pragma: no cover
    from .constants import PDB_MANIFEST_COLUMNS, PDB_MANIFEST_CSV, RAW_OUTPUTS_DIR
    from .ids import make_pdb_key, parse_grid_string
    from .manifest_io import assert_unique, write_manifest


LOGGER = logging.getLogger(__name__)


def _iter_docked_csvs(outputs_dir: str | Path) -> list[Path]:
    """Collect all docked Glide metadata CSVs under the outputs directory."""
    outputs_dir = Path(outputs_dir)
    return sorted(outputs_dir.glob("*_docked/batch_*_docked.csv"))


def build_pdb_manifest(outputs_dir: str | Path) -> pd.DataFrame:
    """Build one row per unique `(uniprot_id, pdb_id)` observed in docked CSVs."""
    rows: list[dict[str, str]] = []
    for csv_path in _iter_docked_csvs(outputs_dir):
        df = pd.read_csv(csv_path, usecols=["s_i_glide_gridfile"])
        for grid in df["s_i_glide_gridfile"].dropna().unique():
            pdb_id, uniprot_id = parse_grid_string(grid)
            rows.append(
                {
                    "pdb_key": make_pdb_key(pdb_id, uniprot_id),
                    "uniprot_id": uniprot_id,
                    "pdb_id": pdb_id,
                }
            )

    pdb_df = (
        pd.DataFrame(rows, columns=list(PDB_MANIFEST_COLUMNS))
        .drop_duplicates()
        .sort_values(["uniprot_id", "pdb_id"])
        .reset_index(drop=True)
    )
    assert_unique(pdb_df, ["pdb_key"], "pdb_manifest")
    return pdb_df


def parse_args() -> argparse.Namespace:
    """CLI for generating `pdb_manifest.csv`."""
    parser = argparse.ArgumentParser(
        description="Build pdb_manifest.csv from docked Glide CSV outputs."
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(RAW_OUTPUTS_DIR),
        help="Directory containing *_docked subdirectories and docked CSVs.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(PDB_MANIFEST_CSV),
        help="Output path for pdb_manifest.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    pdb_df = build_pdb_manifest(args.outputs_dir)
    output_path = write_manifest(pdb_df, args.output_csv)
    LOGGER.info("Wrote pdb manifest: %s (%d rows)", output_path, len(pdb_df))


if __name__ == "__main__":
    main()
