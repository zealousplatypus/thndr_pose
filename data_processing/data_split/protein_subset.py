"""Helpers for protein-subset split manifests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..common.constants import (
    AFFINITY_SPLIT_MANIFEST_CSV,
    PROCESSED_DIR,
    PROTEIN_MANIFEST_CSV,
    SPLIT_CONFLICT_GRAPH_CACHE_PKL,
    SPLIT_MANIFEST_CSV,
    SPLIT_MANIFEST_REPORT_TXT,
)


@dataclass(frozen=True)
class SubsetPaths:
    """Output paths for split-manifest generation."""

    split_manifest_csv: Path
    protein_manifest_csv: Path
    affinity_split_manifest_csv: Path
    split_manifest_report_txt: Path
    conflict_graph_cache_pkl: Path


def normalize_uniprot_ids(uniprot_ids: Sequence[str] | None) -> tuple[str, ...] | None:
    """Return a deduplicated tuple of non-empty UniProt IDs, or None when unset."""
    if uniprot_ids is None:
        return None
    normalized = tuple(sorted({str(uniprot_id).strip() for uniprot_id in uniprot_ids if str(uniprot_id).strip()}))
    if not normalized:
        raise ValueError("At least one non-empty uniprot_id is required when filtering by protein subset.")
    return normalized


def make_protein_subset_slug(uniprot_ids: Sequence[str]) -> str:
    """Build a stable filename slug from UniProt IDs (sorted, underscore-joined)."""
    return "_".join(normalize_uniprot_ids(uniprot_ids) or ())


def resolve_subset_paths(
    processed_dir: Path | None = None,
    slug: str | None = None,
) -> SubsetPaths:
    """Return default manifest paths for global or protein-subset runs."""
    processed_dir = processed_dir or PROCESSED_DIR
    if slug is None:
        return SubsetPaths(
            split_manifest_csv=processed_dir / SPLIT_MANIFEST_CSV.name,
            protein_manifest_csv=processed_dir / PROTEIN_MANIFEST_CSV.name,
            affinity_split_manifest_csv=processed_dir / AFFINITY_SPLIT_MANIFEST_CSV.name,
            split_manifest_report_txt=processed_dir / SPLIT_MANIFEST_REPORT_TXT.name,
            conflict_graph_cache_pkl=processed_dir / SPLIT_CONFLICT_GRAPH_CACHE_PKL.name,
        )
    prefix = f"{slug}_"
    return SubsetPaths(
        split_manifest_csv=processed_dir / f"{prefix}split_manifest.csv",
        protein_manifest_csv=processed_dir / f"{prefix}protein_manifest.csv",
        affinity_split_manifest_csv=processed_dir / f"{prefix}affinity_split_manifest.csv",
        split_manifest_report_txt=processed_dir / f"{prefix}split_manifest_report.txt",
        conflict_graph_cache_pkl=processed_dir / f"{prefix}split_conflict_graph.pkl",
    )


def assert_outputs_writable(paths: Sequence[Path], overwrite: bool) -> None:
    """Fail if any output path exists and overwrite is disabled."""
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output path(s) already exist: {joined}. Pass --overwrite to replace them."
        )


def filter_ligand_protein_maps(
    ligand_proteins: dict[str, set[str]],
    protein_ligands: dict[str, set[str]],
    include_uniprot_ids: Sequence[str],
) -> tuple[list[str], dict[str, set[str]], dict[str, set[str]]]:
    """Restrict ligand/protein maps to a UniProt subset."""
    del protein_ligands  # rebuilt from filtered ligand memberships
    subset = set(normalize_uniprot_ids(include_uniprot_ids) or ())
    filtered_ligand_proteins: dict[str, set[str]] = {}
    for ligand, proteins in ligand_proteins.items():
        kept = proteins & subset
        if kept:
            filtered_ligand_proteins[ligand] = kept

    filtered_protein_ligands: dict[str, set[str]] = {protein: set() for protein in subset}
    for ligand, proteins in filtered_ligand_proteins.items():
        for protein in proteins:
            filtered_protein_ligands[protein].add(ligand)

    ligands = sorted(filtered_ligand_proteins)
    if not ligands:
        raise ValueError(
            f"No ligands remain for uniprot subset {sorted(subset)}. "
            "Check affinity_manifest.csv and --uniprot-ids."
        )
    missing_proteins = {protein for protein in subset if not filtered_protein_ligands[protein]}
    if missing_proteins:
        raise ValueError(
            f"Requested uniprot_ids have no affinity rows: {sorted(missing_proteins)}"
        )
    return ligands, filtered_ligand_proteins, filtered_protein_ligands
