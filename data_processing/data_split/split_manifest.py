"""Build a ligand-level split manifest from an affinity manifest.

This module intentionally knows only about affinity examples and ligand
similarity. It does not apply experiment-specific PDB, pose, or selector
filters; those belong in the training/evaluation layer.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from ..common.constants import (
    ACTIVE_SPLIT_NAMES,
    AFFINITY_MANIFEST_CSV,
    ALL_SPLIT_NAMES,
    DEFAULT_FINGERPRINT_RADIUS,
    DEFAULT_FINGERPRINT_SIZE,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TANIMOTO_THRESHOLD,
    DEFAULT_TEST_FRACTION,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VAL_FRACTION,
    MIN_LIGANDS_FOR_REQUIRED_VAL_TEST,
    PROTEIN_MANIFEST_CSV,
    SPLIT_CONFLICT_GRAPH_CACHE_PKL,
    SPLIT_MANIFEST_CSV,
    SPLIT_MANIFEST_REPORT_TXT,
)
from ..common.manifest_io import assert_unique, ensure_parent_dir, read_csv_checked, write_manifest


LOGGER = logging.getLogger(__name__)
DEFAULT_NUM_RESTARTS = 3
CONFLICT_GRAPH_CACHE_VERSION = 1
PerProteinStats = dict[str, dict[str, int | float]]


@dataclass(frozen=True)
class RestartResult:
    """One candidate split assignment from a randomized restart."""

    restart_index: int
    seed: int
    assignments: dict[str, str]
    splits: dict[str, set[str]]
    ligand_proteins: dict[str, set[str]]
    protein_ligands: dict[str, set[str]]
    target_fractions: dict[str, float]

    @property
    def test_size(self) -> int:
        return len(self.splits["test"])

    @property
    def val_size(self) -> int:
        return len(self.splits["val"])

    @property
    def train_size(self) -> int:
        return len(self.splits["train"])

    @property
    def dropped_size(self) -> int:
        return len(self.splits["dropped"])

    @property
    def kept_size(self) -> int:
        return self.train_size + self.val_size + self.test_size

    @property
    def per_protein_balance_penalty(self) -> float:
        return _per_protein_balance_penalty(
            protein_ligands=self.protein_ligands,
            assignments=self.assignments,
            target_fractions=self.target_fractions,
        )

    @property
    def score(self) -> tuple[int, int, int, int, int, int]:
        # Held-out size matters most; retained examples matter after that.
        return (
            self.test_size,
            self.val_size,
            self.train_size,
            self.kept_size,
            -int(round(self.per_protein_balance_penalty * 1_000_000)),
            -self.restart_index,
        )


def _read_ligand_proteins(
    affinity_csv: str | Path,
) -> tuple[list[str], dict[str, set[str]], dict[str, set[str]]]:
    """Read unique ligands and their protein memberships from an affinity manifest."""
    affinity_df = read_csv_checked(affinity_csv, ["ligand", "uniprot_id"])
    affinity_df = affinity_df.dropna(subset=["ligand", "uniprot_id"])
    affinity_df["ligand"] = affinity_df["ligand"].astype(str)
    affinity_df["uniprot_id"] = affinity_df["uniprot_id"].astype(str)

    ligand_proteins: dict[str, set[str]] = {}
    protein_ligands: dict[str, set[str]] = {}
    for ligand, uniprot_id in affinity_df[["ligand", "uniprot_id"]].itertuples(index=False):
        ligand_proteins.setdefault(ligand, set()).add(uniprot_id)
        protein_ligands.setdefault(uniprot_id, set()).add(ligand)

    ligands = sorted(ligand_proteins)
    if not ligands:
        raise ValueError("Affinity manifest contains no ligands to split.")
    return ligands, ligand_proteins, protein_ligands


def _build_ligand_index_map(ligands: list[str]) -> dict[str, int]:
    """Return a stable ligand string -> contiguous integer index map."""
    return {ligand: index for index, ligand in enumerate(ligands)}


def _build_protein_index_map(protein_ligands: dict[str, set[str]]) -> dict[str, int]:
    """Return a stable uniprot_id -> contiguous integer index map."""
    proteins = sorted(protein_ligands)
    return {protein: index for index, protein in enumerate(proteins)}


def _build_split_manifest_df(
    ligands: list[str],
    assignments: dict[str, str],
) -> pd.DataFrame:
    """Build the ligand-level split manifest with contiguous ligand indices."""
    ligand_to_idx = _build_ligand_index_map(ligands)
    split_df = pd.DataFrame(
        [
            {
                "ligand": ligand,
                "ligand_idx": ligand_to_idx[ligand],
                "split": assignments[ligand],
            }
            for ligand in ligands
        ]
    ).sort_values(["split", "ligand_idx", "ligand"], ignore_index=True)
    assert_unique(split_df, ["ligand"], "split_manifest")
    assert_unique(split_df, ["ligand_idx"], "split_manifest ligand_idx")
    return split_df


def _build_protein_manifest_df(protein_ligands: dict[str, set[str]]) -> pd.DataFrame:
    """Build the protein index manifest for embedding lookup tables."""
    proteins = sorted(protein_ligands)
    protein_df = pd.DataFrame(
        {
            "uniprot_id": proteins,
            "protein_idx": list(range(len(proteins))),
        }
    )
    assert_unique(protein_df, ["uniprot_id"], "protein_manifest")
    assert_unique(protein_df, ["protein_idx"], "protein_manifest protein_idx")
    return protein_df


def _build_fingerprints(
    ligands: list[str],
    radius: int,
    fp_size: int,
) -> dict[str, DataStructs.cDataStructs.ExplicitBitVect]:
    """Convert canonical ligand SMILES strings into Morgan fingerprints."""
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fingerprints: dict[str, DataStructs.cDataStructs.ExplicitBitVect] = {}
    for ligand in ligands:
        mol = Chem.MolFromSmiles(ligand)
        if mol is None:
            raise ValueError(f"Failed to parse ligand SMILES from affinity manifest: {ligand}")
        fingerprints[ligand] = fpgen.GetFingerprint(mol)
    return fingerprints


def _build_conflict_graph(
    ligands: list[str],
    fingerprints: dict[str, DataStructs.cDataStructs.ExplicitBitVect],
    tanimoto_threshold: float,
) -> dict[str, set[str]]:
    """Return an undirected graph connecting ligand pairs above the threshold."""
    conflict_graph = {ligand: set() for ligand in ligands}
    for left_index, left_ligand in enumerate(ligands):
        left_fp = fingerprints[left_ligand]
        for right_ligand in ligands[left_index + 1 :]:
            similarity = DataStructs.TanimotoSimilarity(left_fp, fingerprints[right_ligand])
            if similarity >= tanimoto_threshold:
                conflict_graph[left_ligand].add(right_ligand)
                conflict_graph[right_ligand].add(left_ligand)
    return conflict_graph


def _make_conflict_graph_cache_key(
    ligands: list[str],
    tanimoto_threshold: float,
    radius: int,
    fp_size: int,
) -> dict[str, object]:
    """Build a cache key that invalidates when ligand membership or params change."""
    ligand_hasher = hashlib.sha256()
    for ligand in ligands:
        ligand_hasher.update(ligand.encode("utf-8"))
        ligand_hasher.update(b"\0")
    return {
        "version": CONFLICT_GRAPH_CACHE_VERSION,
        "num_ligands": len(ligands),
        "ligand_sha256": ligand_hasher.hexdigest(),
        "tanimoto_threshold": tanimoto_threshold,
        "radius": radius,
        "fp_size": fp_size,
    }


def _load_conflict_graph_cache(
    cache_path: str | Path | None,
    cache_key: dict[str, object],
) -> dict[str, set[str]] | None:
    """Load a cached conflict graph if it matches the ligand set and params."""
    if cache_path is None:
        return None
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError) as exc:
        LOGGER.warning("Ignoring unreadable conflict-graph cache %s: %s", cache_path, exc)
        return None

    if not isinstance(payload, dict):
        LOGGER.warning("Ignoring malformed conflict-graph cache %s: expected dict payload", cache_path)
        return None
    if payload.get("cache_key") != cache_key:
        return None

    conflict_graph = payload.get("conflict_graph")
    if not isinstance(conflict_graph, dict):
        LOGGER.warning("Ignoring malformed conflict-graph cache %s: missing conflict_graph", cache_path)
        return None

    LOGGER.info("Loaded cached conflict graph: %s", cache_path)
    return conflict_graph


def _write_conflict_graph_cache(
    cache_path: str | Path | None,
    cache_key: dict[str, object],
    conflict_graph: dict[str, set[str]],
) -> None:
    """Persist the conflict graph for reuse by future runs."""
    if cache_path is None:
        return
    cache_path = ensure_parent_dir(cache_path)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {"cache_key": cache_key, "conflict_graph": conflict_graph},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    LOGGER.info("Saved conflict-graph cache: %s", cache_path)


def _get_conflict_graph(
    ligands: list[str],
    tanimoto_threshold: float,
    radius: int,
    fp_size: int,
    cache_path: str | Path | None,
) -> dict[str, set[str]]:
    """Load the conflict graph from cache when possible, otherwise rebuild it."""
    cache_key = _make_conflict_graph_cache_key(
        ligands=ligands,
        tanimoto_threshold=tanimoto_threshold,
        radius=radius,
        fp_size=fp_size,
    )
    cached_graph = _load_conflict_graph_cache(cache_path=cache_path, cache_key=cache_key)
    if cached_graph is not None:
        return cached_graph

    LOGGER.info("Building ligand conflict graph from scratch")
    fingerprints = _build_fingerprints(ligands, radius=radius, fp_size=fp_size)
    conflict_graph = _build_conflict_graph(
        ligands=ligands,
        fingerprints=fingerprints,
        tanimoto_threshold=tanimoto_threshold,
    )
    _write_conflict_graph_cache(
        cache_path=cache_path,
        cache_key=cache_key,
        conflict_graph=conflict_graph,
    )
    return conflict_graph


def _validate_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    """Validate configured split fractions."""
    fractions = {
        "train": train_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }
    for split_name, value in fractions.items():
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{split_name} fraction must be between 0 and 1; got {value}")

    total = train_fraction + val_fraction + test_fraction
    if total > 1.0 + 1e-9:
        raise ValueError(
            "Split fractions must sum to at most 1.0 so ligands can be dropped; "
            f"got {total:.4f}"
        )


def _compute_target_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, float]:
    """Return desired active split fractions."""
    _validate_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    return {
        "train": train_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }


def _compute_target_counts(
    total_ligands: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    """Return desired ligand counts for active splits."""
    _validate_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    test_target = min(total_ligands, int(round(total_ligands * test_fraction)))
    val_target = min(total_ligands - test_target, int(round(total_ligands * val_fraction)))
    train_target = min(
        total_ligands - test_target - val_target,
        int(round(total_ligands * train_fraction)),
    )
    return {
        "test": test_target,
        "val": val_target,
        "train": train_target,
    }


def _randomized_order(
    ligands: list[str],
    conflict_graph: dict[str, set[str]],
    rng: random.Random,
) -> list[str]:
    """Return a seeded ligand order with a light low-conflict bias."""
    decorated = [(len(conflict_graph[ligand]), rng.random(), ligand) for ligand in ligands]
    decorated.sort()
    return [ligand for _, _, ligand in decorated]


def _can_join_split(
    ligand: str,
    split_name: str,
    splits: dict[str, set[str]],
    conflict_graph: dict[str, set[str]],
) -> bool:
    """Return whether a ligand can join a split without cross-split leakage."""
    blockers = set()
    for other_split in ACTIVE_SPLIT_NAMES:
        if other_split != split_name:
            blockers.update(splits[other_split])
    return conflict_graph[ligand].isdisjoint(blockers)


def _init_protein_counts(protein_ligands: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    """Initialize per-protein split counters to zero."""
    return {
        protein: {split_name: 0 for split_name in ALL_SPLIT_NAMES}
        for protein in protein_ligands
    }


def _split_balance_score(current_fraction: float, target_fraction: float) -> float:
    """Score how much a protein needs more ligands in a split (lower is better)."""
    if current_fraction < target_fraction:
        return target_fraction - current_fraction
    # Penalize overfilled splits so underfilled val/test buckets are preferred.
    return (current_fraction - target_fraction) + 1.0


def _pick_balanced_split(
    ligand: str,
    candidate_splits: list[str],
    ligand_proteins: dict[str, set[str]],
    protein_counts: dict[str, dict[str, int]],
    protein_ligands: dict[str, set[str]],
    target_fractions: dict[str, float],
) -> str:
    """Pick the candidate split that best improves per-protein balance."""
    affected_proteins = ligand_proteins.get(ligand, set())
    best_split = candidate_splits[0]
    best_score = float("inf")
    for split_name in candidate_splits:
        balance_score = 0.0
        for protein in affected_proteins:
            total = len(protein_ligands[protein])
            if total == 0:
                continue
            current_fraction = protein_counts[protein][split_name] / total
            target_fraction = target_fractions[split_name]
            balance_score += _split_balance_score(current_fraction, target_fraction)
        if balance_score < best_score or (balance_score == best_score and split_name < best_split):
            best_score = balance_score
            best_split = split_name
    return best_split


def _update_protein_counts(
    ligand: str,
    split_name: str,
    ligand_proteins: dict[str, set[str]],
    protein_counts: dict[str, dict[str, int]],
) -> None:
    """Increment per-protein counters after assigning a ligand."""
    for protein in ligand_proteins.get(ligand, set()):
        protein_counts[protein][split_name] += 1


def _assign_split_from_order(
    ordered_ligands: list[str],
    conflict_graph: dict[str, set[str]],
    target_counts: dict[str, int],
    forbidden_test_ligands: set[str],
    ligand_proteins: dict[str, set[str]],
    protein_ligands: dict[str, set[str]],
    target_fractions: dict[str, float],
) -> dict[str, set[str]]:
    """Greedily assign ligands to active splits, then drop leftovers."""
    splits = {
        "train": set(),
        "val": set(),
        "test": set(),
        "dropped": set(),
    }
    protein_counts = _init_protein_counts(protein_ligands)

    for ligand in ordered_ligands:
        candidate_splits: list[str] = []
        for split_name in ("test", "val", "train"):
            if split_name == "test" and ligand in forbidden_test_ligands:
                continue
            if len(splits[split_name]) >= target_counts[split_name]:
                continue
            if _can_join_split(ligand, split_name, splits, conflict_graph):
                candidate_splits.append(split_name)

        if candidate_splits:
            split_name = _pick_balanced_split(
                ligand=ligand,
                candidate_splits=candidate_splits,
                ligand_proteins=ligand_proteins,
                protein_counts=protein_counts,
                protein_ligands=protein_ligands,
                target_fractions=target_fractions,
            )
            splits[split_name].add(ligand)
            _update_protein_counts(
                ligand=ligand,
                split_name=split_name,
                ligand_proteins=ligand_proteins,
                protein_counts=protein_counts,
            )
        else:
            splits["dropped"].add(ligand)
            _update_protein_counts(
                ligand=ligand,
                split_name="dropped",
                ligand_proteins=ligand_proteins,
                protein_counts=protein_counts,
            )

    return splits


def _per_protein_balance_penalty(
    protein_ligands: dict[str, set[str]],
    assignments: dict[str, str],
    target_fractions: dict[str, float],
) -> float:
    """Return the sum of absolute fraction deviations across proteins and splits."""
    penalty = 0.0
    for protein, ligands in protein_ligands.items():
        total = len(ligands)
        if total == 0:
            continue
        for split_name in ACTIVE_SPLIT_NAMES:
            count = sum(1 for ligand in ligands if assignments.get(ligand) == split_name)
            actual_fraction = count / total
            penalty += abs(actual_fraction - target_fractions[split_name])
    return penalty


def _compute_per_protein_stats(
    protein_ligands: dict[str, set[str]],
    assignments: dict[str, str],
) -> dict[str, PerProteinStats]:
    """Compute train/val/test/dropped ligand counts and percentages per protein."""
    stats: dict[str, PerProteinStats] = {}
    for protein in sorted(protein_ligands):
        ligands = protein_ligands[protein]
        total = len(ligands)
        counts = {
            split_name: sum(1 for ligand in ligands if assignments.get(ligand) == split_name)
            for split_name in ALL_SPLIT_NAMES
        }
        stats[protein] = {
            "total": total,
            "train": counts["train"],
            "val": counts["val"],
            "test": counts["test"],
            "dropped": counts["dropped"],
            "train_pct": 100.0 * counts["train"] / total if total else 0.0,
            "val_pct": 100.0 * counts["val"] / total if total else 0.0,
            "test_pct": 100.0 * counts["test"] / total if total else 0.0,
        }
    return stats


def _make_restart_result(
    ligands: list[str],
    conflict_graph: dict[str, set[str]],
    target_counts: dict[str, int],
    forbidden_test_ligands: set[str],
    seed: int,
    restart_index: int,
    ligand_proteins: dict[str, set[str]],
    protein_ligands: dict[str, set[str]],
    target_fractions: dict[str, float],
) -> RestartResult:
    """Run one randomized restart and capture the resulting split assignment."""
    restart_seed = seed + restart_index
    rng = random.Random(restart_seed)
    ordered_ligands = _randomized_order(ligands, conflict_graph, rng)
    splits = _assign_split_from_order(
        ordered_ligands=ordered_ligands,
        conflict_graph=conflict_graph,
        target_counts=target_counts,
        forbidden_test_ligands=forbidden_test_ligands,
        ligand_proteins=ligand_proteins,
        protein_ligands=protein_ligands,
        target_fractions=target_fractions,
    )
    assignments = {
        ligand: split_name
        for split_name, split_ligands in splits.items()
        for ligand in split_ligands
    }
    return RestartResult(
        restart_index=restart_index,
        seed=restart_seed,
        assignments=assignments,
        splits=splits,
        ligand_proteins=ligand_proteins,
        protein_ligands=protein_ligands,
        target_fractions=target_fractions,
    )


def _validate_result(
    ligands: list[str],
    conflict_graph: dict[str, set[str]],
    result: RestartResult,
) -> None:
    """Validate the split assignment contract before writing output."""
    expected_ligands = set(ligands)
    assigned_ligands = set(result.assignments)
    if assigned_ligands != expected_ligands:
        missing = sorted(expected_ligands - assigned_ligands)[:10]
        extra = sorted(assigned_ligands - expected_ligands)[:10]
        raise ValueError(
            f"Split assignment mismatch. Missing ligands: {missing}; extra ligands: {extra}"
        )

    invalid_split_names = set(result.assignments.values()) - set(ALL_SPLIT_NAMES)
    if invalid_split_names:
        raise ValueError(f"Found invalid split labels: {sorted(invalid_split_names)}")

    active_splits = tuple(ACTIVE_SPLIT_NAMES)
    for left_index, left_split in enumerate(active_splits):
        for right_split in active_splits[left_index + 1 :]:
            for ligand in result.splits[left_split]:
                conflicts = conflict_graph[ligand] & result.splits[right_split]
                if conflicts:
                    raise ValueError(
                        f"Validation failed: {left_split} ligand {ligand} conflicts "
                        f"with {right_split} ligands {sorted(conflicts)[:5]}"
                    )


def _validate_per_protein_balance(
    protein_ligands: dict[str, set[str]],
    assignments: dict[str, str],
    target_fractions: dict[str, float],
    min_ligands_for_required_val_test: int = MIN_LIGANDS_FOR_REQUIRED_VAL_TEST,
) -> None:
    """Ensure large proteins retain val/test coverage; warn for smaller proteins."""
    per_protein_stats = _compute_per_protein_stats(protein_ligands, assignments)
    worst_deviation = 0.0
    worst_protein = ""
    for protein, stats in per_protein_stats.items():
        total = int(stats["total"])
        val_count = int(stats["val"])
        test_count = int(stats["test"])
        kept = total - int(stats["dropped"])

        for split_name in ACTIVE_SPLIT_NAMES:
            actual_fraction = int(stats[split_name]) / total if total else 0.0
            deviation = abs(actual_fraction - target_fractions[split_name])
            if deviation > worst_deviation:
                worst_deviation = deviation
                worst_protein = protein

        if kept < min_ligands_for_required_val_test:
            if val_count == 0 or test_count == 0:
                LOGGER.warning(
                    "Protein %s has only %d kept ligands and is missing val=%d or test=%d",
                    protein,
                    kept,
                    val_count,
                    test_count,
                )
            continue

        if val_count == 0 or test_count == 0:
            raise ValueError(
                f"Protein {protein} has {kept} kept ligands but is missing "
                f"val={val_count} or test={test_count} ligands"
            )

    LOGGER.info(
        "Worst per-protein fraction deviation: protein=%s, deviation=%.4f",
        worst_protein,
        worst_deviation,
    )


def _make_restart_results(
    ligands: list[str],
    conflict_graph: dict[str, set[str]],
    target_counts: dict[str, int],
    seed: int,
    num_restarts: int,
    ligand_proteins: dict[str, set[str]],
    protein_ligands: dict[str, set[str]],
    target_fractions: dict[str, float],
) -> list[RestartResult]:
    """Build restart candidates while keeping their test sets non-overlapping."""
    restart_results = []
    previously_tested_ligands: set[str] = set()
    for restart_index in range(num_restarts):
        result = _make_restart_result(
            ligands=ligands,
            conflict_graph=conflict_graph,
            target_counts=target_counts,
            forbidden_test_ligands=previously_tested_ligands,
            seed=seed,
            restart_index=restart_index,
            ligand_proteins=ligand_proteins,
            protein_ligands=protein_ligands,
            target_fractions=target_fractions,
        )
        restart_results.append(result)
        previously_tested_ligands.update(result.splits["test"])
        LOGGER.info(
            (
                "Restart %d (seed=%d): train=%d, val=%d, test=%d, "
                "dropped=%d, forbidden_test_for_next=%d, balance_penalty=%.4f"
            ),
            result.restart_index,
            result.seed,
            result.train_size,
            result.val_size,
            result.test_size,
            result.dropped_size,
            len(previously_tested_ligands),
            result.per_protein_balance_penalty,
        )
    return restart_results


def _format_per_protein_report_lines(
    per_protein_stats: dict[str, PerProteinStats],
) -> list[str]:
    """Format per-protein split counts for the text report."""
    lines = [
        "per_protein_ligand_splits:",
        (
            "Percentages are computed over total ligands per uniprot_id "
            "(including dropped ligands)."
        ),
        (
            f"{'uniprot_id':<12} {'total':>6} {'train':>6} {'val':>6} "
            f"{'test':>6} {'dropped':>8} {'train_pct':>10} {'val_pct':>9} {'test_pct':>10}"
        ),
    ]
    for protein, stats in sorted(per_protein_stats.items()):
        lines.append(
            (
                f"{protein:<12} {int(stats['total']):>6} "
                f"{int(stats['train']):>6} {int(stats['val']):>6} "
                f"{int(stats['test']):>6} {int(stats['dropped']):>8} "
                f"{float(stats['train_pct']):>9.1f}% "
                f"{float(stats['val_pct']):>8.1f}% "
                f"{float(stats['test_pct']):>9.1f}%"
            )
        )
    return lines


def _write_split_manifest_report(
    report_txt: str | Path | None,
    ligands: list[str],
    target_counts: dict[str, int],
    restart_results: list[RestartResult],
    best_result: RestartResult,
    total_conflicts: int,
    per_protein_stats: dict[str, PerProteinStats],
) -> Path | None:
    """Write a small text report summarizing restart split sizes."""
    if report_txt is None:
        return None

    report_path = ensure_parent_dir(report_txt)
    lines = [
        "split_manifest restart report",
        "",
        f"total_ligands: {len(ligands)}",
        f"total_conflicts: {total_conflicts}",
        (
            "targets: "
            f"train={target_counts['train']}, "
            f"val={target_counts['val']}, "
            f"test={target_counts['test']}"
        ),
        f"winning_restart: {best_result.restart_index}",
        f"winning_seed: {best_result.seed}",
        f"winning_balance_penalty: {best_result.per_protein_balance_penalty:.6f}",
        "",
        "restart_results:",
    ]
    for result in restart_results:
        winner_suffix = " (winner)" if result.restart_index == best_result.restart_index else ""
        lines.append(
            (
                f"- restart={result.restart_index}, seed={result.seed}, "
                f"train={result.train_size}, val={result.val_size}, "
                f"test={result.test_size}, dropped={result.dropped_size}, "
                f"kept={result.kept_size}, "
                f"balance_penalty={result.per_protein_balance_penalty:.6f}"
                f"{winner_suffix}"
            )
        )

    lines.append("")
    lines.extend(_format_per_protein_report_lines(per_protein_stats))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote split manifest report: %s", report_path)
    return report_path


def build_split_manifest(
    affinity_csv: str | Path = AFFINITY_MANIFEST_CSV,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    fp_size: int = DEFAULT_FINGERPRINT_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
    num_restarts: int = DEFAULT_NUM_RESTARTS,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    conflict_graph_cache: str | Path | None = SPLIT_CONFLICT_GRAPH_CACHE_PKL,
    report_txt: str | Path | None = SPLIT_MANIFEST_REPORT_TXT,
    min_ligands_for_required_val_test: int = MIN_LIGANDS_FOR_REQUIRED_VAL_TEST,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build split and protein manifests from unique ligands in `affinity_manifest.csv`."""
    if num_restarts <= 0:
        raise ValueError(f"num_restarts must be positive; got {num_restarts}")

    ligands, ligand_proteins, protein_ligands = _read_ligand_proteins(affinity_csv)
    conflict_graph = _get_conflict_graph(
        ligands=ligands,
        tanimoto_threshold=tanimoto_threshold,
        radius=radius,
        fp_size=fp_size,
        cache_path=conflict_graph_cache,
    )
    target_fractions = _compute_target_fractions(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    target_counts = _compute_target_counts(
        total_ligands=len(ligands),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    restart_results = _make_restart_results(
        ligands=ligands,
        conflict_graph=conflict_graph,
        target_counts=target_counts,
        seed=seed,
        num_restarts=num_restarts,
        ligand_proteins=ligand_proteins,
        protein_ligands=protein_ligands,
        target_fractions=target_fractions,
    )
    best_result = max(restart_results, key=lambda result: result.score)
    _validate_result(ligands=ligands, conflict_graph=conflict_graph, result=best_result)
    _validate_per_protein_balance(
        protein_ligands=protein_ligands,
        assignments=best_result.assignments,
        target_fractions=target_fractions,
        min_ligands_for_required_val_test=min_ligands_for_required_val_test,
    )

    split_df = _build_split_manifest_df(ligands=ligands, assignments=best_result.assignments)
    protein_df = _build_protein_manifest_df(protein_ligands=protein_ligands)

    total_conflicts = sum(len(neighbors) for neighbors in conflict_graph.values()) // 2
    per_protein_stats = _compute_per_protein_stats(
        protein_ligands=protein_ligands,
        assignments=best_result.assignments,
    )
    LOGGER.info(
        (
            "Winning split: total=%d, targets(train=%d,val=%d,test=%d), "
            "actual(train=%d,val=%d,test=%d,dropped=%d), "
            "base_seed=%d, winning_restart=%d, winning_seed=%d, conflicts=%d, "
            "balance_penalty=%.6f"
        ),
        len(ligands),
        target_counts["train"],
        target_counts["val"],
        target_counts["test"],
        best_result.train_size,
        best_result.val_size,
        best_result.test_size,
        best_result.dropped_size,
        seed,
        best_result.restart_index,
        best_result.seed,
        total_conflicts,
        best_result.per_protein_balance_penalty,
    )
    for split_name in ("train", "val", "test", "dropped"):
        split_count = len(best_result.splits[split_name])
        LOGGER.info("%s: %d ligands (%.3f)", split_name, split_count, split_count / len(ligands))
    for protein, stats in sorted(per_protein_stats.items()):
        LOGGER.info(
            "Protein %s: total=%d, train=%d, val=%d, test=%d, dropped=%d",
            protein,
            int(stats["total"]),
            int(stats["train"]),
            int(stats["val"]),
            int(stats["test"]),
            int(stats["dropped"]),
        )

    _write_split_manifest_report(
        report_txt=report_txt,
        ligands=ligands,
        target_counts=target_counts,
        restart_results=restart_results,
        best_result=best_result,
        total_conflicts=total_conflicts,
        per_protein_stats=per_protein_stats,
    )

    return split_df, protein_df


def parse_args() -> argparse.Namespace:
    """CLI for generating `split_manifest.csv`."""
    parser = argparse.ArgumentParser(
        description=(
            "Build split_manifest.csv from affinity_manifest.csv using ligand "
            "Tanimoto conflicts."
        )
    )
    parser.add_argument(
        "--affinity-csv",
        default=str(AFFINITY_MANIFEST_CSV),
        help="Path to affinity_manifest.csv.",
    )
    parser.add_argument(
        "--tanimoto-threshold",
        type=float,
        default=DEFAULT_TANIMOTO_THRESHOLD,
        help="Ligands with similarity >= this value cannot span active splits.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_FINGERPRINT_RADIUS,
        help="Morgan fingerprint radius.",
    )
    parser.add_argument(
        "--fp-size",
        type=int,
        default=DEFAULT_FINGERPRINT_SIZE,
        help="Morgan fingerprint size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Base random seed for restart generation.",
    )
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=DEFAULT_NUM_RESTARTS,
        help="Number of randomized restarts to evaluate before choosing the best split.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Desired train fraction.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Desired validation fraction.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Desired test fraction.",
    )
    parser.add_argument(
        "--conflict-graph-cache",
        default=str(SPLIT_CONFLICT_GRAPH_CACHE_PKL),
        help="Path to the processed conflict-graph cache pickle.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(SPLIT_MANIFEST_CSV),
        help="Output path for split_manifest.csv.",
    )
    parser.add_argument(
        "--protein-output-csv",
        default=str(PROTEIN_MANIFEST_CSV),
        help="Output path for protein_manifest.csv.",
    )
    parser.add_argument(
        "--report-txt",
        default=str(SPLIT_MANIFEST_REPORT_TXT),
        help="Output path for the restart report text file.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    split_df, protein_df = build_split_manifest(
        affinity_csv=args.affinity_csv,
        tanimoto_threshold=args.tanimoto_threshold,
        radius=args.radius,
        fp_size=args.fp_size,
        seed=args.seed,
        num_restarts=args.num_restarts,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        conflict_graph_cache=args.conflict_graph_cache,
        report_txt=args.report_txt,
    )
    output_path = write_manifest(split_df, args.output_csv)
    protein_output_path = write_manifest(protein_df, args.protein_output_csv)
    LOGGER.info("Wrote split manifest: %s (%d rows)", output_path, len(split_df))
    LOGGER.info("Wrote protein manifest: %s (%d rows)", protein_output_path, len(protein_df))


if __name__ == "__main__":
    main()
