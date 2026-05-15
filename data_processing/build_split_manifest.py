"""Build a ligand-level train/val/test split manifest for the MVP pipeline.

Unlike the older cluster-based approach, this script enforces only the actual
cross-split rule: ligands in different active splits must have Tanimoto
similarity below the configured threshold. Similar ligands may still coexist
within the same split.
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

try:
    from constants import (
        ALL_SPLIT_NAMES,
        AFFINITY_MANIFEST_CSV,
        PDB_MANIFEST_CSV,
        POSE_MANIFEST_CSV,
        DEFAULT_FINGERPRINT_RADIUS,
        DEFAULT_FINGERPRINT_SIZE,
        DEFAULT_SPLIT_SEED,
        DEFAULT_TANIMOTO_THRESHOLD,
        DEFAULT_TEST_FRACTION,
        DEFAULT_TRAIN_FRACTION,
        DEFAULT_VAL_FRACTION,
        SPLIT_CONFLICT_GRAPH_CACHE_PKL,
        SPLIT_MANIFEST_CSV,
    )
    from manifest_io import assert_unique, ensure_parent_dir, read_csv_checked, write_manifest
    from selector_utils import collapse_pose_membership_to_examples, resolve_pose_membership
except ImportError:  # pragma: no cover
    from .constants import (
        ALL_SPLIT_NAMES,
        AFFINITY_MANIFEST_CSV,
        PDB_MANIFEST_CSV,
        POSE_MANIFEST_CSV,
        DEFAULT_FINGERPRINT_RADIUS,
        DEFAULT_FINGERPRINT_SIZE,
        DEFAULT_SPLIT_SEED,
        DEFAULT_TANIMOTO_THRESHOLD,
        DEFAULT_TEST_FRACTION,
        DEFAULT_TRAIN_FRACTION,
        DEFAULT_VAL_FRACTION,
        SPLIT_CONFLICT_GRAPH_CACHE_PKL,
        SPLIT_MANIFEST_CSV,
    )
    from .manifest_io import assert_unique, ensure_parent_dir, read_csv_checked, write_manifest
    from .selector_utils import collapse_pose_membership_to_examples, resolve_pose_membership


LOGGER = logging.getLogger(__name__)
DEFAULT_NUM_RESTARTS = 3
CONFLICT_GRAPH_CACHE_VERSION = 1


@dataclass(frozen=True)
class RestartResult:
    """One candidate split assignment from a randomized restart."""

    restart_index: int
    seed: int
    assignments: dict[str, str]
    splits: dict[str, set[str]]

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
        return self.test_size + self.val_size + self.train_size

    @property
    def score(self) -> tuple[int, int, int, int]:
        # Prefer larger held-out splits first, then more retained ligands.
        # Final negative restart index gives deterministic tie-breaking.
        return (self.test_size, self.val_size, self.kept_size, -self.restart_index)


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
    """Return an undirected graph connecting ligand pairs with high similarity."""
    conflict_graph = {ligand: set() for ligand in ligands}
    for i, left_ligand in enumerate(ligands):
        left_fp = fingerprints[left_ligand]
        for j in range(i + 1, len(ligands)):
            right_ligand = ligands[j]
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
    """Load a cached conflict graph if it matches the current ligand set and params."""
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
        LOGGER.warning("Ignoring malformed conflict-graph cache %s: expected a dict payload", cache_path)
        return None
    if payload.get("cache_key") != cache_key:
        return None

    conflict_graph = payload.get("conflict_graph")
    if not isinstance(conflict_graph, dict):
        LOGGER.warning(
            "Ignoring malformed conflict-graph cache %s: missing conflict_graph dict",
            cache_path,
        )
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
    payload = {
        "cache_key": cache_key,
        "conflict_graph": conflict_graph,
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    LOGGER.info("Saved conflict-graph cache: %s", cache_path)


def _get_conflict_graph(
    ligands: list[str],
    fingerprints: dict[str, DataStructs.cDataStructs.ExplicitBitVect],
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


def _read_forbidden_test_ligands(split_csv: str | Path | None) -> set[str]:
    """Read previously used test ligands that should not reappear in new test."""
    if split_csv is None:
        return set()
    split_df = read_csv_checked(split_csv, ["ligand", "split"])
    return set(split_df.loc[split_df["split"] == "test", "ligand"].tolist())


def _select_experiment_affinity_universe(
    affinity_csv: str | Path,
    pdb_csv: str | Path,
    pose_csv: str | Path,
    uniprot_to_pdb_csv: str | Path | None,
    ligand_to_pose_csv: str | Path | None,
) -> pd.DataFrame:
    """Restrict the affinity table to the currently selected experiment universe."""
    affinity_df = read_csv_checked(affinity_csv, ["uniprot_id", "ligand", "affinity"])
    if uniprot_to_pdb_csv is None and ligand_to_pose_csv is None:
        return affinity_df

    pdb_df = read_csv_checked(pdb_csv, ["uniprot_id", "pdb_key", "pdb_id"])
    pose_df = read_csv_checked(
        pose_csv,
        [
            "pose_id",
            "pdb_key",
            "ligand",
            "uniprot_id",
            "pdb_id",
            "glide_score",
            "pose_rank",
        ],
    )
    pose_membership_df = resolve_pose_membership(
        affinity_df=affinity_df,
        pdb_df=pdb_df,
        pose_df=pose_df,
        uniprot_to_pdb_csv=uniprot_to_pdb_csv,
        ligand_to_pose_csv=ligand_to_pose_csv,
    )
    return collapse_pose_membership_to_examples(pose_membership_df)


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
            "Split fractions must sum to at most 1.0 so ligands can still be dropped; "
            f"got {total:.4f}"
        )


def _compute_target_counts(
    total_ligands: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    """Return desired ligand counts for held-out splits."""
    _validate_fractions(train_fraction, val_fraction, test_fraction)
    test_target = min(total_ligands, int(round(total_ligands * test_fraction)))
    remaining_after_test = max(total_ligands - test_target, 0)
    val_target = min(remaining_after_test, int(round(total_ligands * val_fraction)))
    return test_target, val_target


def _randomized_order(
    ligands: list[str],
    conflict_graph: dict[str, set[str]],
    rng: random.Random,
) -> list[str]:
    """Return a seeded-but-variable ligand order with a light low-degree bias."""
    decorated = [(len(conflict_graph[ligand]), rng.random(), ligand) for ligand in ligands]
    decorated.sort()
    return [ligand for _, _, ligand in decorated]


def _assign_split_from_order(
    ordered_ligands: list[str],
    conflict_graph: dict[str, set[str]],
    forbidden_test_ligands: set[str],
    test_target: int,
    val_target: int,
) -> dict[str, set[str]]:
    """Greedily fill test, then val, then train from one randomized ligand order."""
    splits = {
        "train": set(),
        "val": set(),
        "test": set(),
        "dropped": set(),
    }

    remaining = []
    for ligand in ordered_ligands:
        if len(splits["test"]) < test_target and ligand not in forbidden_test_ligands:
            splits["test"].add(ligand)
        else:
            remaining.append(ligand)

    still_unassigned = []
    for ligand in remaining:
        if len(splits["val"]) >= val_target:
            still_unassigned.append(ligand)
            continue
        if conflict_graph[ligand].isdisjoint(splits["test"]):
            splits["val"].add(ligand)
        else:
            still_unassigned.append(ligand)

    blocked_by_heldout = splits["test"] | splits["val"]
    for ligand in still_unassigned:
        if conflict_graph[ligand].isdisjoint(blocked_by_heldout):
            splits["train"].add(ligand)
        else:
            splits["dropped"].add(ligand)

    return splits


def _make_restart_result(
    ligands: list[str],
    conflict_graph: dict[str, set[str]],
    forbidden_test_ligands: set[str],
    test_target: int,
    val_target: int,
    seed: int,
    restart_index: int,
) -> RestartResult:
    """Run one randomized restart and capture the resulting split assignment."""
    restart_seed = seed + restart_index
    rng = random.Random(restart_seed)
    ordered_ligands = _randomized_order(ligands, conflict_graph, rng)
    splits = _assign_split_from_order(
        ordered_ligands=ordered_ligands,
        conflict_graph=conflict_graph,
        forbidden_test_ligands=forbidden_test_ligands,
        test_target=test_target,
        val_target=val_target,
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
    )


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return Jaccard similarity, treating two empty sets as identical."""
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _log_restart_diversity(best_result: RestartResult, restart_results: list[RestartResult]) -> None:
    """Summarize how much alternate restarts differ from the winning assignment."""
    if len(restart_results) <= 1:
        return

    for result in restart_results:
        if result.restart_index == best_result.restart_index:
            continue
        moved_ligands = sum(
            1
            for ligand, split_name in result.assignments.items()
            if best_result.assignments[ligand] != split_name
        )
        LOGGER.info(
            (
                "Restart %d (seed=%d) vs winner: moved=%d, "
                "test_jaccard=%.3f, val_jaccard=%.3f, train_jaccard=%.3f"
            ),
            result.restart_index,
            result.seed,
            moved_ligands,
            _jaccard_similarity(result.splits["test"], best_result.splits["test"]),
            _jaccard_similarity(result.splits["val"], best_result.splits["val"]),
            _jaccard_similarity(result.splits["train"], best_result.splits["train"]),
        )


def _validate_result(
    ligands: list[str],
    conflict_graph: dict[str, set[str]],
    forbidden_test_ligands: set[str],
    result: RestartResult,
) -> None:
    """Validate the split assignment contract before writing output."""
    assigned_ligands = set(result.assignments)
    expected_ligands = set(ligands)
    if assigned_ligands != expected_ligands:
        missing = sorted(expected_ligands - assigned_ligands)[:10]
        extra = sorted(assigned_ligands - expected_ligands)[:10]
        raise ValueError(
            f"Split assignment mismatch. Missing ligands: {missing}; extra ligands: {extra}"
        )

    split_names = set(result.assignments.values())
    invalid_split_names = split_names - set(ALL_SPLIT_NAMES)
    if invalid_split_names:
        raise ValueError(f"Found invalid split labels: {sorted(invalid_split_names)}")

    for ligand in result.splits["val"]:
        conflicting_heldout = conflict_graph[ligand] & result.splits["test"]
        if conflicting_heldout:
            raise ValueError(
                f"Validation failed: val ligand {ligand} conflicts with test ligands "
                f"{sorted(conflicting_heldout)[:5]}"
            )

    train_blockers = result.splits["test"] | result.splits["val"]
    for ligand in result.splits["train"]:
        conflicting_heldout = conflict_graph[ligand] & train_blockers
        if conflicting_heldout:
            raise ValueError(
                f"Validation failed: train ligand {ligand} conflicts with held-out ligands "
                f"{sorted(conflicting_heldout)[:5]}"
            )

    reused_forbidden = result.splits["test"] & forbidden_test_ligands
    if reused_forbidden:
        raise ValueError(
            "Validation failed: forbidden ligands were reused in test: "
            f"{sorted(reused_forbidden)[:10]}"
        )


def build_split_manifest(
    affinity_csv: str | Path,
    pdb_csv: str | Path = PDB_MANIFEST_CSV,
    pose_csv: str | Path = POSE_MANIFEST_CSV,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    fp_size: int = DEFAULT_FINGERPRINT_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
    num_restarts: int = DEFAULT_NUM_RESTARTS,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    conflict_graph_cache: str | Path | None = SPLIT_CONFLICT_GRAPH_CACHE_PKL,
    forbid_test_from_split_csv: str | Path | None = None,
    uniprot_to_pdb_csv: str | Path | None = None,
    ligand_to_pose_csv: str | Path | None = None,
    report_restart_diversity: bool = False,
) -> pd.DataFrame:
    """Build `split_manifest.csv` with a direct ligand-level assignment policy."""
    if num_restarts <= 0:
        raise ValueError(f"num_restarts must be positive; got {num_restarts}")

    affinity_df = _select_experiment_affinity_universe(
        affinity_csv=affinity_csv,
        pdb_csv=pdb_csv,
        pose_csv=pose_csv,
        uniprot_to_pdb_csv=uniprot_to_pdb_csv,
        ligand_to_pose_csv=ligand_to_pose_csv,
    )
    ligands = sorted(affinity_df["ligand"].drop_duplicates().tolist())
    if not ligands:
        raise ValueError("Affinity manifest contains no ligands to split.")

    fingerprints = _build_fingerprints(ligands, radius=radius, fp_size=fp_size)
    conflict_graph = _get_conflict_graph(
        ligands=ligands,
        fingerprints=fingerprints,
        tanimoto_threshold=tanimoto_threshold,
        radius=radius,
        fp_size=fp_size,
        cache_path=conflict_graph_cache,
    )
    forbidden_test_ligands = _read_forbidden_test_ligands(forbid_test_from_split_csv) & set(ligands)
    test_target, val_target = _compute_target_counts(
        total_ligands=len(ligands),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    restart_results = [
        _make_restart_result(
            ligands=ligands,
            conflict_graph=conflict_graph,
            forbidden_test_ligands=forbidden_test_ligands,
            test_target=test_target,
            val_target=val_target,
            seed=seed,
            restart_index=restart_index,
        )
        for restart_index in range(num_restarts)
    ]
    best_result = max(restart_results, key=lambda result: result.score)
    _validate_result(
        ligands=ligands,
        conflict_graph=conflict_graph,
        forbidden_test_ligands=forbidden_test_ligands,
        result=best_result,
    )

    if report_restart_diversity:
        _log_restart_diversity(best_result, restart_results)

    rows = [
        {
            "ligand": ligand,
            "split": best_result.assignments[ligand],
        }
        for ligand in ligands
    ]
    split_df = pd.DataFrame(rows).sort_values(["split", "ligand"]).reset_index(drop=True)
    assert_unique(split_df, ["ligand"], "split_manifest")

    total_conflicts = sum(len(neighbors) for neighbors in conflict_graph.values()) // 2
    LOGGER.info(
        (
            "Winning split: total=%d, targets(test=%d,val=%d), "
            "actual(train=%d,val=%d,test=%d,dropped=%d), discarded=%d, "
            "forbidden_test=%d, base_seed=%d, winning_restart=%d, winning_seed=%d, conflicts=%d"
        ),
        len(ligands),
        test_target,
        val_target,
        best_result.train_size,
        best_result.val_size,
        best_result.test_size,
        best_result.dropped_size,
        best_result.dropped_size,
        len(forbidden_test_ligands),
        seed,
        best_result.restart_index,
        best_result.seed,
        total_conflicts,
    )
    for split_name in ("train", "val", "test", "dropped"):
        split_count = len(best_result.splits[split_name])
        LOGGER.info(
            "%s: %d ligands (%.3f)",
            split_name,
            split_count,
            split_count / len(ligands),
        )

    return split_df


def parse_args() -> argparse.Namespace:
    """CLI for generating split_manifest.csv."""
    parser = argparse.ArgumentParser(
        description=(
            "Build split_manifest.csv from affinity_manifest.csv using a direct ligand-level "
            "conflict graph and randomized multi-start assignment."
        )
    )
    parser.add_argument(
        "--affinity-csv",
        default=str(AFFINITY_MANIFEST_CSV),
        help="Path to affinity_manifest.csv.",
    )
    parser.add_argument(
        "--pdb-csv",
        default=str(PDB_MANIFEST_CSV),
        help="Path to pdb_manifest.csv, used for experiment-specific selector filtering.",
    )
    parser.add_argument(
        "--pose-csv",
        default=str(POSE_MANIFEST_CSV),
        help="Path to pose_manifest.csv, used for experiment-specific selector filtering.",
    )
    parser.add_argument(
        "--uniprot-to-pdb-csv",
        default=None,
        help="Optional selector CSV restricting which PDBs participate in this experiment.",
    )
    parser.add_argument(
        "--ligand-to-pose-csv",
        default=None,
        help=(
            "Optional selector CSV restricting which pose groups participate in this "
            "experiment. Supports ligand+pdb_id, ligand+pdb_key, or legacy ligand+pose_id."
        ),
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
        help="Desired train fraction (validated but not strictly packed).",
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
        help="Path to a pickle file used to cache the ligand conflict graph.",
    )
    parser.add_argument(
        "--forbid-test-from-split-csv",
        default=None,
        help="Optional previous split_manifest.csv whose test ligands may not reappear in test.",
    )
    parser.add_argument(
        "--report-restart-diversity",
        action="store_true",
        help="Log overlap statistics comparing alternate restarts with the winning split.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(SPLIT_MANIFEST_CSV),
        help="Output path for split_manifest.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    split_df = build_split_manifest(
        affinity_csv=args.affinity_csv,
        pdb_csv=args.pdb_csv,
        pose_csv=args.pose_csv,
        tanimoto_threshold=args.tanimoto_threshold,
        radius=args.radius,
        fp_size=args.fp_size,
        seed=args.seed,
        num_restarts=args.num_restarts,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        conflict_graph_cache=args.conflict_graph_cache,
        forbid_test_from_split_csv=args.forbid_test_from_split_csv,
        uniprot_to_pdb_csv=args.uniprot_to_pdb_csv,
        ligand_to_pose_csv=args.ligand_to_pose_csv,
        report_restart_diversity=args.report_restart_diversity,
    )
    output_path = write_manifest(split_df, args.output_csv)
    LOGGER.info("Wrote split manifest: %s (%d rows)", output_path, len(split_df))


if __name__ == "__main__":
    main()
