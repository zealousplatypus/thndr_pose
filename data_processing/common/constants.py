"""Shared constants for the MVP data-processing pipeline.

This module keeps filenames, default paths, and a few small configuration
values in one place so the manifest-building scripts stay consistent.
"""

from pathlib import Path


# Resolve the MVP root from this file so the scripts can be run from anywhere.
MVP_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = MVP_ROOT / "raw"
PROCESSED_DIR = MVP_ROOT / "processed"
RUNS_DIR = MVP_ROOT / "runs"


# Raw input files/directories used by the manifest builders.
RAW_BINDING_CSV = RAW_DIR / "mol_binding_data.csv"
RAW_OUTPUTS_DIR = RAW_DIR / "output"


# Processed manifest outputs for the generalized MVP pipeline.
AFFINITY_MANIFEST_CSV = PROCESSED_DIR / "affinity_manifest.csv"
PDB_MANIFEST_CSV = PROCESSED_DIR / "pdb_manifest.csv"
POSE_MANIFEST_CSV = PROCESSED_DIR / "pose_manifest.csv"
POSE_LMDB_DIR = PROCESSED_DIR / "pose_lmdb"
POSE_DUPLICATES_CSV = PROCESSED_DIR / "pose_duplicate_keys.csv"
LIGAND_CLUSTER_MANIFEST_CSV = PROCESSED_DIR / "ligand_cluster_manifest.csv"
SPLIT_MANIFEST_CSV = PROCESSED_DIR / "split_manifest.csv"
PROTEIN_MANIFEST_CSV = PROCESSED_DIR / "protein_manifest.csv"
SPLIT_CONFLICT_GRAPH_CACHE_PKL = PROCESSED_DIR / "split_conflict_graph.pkl"
SPLIT_MANIFEST_REPORT_TXT = PROCESSED_DIR / "split_manifest_report.txt"
AFFINITY_SPLIT_MANIFEST_CSV = PROCESSED_DIR / "affinity_split_manifest.csv"
CHEMPROP_EMBEDDINGS_NPY = PROCESSED_DIR / "chemprop_embeddings.float32.npy"
CHEMPROP_SMILES_BY_IDX_JSON = PROCESSED_DIR / "chemprop_smiles_by_idx.json"
CHEMPROP_EMBEDDINGS_METADATA_JSON = PROCESSED_DIR / "chemprop_embeddings_metadata.json"


# Default columns in the raw binding CSV.
RAW_BINDING_REQUIRED_COLUMNS = (
    "uniprot_id",
    "smiles",
    "binding_affinity",
    "pIC50",
)

SPLIT_MANIFEST_COLUMNS = (
    "ligand",
    "ligand_idx",
    "split",
)

PROTEIN_MANIFEST_COLUMNS = (
    "uniprot_id",
    "protein_idx",
)

AFFINITY_MANIFEST_COLUMNS = (
    "uniprot_id",
    "ligand",
    "affinity",
)

AFFINITY_SPLIT_MANIFEST_COLUMNS = (
    "uniprot_id",
    "protein_idx",
    "ligand",
    "ligand_idx",
    "affinity",
    "split",
)

PDB_MANIFEST_COLUMNS = (
    "pdb_key",
    "uniprot_id",
    "pdb_id",
)

POSE_MANIFEST_COLUMNS = (
    "pose_id",
    "pose_hash",
    "pdb_key",
    "ligand",
    "grid",
    "uniprot_id",
    "pdb_id",
    "glide_score",
    "pose_rank",
    "is_top_rank",
    "source_sdf",
)

# Default clustering / splitting parameters for bag-style training manifests.
DEFAULT_TANIMOTO_THRESHOLD = 0.3
DEFAULT_FINGERPRINT_RADIUS = 2
DEFAULT_FINGERPRINT_SIZE = 248
DEFAULT_SPLIT_SEED = 7
DEFAULT_TRAIN_FRACTION = 0.98
DEFAULT_VAL_FRACTION = 0.01
DEFAULT_TEST_FRACTION = 0.01
MIN_LIGANDS_FOR_REQUIRED_VAL_TEST = 100
ACTIVE_SPLIT_NAMES = ("train", "val", "test")
ALL_SPLIT_NAMES = ACTIVE_SPLIT_NAMES + ("dropped",)