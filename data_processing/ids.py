"""Helpers for canonical chemistry strings and stable manifest IDs.

The goal of this module is to make all ID construction deterministic and easy
to reuse across manifest-building scripts. Future scripts should import these
helpers instead of rebuilding IDs ad hoc.
"""

from __future__ import annotations

from rdkit import Chem


def canonicalize_smiles(smiles: str) -> str:
    """Return a canonical RDKit SMILES string.

    We canonicalize before assigning ligand identity so equivalent ligands map
    to the same `ligand_id` even if the source CSV wrote the SMILES in
    different ways.
    """
    if smiles is None:
        raise ValueError("SMILES is missing")

    smiles = str(smiles).strip()
    if not smiles:
        raise ValueError("SMILES is empty")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    return Chem.MolToSmiles(mol, canonical=True)


def make_ligand_id(smiles: str) -> str:
    """Use canonical SMILES directly as the readable ligand identifier."""
    return canonicalize_smiles(smiles)


def make_pdb_key(pdb_id: str, uniprot_id: str) -> str:
    """Combine a PDB structure and target identifier into one stable key."""
    pdb_id = str(pdb_id).strip()
    uniprot_id = str(uniprot_id).strip()
    if not pdb_id or not uniprot_id:
        raise ValueError("pdb_id and uniprot_id are both required")
    return f"{pdb_id}_{uniprot_id}"


def parse_grid_string(grid: str) -> tuple[str, str]:
    """Split a grid string like `4LDO_P07550` into `(pdb_id, uniprot_id)`."""
    grid = str(grid).strip()
    parts = grid.split("_", maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Grid string must look like 'PDB_UNIPROT', got: {grid}")
    return parts[0], parts[1]


def make_pose_id(
    pdb_key: str,
    ligand: str,
    pose_number: int | str,
) -> str:
    """Build a stable pose ID for one docked ligand pose.

    Pose IDs are keyed by the PDB context, the ligand identity, and the
    per-ligand pose rank within that PDB context.
    """
    ligand = canonicalize_smiles(ligand)
    return f"{pdb_key}_{ligand}_{pose_number}"
