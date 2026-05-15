*Data Processing Pipeline:*

data_processing/
    common/
        constants.py
        ids.py: canonicalize smiles, make_ligand_id
        manifest_io.py: read with required-column validation, write manifests, ensure output directories, and assert uniqueness.

    data_split/ 
        affinity_manifest.py: safe conversion of mol_binding_data.csv -> affinity_manifest.csv
        split_manifest.py: safe conversion of affinity_manifest.csv -> split_manifest.csv
        affinity_split_manifest.py: safe conversion of split_manifest.csv + affinity_manifest.csv -> affinity_split_manifest.csv

    ligand_embeddings/
        pose_manifest_and_lmdb.py

    protein_embeddings
        pyg_manifest_and_lmdb.py

processed/
    pose_lmdb/
    pyg_lmdb/
    ...csv


Run manifest builders from the repository root with module syntax, for example:
        `python -m data_processing.data_split.affinity_manifest`