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


*Models:*

Run all model commands from the repository root.

**Linear Glide score baseline** (`models/linear_glide_score/`)

Fits a train-only line from minimum Glide score to affinity and writes predictions, scatter plots, and result summaries. There is no separate training step; evaluation performs the fit and scoring in one pass. Only ligands with a matching `(uniprot_id, ligand)` entry in the pose manifest are included.

```bash
python -m models.linear_glide_score.linear_evaluation --experiment configs/linear_glide_score_test.json
```

Without an experiment config, pass manifests and output directory directly:

```bash
python -m models.linear_glide_score.linear_evaluation \
  --pose-csv archive/pose_manifest.csv \
  --affinity-split-csv processed/affinity_split_manifest.csv \
  --output-dir runs/linear_glide_score
```

**Chemprop + ESM affinity** (`models/chemprop_esm_affinity/`)

Train on train/val splits and write checkpoints, loss curves, and train/val predictions:

```bash
python -m models.chemprop_esm_affinity.train --experiment configs/chemprop_esm_affinity_example.json
```

Validate config and data without training:

```bash
python -m models.chemprop_esm_affinity.train --experiment configs/chemprop_esm_affinity_example.json --dry-run
```

Evaluate a saved checkpoint on the held-out test split (test is not run during training):

```bash
python -m models.chemprop_esm_affinity.evaluate_test \
  --experiment runs/chemprop_esm_affinity_P07550/experiment.json \
  --checkpoint runs/chemprop_esm_affinity_P07550/checkpoints/best.ckpt
```

Example configs live in `configs/`. Run outputs are written under `runs/<experiment_name>/`.