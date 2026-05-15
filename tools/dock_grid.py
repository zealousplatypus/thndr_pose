import subprocess
import time
import pandas as pd
import click
from pathlib import Path
from rdkit import Chem
"""
Test labeled ligand docking workflow
	a. Make a function to filter the csv by protein name.
	b. Make a function that labels the ligands.
	c. Call a then b on P07550-4LDO.
	d. Manually delete all except the first two rows.
	e. Ligprep on this mini-csv
	f. Test docking with grid_v6.zip. How are the poses titled? Can we group all poses into 		{ligand}_{uniprot}_{pdb}.maegz ?
	
	g. Write a protocol for grid preparation. 
	h. Prepare a grid for P07550_4LDO according to this protocol
	i. Test dock mini.maegz
	
	j. Make a function that batches all ligands into csvs of ~100 ligands each. 
	k. Make a function that takes the output of function j —> maegz files. 
	l. Ideally take the output of k and group into sets of poses per ligand of form 
		{ligand}_{pdb}_{uniprot}.maegz


Okay let’s just obey these steps because they were made by a smarter Zane.
They were made by far from the smartest Zane, but he’s was far smarter than I am now. 


- Processing
	- dock_grid.py
		- def label(csv)->csv
			“””Canonicalizes SMILES and uses readable ligand IDs in the LigPrep input CSV”””
		- def filter(csv, uniprot_id)
			“””Given a csv, outputs a csv with the filtered ”””

		- def main({pdb_uniprot.zip}, outdir):
			 “””Given a grid zip with title {pdb}_{uniprot}.zip outputs maegz files containing all docked 				poses of form {ligand}_{pdb}_{uniprot}.maegz
				”””
			"""

"""Return a canonical RDKit SMILES string suitable for readable ligand IDs."""
def canonicalize_smiles(smiles):
	mol = Chem.MolFromSmiles(str(smiles).strip())
	if mol is None:
		raise ValueError(f"Failed to parse SMILES: {smiles}")
	return Chem.MolToSmiles(mol, canonical=True)


"""
Canonicalizes SMILES and stores the readable ligand identity in `title`.
LigPrep will carry this title into downstream docking outputs, which keeps
the metadata easier to interpret than a hash-based ID.
"""
def label(csv):
	csv = csv.copy()
	csv["SMILES"] = csv["smiles"]
	csv["canonical_smiles"] = csv["SMILES"].apply(canonicalize_smiles)
	csv["ligand_id"] = csv["canonical_smiles"]
	csv["title"] = csv["canonical_smiles"]
	# Equivalent SMILES strings should collapse onto the same ligand identity.
	if csv.duplicated(subset=["title", "uniprot_id"]).any():
		raise ValueError(
			"Duplicate canonical ligand titles found for the same uniprot_id"
		)
	return csv

"""
Given a csv, outputs a csv with the filtered uniprot_id
"""
def filter(csv, uniprot_id):
    return csv[csv["uniprot_id"] == uniprot_id]


def _tools_dir():
	"""Resolve tools dir whether dock_grid.py is at project root or inside tools/."""
	script_dir = Path(__file__).resolve().parent
	return script_dir if (script_dir / "glide_dock.py").exists() else script_dir / "tools"


def _wait_for_outputs(output_groups, label, timeout_sec=48 * 3600, interval=120):
	"""Wait until each output group has at least one file present."""
	elapsed = 0
	while not all(any(path.exists() for path in group) for group in output_groups) and elapsed < timeout_sec:
		print(f"Waiting for {len(output_groups)} {label} ... ({elapsed // 60}m)")
		time.sleep(min(interval, timeout_sec - elapsed))
		elapsed += interval
	if not all(any(path.exists() for path in group) for group in output_groups):
		raise TimeoutError(f"{label.capitalize()} did not appear within 48h")


"""
Run LigPrep on a SMILES CSV in batches via tools/sbatch.py.
Writes batch CSVs (batch_0.csv, ...) and a jobs CSV, then runs sbatch so each batch
runs as its own job (or sequentially if num_jobs=0). Output .maegz files are written
to batch_maegz_dir as batch_0.maegz, batch_1.maegz, ...

Input:
	- df: DataFrame with columns SMILES and title
	- batch_maegz_dir: directory to save batch CSVs and .maegz files
	- batch_size: number of ligands per batch
	- overwrite: if False, skip batches whose output .maegz already exists
	- num_jobs: -1 = one SLURM job per batch, 0 = run commands directly (sequential)
	- log_dir: where sbatch writes scripts and .out (default: batch_maegz_dir.parent / "logs")
	- time_hours: SLURM time limit per job
"""
def ligprep_job(df, batch_maegz_dir, batch_size=100, overwrite=True, keep_logs=False, num_jobs=-1, log_dir=None, time_hours=2):
	batch_maegz_dir = Path(batch_maegz_dir)
	batch_maegz_dir.mkdir(parents=True, exist_ok=True)
	tools_dir = _tools_dir()
	ligprep_script = tools_dir / "glide_ligprep.py"

	n = len(df)
	if not batch_size or n == 0:
		batches = [df] if n else []
	else:
		k, r = n // batch_size, n % batch_size
		batches = [
			df.iloc[i * batch_size : (i + 1) * batch_size]
			for i in range(k)
		]
		if r > 0:
			batches.append(df.iloc[k * batch_size :])

	rows = []
	for i, batch in enumerate(batches):
		output_mae = batch_maegz_dir / f"batch_{i}.maegz"
		if not overwrite and output_mae.exists():
			continue
		batch_csv = batch_maegz_dir / f"batch_{i}.csv"
		batch.to_csv(batch_csv, index=False)
		cmd = f"python3 {ligprep_script}"
		overwrite_opt = "--overwrite" if overwrite else ""
		rows.append({"command": cmd, "input_smiles": str(batch_csv.resolve()), "output_mae": str(output_mae.resolve()), "overwrite_opt": overwrite_opt})

	if not rows:
		print("No ligprep batches to run (all outputs exist and overwrite=False)")
		return None

	expected_outputs = [Path(r["output_mae"]) for r in rows]
	jobs_df = pd.DataFrame(rows)
	grid_name = batch_maegz_dir.name.replace("_ligands_maegz", "")
	jobs_csv = batch_maegz_dir.parent / f"ligprep_jobs_{grid_name}.csv"
	jobs_df.to_csv(jobs_csv, index=False)
	jobs_csv_abs = jobs_csv.resolve()
	log_dir = Path(log_dir) if log_dir else batch_maegz_dir.parent / "logs"
	log_dir = log_dir / f"{grid_name}_log"
	log_dir = log_dir.resolve()
	sbatch_script = tools_dir / "sbatch.py"
	args = ["python3", str(sbatch_script), str(jobs_csv_abs), "-n", str(num_jobs), "-l", str(log_dir), "-t", str(time_hours), "-j", f"ligprep_{grid_name}"]
	subprocess.run(args, cwd=tools_dir, check=True)
	# When we submitted SLURM jobs, caller may wait for these before docking
	return expected_outputs if num_jobs != 0 else None


"""
Run Glide docking on each batch .maegz in batch_maegz_dir using the given grid zip.
Builds a jobs CSV and runs tools/sbatch.py (optionally with num_jobs=0 to run directly).
Glide is asked to write batch_0_docked.maegz, batch_1_docked.maegz, ...
"""
def dock_job(batch_maegz_dir, grid_zip, docked_maegz_dir=None, num_jobs=-1, log_dir=None, time_hours=8, **sbatch_kw):
	batch_maegz_dir = Path(batch_maegz_dir)
	grid_zip = Path(grid_zip)
	if docked_maegz_dir is None:
		docked_maegz_dir = batch_maegz_dir.parent / f"{batch_maegz_dir.name.replace('_ligands_maegz', '_docked')}"
	docked_maegz_dir = Path(docked_maegz_dir)
	docked_maegz_dir.mkdir(parents=True, exist_ok=True)

	tools_dir = _tools_dir()
	glide_dock_script = tools_dir / "glide_dock.py"
	grid_abs = grid_zip.resolve()

	batch_files = sorted(batch_maegz_dir.glob("batch_*.maegz"))
	if not batch_files:
		print(f"No batch_*.maegz files in {batch_maegz_dir}")
		return

	rows = []
	requested_outputs = []
	for bf in batch_files:
		# batch_0.maegz -> request batch_0_docked.maegz -> batch_0_docked_{pv|raw}.maegz
		stem = bf.stem
		out_maegz = docked_maegz_dir / f"{stem}_docked.maegz"
		requested_outputs.append(out_maegz.resolve())
		rows.append(
			{
				"command": f"python3 {glide_dock_script.resolve()}",
				"grid": str(grid_abs),
				"ligand": str(bf.resolve()),
				"output": str(out_maegz.resolve()),
				"keep_logs": "--keep-logs",
			}
		)

	jobs_df = pd.DataFrame(rows)
	# sbatch expects first column "command", then args in order
	grid_name = batch_maegz_dir.name.replace("_ligands_maegz", "")
	jobs_csv = batch_maegz_dir.parent / f"dock_jobs_{grid_name}.csv"
	jobs_df.to_csv(jobs_csv, index=False)

	log_dir = Path(log_dir) if log_dir else batch_maegz_dir.parent / "logs"
	log_dir = log_dir / f"{grid_name}_log"
	log_dir = log_dir.resolve()
	jobs_csv_abs = jobs_csv.resolve()
	sbatch_script = tools_dir / "sbatch.py"
	subprocess.run(
		["python3", str(sbatch_script), str(jobs_csv_abs), "-n", str(num_jobs), "-l", str(log_dir), "-t", str(time_hours), "-j", f"glide_dock_{grid_name}"],
		cwd=tools_dir,
		check=True,
	)
	return requested_outputs


"""
Given a grid zip with title {pdb}_{uniprot}.zip, generate LigPrep batch CSVs
and docked structure outputs with readable ligand titles carried through from
canonical SMILES.
"""
@click.command()
@click.argument("pdb_uniprot_grid", type=click.Path(dir_okay=False, readable=True))
@click.argument("binding_csv", type=click.Path(dir_okay=False, readable=True))
@click.argument("outdir", type=click.Path(dir_okay=True, writable=True))
def main(pdb_uniprot_grid, binding_csv, outdir):
	pdb_uniprot_grid = Path(pdb_uniprot_grid)	
	outdir = Path(outdir)
	outdir.mkdir(parents=True, exist_ok=True)
	
	protein_ids = pdb_uniprot_grid.stem.split('_')
	pdb = protein_ids[0]
	uniprot = protein_ids[1]

	# generate the ligands csv for ligprep
	df = pd.read_csv(binding_csv)
	df_labeled = label(df)
	protein_ligand_df = filter(df_labeled, uniprot)
	smiles_df = protein_ligand_df[["SMILES", "title"]]

	# ligprep the csv — batch size 100, submitted via sbatch (one job per batch)
	batch_maegz_dir = outdir / f"{pdb}_{uniprot}_ligands_maegz"
	ligprep_outputs = None
	if not batch_maegz_dir.exists():
		ligprep_outputs = ligprep_job(smiles_df, batch_maegz_dir, batch_size=100, overwrite=True, num_jobs=-1, log_dir=outdir / "logs", time_hours=4)
	else:
		print(f"Skipping ligprep: {batch_maegz_dir} already exists")

	# If ligprep was submitted to SLURM, wait for all batch .maegz before submitting dock jobs (timeout 48h)
	if ligprep_outputs:
		_wait_for_outputs([[p] for p in ligprep_outputs], "ligprep batch(es)")

	# dock each batch .maegz with the grid via tools/sbatch.py
	docked_dir = outdir / f"{pdb}_{uniprot}_docked"
	if not docked_dir.exists():
		dock_job(batch_maegz_dir, pdb_uniprot_grid, docked_maegz_dir=docked_dir, num_jobs=-1, log_dir=outdir / "logs")
	else:
		print(f"Skipping dock_job: {docked_dir} already exists")



if __name__ == "__main__":
	main()

