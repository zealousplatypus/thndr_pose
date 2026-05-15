import os
import shlex
import subprocess
from pathlib import Path

import click
import pandas as pd


def _group_docked_outputs(docked_dir):
    """
    Group Glide docked outputs by base name and prefer `_pv.maegz` over `_raw.maegz`.
    Returns tuples of (source_maegz, output_sdf).
    """
    candidates = {}
    for path in sorted(docked_dir.glob("*.maegz")):
        if path.name.endswith("_pv.maegz"):
            base = path.name[: -len("_pv.maegz")]
            candidates.setdefault(base, {})["pv"] = path
        elif path.name.endswith("_raw.maegz"):
            base = path.name[: -len("_raw.maegz")]
            candidates.setdefault(base, {})["raw"] = path

    conversions = []
    for base in sorted(candidates):
        grouped = candidates[base]
        source = grouped.get("pv") or grouped.get("raw")
        if source is None:
            continue
        conversions.append((source, docked_dir / f"{base}.sdf"))
    return conversions


def _tools_dir():
    return Path(__file__).resolve().parent


@click.command()
@click.argument("docked_dir", type=click.Path(file_okay=False, exists=True, readable=True))
@click.option("--overwrite", is_flag=True, help="Overwrite existing SDF outputs")
@click.option(
    "-n",
    "--num-jobs",
    type=int,
    default=-1,
    show_default=True,
    help="Number of sbatch workers; 0 runs conversions directly",
)
@click.option(
    "-l",
    "--log-dir",
    type=click.Path(file_okay=False, writable=True),
    default=None,
    help="Directory for sbatch logs (defaults to docked_dir parent / logs)",
)
@click.option(
    "-t",
    "--time-hours",
    type=int,
    default=2,
    show_default=True,
    help="SLURM time limit per conversion job in hours",
)
def main(docked_dir, overwrite, num_jobs, log_dir, time_hours):
    """
    Convert docked Glide `_pv.maegz` / `_raw.maegz` files in one directory to SDF.
    When both outputs exist for the same base name, `_pv.maegz` is preferred.
    """
    docked_dir = Path(docked_dir).resolve()
    structconvert = Path(os.path.expandvars("$SCHRODINGER/utilities/structconvert"))
    tools_dir = _tools_dir()
    sbatch_script = tools_dir / "sbatch.py"

    if not structconvert.exists():
        raise FileNotFoundError(
            f"Could not find structconvert at {structconvert}. Is SCHRODINGER set?"
        )

    conversions = _group_docked_outputs(docked_dir)
    if not conversions:
        click.echo(f"No *_pv.maegz or *_raw.maegz files found in {docked_dir}")
        return 0

    rows = []
    skipped = 0
    for source_maegz, output_sdf in conversions:
        if output_sdf.exists() and not overwrite:
            click.echo(f"Skipping existing {output_sdf.name}")
            skipped += 1
            continue

        command = (
            f"{shlex.quote(str(structconvert))} "
            f"{shlex.quote(str(source_maegz))} "
            f"{shlex.quote(str(output_sdf))}"
        )
        rows.append(
            {
                "command": "bash",
                "shell_flag": "-lc",
                "shell_command": shlex.quote(command),
            }
        )

    if not rows:
        click.echo(
            f"No conversions to submit in {docked_dir}: 0 submitted, {skipped} skipped."
        )
        return 0

    jobs_df = pd.DataFrame(rows)
    jobs_csv = docked_dir.parent / f"{docked_dir.name}_sdf_jobs.csv"
    jobs_df.to_csv(jobs_csv, index=False)

    log_dir = Path(log_dir) if log_dir else docked_dir.parent / "logs"
    log_dir = log_dir / f"{docked_dir.name}_sdf"
    log_dir = log_dir.resolve()

    subprocess.run(
        [
            "python3",
            str(sbatch_script),
            str(jobs_csv.resolve()),
            "-n",
            str(num_jobs),
            "-l",
            str(log_dir),
            "-t",
            str(time_hours),
            "-j",
            f"structconvert_{docked_dir.name}",
        ],
        cwd=tools_dir,
        check=True,
    )
    click.echo(
        f"Submitted {len(rows)} conversion job(s) for {docked_dir}; {skipped} skipped."
    )
    return 0


if __name__ == "__main__":
    main()
