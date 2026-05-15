import subprocess
import shutil

from pathlib import Path
import click


GLIDE_FAST_DOCK = """
DOCKING_METHOD confgen
PRECISION SP
POSES_PER_LIG 1
WRITE_CSV true
"""

GLIDE_COMBINDVS_DOCK = """
DOCKING_METHOD confgen
PRECISION SP
POSES_PER_LIG 30
POSTDOCK_NPOSE 30
WRITE_CSV true
"""

GLIDE_COMBIND_DOCK = """
DOCKING_METHOD confgen
PRECISION SP
POSES_PER_LIG 100
POSTDOCK_NPOSE 100
WRITE_CSV true
"""


@click.command()
@click.argument("grid", type=click.Path(dir_okay=False, readable=True))
@click.argument("ligand", type=click.Path(dir_okay=False, readable=True))
@click.argument("output", type=click.Path(dir_okay=False, writable=True))
@click.option("--overwrite", is_flag=True, help="Overwrite output if it already exists")
@click.option(
    "--keep-logs", is_flag=True, help="Keep log files from running this command"
)
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["fast", "combindvs", "combind"]),
    default="combindvs",
    show_default=True,
    show_choices=True,
    help="Docking mode to run in",
)
def main(grid, ligand, output, overwrite, keep_logs, mode):
    """
    Glide docking helper utility. Pass in the input path to a protein grid zip
    file, the ligand ligprep and combind prepped .maegz file, and the output MAE file to generate.
    """
    grid = Path(grid)
    ligand = Path(ligand)
    output = Path(output)
    output_csv = output.parent / f"{output.stem}.csv"

    # If we're not overwriting and prepped structure already exists, we're done
    if not overwrite and output_csv.exists():
        click.echo(f"Found output at {output_csv}, Glide docking complete.")
        return 0

    # Make scratch files
    temp_scratch = output.parent / f"{output.stem}_scratch"
    temp_scratch.mkdir(exist_ok=True)

    # Generate docking settings based on user choice and run
    dock_input = None
    if mode == "fast":
        dock_input = GLIDE_FAST_DOCK
    elif mode == "combindvs":
        dock_input = GLIDE_COMBINDVS_DOCK
    elif mode == "combind":
        dock_input = GLIDE_COMBIND_DOCK
    dock_input += f"GRIDFILE {grid.absolute()}\n"
    dock_input += f"LIGANDFILE {ligand.absolute()}\n"
    dock_settings = output.parent / f"{output.stem}.in"
    dock_settings.write_text(dock_input)
    subprocess.run(
        f"glide -WAIT -TMPDIR {temp_scratch} {dock_settings.absolute()}",
        shell=True,
        cwd=dock_settings.parent.absolute(),
    )

    # If we're keeping logs, we remove tmpdir and we're done
    shutil.rmtree(temp_scratch)
    if keep_logs:
        return 0

    # Otherwise, we now start cleaning up the logs, by iterating through all possible log files and removing them
    log_files = []
    files_to_check = [x for x in Path.cwd().iterdir() if x.is_file()]
    files_to_check += [x for x in dock_settings.parent.iterdir() if x.is_file()]
    for file in files_to_check:
        if output.stem in file.stem and "skip" in file.stem:
            log_files.append(file)
        elif output.stem in file.stem and "raw" in file.stem:
            log_files.append(file)
        elif output.stem in file.stem and file.suffix == ".log":
            log_files.append(file)
        elif output.stem in file.stem and file.suffix == ".json":
            log_files.append(file)
    for log_file in log_files:
        if log_file.exists():
            log_file.unlink()


if __name__ == "__main__":
    main()
