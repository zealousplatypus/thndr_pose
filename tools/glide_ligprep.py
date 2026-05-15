import subprocess

from pathlib import Path
import click


@click.command()
@click.argument("input_smiles", type=click.Path(dir_okay=False, readable=True))
@click.argument("output_mae", type=click.Path(dir_okay=False, writable=True))
@click.option("--overwrite", is_flag=True, help="Overwrite output if it already exists")
@click.option(
    "--keep-logs", is_flag=True, help="Keep log files from running this command"
)
def main(input_smiles, output_mae, overwrite, keep_logs):
    """
    Glide LigPrep helper utility. Pass in the input path to a ligand smiles
    file and the output MAE file to generate.
    """
    input_smiles = Path(input_smiles)
    output_mae = Path(output_mae)

    # If we're not overwriting and prepped structure already exists, we're done
    if not overwrite and output_mae.exists():
        click.echo(
            f"Found {input_smiles} already converted to {output_mae}, ligprep complete."
        )
        return 0

    # Then, run ligprep with scratch dir
    temp_scratch = output_mae.parent / f"{output_mae.stem}_scratch"
    temp_scratch.mkdir(exist_ok=True)
    subprocess.run(
        f"ligprep -WAIT -NONICE -TMPDIR {temp_scratch} -epik -ismi {input_smiles} -omae {output_mae}",
        shell=True,
    )

    # Scratch output is intentionally preserved for debugging.
    if keep_logs:
        return 0

    # Otherwise, we now start cleaning up the logs, by iterating through all possible log files and removing them
    log_files = []
    files_to_check = [x for x in Path.cwd().iterdir() if x.is_file()]
    files_to_check += [x for x in output_mae.parent.iterdir() if x.is_file()]
    for file in files_to_check:
        if input_smiles.stem in file.stem and "dropped" in file.stem:
            log_files.append(file.absolute())
        elif output_mae.stem in file.stem and file.suffix == ".log":
            log_files.append(file.absolute())
    for log_file in log_files:
        log_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
