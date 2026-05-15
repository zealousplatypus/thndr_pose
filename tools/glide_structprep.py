import subprocess
import shutil

from pathlib import Path
import click


GLIDE_GRID_INPUT = """
INNERBOX 15,15,15
OUTERBOX 30,30,30
LIGAND_INDEX 2
"""


@click.command()
@click.argument("protein", type=click.Path(dir_okay=False, readable=True))
@click.argument("ligand", type=click.Path(dir_okay=False, readable=True))
@click.argument("output", type=click.Path(dir_okay=False, writable=True))
@click.option("--overwrite", is_flag=True, help="Overwrite output if it already exists")
@click.option(
    "--keep-logs", is_flag=True, help="Keep log files from running this command"
)
def main(protein, ligand, output, overwrite, keep_logs):
    """
    Glide StructPrep helper utility. Pass in the input path to a protein
    file, the ligand file, and the output .zip grid files to generate.
    This script also keeps the intermediate structprep output (before grid generation)
    as an additional file that has the same name as the protein file folder name but with the suffix
    "_prepped.mae".
    """
    protein = Path(protein)
    ligand = Path(ligand)
    output = Path(output)

    # If we're not overwriting and prepped structure already exists, we're done
    if not overwrite and output.exists():
        click.echo(
            f"Found {protein} already converted to {output}, structprep complete."
        )
        return 0

    # First, do file conversion if necessary, or just copy
    temp_protein = output.parent / f"{output.stem}_protein_tmp.mae"
    if protein.suffix == ".mae":
        temp_protein.write_bytes(protein.read_bytes())
    else:
        subprocess.run(
            f"$SCHRODINGER/utilities/structconvert {protein.absolute()} {temp_protein.absolute()}",
            shell=True,
        )
    temp_ligand = output.parent / f"{output.stem}_ligand_tmp.mae"
    if ligand.suffix == ".mae":
        temp_ligand.write_bytes(ligand.read_bytes())
    else:
        subprocess.run(
            f"$SCHRODINGER/utilities/structconvert {ligand.absolute()} {temp_ligand.absolute()}",
            shell=True,
        )

    # Then, run structprep and append our ligand to the result
    output_structprep_mae = (
        output.parent / f"{protein.absolute().parent.name}_prepped.mae" # would it make more sense or it to be protein.stem?
    )
    temp_structprep = output.parent / f"{output.stem}_structprep_tmp.mae"
    temp_scratch = output.parent / f"{output.stem}_scratch"
    temp_scratch.mkdir(exist_ok=True)

    preppedir = output.parent
    # subprocess.run(
    #     f"$SCHRODINGER/utilities/prepwizard -rehtreat -watdist 0 -WAIT -NONICE -TMPDIR {temp_scratch.absolute()} "
    #     f"{temp_protein.absolute()} {temp_structprep.absolute()}",
    #     shell=True,
    # )
    subprocess.run(
        f"$SCHRODINGER/utilities/prepwizard -rehtreat -watdist 0 -WAIT "
        f"{temp_protein.name} {temp_structprep.name}",
        shell=True,
        check=True,
        cwd=preppedir,
    )

    # subprocess.run(
    #     f"$SCHRODINGER/utilities/structcat -imae {temp_structprep.absolute()} "
    #     f"-imae {temp_ligand.absolute()} -omae {output_structprep_mae.absolute()}",
    #     shell=True,
    # )
    subprocess.run(
        f"$SCHRODINGER/utilities/structcat -imae {temp_structprep.name} "
        f"-imae {temp_ligand.name} -omae {output_structprep_mae.name}",
        shell=True,
        check=True,
        cwd=preppedir,
    )

    # Now remove log file unless we're supposed to keep it
    out_log = Path.cwd() / temp_protein.with_suffix(".log").name
    if not keep_logs and out_log.exists():
        out_log.unlink()

    # Cleanup our explicit temp files
    temp_protein.unlink()
    temp_ligand.unlink()
    temp_structprep.unlink()

    # Now, we're ready for pt 2 -- generating the docking grid
    grid_settings = output.parent / f"{output.stem}_grid_gen.in"
    grid_input = GLIDE_GRID_INPUT + f"RECEP_FILE {output_structprep_mae.absolute()}\n"
    grid_input += f"GRIDFILE {output.absolute()}\n"
    grid_settings.write_text(grid_input)
    subprocess.run(
        f"glide -WAIT -TMPDIR {temp_scratch} {grid_settings.absolute()}",
        shell=True,
        cwd=grid_settings.parent.absolute(),
    )

    # Cleanup pt 2
    out_log = grid_settings.with_suffix(".log")
    if not keep_logs and out_log.exists():
        out_log.unlink()
    out_log = Path.cwd() / grid_settings.with_suffix(".log").name
    if not keep_logs and out_log.exists():
        out_log.unlink()
    grid_settings.unlink()
    shutil.rmtree(temp_scratch)


if __name__ == "__main__":
    main()
