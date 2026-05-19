from pathlib import Path
import subprocess

import click
import pandas as pd
from tqdm import tqdm

# Magic constants
MAX_NUM_JOBS = 1000


@click.command()
@click.argument("INPUT_CSV")
@click.option(
    "-l",
    "--log-dir",
    type=click.Path(file_okay=False, writable=True),
    default=Path("logs"),
    show_default=True,
    help="Path to subfolder to write job output data to",
)
@click.option(
    "-n",
    "--num-jobs",
    type=int,
    default=-1,
    show_default=True,
    help="Number of jobs to submit, where -1 means as many as needed, 0 starts running tasks directly, with max of 1K",
)
@click.option(
    "-p",
    "--partition",
    type=str,
    default="rondror,owners",
    show_default=True,
    help="Partitions to submit to",
)
@click.option(
    "-t",
    "--time",
    type=int,
    default=4,
    show_default=True,
    help="Time in hours to run job for",
)
@click.option(
    "-j",
    "--job-name",
    type=str,
    default="",
    show_default=True,
    help="Job name for all submitted jobs, by default empty and set based on INPUT_CSV",
)
@click.option(
    "-c",
    "--cpus",
    type=int,
    default=1,
    show_default=True,
    help="CPUs to request per submitted job",
)
@click.option(
    "-g",
    "--gpus",
    type=int,
    default=0,
    show_default=True,
    help="GPUs to request per submitted job",
)
@click.option(
    "--additional-args",
    type=str,
    default="",
    show_default=True,
    help="Additional arguments to pass to sbatch",
)
def main(
    input_csv, log_dir, num_jobs, partition, time, job_name, cpus, gpus, additional_args
):
    """
    Program to submit batches of jobs read from CSV file to Sherlock SLURM, and
    get notifications when jobs finish running. Note that the CSV must have
    a "command" column, which has a full filepath to a script to run, with all
    the remaining columns being passed in sequence as positional arguments.
    """
    # Load jobs in
    jobs_df = pd.read_csv(input_csv)

    # Confirm we have jobs, our command column exists and is first, and setup num_jobs
    total_tasks = jobs_df.shape[0]
    assert total_tasks > 0, f"{input_csv} is empty, shape {jobs_df.shape}"
    assert "command" == jobs_df.columns[0], IndexError(
        f"Expected 'command' column in {input_csv}, but only found {jobs_df.columns}"
    )
    if num_jobs < 0:
        num_jobs = min(MAX_NUM_JOBS, total_tasks)
    elif num_jobs > 0:
        num_jobs = min(num_jobs, total_tasks)

    # Now make all the commands we'll be running into a list of strings
    task_commands = []
    for _, row in jobs_df.iterrows():
        command = " ".join(str(part) for part in row.to_list())
        task_commands.append(command)

    # If we run directly on this node (num_jobs == 0), we just go for it
    if num_jobs == 0:
        click.echo(f"Running {total_tasks} jobs directly on node ...")
        for command in task_commands:
            subprocess.run(command, shell=True)
        return

    # Otherwise, set up batching
    job_batches = [[] for _ in range(num_jobs)]
    for i, command in enumerate(task_commands):
        job_batches[i % num_jobs].append(command)

    # Logging directory
    log_dir = Path(log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Base job script, we add to this for each batch
    if not job_name:
        job_name = task_commands[0].split()[0]
    job_script = (
        "#!/bin/env bash\n"
        f"#SBATCH -p {partition}\n"
        f"#SBATCH -t {time}:00:00\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH -c {cpus}\n"
    )
    if gpus > 0:
        job_script += f"#SBATCH -G {gpus}\n"
    if additional_args:
        job_script += f"#SBATCH {additional_args}\n"

    # Now, actually start submitting jobs
    click.echo(f"Submitting {num_jobs} total jobs ...")
    for i, job_batch in enumerate(tqdm(job_batches)):
        batch_output = log_dir / f"{job_name}_batch_{i}.out"
        curr_batch_script = job_script + f"#SBATCH -o {batch_output}\n"
        curr_batch_script += "\n".join(job_batch) + "\n"
        script_file = log_dir / f"{job_name}_batch_{i}.sh"
        script_file.write_text(curr_batch_script)
        subprocess.run(f"sbatch {script_file}", shell=True)
    # We're done
    click.echo("We're done!")


if __name__ == "__main__":
    main()
