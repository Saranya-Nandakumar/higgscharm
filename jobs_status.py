"""
This script inspects Condor job outputs, identifies missing jobs, analyzes recent xrootd-related failures, and optionally regenerates input filesets with problematic sites blacklisted and resubmits only the missing jobs
"""

import yaml
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

from analysis.utils import make_output_directory
from analysis.filesets.xrootd_sites import xroot_to_site
from analysis.filesets.utils import (
    divide_list,
    modify_site_list,
    extract_xrootd_errors,
    get_nano_version,
)
from submit_condor import move_proxy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--workflow",
        dest="workflow",
        required=True,
        type=str,
        choices=[
            f.stem for f in (Path.cwd() / "analysis" / "workflows").glob("*.yaml")
        ],
        help="Workflow name (must correspond to a YAML file in analysis/workflows)",
    )
    parser.add_argument(
        "-y",
        "--year",
        type=str,
        required=True,
        choices=[
            "2016preVFP",
            "2016postVFP",
            "2017",
            "2018",
            "2022preEE",
            "2022postEE",
            "2023preBPix",
            "2023postBPix",
            "2024",
        ],
        help="Dataset year",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Read job outputs from EOS instead of local storage",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "parquet"],
        help="Output file format of the produced histograms",
    )
    parser.add_argument(
        "--hours_ago",
        type=int,
        default=8,
        help="Only consider .err files modified within the last N hours",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete all job outputs, logs, and filesets for this workflow/year and rerun",
    )
    parser.add_argument(
        "--single_fetch",
        action="store_true",
        help="Run fetch.py only once by merging all failing sites and datasets into a single group",
    )
    return parser.parse_args()


def get_jobs_info(job_dir, output_dir, log_dir, output_format, hours_ago=3):
    """
    Inspect the job directory structure and determine:
      1. Which jobs were expected to run (from jobnum.txt)
      2. Which jobs successfully produced output files
      3. Which jobs produced recent error logs

    error logs are collected *per dataset*, which allows later
    analysis and remediation to remain dataset-local rather than global.

    Parameters
    ----------
    job_dir : Path
        Directory containing one subdirectory per dataset with Condor configs.
    output_dir : Path
        Directory containing produced output files.
    log_dir : Path
        Directory containing Condor log files (.err, .out, .log).
    output_format : str
        File extension of produced outputs (e.g. "coffea", "parquet").
    hours_ago : int
        Only .err files modified within this many hours are considered relevant.

    Returns
    -------
    tuple
        jobnum : dict
            dataset -> list of expected job numbers (strings)
        jobnum_done : dict
            dataset -> list of job numbers that produced output
        error_files : dict
            dataset -> list of Path objects pointing to recent .err files
    """
    jobnum, jobnum_done = {}, {}
    error_files = {}

    for dataset_dir in job_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset = dataset_dir.name
        jobnum_path = dataset_dir / "jobnum.txt"
        if not jobnum_path.exists():
            raise FileNotFoundError(
                f"Missing jobnum.txt for dataset '{dataset}'. "
                f"Expected at: {jobnum_path}"
            )

        # Read expected job numbers
        jobnum[dataset] = jobnum_path.read_text().splitlines()

        # Discover completed jobs by looking for output files
        output_files = list((output_dir / dataset).glob(f"*.{output_format}"))
        jobnum_done[dataset] = [f.stem.replace(f"{dataset}_", "") for f in output_files]

        # Collect recent error logs for this dataset
        x_hours_ago = datetime.now() - timedelta(hours=hours_ago)
        dataset_errs = []
        for err_file in (log_dir / dataset).glob("*.err"):
            if datetime.fromtimestamp(err_file.stat().st_mtime) > x_hours_ago:
                dataset_errs.append(err_file)

        if dataset_errs:
            error_files[dataset] = dataset_errs

    return jobnum, jobnum_done, error_files


def print_job_status(jobnum, jobnum_done):
    """
    Print a concise but informative summary of the job execution state.

    The function reports:
      - Total expected jobs
      - Total completed jobs
      - Total missing jobs
      - A YAML-formatted list of datasets with missing jobs

    Parameters
    ----------
    jobnum : dict
        dataset -> list of expected job numbers
    jobnum_done : dict
        dataset -> list of completed job numbers

    Returns
    -------
    tuple
        jobnum_missing : dict
            dataset -> set of missing job numbers
        datasets_with_missing : list
            List of dataset names with at least one missing job
    """
    jobnum_missing = {d: set(jobnum[d]) - set(jobnum_done.get(d, [])) for d in jobnum}
    n_expected = sum(len(v) for v in jobnum.values())
    n_done = sum(len(v) for v in jobnum_done.values())
    n_missing = sum(len(v) for v in jobnum_missing.values())

    print("------------------------------------------------------------")
    print("JOB STATUS SUMMARY")
    print("------------------------------------------------------------")
    print(f"Total jobs expected : {n_expected}")
    print(f"Total jobs finished : {n_done}")
    print(f"Total jobs missing  : {n_missing}")

    datasets_with_missing = [d for d in jobnum_missing if jobnum_missing[d]]
    if n_missing:
        print("------------------------------------------------------------")
        print(
            f"Datasets with missing jobs ({len(datasets_with_missing)}/{len(jobnum)}):"
        )
        print("------------------------------------------------------------")
        print(
            yaml.dump(
                datasets_with_missing,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
            )
        )

    return jobnum_missing, datasets_with_missing


def analyze_xrootd_errors_by_dataset(error_files):
    """
    Analyze xrootd-related failures on a per-dataset basis.

    For each dataset, this function:
      1. Parses its recent .err files
      2. Extracts xrootd endpoints or error patterns
      3. Maps them to physical sites using `xroot_to_site`
      4. Produces a set of failing sites per dataset

    This dataset-level granularity avoids the pathological behavior
    of blacklisting a site globally when it only affects one dataset.

    Parameters
    ----------
    error_files : dict
        dataset -> list of Path objects pointing to .err files

    Returns
    -------
    dict
        dataset -> set of sites that exhibited xrootd errors
    """
    dataset_sites = {}

    for dataset, err_files in error_files.items():
        xrootd_errs = extract_xrootd_errors(err_files)

        # Translate xrootd endpoints to site names where possible
        sites = {xroot_to_site[err] for err in xrootd_errs if err in xroot_to_site}
        if sites:
            dataset_sites[dataset] = sites

        # Warn explicitly if some endpoints could not be mapped
        for err in xrootd_errs:
            if err not in xroot_to_site:
                print(
                    f"[{dataset}] Could not map xrootd endpoint '{err}' to a site name"
                )

    if not dataset_sites:
        print(f"No xrootd site failures detected in recent error logs.\n")
    return dataset_sites


def group_datasets_by_sites(dataset_sites):
    """
    Group datasets by identical sets of failing sites.

    Many datasets often fail on exactly the same sites.
    Rather than regenerating input filesets separately
    for each dataset, we group datasets by their failing-site
    signature and regenerate filesets once per group.

    Example:
        Input:
            {
              "A": {"T2_US_Florida", "T2_FR_GRIF"},
              "B": {"T2_US_Florida", "T2_FR_GRIF"},
              "C": {"T2_IT_Pisa"},
            }

        Output:
            {
              frozenset({"T2_US_Florida", "T2_FR_GRIF"}): ["A", "B"],
              frozenset({"T2_IT_Pisa"}): ["C"],
            }

    Parameters
    ----------
    dataset_sites : dict
        dataset -> set of failing sites

    Returns
    -------
    dict
        frozenset(failing sites) -> list of datasets
    """
    groups = defaultdict(list)
    for dataset, sites in dataset_sites.items():
        groups[frozenset(sites)].append(dataset)
    return groups


def update_input_filesets_for_group(
    sites_to_blacklist, year, fileset_dir, job_dir, datasets
):
    """
    Regenerate input filesets for a group of datasets that share the same
    failing sites.

    The procedure is:
      1. Reset all sites to "white" (allowed)
      2. Blacklist only the sites known to be problematic for this group
      3. Run fetch.py once for all datasets in the group
      4. Regenerate partition files (partitions.json) for each dataset

    This minimizes expensive calls to fetch.py while still preserving
    dataset-level correctness.

    Parameters
    ----------
    sites_to_blacklist : iterable
        Collection of site names to blacklist for this group
    year : str
        Dataset year
    fileset_dir : Path
        Directory containing JSON filesets
    job_dir : Path
        Directory containing Condor job files
    datasets : list
        List of dataset names to regenerate
    """
    print("------------------------------------------------------------")
    print("REGENERATING FILESETS FOR DATASET GROUP")
    print("------------------------------------------------------------")
    print(f"Datasets           : {datasets}")
    print(f"Blacklisted sites  : {sorted(sites_to_blacklist)}")

    # First, reset all sites to "white" to ensure no cross-group contamination.
    # This guarantees that each group is regenerated with exactly its own
    # blacklist, and nothing else.
    for site in xroot_to_site.values():
        modify_site_list(year, site, "white")

    # Apply the blacklist for this specific group.
    for site in sites_to_blacklist:
        modify_site_list(year, site, "black")

    # Run fetch.py once for the entire dataset group.
    samples_str = " ".join(datasets)
    print("Running fetch.py for this group...")
    subprocess.run(
        ["python3", "fetch.py", "--year", year, "--samples", samples_str],
        check=True,
    )

    # Load the regenerated filesets
    nano_version = get_nano_version(year)
    fileset_path = fileset_dir / f"fileset_{year}_nanov{nano_version}_lxplus.json"
    all_filesets = json.loads(fileset_path.read_text())

    # Regenerate partitions.json for each dataset in the group
    for dataset in datasets:
        if dataset not in all_filesets:
            print(f"[{dataset}] Not found in fileset JSON — skipping")
            continue

        root_files = all_filesets[dataset]
        args_json = job_dir / dataset / "arguments.json"
        if not args_json.exists():
            print(f"[{dataset}] Missing arguments.json — cannot repartition")
            continue

        nfiles = json.loads(args_json.read_text())["nfiles"]
        root_files_list = divide_list(root_files, nfiles)

        partition_dataset = {
            i
            + 1: {(f"{dataset}_{i+1}" if len(root_files_list) > 1 else dataset): chunk}
            for i, chunk in enumerate(root_files_list)
        }

        partition_file = job_dir / dataset / "partitions.json"
        with open(partition_file, "w") as json_file:
            json.dump(partition_dataset, json_file, indent=4)

    print("Fileset regeneration for this group completed.\n")


def resubmit_jobs(job_dir, jobnum_missing, datasets_with_missing_jobs, workflow, year):
    """
    Prepare and resubmit only the jobs that are missing output files.

    For each affected dataset:
      1. Write a missing.txt file listing the missing job numbers
      2. Patch the Condor submission file to use missing.txt instead of jobnum.txt
      3. Submit the modified job description to Condor

    A backup of the original submission file (*_all.sub) is created once
    per dataset to avoid destructive overwrites.

    Parameters
    ----------
    job_dir : Path
        Directory containing Condor job files
    jobnum_missing : dict
        dataset -> set of missing job numbers
    datasets_with_missing_jobs : list
        List of dataset names with missing jobs
    workflow : str
        Workflow name
    year : str
        Dataset year
    """
    print("------------------------------------------------------------")
    print("RESUBMITTING MISSING JOBS")
    print("------------------------------------------------------------")

    # Refresh the AFS proxy copy each .sub file's X509PATH argument points
    # to. Unlike a fresh `submit_condor.py --submit` run, this resubmit path
    # reuses an existing .sub file verbatim -- without this, jobs silently
    # run against whatever proxy snapshot was copied at the campaign's last
    # real submission, which can be hours/days stale and expire mid-queue
    # (seen 2026-08-26/27: ~30% of a resubmitted batch failed on
    # "Auth failed: No protocols left to try" from an expired AFS proxy
    # copy, misdiagnosed initially as site-wide XRootD congestion).
    move_proxy()

    to_resubmit = []
    for dataset in datasets_with_missing_jobs:
        missing_jobs = sorted(jobnum_missing[dataset])
        missing_file = job_dir / dataset / "missing.txt"
        with open(missing_file, "w") as f:
            print(*missing_jobs, sep="\n", file=f)

        condor_file = job_dir / dataset / f"{workflow}_{dataset}.sub"
        condor_backup = condor_file.with_name(condor_file.stem + "_all.sub")
        if not condor_backup.exists():
            subprocess.run(["cp", str(condor_file), str(condor_backup)], check=True)

        submit_text = condor_file.read_text().replace("jobnum.txt", "missing.txt")
        condor_file.write_text(submit_text)
        to_resubmit.append(str(condor_file))

    for submit_file in to_resubmit:
        print(f"Submitting {submit_file}")
        subprocess.run(["condor_submit", submit_file], check=True)

    print("Job resubmission completed.\n")


def main():
    args = parse_args()

    # Optional full reset
    if args.reset:
        print("------------------------------------------------------------")
        print("RESET REQUESTED — REMOVING ALL JOB ARTIFACTS AND RESTARTING")
        print("------------------------------------------------------------")

        nano_version = get_nano_version(args.year)
        subprocess.run(
            ["rm", "-rf", f"condor/{args.workflow}/{args.year}"], check=False
        )
        subprocess.run(
            ["rm", "-rf", f"condor/logs/{args.workflow}/{args.year}"], check=False
        )
        subprocess.run(
            ["rm", "-rf", f"analysis/filesets/{args.year}_sites.yaml"], check=False
        )
        subprocess.run(
            [
                "rm",
                "-rf",
                f"analysis/filesets/fileset_{args.year}_nanov{nano_version}_lxplus.json",
            ],
            check=False,
        )

        reset_cmd = [
            "python3",
            "runner.py",
            "-w",
            args.workflow,
            "-y",
            args.year,
            "--output_format",
            args.output_format,
        ]
        if args.eos:
            reset_cmd.append("--eos")

        print("Re-running full workflow via runner.py...")
        subprocess.run(reset_cmd, check=True)
        print("Reset complete.\n")

    # Directory setup
    output_dir = Path(make_output_directory(args))
    base_dir = Path.cwd()
    condor_dir = base_dir / "condor"
    job_dir = condor_dir / args.workflow / args.year
    log_dir = condor_dir / "logs" / args.workflow / args.year
    fileset_dir = base_dir / "analysis" / "filesets"

    # Discover job state
    jobnum, jobnum_done, error_files = get_jobs_info(
        job_dir, output_dir, log_dir, args.output_format, args.hours_ago
    )

    jobnum_missing, datasets_with_missing_jobs = print_job_status(jobnum, jobnum_done)

    if not datasets_with_missing_jobs:
        return

    # Analyze xrootd errors per dataset
    dataset_sites = analyze_xrootd_errors_by_dataset(error_files)

    # Restrict to datasets that actually have missing jobs
    dataset_sites = {
        d: sites
        for d, sites in dataset_sites.items()
        if d in datasets_with_missing_jobs
    }

    if not dataset_sites:
        print("No xrootd-related failures detected for datasets with missing jobs.")
        print("You may want to inspect other error modes or resubmit directly.\n")
    else:
        # Group datasets by identical failing-site sets
        groups = group_datasets_by_sites(dataset_sites)
        if args.single_fetch:
            all_sites = set().union(*dataset_sites.values())
            all_datasets = sorted(dataset_sites.keys())
            groups = {frozenset(all_sites): all_datasets}
        else:
            groups = group_datasets_by_sites(dataset_sites)

        # Optional fileset regeneration
        if input("Regenerate input filesets for these groups? (y/n): ").lower() in [
            "y",
            "yes",
        ]:
            for sites, datasets in groups.items():
                update_input_filesets_for_group(
                    sites, args.year, fileset_dir, job_dir, datasets
                )

    # Optional job resubmission
    if input("Resubmit missing jobs now? (y/n): ").lower() in ["y", "yes"]:
        resubmit_jobs(
            job_dir,
            jobnum_missing,
            datasets_with_missing_jobs,
            args.workflow,
            args.year,
        )


if __name__ == "__main__":
    main()
