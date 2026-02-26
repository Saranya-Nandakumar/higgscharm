import argparse
import subprocess
from pathlib import Path
from analysis.filesets.utils import (
    fileset_checker,
    get_datasets_to_run_over,
)


if __name__ == "__main__":
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
        help="workflow to run",
    )
    parser.add_argument(
        "-y",
        "--year",
        dest="year",
        type=str,
        choices=[
            "2016preVFP",
            "2016postVFP",
            "2017",
            "2018",
            "2022preEE",
            "2022postEE",
            "2023preBPix",
            "2023postBPix",
            "2024"
        ],
        help="dataset year",
    )
    parser.add_argument(
        "--nfiles",
        dest="nfiles",
        type=int,
        default=15,
        help="number of root files to include in each dataset partition (default 15)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Enable Condor job submission. If not provided, it just builds condor files",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Enable saving outputs to /eos",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "parquet"],
        help="format of output file",
    )
    parser.add_argument(
        "--use_6class",
        action="store_true",
        help="Use 6-class MVA labels (separate gg->ZZ and qq->ZZ). Default is 5-class (combined).",
    )
    parser.add_argument(
        "--mva_model_path",
        type=str,
        default=None,
        help="Path to ONNX or PyTorch model for MVA inference. If not provided, MVA scores are not computed.",
    )
    parser.add_argument(
        "--use_pytorch",
        action="store_true",
        help="Use PyTorch model instead of ONNX for MVA inference.",
    )
    parser.add_argument(
        "--mva_hidden_dim",
        type=int,
        default=384,
        help="Hidden dimension for PyTorch MVA model (default: 64).",
    )
    parser.add_argument(
        "--mva_num_layers",
        type=int,
        default=12,
        help="Number of layers for PyTorch MVA model (default: 2).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mlp",
        choices=["mlp", "part"],
        help="MVA model type: 'mlp' for MLP_HcZZ_MW_Deep, 'part' for ParticleTransformer2_HcZZ (default: mlp).",
    )
    parser.add_argument(
        "--mass_window",
        action="store_true",
        help="Apply Higgs mass window cut (100 < m4l < 150 GeV) before MVA inference.",
    )
    args = parser.parse_args()

    # get datasets to run over for selected workflow and year
    datasets_to_run_over = get_datasets_to_run_over(args.workflow, args.year)

    # check if the input fileset for the given year exists, generate it otherwise
    fileset_checker(samples=datasets_to_run_over, year=args.year)

    # submit (or prepare) a job for each dataset using the given arguments
    cmd = ["python3", "submit_condor.py"]
    for dataset in datasets_to_run_over:
        cmd_args = [
            "--workflow",
            args.workflow,
            "--year",
            args.year,
            "--dataset",
            dataset,
            "--nfiles",
            str(args.nfiles),
            "--output_format",
            args.output_format,
        ]
        if args.submit:
            cmd_args.append("--submit")
        if args.eos:
            cmd_args.append("--eos")
        if args.use_6class:
            cmd_args.append("--use_6class")
        if args.mva_model_path:
            cmd_args.extend(["--mva_model_path", args.mva_model_path])
        if args.use_pytorch:
            cmd_args.append("--use_pytorch")
            cmd_args.extend(["--mva_hidden_dim", str(args.mva_hidden_dim)])
            cmd_args.extend(["--mva_num_layers", str(args.mva_num_layers)])
            cmd_args.extend(["--model_type", args.model_type])
        if args.mass_window:
            cmd_args.append("--mass_window")
        subprocess.run(cmd + cmd_args)
