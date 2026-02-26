import os
import json
import argparse
from pathlib import Path
from coffea import processor
from coffea.util import save
from coffea.nanoevents import NanoAODSchema
from analysis.utils import write_root
from analysis.processors.base import BaseProcessor


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
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--partition_json",
        dest="partition_json",
        type=str,
        help="json with partition dataset",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "root", "parquet"],
        help="format of output histogram",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Enable saving outputs to /eos",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="User name",
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
        default=64,
        help="Hidden dimension for PyTorch MVA model (default: 64).",
    )
    parser.add_argument(
        "--mva_num_layers",
        type=int,
        default=2,
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

    # set output location (used when --output_format parquet)
    if args.eos:
        output_location = f"root://eosuser.cern.ch//eos/user/{args.user[0]}/{args.user}/higgscharm/outputs/"
    else:
        output_location = (
            f"/afs/cern.ch/user/{args.user[0]}/{args.user}/public/higgscharm/outputs/"
        )

    # load partition_fileset, run processor and save output
    with open(args.partition_json) as f:
        partition_fileset = json.load(f)
    out = processor.run_uproot_job(
        partition_fileset,
        treename="Events",
        processor_instance=BaseProcessor(
            workflow=args.workflow,
            year=args.year,
            output_format=args.output_format,
            output_location=output_location,
            use_6class=args.use_6class,
            mva_model_path=args.mva_model_path,
            use_pytorch=args.use_pytorch,
            mva_hidden_dim=args.mva_hidden_dim,
            mva_num_layers=args.mva_num_layers,
            model_type=args.model_type,
            mass_window=args.mass_window,
        ),
        executor=processor.futures_executor,
        executor_args={"schema": NanoAODSchema, "workers": 4},
    )
    savepath = f"{args.output_path}/{args.dataset}"
    if args.output_format in ["coffea", "parquet"]:
        save(out, f"{savepath}.coffea")
    elif args.output_format == "root":
        write_root(out, savepath, args)
