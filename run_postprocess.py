import gc
import yaml
import json
import glob
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from coffea.util import load, save
from coffea.processor import accumulate
from analysis.postprocess.plotter import CoffeaPlotter
from analysis.workflows.config import WorkflowConfigBuilder
from analysis.filesets.utils import get_workflow_key_process_map, get_process_sample_map
from analysis.postprocess.postprocessor import (
    save_histograms_by_sample,
    save_histograms_by_process,
)
from analysis.postprocess.utils import (
    print_header,
    setup_logger,
    clear_output_directory,
    df_to_latex,
    combine_event_tables,
    combine_cutflows,
    format_cutflow_with_efficiency,
    merge_parquets_by_sample,
    load_processed_histograms,
    get_results_report,
)
from analysis.postprocess.mva_inference import MVAPostProcessor


OUTPUT_DIR = Path.cwd() / "outputs"


def parse_arguments():
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
        required=True,
        choices=[
            "2016preVFP",
            "2016postVFP",
            "2016",
            "2017",
            "2018",
            "2022preEE",
            "2022postEE",
            "2022",
            "2023preBPix",
            "2023postBPix",
            "2023",
            "2024",
        ],
        help="Data year",
    )
    parser.add_argument(
        "--log", action="store_true", help="Enable log scale for y-axis"
    )
    parser.add_argument(
        "--postprocess", action="store_true", help="Enable postprocessing"
    )
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument(
        "--yratio_limits",
        type=float,
        nargs=2,
        default=(0.5, 1.5),
        help="Set y-axis ratio limits",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
        help="Output file extension for plots",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        default="process",
        help="Axis to group by (e.g., 'process', or a JSON dict)",
    )
    parser.add_argument(
        "--pass_axis",
        type=str,
        default=None,
        help="Binary axis (e.g., 'is_passing_lepton')",
    )
    parser.add_argument(
        "--nocutflow", action="store_true", help="Enable postprocessing"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="coffea",
        choices=["coffea", "parquet"],
        help="Format of output files",
    )
    parser.add_argument(
        "--skipmerging", action="store_true", help="Skip parquet outputs merging"
    )
    parser.add_argument("--blind", action="store_true", help="Blind data")
    parser.add_argument(
        "--mva-inference",
        action="store_true",
        help="Run MVA inference on merged parquet files",
    )
    parser.add_argument(
        "--mva-config",
        type=str,
        default=None,
        help="Path to b-hive config YAML for MVA inference",
    )
    parser.add_argument(
        "--mva-model",
        type=str,
        default=None,
        help="Path to trained model .pt file for MVA inference",
    )
    parser.add_argument(
        "--mva-output",
        type=str,
        default=None,
        help="Output directory for MVA-scored parquets (default: outputs/<workflow>_mvascores/<year>)",
    )
    parser.add_argument(
        "--mva-mass-window",
        action="store_true",
        help="Apply mass window cut (100 < m4l < 150 GeV) before running MVA inference",
    )
    return parser.parse_args()


def check_output_dir(workflow: str, year: str) -> Path:
    """
    Verify that the output directory exists for the given workflow and year.
    - For years 2022 and 2023, both pre/post sub-year directories must exist
      before creating the parent directory.
    - Returns the valid Path if successful.
    - Raises FileNotFoundError if required directories are missing.
    """

    output_dir = OUTPUT_DIR / workflow / year

    if output_dir.exists():
        return output_dir

    # Years that require both pre and post subdirectories
    aux_map = {
        "2016": ["2016preVFP", "2016postVFP"],
        "2022": ["2022preEE", "2022postEE"],
        "2023": ["2023preBPix", "2023postBPix"],
    }

    if year in aux_map:
        pre_year, post_year = [OUTPUT_DIR / workflow / y for y in aux_map[year]]

        # Collect missing subdirectories
        missing = [str(p) for p in (pre_year, post_year) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required directories for year {year}: {', '.join(missing)}"
            )

        # Create the parent directory if both pre and post exist
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    # Case for sub-years or any other invalid year
    raise FileNotFoundError(f"Could not find outputs at {output_dir}")


def get_sample_name(filename: str, year: str) -> str:
    """return sample name from filename"""
    sample_name = Path(filename).stem
    if sample_name.rsplit("_")[-1].isdigit():
        sample_name = "_".join(sample_name.rsplit("_")[:-1])
    return sample_name.replace(f"{year}_", "")


def load_year_histograms(workflow: str, year: str):
    """load and merge histograms from pre/post years"""
    aux_map = {
        "2016": ["2016preVFP", "2016postVFP"],
        "2022": ["2022preEE", "2022postEE"],
        "2023": ["2023preBPix", "2023postBPix"],
    }
    pre_year, post_year = aux_map[year]
    base_path = OUTPUT_DIR / workflow
    pre_file = base_path / pre_year / f"{pre_year}_processed_histograms.coffea"
    post_file = base_path / post_year / f"{post_year}_processed_histograms.coffea"
    return accumulate([load(pre_file), load(post_file)])


def load_histogram_file(path: Path):
    return load(path) if path.exists() else None


def plot_variable(variable: str, group_by, histogram_config) -> bool:
    """decide whether to plot a given variable under group_by mode"""
    if isinstance(group_by, str) and group_by == "process":
        return True
    for hist_key, variables in histogram_config.layout.items():
        if variable in variables and group_by["name"] in variables:
            return group_by["name"] != variable
    return False


if __name__ == "__main__":
    args = parse_arguments()

    try:
        group_by = json.loads(args.group_by)
    except json.JSONDecodeError:
        group_by = args.group_by

    output_dir = check_output_dir(args.workflow, args.year)
    clear_output_directory(output_dir, "txt")
    setup_logger(output_dir)

    config_builder = WorkflowConfigBuilder(workflow=args.workflow)
    workflow_config = config_builder.build_workflow_config()
    histogram_config = workflow_config.histogram_config
    event_selection = workflow_config.event_selection
    categories = event_selection["categories"]
    processed_histograms = None

    if "data" not in workflow_config.datasets:
        args.blind = True

    if args.postprocess and (args.year not in ["2016", "2022", "2023"]):
        print_header(f"Running postprocess for {args.year}")
        logging.info(f"Reading outputs from: {output_dir}")

        output_files = [
            f
            for f in glob.glob(f"{output_dir}/*/*.coffea", recursive=True)
            if not Path(f).stem.startswith("cutflow")
            and not Path(f).stem.startswith("processed")
        ]

        grouped_outputs = defaultdict(list)
        for output_file in output_files:
            sample_name = get_sample_name(output_file, args.year)
            grouped_outputs[sample_name].append(output_file)

        process_samples_map = get_process_sample_map(grouped_outputs.keys(), args.year)

        if args.output_format == "parquet":
            if not args.skipmerging:
                merge_parquets_by_sample(output_dir, args.year, categories)

        for sample in grouped_outputs:
            save_histograms_by_sample(
                grouped_outputs,
                sample,
                args.year,
                output_dir,
                categories,
                workflow_config,
                args.nocutflow,
                args.output_format,
                args.skipmerging,
            )
            gc.collect()

        for process in process_samples_map:
            save_histograms_by_process(
                process,
                output_dir,
                process_samples_map,
                categories,
                args.nocutflow,
                args.output_format,
            )
            gc.collect()

        processed_histograms = load_processed_histograms(
            args.year,
            output_dir,
            process_samples_map,
        )

        for category in categories:
            logging.info(f"category: {category}")
            category_dir = output_dir / str(category)

            if not args.nocutflow:
                print_header(f"Cutflow")
                cutflow_df = pd.DataFrame()
                for process in process_samples_map:
                    cutflow_file = category_dir / f"cutflow_{category}_{process}.csv"
                    cutflow_df = pd.concat(
                        [cutflow_df, pd.read_csv(cutflow_file, index_col=[0])], axis=1
                    )

                columns_to_drop = []
                key_process_map = get_workflow_key_process_map(
                    workflow_config, args.year
                )
                if "signal" in workflow_config.datasets:
                    signal_keys = [k for k in workflow_config.datasets["signal"]]
                    signals = [key_process_map[key] for key in signal_keys]
                    columns_to_drop += signals

                if not args.blind:
                    columns_to_drop += ["Data"]

                total_background = cutflow_df.drop(columns=columns_to_drop).sum(axis=1)
                cutflow_df["Total Background"] = total_background

                cutflow_index = event_selection["categories"][category]
                cutflow_df = cutflow_df.loc[cutflow_index]

                if not args.blind:
                    to_process = ["Data", "Total Background"]
                else:
                    to_process = ["Total Background"]
                cutflow_df = cutflow_df[
                    to_process
                    + [
                        process
                        for process in cutflow_df.columns
                        if process not in to_process
                    ]
                ]
                logging.info(
                    f'{cutflow_df.applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else "")}\n'
                )
                cutflow_df.to_csv(f"{category_dir}/cutflow_{category}.csv")
                logging.info("\n")

            if args.workflow in ["ztoee", "ztomumu"]:
                print_header(f"Results")
                results_df = get_results_report(
                    processed_histograms,
                    workflow_config,
                    category,
                    columns_to_drop,
                    args.blind,
                )
                logging.info(
                    results_df.applymap(lambda x: f"{x:.5f}" if pd.notnull(x) else "")
                )
                logging.info("\n")
                results_df.to_csv(f"{category_dir}/results_{category}.csv")

                # save latex table
                latex_table = df_to_latex(results_df, args.blind)
                with open(category_dir / f"results_{category}.txt", "w") as f:
                    f.write(latex_table)

    if args.year in ["2016", "2022", "2023"]:
        if args.postprocess:
            print_header(f"Running postprocess for {args.year}")
            # load and accumulate processed histograms
            processed_histograms = load_year_histograms(args.workflow, args.year)
            save(
                processed_histograms,
                f"{output_dir}/{args.year}_processed_histograms.coffea",
            )
            identifier_map = {"2016": "VFP", "2022": "EE", "2023": "BPix"}
            identifier = identifier_map[args.year]
            for category in categories:
                logging.info(f"category: {category}")
                category_dir = OUTPUT_DIR / args.workflow / args.year / category
                if not category_dir.exists():
                    category_dir.mkdir(parents=True, exist_ok=True)
                if args.workflow in ["ztoee", "ztomumu"]:
                    # load and combine results tables
                    results_pre = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}pre{identifier}"
                        / category
                        / f"results_{category}.csv",
                        index_col=0,
                    )
                    results_post = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}post{identifier}"
                        / category
                        / f"results_{category}.csv",
                        index_col=0,
                    )
                    combined_results = combine_event_tables(
                        results_pre, results_post, args.blind
                    )

                    print_header(f"Results")
                    logging.info(
                        combined_results.applymap(
                            lambda x: f"{x:.5f}" if pd.notnull(x) else ""
                        )
                    )
                    logging.info("\n")

                    combined_results.to_csv(category_dir / f"results_{category}.csv")

                    # save latex table
                    latex_table = df_to_latex(combined_results, args.blind)
                    with open(category_dir / f"results_{category}.txt", "w") as f:
                        f.write(latex_table)

                # load and combine cutflow tables
                if not args.nocutflow:
                    print_header(f"Cutflow")
                    cutflow_pre = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}pre{identifier}"
                        / category
                        / f"cutflow_{category}.csv",
                        index_col=0,
                    )
                    cutflow_post = pd.read_csv(
                        OUTPUT_DIR
                        / args.workflow
                        / f"{args.year}post{identifier}"
                        / category
                        / f"cutflow_{category}.csv",
                        index_col=0,
                    )
                    combined_cutflow = combine_cutflows(cutflow_pre, cutflow_post)
                    combined_cutflow.to_csv(category_dir / f"cutflow_{category}.csv")
                    logging.info(
                        combined_cutflow.applymap(
                            lambda x: f"{x:.2f}" if pd.notnull(x) else ""
                        )
                    )
                    logging.info("\n")

                    # compute efficiencies
                    print_header(f"Efficiency")
                    eff_df = pd.DataFrame(index=combined_cutflow.index)
                    for col in combined_cutflow.columns:
                        eff_df[col] = (
                            combined_cutflow[col] / combined_cutflow[col].iloc[0] * 100
                        )
                    eff_df.to_csv(category_dir / f"eff_{category}.csv")
                    logging.info(eff_df)
                    logging.info("\n")

                    cutflow_eff = format_cutflow_with_efficiency(
                        combined_cutflow, eff_df
                    )
                    cutflow_eff.to_csv(category_dir / f"cutflow_eff_{category}.csv")

    if args.mva_inference:
        if not args.mva_config or not args.mva_model:
            raise ValueError(
                "--mva-config and --mva-model are required when using --mva-inference"
            )
        print_header(f"Running MVA inference for {args.year}")
        logging.info(f"Config: {args.mva_config}")
        logging.info(f"Model:  {args.mva_model}")

        # Resolve output dir: default to outputs/<workflow>_mvascores/<year>
        if args.mva_output:
            mva_output_dir = Path(args.mva_output)
        else:
            mva_output_dir = OUTPUT_DIR / f"{args.workflow}_mvascores" / args.year
        mva_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"MVA output: {mva_output_dir}")

        processor = MVAPostProcessor(
            bhive_config_path=args.mva_config,
            bhive_model_path=args.mva_model,
            apply_mass_window=args.mva_mass_window,
            era=args.year,
        )
        processor._load_model()

        import hist as hist_lib

        # Added 2026-08-14: histogram every MVA input feature (global + jagged
        # candidate features), not just the score outputs -- for post-scoring
        # data/MC-style validation plots of exactly what the model saw.
        # Range heuristics are name-pattern-based since features span very different
        # scales (pt in [0,300], eta in [-6,6], flags in {0,1}, ...); anything outside
        # its assigned range lands in flow (over/underflow), it is not silently dropped.
        def _feature_axis(col):
            name = col.lower()
            if name.startswith("jet_is_") or name.startswith("lepton_is_") or "_flag" in name \
                    or name in ("in_mass_window", "lepton_isPFcand".lower()):
                return hist_lib.axis.Regular(2, -0.5, 1.5, name=col, label=col, flow=True)
            if "eta" in name:
                return hist_lib.axis.Regular(50, -6, 6, name=col, label=col, flow=True)
            if "phi" in name:
                return hist_lib.axis.Regular(50, -3.2, 3.2, name=col, label=col, flow=True)
            if name in ("m4l",) or name.endswith("_mass") or name == "mass":
                return hist_lib.axis.Regular(50, 60, 200, name=col, label=col, flow=True)
            if "deltar" in name or "dphi" in name:
                return hist_lib.axis.Regular(50, 0, 6, name=col, label=col, flow=True)
            if name in ("n_cjet", "nsv", "jet_multiplicity", "n_lepton") or "multiplicity" in name:
                return hist_lib.axis.Regular(15, 0, 15, name=col, label=col, flow=True)
            if "charge" in name:
                return hist_lib.axis.Regular(5, -2.5, 2.5, name=col, label=col, flow=True)
            if "iso" in name or "sip3d" in name or "mva" in name:
                return hist_lib.axis.Regular(50, 0, 5, name=col, label=col, flow=True)
            if "pt" in name or "ht" in name:
                return hist_lib.axis.Regular(50, 0, 300, name=col, label=col, flow=True)
            # generic fallback for anything unrecognized (still gets a histogram, just a
            # wide default range with flow catching outliers)
            return hist_lib.axis.Regular(50, -10, 10, name=col, label=col, flow=True)

        def _build_mva_histograms(class_names, processor):
            """Build one histogram per MVA score column (signal + per-class) PLUS one
            per MVA input feature: global_features/lt_candidates (scalar, one value per
            event) and cpf_candidates/vtx_features/npf_candidates (jagged -- one value
            per jet/Z/lepton, flattened across candidates at fill time)."""
            hists = {}
            score_cols = ["mva_signal_score"] + [f"mva_score_{c}" for c in class_names]
            scalar_cols = list(dict.fromkeys(
                score_cols
                + processor.config.get("global_features", [])
                + processor.config.get("lt_candidates", [])
            ))
            jagged_cols = list(dict.fromkeys(
                processor.config.get("cpf_candidates", [])
                + processor.config.get("vtx_features", [])
                + processor.config.get("npf_candidates", [])
            ))
            for col in scalar_cols:
                axis = (hist_lib.axis.Regular(50, 0, 1, name=col, label=col) if col in score_cols
                        else _feature_axis(col))
                hists[col] = hist_lib.Hist(
                    axis,
                    hist_lib.axis.StrCategory([], name="process", growth=True),
                    hist_lib.axis.StrCategory([], name="variation", growth=True),
                    hist_lib.storage.Weight(),
                )
            for col in jagged_cols:
                hists[col] = hist_lib.Hist(
                    _feature_axis(col),
                    hist_lib.axis.StrCategory([], name="process", growth=True),
                    hist_lib.axis.StrCategory([], name="variation", growth=True),
                    hist_lib.storage.Weight(),
                )
            return hists, set(jagged_cols)

        def _fill_mva_histograms(hists, jagged_cols, df, process_name, weight_col="weight_nominal"):
            """Fill score/feature histograms from a scored DataFrame. Scalar columns fill
            directly; jagged (list-valued) columns are exploded so every jet/Z/lepton
            entry gets its own fill, with the event weight repeated per entry (a jet-level
            histogram, not an event-level one -- an event with 3 jets contributes 3
            entries, each carrying that event's full weight, same convention coffea's
            fill_histograms uses for jagged axes elsewhere in this codebase)."""
            # Only fill events with valid scores (apply_mass_window sets -1 outside window)
            mask = df["mva_signal_score"] >= 0
            df_valid = df[mask]
            if df_valid.empty:
                return
            event_weights = (
                df_valid[weight_col].fillna(1.0)
                if weight_col in df_valid.columns
                else pd.Series(np.ones(len(df_valid)), index=df_valid.index)
            )
            for col, h in hists.items():
                if col not in df_valid.columns:
                    continue
                if col in jagged_cols:
                    exploded = df_valid[col].explode()
                    exploded = exploded.dropna()
                    if exploded.empty:
                        continue
                    w = event_weights.loc[exploded.index].values
                    h.fill(**{col: exploded.astype(float).values}, process=process_name,
                           variation="nominal", weight=w)
                else:
                    h.fill(**{col: df_valid[col].values}, process=process_name,
                           variation="nominal", weight=event_weights.values)

        # Run on merged parquets (parquets_<sample>/ dirs created by merge step)
        merged_dirs = sorted(output_dir.glob("parquets_*"))
        if not merged_dirs:
            logging.warning(
                "No merged parquet dirs found (parquets_*/). "
                "Run with --postprocess first, or ensure --output_format parquet was used."
            )
        for merged_dir in merged_dirs:
            sample_name = merged_dir.name.replace("parquets_", "")
            sample_hists, jagged_cols = _build_mva_histograms(processor.class_names, processor)

            for pq_file in sorted(merged_dir.rglob("*.parquet")):
                df = pd.read_parquet(pq_file)
                if df.empty:
                    continue
                features = processor.prepare_features(df)
                result = processor.predict(features)
                for i, cls in enumerate(processor.class_names):
                    df[f"mva_score_{cls}"] = result["scores"][:, i]
                df["mva_signal_score"] = result["signal_score"]
                df["mva_class_prediction"] = result["class_prediction"]

                # Save scored parquet
                rel = pq_file.relative_to(merged_dir)
                out_file = mva_output_dir / sample_name / rel
                out_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_file, index=False)

                # Accumulate into histograms
                _fill_mva_histograms(sample_hists, jagged_cols, df, sample_name)
                logging.info(
                    f"  {sample_name}/{rel}: {len(df)} events, "
                    f"mean signal score={df['mva_signal_score'].mean():.4f}"
                )

            # Save per-sample MVA histogram coffea file
            hist_file = mva_output_dir / f"{sample_name}_mva_scores.coffea"
            save(sample_hists, hist_file)
            logging.info(f"  Saved histograms: {hist_file}")

        logging.info("MVA inference complete")

    if args.plot:
        subprocess.run("python3 analysis/postprocess/build_color_map.py", shell=True)
        if not args.postprocess:
            postprocess_file = output_dir / f"{args.year}_processed_histograms.coffea"
            processed_histograms = load_histogram_file(postprocess_file)
            if processed_histograms is None:
                raise ValueError(
                    f"Postprocess file not found. Please run:\n"
                    f"  'python3 run_postprocess.py -w {args.workflow} -y {args.year} --postprocess'"
                )

        print_header(f"Running plotter for {args.year}")
        plotter = CoffeaPlotter(
            workflow=args.workflow,
            processed_histograms=processed_histograms,
            year=args.year,
            output_dir=output_dir,
            group_by=group_by,
            pass_axis=args.pass_axis,
        )
        for category in categories:
            logging.info(
                f"Plotting histograms by '{group_by if group_by == 'process' else group_by['name']}' for category '{category}'"
            )
            for variable in workflow_config.histogram_config.variables:
                if args.pass_axis:
                    if variable == args.pass_axis:
                        continue
                    if histogram_config.layout == "individual":
                        print("There's only individual axes!")
                        break
                    proceed = False
                    for key, variables in histogram_config.layout.items():
                        if (variable in variables) and (args.pass_axis in variables):
                            proceed = True
                            break
                    if not proceed:
                        continue
                if plot_variable(variable, group_by, workflow_config.histogram_config):
                    logging.info(variable)
                    plotter.plot_histograms(
                        variable=variable,
                        category=category,
                        yratio_limits=args.yratio_limits,
                        log=args.log,
                        extension=args.extension,
                        blind=args.blind,
                    )
            if args.workflow.startswith("zplusl_"):
                plotter.plot_fake_rate(category)
            subprocess.run(
                f"tar -zcvf {output_dir}/{category}/{args.workflow}_{args.year}_plots.tar.gz {output_dir}/{category}/*.{args.extension}",
                shell=True,
            )
