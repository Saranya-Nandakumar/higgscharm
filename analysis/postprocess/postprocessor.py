import copy
import yaml
import glob
import logging
import numpy as np
import pandas as pd
import awkward as ak
import dask.dataframe as dd
from pathlib import Path
from coffea.util import load, save
from coffea.processor import accumulate
from analysis.filesets.utils import get_dataset_config
from analysis.histograms import HistBuilder, fill_histogram
from analysis.postprocess.utils import (
    print_header,
    get_variations_keys,
    find_kin_and_axis,
    get_lumi_weight,
    accumulate_histograms,
    accumulate_metadata,
    get_process_dict,
    save_cutflows,
    accumulate_and_save_cutflows,
    concat_tables_schema_safe,
)


def fill_histograms_from_parquets(
    year, sample, categories, workflow_config, output_dir
):
    """Build and fill histograms from parquet files for a given sample"""
    dataset_config = get_dataset_config(year)
    histogram_config = workflow_config.histogram_config
    variables = list(histogram_config.axes.keys())
    histograms = HistBuilder(workflow_config).build_individual_histogram()
    process_dict = get_process_dict(output_dir, year, categories)
    sample_histograms = copy.deepcopy(histograms)

    for category in categories:
        logging.info(f"Filling {sample} histograms")

        # merge sample parquets
        sample_df_file = output_dir / f"{sample}.parquet"
        if sample_df_file.exists():
            sample_df = pd.read_parquet(sample_df_file)
        else:
            sample_parquets = glob.glob(
                f"{output_dir}/parquets_{sample}/{category}/*.parquet"
            )
            if not sample_parquets:
                logging.warning(f"No parquet files found for sample {sample}, category {category} — skipping")
                continue
            # pyarrow schema-safe read instead of dd.read_parquet(...).compute() -- same
            # fix as analysis/postprocess/utils.py::merge_parquets (2026-08-14). The
            # per-partition merge there unions schemas WITHIN each partition's own chunk
            # files, but different partitions of the same dataset can still end up with
            # different merged schemas from each other (e.g. EGamma0v1D's partition 1 vs.
            # partition 4, if the workflow yaml's feature list changed between when each
            # was condor-run) -- this second, across-partition aggregation needs the same
            # treatment, confirmed necessary by a real crash here (and a second real crash,
            # a genuine type mismatch on `nSV`, int64 vs double -- see
            # concat_tables_schema_safe's docstring).
            import pyarrow.parquet as pq
            _tables = [pq.read_table(f) for f in sample_parquets]
            sample_df = concat_tables_schema_safe(_tables).to_pandas()
            sample_df = sample_df.replace({None: np.nan})
            sample_df.to_parquet(
                f"{output_dir}/{sample}.parquet", engine="pyarrow", index=False
            )

        # build variables map
        variables_map = {}
        variables_mask_map = {}
        for variable in variables:
            if variable not in sample_df.columns:
                logging.info(f"Could not found variable {variable} for sample {sample}")
                continue
            variable_array = sample_df[variable].values
            if variable_array.dtype.type is np.object_:
                first_valid = next((x for x in variable_array if x is not None), None)
                if isinstance(first_valid, np.ndarray):
                    # Jagged column (e.g. cjets_pt): convert to ak.Array for flatten support
                    variable_array = ak.Array(
                        [x.tolist() if x is not None else [] for x in variable_array]
                    )
                else:
                    # Scalar object column (bool-like)
                    variable_array = np.array(
                        [x if x is not None else np.nan for x in variable_array],
                        dtype=bool,
                    )
            variables_map[variable] = variable_array

        # compute nominal weights
        partial_weights = list(
            set(
                [
                    w.replace("Up", "").replace("Down", "")
                    for w in sample_df.columns
                    if w.startswith("weight") and "nominal" not in w
                ]
            )
        )
        nominal_weights = sample_df[partial_weights].prod(axis=1).values
        if len(partial_weights) > 0:
            logging.info(
                f"weights: {[w.replace('weight_','') for w in partial_weights]}"
            )

        # fill nominal histograms
        sample_histograms = copy.deepcopy(histograms)
        # Force individual fill (stacked multi-D histograms are too large for postprocess)
        original_layout = histogram_config.layout
        histogram_config.layout = "individual"
        fill_args = {
            "histograms": sample_histograms,
            "histogram_config": histogram_config,
            "variables_map": variables_map,
            "category": category,
            "flow": True,
            "weights": nominal_weights,
            "variation": "nominal",
        }
        fill_histogram(**fill_args)

        # fill syst variation histograms
        # Skip gracefully (nominal histogram above is unaffected) rather than crash for
        # samples absent from the fileset registry -- same class of gap as
        # filesets/utils.py::get_process_sample_map (private datasets condor-run directly,
        # e.g. 2023postBPixHB, never added to <era>_nanov<n>.yaml). Added 2026-08-14.
        if sample in dataset_config and dataset_config[sample]["era"] in ["mc", "signal"]:
            for syst in partial_weights:
                for variation in ["Up", "Down"]:
                    syst_name = f"{syst}{variation}"
                    if syst_name in sample_df.columns:
                        fill_args["weights"] = sample_df[syst_name].values
                        fill_args["variation"] = syst_name.replace("weight_", "")
                        fill_histogram(**fill_args)

        histogram_config.layout = original_layout

    return sample_histograms


def save_histograms_by_sample(
    grouped_outputs,
    sample,
    year,
    output_dir,
    categories,
    workflow_config,
    nocutflow,
    output_format,
    skipmerging,
):
    """Accumulate, scale, and save histograms for a single sample"""
    print_header(f"Processing {sample} outputs")

    # Skip entirely (not a degraded/partial fill) if this sample is absent from the
    # fileset registry -- get_lumi_weight() needs a real xsec to scale by, and there is
    # no safe default (weight=1 would silently mis-scale, not just omit, the saved
    # histogram/cutflow). Same class of gap as get_process_sample_map/
    # fill_histograms_from_parquets (private datasets condor-run directly, e.g.
    # 2023postBPixHB, never added to <era>_nanov<n>.yaml). Added 2026-08-14.
    if sample not in get_dataset_config(year):
        logging.warning(
            f"  '{sample}' not in fileset registry for {year}, skipping "
            f"save_histograms_by_sample entirely (no xsec to scale by)"
        )
        return

    # get histograms
    if output_format == "coffea":
        histograms = accumulate_histograms(grouped_outputs, sample)
    elif output_format == "parquet":
        histograms = fill_histograms_from_parquets(
            year, sample, categories, workflow_config, output_dir
        )
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    # accumulate metadata and compute lumi weight
    metadata = accumulate_metadata(grouped_outputs, sample)
    weight = get_lumi_weight(year, sample, metadata)

    # scale histograms by lumi-xsec weight
    scaled_histograms = {
        variable: histograms[variable] * weight for variable in histograms
    }
    save(scaled_histograms, Path(output_dir) / f"{sample}.coffea")

    # save cutflows if requested
    if not nocutflow:
        save_cutflows(metadata, categories, sample, weight, output_dir)


def save_histograms_by_process(
    process: str,
    output_dir: str,
    process_samples_map: dict,
    categories: list,
    nocutflow: bool,
    output_format: str,
):
    """Accumulate and save all outputs for a given physics process"""
    print_header(f"Processing {process} outputs")

    # accumulate and save all histograms into a single dictionary
    coffea_files = []
    for sample in process_samples_map[process]:
        coffea_files += glob.glob(f"{output_dir}/{sample}*.coffea", recursive=True)

    logging.info(f"Accumulating histograms for process {process}")
    hist_to_accumulate = [load(f) for f in coffea_files]
    output_histograms = {process: accumulate(hist_to_accumulate)}
    save(output_histograms, Path(output_dir) / f"{process}.coffea")

    # accumulate and save all parquets into a single parquet file
    if output_format == "parquet":
        logging.info(f"Accumulating parquets for process {process}")
        parquet_files = []
        for sample in process_samples_map[process]:
            parquet_files += glob.glob(
                f"{output_dir}/{sample}*.parquet", recursive=True
            )
        if not parquet_files:
            # Legitimately possible: every sample in this process group can have zero
            # parquet output for a given era (e.g. 2022postEE's `dy_nlo` fileset entries
            # point to a stale DAS tag -- pre-existing, documented, unrelated to this
            # pipeline -- so DY+Jets has no real events at all for that one era). Skip
            # rather than crash `pd.concat([])` (`ValueError: No objects to concatenate`).
            logging.warning(f"No parquet files found for process {process} -- skipping "
                             f"process-level parquet merge")
        else:
            process_df = pd.concat(
                [pd.read_parquet(f) for f in parquet_files], ignore_index=True
            )
            process_df.to_parquet(Path(output_dir) / f"{process}.parquet")

    # accumulate and save cutflows if requested
    if not nocutflow:
        accumulate_and_save_cutflows(
            process, process_samples_map, output_dir, categories
        )