import os
import yaml
import glob
import hist
import pickle
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from collections import defaultdict
from coffea.util import load, save
from coffea.processor import accumulate
from analysis.filesets.utils import get_dataset_config


def setup_logger(output_dir):
    """Set up the logger to log to a file in the specified output directory."""
    output_file_path = os.path.join(output_dir, "output.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(output_file_path), logging.StreamHandler()],
    )


def open_output(fname: str) -> dict:
    with open(fname, "rb") as f:
        output = pickle.load(f)
    return output


def print_header(text):
    logging.info("-" * 90)
    logging.info(text)
    logging.info("-" * 90)


def divide_by_binwidth(histogram):
    bin_width = histogram.axes.edges[0][1:] - histogram.axes.edges[0][:-1]
    return histogram / bin_width


def clear_output_directory(output_dir, ext):
    """Delete all result files in the output directory with extension 'ext'"""
    files = glob.glob(os.path.join(output_dir, f"*.{ext}"))
    for file in files:
        os.remove(file)


def combine_event_tables(df1, df2, blind):
    assert all(df1.index == df2.index), "DataFrame index don't match!"
    combined = pd.DataFrame(index=df1.index)
    combined["events"] = df1["events"] + df2["events"]
    combined["stat err"] = np.sqrt(df1["stat err"] ** 2 + df2["stat err"] ** 2)
    combined["syst err up"] = np.sqrt(df1["syst err up"] ** 2 + df2["syst err up"] ** 2)
    combined["syst err down"] = np.sqrt(
        df1["syst err up"] ** 2 + df2["syst err down"] ** 2
    )
    if not blind:
        data = combined.loc["Data", "events"]
        total_bkg = combined.loc["Total background", "events"]
        combined.loc[
            "Data/Total background",
            ["events", "stat err", "syst err up", "syst err down"],
        ] = [
            data / total_bkg,
            np.nan,
            np.nan,
            np.nan,
        ]
    return combined


def combine_cutflows(df1, df2):
    if not df1.index.equals(df2.index):
        raise ValueError("Los índices (etiquetas de cortes) no coinciden.")
    combined = df1.add(df2, fill_value=0)
    return combined


def format_cutflow_with_efficiency(events_df, eff_df):
    combined = pd.DataFrame(index=events_df.index, columns=events_df.columns)
    for col in events_df.columns:
        for idx in events_df.index:
            val = events_df.loc[idx, col]
            eff = eff_df.loc[idx, col]
            combined.loc[idx, col] = f"{val} ({eff:.2f}%)"
    return combined


def get_variations_keys(processed_histograms):
    variations = {}
    for process, histogram_dict in processed_histograms.items():
        if process == "Data":
            continue
        for feature in histogram_dict:
            helper_histogram = histogram_dict[feature]
            variations = [
                var for var in helper_histogram.axes["variation"] if var != "nominal"
            ]
            break
        break
    variations = list(
        set([var.replace("Up", "").replace("Down", "") for var in variations])
    )
    return variations


def find_kin_and_axis(processed_histograms, name="multiplicity"):
    for process, histogram_dict in processed_histograms.items():
        if process == "Data":
            continue
        for kin, hist in histogram_dict.items():
            for axis_name in hist.axes.name:
                if axis_name != "variation" and name in axis_name:
                    return kin, axis_name
    raise ValueError(f"No histogram with a '{name}' axis found.")


def concat_tables_schema_safe(tables):
    """pa.concat_tables(tables, promote=True) unions schemas (a column missing from some
    tables gets filled null) but does NOT reconcile a column present in multiple tables
    with genuinely different types -- confirmed real 2026-08-14: `nSV` is stored as
    int64 in some chunk files and double (float) in others, a type drift (not just a
    missing-column drift) that raises `ArrowInvalid: Unable to merge: Field nSV has
    incompatible types`. Cast any such numeric-vs-numeric mismatch to float64 (lossless
    for realistic count-like values) before the union concat. A non-numeric type clash
    is a more serious, not-safe-to-silently-resolve problem -- left to raise rather than
    guessed at. Shared by merge_parquets() (within-partition) and
    postprocessor.py::fill_histograms_from_parquets() (across-partition) -- both hit
    this independently, keep them using the same helper rather than drifting.
    """
    import pyarrow as pa
    import pyarrow.types as pat

    col_types = {}
    for t in tables:
        for name, typ in zip(t.schema.names, t.schema.types):
            col_types.setdefault(name, set()).add(typ)
    to_cast = {
        name: pa.float64()
        for name, types in col_types.items()
        if len(types) > 1 and all(pat.is_integer(ty) or pat.is_floating(ty) for ty in types)
    }
    if to_cast:
        cast_tables = []
        for t in tables:
            for name, target_type in to_cast.items():
                if name in t.schema.names:
                    idx = t.schema.get_field_index(name)
                    t = t.set_column(idx, name, t.column(name).cast(target_type))
            cast_tables.append(t)
        tables = cast_tables
    return pa.concat_tables(tables, promote=True)


def merge_parquets(inpath, outpath, sample_name):
    """Merge all per-chunk parquet files for one dataset partition into one file.

    Reads each file with pyarrow and concatenates via concat_tables_schema_safe() (schema
    union + numeric type promotion) rather than `dask.dataframe.read_parquet(...).compute()`
    over a glob -- the latter requires every file to share an IDENTICAL schema and raises
    a bare `KeyError` naming the missing columns otherwise. Confirmed 2026-08-14 this isn't
    hypothetical: several non-analysis datasets (data, DY -- not in create_datacards.py's
    PROCESS_MAPPING, so never read downstream) have files from before/after the workflow
    yaml added `nSV`/`jet_btagUParTAK4*`/`jet_h_dphi_inclusive` columns, mixed within the
    same dataset partition and never reprocessed to match. The all-physics-process
    (Signal/qqZZ/ggZZ/Other_Higgs) directories were checked and are schema-clean, but this
    generic loop processes every dataset it finds, physics-relevant or not, so a mismatch
    anywhere used to abort the entire --postprocess run.
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{inpath}/*.parquet"))
    if not files:
        # Legitimately empty: e.g. all-hadronic MC (W->4Q, tt->4Q) produces zero rows for
        # a leptonic selection, and parquet_writer.py::dump_pa_table skips writing empty
        # tables by design -- zero files here is a correct, expected outcome, not a
        # failure. Skip writing a merged file rather than crash (pa.concat_tables raises
        # ArrowInvalid on an empty table list).
        return
    tables = [pq.read_table(f) for f in files]
    table = concat_tables_schema_safe(tables)
    df = table.to_pandas()
    outpath = Path(outpath)
    if not outpath.exists():
        outpath.mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{outpath}/{sample_name}.parquet", engine="pyarrow", index=False)


def accumulate_metadata(grouped_outputs, sample):
    grouped_metadata = {}
    for fname in grouped_outputs[sample]:
        output = load(fname)
        if not output:
            continue
        for meta_key, meta_val in output["metadata"].items():
            grouped_metadata.setdefault(meta_key, []).append(meta_val)
    return {k: accumulate(v) for k, v in grouped_metadata.items()}


def accumulate_histograms(grouped_outputs, sample):
    grouped_histograms = []
    for fname in grouped_outputs[sample]:
        output = load(fname)
        if output:
            grouped_histograms.append(output["histograms"])
    return accumulate(grouped_histograms)


def get_lumi_weight(year, sample, metadata):
    lumi_file = Path.cwd() / "analysis" / "postprocess" / "luminosity.yaml"
    with open(lumi_file, "r") as f:
        luminosities = yaml.safe_load(f)

    dataset_config = get_dataset_config(year)
    xsec = dataset_config[sample]["xsec"]
    sumw = metadata["sumw"]
    weight = 1
    if dataset_config[sample]["era"] in ["mc", "signal"]:
        weight = (luminosities[year] * xsec) / sumw

    logging.info(f"luminosity [1/pb]: {luminosities[year]}")
    logging.info(f"xsec [pb]: {xsec}")
    logging.info(f"sumw: {sumw}")
    logging.info(f"weight: {weight}")

    return weight


def save_cutflows(metadata, categories, sample, weight, output_dir):
    for category in categories:
        if category not in metadata:
            logging.info(f"zero events for category {category}")
            continue

        logging.info(f"saving {sample} cutflow for category {category}")

        category_dir = Path(output_dir) / category
        if not category_dir.exists():
            category_dir.mkdir(parents=True, exist_ok=True)

        scaled_cutflow = {
            cut: nevents * weight
            for cut, nevents in metadata[category]["cutflow"].items()
        }
        processed_cutflow = {sample: scaled_cutflow}
        cutflow_file = category_dir / f"cutflow_{category}_{sample}.coffea"
        save(processed_cutflow, cutflow_file)


def get_process_dict(output_dir, year, categories):
    dataset_config = get_dataset_config(year)

    process_dict = defaultdict(list)

    folders = glob.glob(str(output_dir / "*"))
    for folder in folders:
        folder_path = Path(folder)
        name = folder_path.name
        for category in categories:
            if name == category:
                continue
            if category not in process_dict:
                process_dict[category] = {}

            folder_content = glob.glob(f"{folder_path}/*")

            # case where there are multiple partitions and the folder includes only .coffea files
            if any(".coffea" in f for f in folder_content) and not any(
                category in f for f in folder_content
            ):
                if name not in process_dict[category]:
                    process_dict[category][name] = []
            # case where there is only one partition and the main folder includes both the .coffea file and the category folder
            elif any(".coffea" in f for f in folder_content) and any(
                category in f for f in folder_content
            ):
                for f in folder_content:
                    if category in f:
                        if name not in process_dict[category]:
                            process_dict[category][name] = [str(folder_path)]
            # case where there's only the category folder
            else:
                actual_name = name.rsplit("_", 1)[0]
                if actual_name in process_dict[category]:
                    process_dict[category][actual_name].append(str(folder_path))

    return dict(process_dict)


def merge_parquets_by_sample(output_dir, year, categories):
    """Merge parquet files from subfolders into a single output path per sample and category."""
    print_header("Merging parquet outputs by sample")
    process_dict = get_process_dict(output_dir, year, categories)
    for category in categories:
        for name, subfolders in process_dict[category].items():
            outpath = f"{output_dir}/parquets_{name}/{category}"
            if len(subfolders) != 0:
                logging.info(
                    f"Merging {name} outputs into {len(subfolders)} "
                    f"{'partition' if len(subfolders) == 1 else 'partitions'}"
                )
                for i, subfolder in enumerate(subfolders, start=1):
                    inpath = f"{subfolder}/{category}"
                    merge_parquets(inpath, outpath, f"{name}_{i}")


def accumulate_and_save_cutflows(process, process_samples_map, output_dir, categories):
    """Accumulate cutflows from all samples in a process and save them per category."""
    for category in categories:
        category_dir = Path(f"{output_dir}/{category}")
        df_total = pd.DataFrame()

        for sample in process_samples_map[process]:
            cutflow_file = category_dir / f"cutflow_{category}_{sample}.coffea"
            if cutflow_file.exists():
                df_sample = pd.DataFrame(load(cutflow_file).values())
                df_total = pd.concat([df_total, df_sample])

        if not df_total.empty:
            cutflow_df = pd.DataFrame(df_total.sum())
            cutflow_df.columns = [process]
            cutflow_csv = category_dir / f"cutflow_{category}_{process}.csv"
            logging.info(
                f"Saving cutflow for process {process} and category {category}"
            )
            cutflow_df.to_csv(cutflow_csv)


def load_processed_histograms(
    year: str,
    output_dir: str,
    process_samples_map: dict,
):
    processed_histograms = {}
    for process in process_samples_map:
        processed_histograms.update(load(f"{output_dir}/{process}.coffea"))
    save(processed_histograms, f"{output_dir}/{year}_processed_histograms.coffea")
    return processed_histograms


def get_results_report(
    processed_histograms, workflow_config, category, columns_to_drop, blind
):
    kin, aux_var = find_kin_and_axis(processed_histograms)
    nominal = {}
    variations = {}
    mcstat_err = {}
    bin_error_up = {}
    bin_error_down = {}
    for process in processed_histograms:
        aux_hist = processed_histograms[process][kin]
        nominal_selector = {"variation": "nominal"}
        if "category" in aux_hist.axes.name:
            nominal_selector["category"] = category
        nominal_hist = aux_hist[nominal_selector].project(aux_var)
        nominal[process] = nominal_hist

        mcstat_err[process] = {}
        bin_error_up[process] = {}
        bin_error_down[process] = {}
        mcstat_err2 = nominal_hist.variances()
        mcstat_err[process] = np.sum(np.sqrt(mcstat_err2))
        err2_up = mcstat_err2
        err2_down = mcstat_err2

        if process == "Data":
            continue

        for variation in get_variations_keys(processed_histograms):
            if f"{variation}Up" not in aux_hist.axes["variation"]:
                continue
            selectorup = {"variation": f"{variation}Up"}
            selectordown = {"variation": f"{variation}Down"}
            if "category" in aux_hist.axes.name:
                selectorup["category"] = category
                selectordown["category"] = category
            var_up = aux_hist[selectorup].project(aux_var).values()
            var_down = aux_hist[selectordown].project(aux_var).values()
            # Compute the uncertainties corresponding to the up/down variations
            err_up = var_up - nominal_hist.values()
            err_down = var_down - nominal_hist.values()
            # Compute the flags to check which of the two variations (up and down) are pushing the nominal value up and down
            up_is_up = err_up > 0
            down_is_down = err_down < 0
            # Compute the flag to check if the uncertainty is one-sided, i.e. when both variations are up or down
            is_onesided = up_is_up ^ down_is_down
            # Sum in quadrature of the systematic uncertainties taking into account if the uncertainty is one- or double-sided
            err2_up_twosided = np.where(up_is_up, err_up**2, err_down**2)
            err2_down_twosided = np.where(up_is_up, err_down**2, err_up**2)
            err2_max = np.maximum(err2_up_twosided, err2_down_twosided)
            err2_up_onesided = np.where(is_onesided & up_is_up, err2_max, 0)
            err2_down_onesided = np.where(is_onesided & down_is_down, err2_max, 0)
            err2_up_combined = np.where(is_onesided, err2_up_onesided, err2_up_twosided)
            err2_down_combined = np.where(
                is_onesided, err2_down_onesided, err2_down_twosided
            )
            # Sum in quadrature of the systematic uncertainty corresponding to a MC sample
            err2_up += err2_up_combined
            err2_down += err2_down_combined

        bin_error_up[process] = np.sum(np.sqrt(err2_up))
        bin_error_down[process] = np.sum(np.sqrt(err2_down))

    mcs = []
    results = {}
    for process in nominal:
        results[process] = {}
        results[process]["events"] = np.sum(nominal[process].values())
        if process == "Data":
            results[process]["stat err"] = np.sqrt(np.sum(nominal[process].values()))
        else:
            if process not in columns_to_drop:
                mcs.append(process)
            results[process]["stat err"] = mcstat_err[process]
            results[process]["syst err up"] = bin_error_up[process]
            results[process]["syst err down"] = bin_error_down[process]
    df = pd.DataFrame(results)
    df["Total background"] = df.loc[["events"], mcs].sum(axis=1)
    df.loc["stat err", "Total background"] = np.sqrt(
        np.sum(df.loc["stat err", mcs] ** 2)
    )
    df.loc["syst err up", "Total background"] = np.sqrt(
        np.sum(df.loc["syst err up", mcs] ** 2)
    )
    df.loc["syst err down", "Total background"] = np.sqrt(
        np.sum(df.loc["syst err down", mcs] ** 2)
    )
    df = df.T
    if not blind:
        df.loc["Data/Total background"] = (
            df.loc["Data", ["events"]] / df.loc["Total background", ["events"]]
        )
    return df


def df_to_latex(df, blind, table_title="Events"):
    output = rf"""\begin{{table}}[h!]
\centering
\begin{{tabular}}{{@{{}} l c @{{}}}}
\hline
 & \textbf{{{table_title}}} \\
\hline
"""

    total_background_value = None
    data_value = None

    for label, row in df.iterrows():
        events = row["events"]
        stat_err = row["stat err"] if pd.notna(row["stat err"]) else None
        syst_err_up = row["syst err up"] if pd.notna(row["syst err up"]) else None
        syst_err_down = row["syst err down"] if pd.notna(row["syst err down"]) else None

        events_f = f"{float(events):.2f}"
        stat_err_f = f"{float(stat_err):.2f}" if stat_err is not None else "nan"
        syst_err_up_f = (
            f"{float(syst_err_up):.2f}" if syst_err_up is not None else "nan"
        )
        syst_err_down_f = (
            f"{float(syst_err_down):.2f}" if syst_err_down is not None else "nan"
        )

        if label not in ["Data", "Total background", "Data/Total background"]:
            output += f"{label} & ${events_f} \\pm {stat_err_f} \\, (\\text{{stat}}) \\pm {syst_err_up_f} \\, (\\text{{syst up}})$ \\pm {syst_err_down_f} \\, (\\text{{syst down}})$\\\\\n"

    output += r"\hline" + "\n"

    # Total background
    bg = df.loc["Total background"]
    total_background_value = bg["events"]
    stat_err_bg = bg["stat err"]
    syst_err_up_bg = bg["syst err up"]
    syst_err_down_bg = bg["syst err down"]

    output += f"Total Background & ${float(total_background_value):.2f} \\pm {float(stat_err_bg):.2f} \\, (\\text{{stat}}) \\pm {float(syst_err_up_bg):.2f} \\, (\\text{{syst up}})$ \\pm {float(syst_err_down_bg):.2f} \\, (\\text{{syst down}})$ \\\\ \n"

    # Data
    if not blind:
        data_value = df.loc["Data"]["events"]
        output += f"Data & ${float(data_value):.0f}$ \\\\ \n"

        output += r"\hline" + "\n"

        # Data/Total Background
        if total_background_value and data_value:
            ratio = data_value / total_background_value
            output += f"Data/Total Background & ${ratio:.2f}$ \\\\ \n"

    output += r"""\hline
\end{tabular}
\end{table}"""
    return output
