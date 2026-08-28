#!/usr/bin/env python3
"""
3-ratio-decomposition variant of create_datacards_ctag2d.py -- ctag2d SF scheme +
joint (r1,r2,r3) discriminant instead of the plain mva_score_Signal score.

r1 = Ps/(Ps+PqqZZ), r2 = Ps/(Ps+PggZZ), r3 = Ps/(Ps+POH), binned jointly via the
"nested"/conditional quantile tree (see second-brain/Notes/SM-3ratio-kappa-decomposition-note.md
Bug 2 for why per-axis independent quantiles are wrong for these strongly-correlated ratios).
Ported from /eos/home-s/snandaku/b-hive_ttcc/combine/create_datacards_3ratio.py (old ctag_light
scheme, non-ctag2d scored parquets) -- this copy swaps in the ctag2d systematics/sumw/process-
mapping wiring from create_datacards_ctag2d.py so the comparison against the ctag2d plain-score
production numbers is apples-to-apples (same scored parquets, same systematics, same mass
windows, only the discriminant differs).

No Z+X support (deliberately, matching the current no-ZX-only production scope, 2026-08-17).
Writes BOTH a full-systematics and a stat-only datacard per run (identical rates, stat-only
just omits every lnN/shape row) so one data load covers both combine invocations.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot
from collections import defaultdict

ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]

LUMINOSITY_PB = {
    "2022preEE":    7980.4,
    "2022postEE":  26671.7,
    "2023preBPix": 17794.0,
    "2023postBPix": 9450.0,
}
TOTAL_LUMI_PB = sum(LUMINOSITY_PB.values())

PROCESS_XS_PB = {
    "Signal":      2.6381e-05,
    "ggZZ":        3 * 0.003 + 3 * 0.006,
    "qqZZ":        1.39,
    "Other_Higgs": (
        0.01434 + 0.00112 + 0.000775 + 0.000156 + 0.000244
        + 0.00312 + 0.000144 + 0.000173
    ),
}
EXPECTED_YIELDS = {p: xs * TOTAL_LUMI_PB for p, xs in PROCESS_XS_PB.items()}

CLASS_NAMES = ["qqZZ", "ggZZ", "Signal", "Other_Higgs"]

PROCESS_MAPPING = {
    "SomeSMSignal":                 "Signal",
    "HPlusCharm":                   "Signal",
    "ZZto4L":                       "qqZZ",
    "GluGlutoContinto2Zto4E":       "ggZZ",
    "GluGlutoContinto2Zto4Mu":      "ggZZ",
    "GluGlutoContinto2Zto4Tau":     "ggZZ",
    "GluGluToContinto2Zto2E2Mu":    "ggZZ",
    "GluGluToContinto2Zto2E2Tau":   "ggZZ",
    "GluGluToContinto2Zto2Mu2Tau":  "ggZZ",
    "GluGluHtoZZto4L":              "Other_Higgs",
    "VBFHto2Zto4L":                 "Other_Higgs",
    "WminusH_Hto2Zto4L":           "Other_Higgs",
    "WplusH_Hto2Zto4L":            "Other_Higgs",
    "ZHto2Zto4L":                   "Other_Higgs",
    "TTH_Hto2Z":                    "Other_Higgs",
    "bbH_Hto2Zto4L":               "Other_Higgs",
    "HPlusBottom":                  "Other_Higgs",
    "2022preEEHB":                  "Other_Higgs",
    "2022postEEHB":                 "Other_Higgs",
    "2023preBPixHB":                "Other_Higgs",
    "2023postBPixHB":               "Other_Higgs",
}

# ctag2d systematics (matches create_datacards_ctag2d.py exactly: CMS_ctag2d
# single 2D SF replaces the 1D CMS_ctag_b/CMS_ctag_c/CMS_ctag_light rows)
SYSTEMATICS = {
    "lhe_pdf":        {"col": "weight_lhe_pdf",       "era_dep": False},
    "lhe_alphaS":     {"col": "weight_lhe_alphaS",    "era_dep": False},
    "scalevar_muR":   {"col": "weight_scalevar_muR",  "era_dep": False},
    "scalevar_muF":   {"col": "weight_scalevar_muF",  "era_dep": False},
    "ps_isr":         {"col": "weight_ps_isr",        "era_dep": False},
    "ps_fsr":         {"col": "weight_ps_fsr",        "era_dep": False},
    "CMS_pileup":     {"col": "weight_CMS_pileup",    "era_dep": True},
    "CMS_ctag2d":     {"col": "weight_CMS_ctag2d",    "era_dep": False},
    # Muon ID efficiency SF (loose WP, wired 2026-08-21) -- see
    # create_datacards_ctag2d.py's comment for the same row; this file was
    # already missing the 2026-08-17 CMS_eff_e_reco_* addition those two
    # scripts have (pre-existing drift, not fixed here -- out of scope for
    # this change), but CMS_eff_m_id is added to keep the 3-way "edit all
    # three together" convention for at least this new systematic.
    "CMS_eff_m_id":   {"col": "weight_CMS_eff_m_id",  "era_dep": True},
}

# lhe_alphaS EXCLUDED EVERYWHERE, 2026-08-19 (under investigation, not yet
# fixed) -- analysis/corrections/lhepdf.py's abs() bug forces Up/Down to
# always move the same direction for every process. See
# create_datacards_ctag2d_100150.py's USABLE_SYST comment / README_HcZZ.md
# for the full writeup.
USABLE_SYST = {
    "qqZZ":        set(SYSTEMATICS.keys()) - {"lhe_alphaS"},
    "Other_Higgs": set(SYSTEMATICS.keys()) - {"lhe_pdf", "scalevar_muR", "scalevar_muF", "lhe_alphaS"},
    "ggZZ":        {"ps_isr", "ps_fsr", "CMS_pileup", "CMS_ctag2d", "CMS_eff_m_id"},
    "Signal":      {"ps_isr", "ps_fsr", "CMS_pileup", "CMS_ctag2d", "CMS_eff_m_id"},
}


def era_year(era):
    return "2022" if era.startswith("2022") else "2023"


def syst_col(syst_name, direction, era):
    spec = SYSTEMATICS[syst_name]
    base = spec["col"]
    if spec["era_dep"]:
        base = f"{base}_{era_year(era)}"
    return f"{base}{direction}"


def get_process_from_path(filepath):
    for key in sorted(PROCESS_MAPPING, key=len, reverse=True):
        if key in filepath:
            return PROCESS_MAPPING[key]
    return None


def load_sumw(sumw_dir):
    import json
    sumw_by_proc = defaultdict(float)
    n_files = defaultdict(int)
    for era in ERAS:
        era_dir = os.path.join(sumw_dir, era)
        if not os.path.isdir(era_dir):
            print(f"  WARNING: sumw era dir not found: {era_dir}")
            continue
        for dataset_partition in os.listdir(era_dir):
            proc = get_process_from_path(dataset_partition)
            if proc is None:
                continue
            sumw_glob = os.path.join(era_dir, dataset_partition, "sumw", "*.json")
            for jf in glob.glob(sumw_glob):
                try:
                    with open(jf) as f:
                        rec = json.load(f)
                except Exception as e:
                    print(f"  WARNING: {jf}: {e}")
                    continue
                sumw_by_proc[proc] += rec["sumw"]
                n_files[proc] += 1
    print(f"\n  {'Process':<15} {'n_sumw_files':>14} {'sumw_generated':>16}")
    print("  " + "-" * 47)
    for proc in CLASS_NAMES:
        print(f"  {proc:<15} {n_files.get(proc, 0):>14} {sumw_by_proc.get(proc, 0.0):>16.2f}")
    return sumw_by_proc


def compute_ratios(df):
    ps  = df["mva_score_Signal"].values
    pqq = df["mva_score_qqZZ"].values
    pgg = df["mva_score_ggZZ"].values
    poh = df["mva_score_Other_Higgs"].values

    def ratio(num, denom_extra):
        denom = num + denom_extra
        return np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)

    r1 = ratio(ps, pqq)
    r2 = ratio(ps, pgg)
    r3 = ratio(ps, poh)
    return np.stack([r1, r2, r3], axis=1)


def load_mc_scored_parquets(scored_dir, sumw_by_proc, mass_window):
    raw_weight  = defaultdict(float)
    proc_ratios = defaultdict(list)
    proc_weights = defaultdict(list)
    proc_syst_weights = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    score_cols = ["mva_score_Signal", "mva_score_qqZZ", "mva_score_ggZZ", "mva_score_Other_Higgs"]

    for era in ERAS:
        era_dir = os.path.join(scored_dir, era)
        if not os.path.isdir(era_dir):
            print(f"  WARNING: era dir not found: {era_dir}")
            continue

        for sample_name in os.listdir(era_dir):
            proc = get_process_from_path(sample_name)
            if proc is None:
                continue
            sample_dir = os.path.join(era_dir, sample_name, "base")
            if not os.path.isdir(sample_dir):
                continue

            usable = USABLE_SYST.get(proc, set())
            syst_cols_wanted = []
            for syst in usable:
                for direction in ("Up", "Down"):
                    syst_cols_wanted.append(syst_col(syst, direction, era))

            for pq_file in glob.glob(os.path.join(sample_dir, "*.parquet")):
                base_cols = ["zz_mass_inclusive", "weight_nominal"] + score_cols
                try:
                    df = pq.read_table(pq_file, columns=base_cols + syst_cols_wanted).to_pandas()
                except Exception:
                    try:
                        available = pq.ParquetFile(pq_file).schema.names
                        cols = base_cols + [c for c in syst_cols_wanted if c in available]
                        df = pq.read_table(pq_file, columns=cols).to_pandas()
                    except Exception as e:
                        print(f"  WARNING: {pq_file}: {e}")
                        continue

                w_all = df["weight_nominal"].fillna(0).values
                raw_weight[proc] += w_all.sum()

                mask = (df["zz_mass_inclusive"] >= mass_window[0]) & \
                       (df["zz_mass_inclusive"] < mass_window[1])
                df_win = df[mask]
                valid = df_win["mva_score_Signal"] >= 0
                df_win = df_win[valid]
                if len(df_win) == 0:
                    continue

                proc_ratios[proc].append(compute_ratios(df_win))
                nominal_win = df_win["weight_nominal"].fillna(0).values
                proc_weights[proc].append(nominal_win)

                for syst in usable:
                    for direction in ("Up", "Down"):
                        col = syst_col(syst, direction, era)
                        if col in df_win.columns:
                            proc_syst_weights[proc][syst][direction].append(
                                df_win[col].fillna(0).values
                            )
                        else:
                            proc_syst_weights[proc][syst][direction].append(nominal_win)

    mc_data = {}
    print(f"\n  {'Process':<15} {'raw_wgt':>12} {'sumw_gen':>14} {'baseline_eff':>13} {'expected':>10} {'in_win_yield':>14}")
    print("  " + "-" * 82)

    for proc in CLASS_NAMES:
        raw = raw_weight.get(proc, 0.0)
        sumw_gen = sumw_by_proc.get(proc, 0.0)
        exp = EXPECTED_YIELDS.get(proc, 0.0)
        baseline_eff = raw / sumw_gen if sumw_gen > 0 else float("nan")

        ratios  = np.concatenate(proc_ratios[proc], axis=0) if proc_ratios[proc] else np.zeros((0, 3))
        weights = np.concatenate(proc_weights[proc]) if proc_weights[proc] else np.array([])

        if sumw_gen > 0 and exp > 0:
            scale = exp / sumw_gen
            scaled_weights = weights * scale
        else:
            print(f"  WARNING: no sumw for {proc} -- falling back to post-selection raw_weight")
            scale = exp / raw if raw > 0 and exp > 0 else 1.0
            scaled_weights = weights * scale

        syst_scaled = defaultdict(dict)
        for syst, by_dir in proc_syst_weights[proc].items():
            for direction, arrs in by_dir.items():
                w = np.concatenate(arrs) if arrs else np.array([])
                syst_scaled[syst][direction] = w * scale

        mc_data[proc] = (ratios, scaled_weights, syst_scaled)
        in_win = scaled_weights.sum()
        print(f"  {proc:<15} {raw:>12.2f} {sumw_gen:>14.2f} {baseline_eff:>13.4%} {exp:>10.4f} {in_win:>14.4f}")

    return mc_data


def quantile_edges_fixed(values, weights, n_bins):
    if len(values) == 0:
        return np.linspace(0, 1, n_bins + 1)
    w = np.abs(weights)
    order = np.argsort(values)
    v_sorted, w_sorted = values[order], w[order]
    cumw = np.cumsum(w_sorted)
    if cumw[-1] <= 0:
        return np.linspace(0, 1, n_bins + 1)
    cumw /= cumw[-1]
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.interp(quantiles, cumw, v_sorted)
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def build_nested_tree(mc_data, n1, n2, n3, reference_proc="qqZZ"):
    ratios, weights, _ = mc_data[reference_proc]
    r1, r2, r3 = ratios[:, 0], ratios[:, 1], ratios[:, 2]
    w = weights

    r1_edges = quantile_edges_fixed(r1, w, n1)
    r2_edges_by_i = []
    r3_edges_by_ij = []

    for i in range(n1):
        lo, hi = r1_edges[i], r1_edges[i + 1]
        mask1 = (r1 >= lo) & (r1 <= hi if i == n1 - 1 else r1 < hi)
        r2_i, w_i = r2[mask1], w[mask1]
        r2_edges_i = quantile_edges_fixed(r2_i, w_i, n2)
        r2_edges_by_i.append(r2_edges_i)

        r3_row = []
        for j in range(n2):
            lo2, hi2 = r2_edges_i[j], r2_edges_i[j + 1]
            mask2 = mask1 & ((r2 >= lo2) & (r2 <= hi2 if j == n2 - 1 else r2 < hi2))
            r3_ij, w_ij = r3[mask2], w[mask2]
            r3_row.append(quantile_edges_fixed(r3_ij, w_ij, n3))
        r3_edges_by_ij.append(r3_row)

    return r1_edges, r2_edges_by_i, r3_edges_by_ij


def assign_nested_bins(ratios, tree, n1, n2, n3):
    r1_edges, r2_edges_by_i, r3_edges_by_ij = tree
    r1v, r2v, r3v = ratios[:, 0], ratios[:, 1], ratios[:, 2]
    n = len(r1v)
    flat_idx = np.zeros(n, dtype=np.int64)

    i_idx = np.clip(np.searchsorted(r1_edges, r1v, side="right") - 1, 0, n1 - 1)

    for i in range(n1):
        sel_i = np.where(i_idx == i)[0]
        if len(sel_i) == 0:
            continue
        r2_edges_i = r2_edges_by_i[i]
        j_idx = np.clip(np.searchsorted(r2_edges_i, r2v[sel_i], side="right") - 1, 0, n2 - 1)

        for j in range(n2):
            sel_j_local = np.where(j_idx == j)[0]
            if len(sel_j_local) == 0:
                continue
            idx_global = sel_i[sel_j_local]
            r3_edges_ij = r3_edges_by_ij[i][j]
            k_idx = np.clip(np.searchsorted(r3_edges_ij, r3v[idx_global], side="right") - 1, 0, n3 - 1)
            flat_idx[idx_global] = i * n2 * n3 + j * n3 + k_idx

    return flat_idx


def build_histograms(mc_data, tree, n1, n2, n3):
    n_bins = n1 * n2 * n3
    histograms = {}
    print(f"\n  {'Process':<15} {'n_events':>10} {'hist_integral':>14}")
    print("  " + "-" * 42)

    for proc in CLASS_NAMES:
        ratios, weights, syst_scaled = mc_data[proc]
        if len(weights) == 0:
            histograms[proc] = np.zeros(n_bins)
            print(f"  {proc:<15} {'(empty)':>10}")
            continue
        flat_idx = assign_nested_bins(ratios, tree, n1, n2, n3)
        h = np.bincount(flat_idx, weights=weights, minlength=n_bins).astype(np.float64)
        histograms[proc] = h
        print(f"  {proc:<15} {len(weights):>10} {h.sum():>14.4f}")

        for syst, by_dir in syst_scaled.items():
            for direction, w in by_dir.items():
                if len(w) != len(weights):
                    print(f"  WARNING: {proc}/{syst}{direction} length mismatch "
                          f"({len(w)} vs {len(weights)}) -- skipping")
                    continue
                hs = np.bincount(flat_idx, weights=w, minlength=n_bins).astype(np.float64)
                histograms[f"{proc}_{syst}{direction}"] = hs

    return histograms, n_bins


def write_root_file(histograms, n_bins, output_dir, filename):
    path = os.path.join(output_dir, filename)
    nominal_procs = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"]
    flat_edges = np.arange(n_bins + 1, dtype=np.float64)
    with uproot.recreate(path, compression=None) as f:
        for proc, h in histograms.items():
            f[f"h_{proc}"] = (h, flat_edges)
        data_obs = sum(histograms[p] for p in nominal_procs if p in histograms)
        f["h_data_obs"] = (data_obs, flat_edges)
    print(f"\n  ROOT file: {path}")
    return path


def write_datacard(histograms, root_filename, output_dir, mass_window, stat_only=False):
    dc_path = os.path.join(output_dir, "datacard_3ratio_statonly.txt" if stat_only
                            else "datacard_3ratio.txt")
    procs    = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"]
    proc_idx = [0, 1, 2, 3]
    yields   = {p: histograms[p].sum() for p in procs}
    col_w = 16

    def row(label, values):
        return label + "".join(f"{v:<{col_w}}" for v in values) + "\n"

    with open(dc_path, "w") as dc:
        dc.write(f"# H+c -> ZZ -> 4l: 3-ratio-decomposition MVA, ctag2d SF, NO Z+X"
                  f"{' (stat-only)' if stat_only else ''}\n")
        dc.write(f"# Mass window: {mass_window[0]}-{mass_window[1]} GeV  |  Lumi: {TOTAL_LUMI_PB:.1f} pb^-1\n")
        dc.write(f"# Discriminant: joint (r1,r2,r3) = (Ps/(Ps+PqqZZ), Ps/(Ps+PggZZ), Ps/(Ps+POH)), flattened\n")
        dc.write(f"imax 1\njmax {len(procs)-1}\nkmax {'0' if stat_only else '*'}\n")
        dc.write("-" * 80 + "\n")
        dc.write(f"shapes * * {root_filename} h_$PROCESS h_$PROCESS_$SYSTEMATIC\n")
        dc.write("-" * 80 + "\n")
        dc.write("bin         hczz\n")
        dc.write("observation -1\n")
        dc.write("-" * 80 + "\n")
        dc.write(row("bin         ", ["hczz"] * len(procs)))
        dc.write(row("process     ", procs))
        dc.write(row("process     ", [str(i) for i in proc_idx]))
        dc.write(row("rate        ", [f"{yields[p]:.6f}" for p in procs]))
        dc.write("-" * 80 + "\n")

        if not stat_only:
            def syst_row(name, stype, vals):
                line = f"{name:<25}{stype:<8}"
                for p in procs:
                    line += f"{vals.get(p, '-'):<{col_w}}"
                return line + "\n"

            dc.write(syst_row("lumi_Run3",     "lnN", {p: "1.014" for p in procs}))
            # QCDscale_gg / pdf_gg / kfactor_ggZZ: values matched to HIG-24-013
            # (AN2023_157_v10) Table 15 2026-08-28 -- see create_datacards_ctag2d.py
            # for the full rationale (was one 5% pdf_gg placeholder + a
            # mislabeled "QCDscale_ggZZ" that was actually the k-factor).
            dc.write(syst_row("QCDscale_gg",   "lnN", {"ggZZ": "1.039", "Signal": "1.039"}))
            dc.write(syst_row("pdf_gg",        "lnN", {"ggZZ": "1.032", "Signal": "1.032"}))
            dc.write(syst_row("kfactor_ggZZ",  "lnN", {"ggZZ": "1.10"}))
            dc.write(f"\n# Shape systematics from per-event weight variations\n")
            for syst in SYSTEMATICS:
                vals = {p: "1" for p in procs if syst in USABLE_SYST.get(p, set())}
                if not vals:
                    continue
                dc.write(syst_row(syst, "shape", vals))

            # autoMCStats, added 2026-08-18 -- see create_datacards_ctag2d.py for
            # the sparse-binning caveat (same underlying histograms/binning risk
            # applies here too, arguably more so given the 3-ratio joint binning
            # is sparser per-cell than the 1D plain-score binning).
            dc.write(f"\nhczz autoMCStats 10\n")

    print(f"  Datacard:  {dc_path}")
    return dc_path


def print_summary(histograms, mass_window):
    procs = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"]
    print(f"\n{'Process':<15} {'Yield in [{},{}] GeV'.format(*mass_window):>20}")
    print("-" * 38)
    total_bkg = 0
    for p in procs:
        y = histograms[p].sum()
        if p != "Signal":
            total_bkg += y
        print(f"  {p:<13} {y:>12.4f}")
    print("-" * 38)
    print(f"  {'Total bkg':<13} {total_bkg:>12.4f}")
    sig = histograms["Signal"].sum()
    if total_bkg > 0:
        print(f"  S/sqrt(B) = {sig/total_bkg**0.5:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-dir",
        # _v2 (default since 2026-08-28) is the rebuild that also carries
        # weight_CMS_eff_m_id_<year>Up/Down (muon efficiency SF); the old
        # _scored dir predates that column and silently reproduces a stale
        # pre-muon-SF result.
        default="/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored_v2")
    parser.add_argument("--sumw-dir",
        default="/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mass-window", type=float, nargs=2, default=[90, 160])
    parser.add_argument("--n-bins-per-axis", type=str, default="2,2,5")
    args = parser.parse_args()

    mass_window = tuple(args.mass_window)
    n1, n2, n3 = [int(x) for x in args.n_bins_per_axis.split(",")]

    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("H+c -> ZZ -> 4l  |  ctag2d 3-ratio-decomposition datacard builder")
    print(f"Mass window : {mass_window[0]}-{mass_window[1]} GeV")
    print(f"Binning     : nested tree ({n1},{n2},{n3}) = {n1*n2*n3} bins")
    print("=" * 70)

    sumw_by_proc = load_sumw(args.sumw_dir)
    mc_data = load_mc_scored_parquets(args.scored_dir, sumw_by_proc, mass_window)

    tree = build_nested_tree(mc_data, n1, n2, n3)
    histograms, n_bins = build_histograms(mc_data, tree, n1, n2, n3)

    root_file = write_root_file(histograms, n_bins, args.output, "histograms_3ratio.root")
    write_datacard(histograms, os.path.basename(root_file), args.output, mass_window, stat_only=False)
    write_datacard(histograms, os.path.basename(root_file), args.output, mass_window, stat_only=True)

    print_summary(histograms, mass_window)
    print(f"\n[Done] All files in: {args.output}")


if __name__ == "__main__":
    main()
