#!/usr/bin/env python3
"""
Fresh plain-vs-kappa-vs-3ratio comparison, matched to the CURRENT ctag2d+JES/JER
production pipeline (create_datacards_ctag2d_100150.py, 2026-09-08 reference
r=412.0/kappa_c=74.11). Single MC-loading pass builds all three discriminant
variants together so every variant shares identical process rates within a run
(avoids the EOS-listing-flakiness trap documented in second-brain
Notes/SM-3ratio-kappa-decomposition-note.md Bug 1).

Discriminants, all built from the same 4-class softmax (Pqq, Pgg, Ps, Poh):
  1. plain  : Ps alone, 1D quantile-binned.
  2. kappa  : Ds = Ps / (Ps + k_qq*Pqq + k_gg*Pgg + k_oh*Poh), weights
              re-optimized fresh against THIS run's real in-window expected
              yields (scipy differential_evolution, S/sqrt(B) objective,
              mirrors kappa_saranya.ipynb's DiscriminantOptimizer -- reusing
              any previously-documented kappa values would be the exact
              stale-weights bug already found and fixed once, see the note
              above). 1D quantile-binned.
  3. ratio  : joint (r1,r2,r3) = (Ps/(Ps+Pqq), Ps/(Ps+Pgg), Ps/(Ps+Poh)),
              nested/conditional-quantile-tree binned (ported from
              create_datacards_ctag2d_3ratio.py).

Scope matches current production exactly: ctag2d SF scheme, [100,150] GeV mass
window, MC-only (--skip-zx, no ZX code path at all here -- ZX is a separate,
still-in-progress axis, out of scope), JES/JER included via --jecshifts-dir.
Does NOT modify create_datacards_ctag2d_100150.py or
create_datacards_ctag2d_3ratio.py (both untouched, per standing project
convention -- new comparison logic lives only here).

Writes, per variant (plain/kappa/ratio) x (full-syst/stat-only) = 6 datacards.
"""

import os
import glob
import argparse
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
from scipy.optimize import differential_evolution, NonlinearConstraint
import uproot

# ---------------------------------------------------------------------------
# Constants -- copied verbatim from create_datacards_ctag2d_100150.py
# ---------------------------------------------------------------------------

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
SCORE_COLS = ["mva_score_qqZZ", "mva_score_ggZZ", "mva_score_Signal", "mva_score_Other_Higgs"]
CLASS_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
SIG_IDX = CLASS_IDX["Signal"]

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

MASS_WINDOW = (100, 150)

# ctag2d SYSTEMATICS -- copied verbatim from create_datacards_ctag2d_100150.py
SYSTEMATICS = {
    "lhe_pdf":        {"col": "weight_lhe_pdf",       "era_dep": False},
    "lhe_alphaS":     {"col": "weight_lhe_alphaS",    "era_dep": False},
    "scalevar_muR":   {"col": "weight_scalevar_muR",  "era_dep": False},
    "scalevar_muF":   {"col": "weight_scalevar_muF",  "era_dep": False},
    "ps_isr":         {"col": "weight_ps_isr",        "era_dep": False},
    "ps_fsr":         {"col": "weight_ps_fsr",        "era_dep": False},
    "CMS_pileup":     {"col": "weight_CMS_pileup",    "era_dep": True},
    "CMS_ctag2d":     {"col": "weight_CMS_ctag2d",    "era_dep": True},
    "CMS_eff_e_reco_20to75":  {"col": "weight_CMS_eff_e_reco_20to75",  "era_dep": True},
    "CMS_eff_e_reco_above75": {"col": "weight_CMS_eff_e_reco_above75", "era_dep": True},
    "CMS_eff_e_reco_below20": {"col": "weight_CMS_eff_e_reco_below20", "era_dep": True},
    "CMS_eff_m_id":           {"col": "weight_CMS_eff_m_id",           "era_dep": True},
    "higgs_plus_c":           {"col": "weight_higgs_plus_c",           "era_dep": False},
}

EFF_E_RECO = {"CMS_eff_e_reco_20to75", "CMS_eff_e_reco_above75", "CMS_eff_e_reco_below20"}
USABLE_SYST = {
    "qqZZ":        set(SYSTEMATICS.keys()) - {"lhe_alphaS", "higgs_plus_c"},
    "Other_Higgs": set(SYSTEMATICS.keys()) - {"lhe_pdf", "scalevar_muR", "scalevar_muF", "lhe_alphaS"},
    "ggZZ":        {"ps_isr", "ps_fsr", "CMS_pileup", "CMS_ctag2d", "CMS_eff_m_id"} | EFF_E_RECO,
    "Signal":      {"ps_isr", "ps_fsr", "CMS_pileup", "CMS_ctag2d", "CMS_eff_m_id"} | EFF_E_RECO,
}

JECSHIFTS_SYSTEMATICS = ["CMS_scale_j", "CMS_res_j", "CMS_scale_m", "CMS_res_m"]
JECSHIFTS_USABLE_PROCS = {"Signal", "ggZZ", "qqZZ", "Other_Higgs"}


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


# ---------------------------------------------------------------------------
# sumw -- copied verbatim (dedup logic included) from create_datacards_ctag2d_100150.py
# ---------------------------------------------------------------------------

def load_sumw(sumw_dir):
    import json
    import re as _re

    sumw_by_proc = defaultdict(float)
    n_files = defaultdict(int)

    for era in ERAS:
        era_dir = os.path.join(sumw_dir, era)
        if not os.path.isdir(era_dir):
            print(f"  WARNING: sumw era dir not found: {era_dir}")
            continue

        _chunk_re = _re.compile(r'_(\d+)-(\d+)\.json$')
        latest_by_chunk = {}
        for dataset_partition in os.listdir(era_dir):
            proc = get_process_from_path(dataset_partition)
            if proc is None:
                continue
            sumw_glob = os.path.join(era_dir, dataset_partition, "sumw", "*.json")
            for jf in glob.glob(sumw_glob):
                base = os.path.basename(jf)
                guid = base.split("_%2F")[0]
                m = _chunk_re.search(base)
                chunk_range = (m.group(1), m.group(2)) if m else None
                key = (guid, chunk_range)
                mtime = os.path.getmtime(jf)
                prev = latest_by_chunk.get(key)
                if prev is None or mtime > prev[1]:
                    latest_by_chunk[key] = (jf, mtime, proc)

        for jf, _mtime, proc in latest_by_chunk.values():
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


# ---------------------------------------------------------------------------
# MC loading -- all 4 class-probability columns kept per event
# ---------------------------------------------------------------------------

def load_mc_scored_parquets(scored_dir, sumw_by_proc):
    import gc

    raw_weight  = defaultdict(float)
    proc_scores4 = defaultdict(list)   # list of per-era (N,4) float32 arrays
    proc_weights = defaultdict(list)
    proc_syst_weights = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for era in ERAS:
        era_dir = os.path.join(scored_dir, era)
        if not os.path.isdir(era_dir):
            print(f"  WARNING: era dir not found: {era_dir}")
            continue

        # Per-era accumulation, concatenated+downcast to float32 and merged
        # into the process-level list at the END OF EACH ERA (not per-file) --
        # keeps only a handful of era-sized arrays alive at once instead of
        # thousands of small per-file fragments, and halves the footprint of
        # every array via float32 (scores/weights don't need float64
        # precision for histogramming). gc.collect() after each era to
        # release freed per-file pandas DataFrames promptly.
        era_scores4 = defaultdict(list)
        era_weights = defaultdict(list)
        era_syst_weights = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

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
                base_cols = ["zz_mass_inclusive", "weight_nominal"] + SCORE_COLS
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

                mask = (df["zz_mass_inclusive"] >= MASS_WINDOW[0]) & \
                       (df["zz_mass_inclusive"] < MASS_WINDOW[1])
                df_win = df[mask]
                valid = df_win["mva_score_Signal"] >= 0
                df_win = df_win[valid]
                if len(df_win) == 0:
                    continue

                scores4 = df_win[SCORE_COLS].values.astype(np.float32)
                era_scores4[proc].append(scores4)
                nominal_win = df_win["weight_nominal"].fillna(0).values.astype(np.float32)
                era_weights[proc].append(nominal_win)

                for syst in usable:
                    for direction in ("Up", "Down"):
                        col = syst_col(syst, direction, era)
                        if col in df_win.columns:
                            era_syst_weights[proc][syst][direction].append(
                                df_win[col].fillna(0).values.astype(np.float32)
                            )
                        else:
                            era_syst_weights[proc][syst][direction].append(nominal_win)

                del df, df_win

        # Fold this era's accumulated arrays into the process-level lists,
        # then drop the era-local dicts and reclaim memory before moving on.
        for proc, arrs in era_scores4.items():
            proc_scores4[proc].append(np.concatenate(arrs, axis=0))
        for proc, arrs in era_weights.items():
            proc_weights[proc].append(np.concatenate(arrs))
        for proc, by_syst in era_syst_weights.items():
            for syst, by_dir in by_syst.items():
                for direction, arrs in by_dir.items():
                    proc_syst_weights[proc][syst][direction].append(np.concatenate(arrs))
        del era_scores4, era_weights, era_syst_weights
        gc.collect()
        print(f"  [era {era} done, memory reclaimed]")

    mc_data = {}
    print(f"\n  {'Process':<15} {'raw_wgt':>12} {'sumw_gen':>14} {'baseline_eff':>13} {'expected':>10} {'in_win_yield':>14}")
    print("  " + "-" * 82)

    for proc in CLASS_NAMES:
        raw = raw_weight.get(proc, 0.0)
        sumw_gen = sumw_by_proc.get(proc, 0.0)
        exp = EXPECTED_YIELDS.get(proc, 0.0)
        baseline_eff = raw / sumw_gen if sumw_gen > 0 else float("nan")

        scores4 = np.concatenate(proc_scores4[proc], axis=0) if proc_scores4[proc] else np.zeros((0, 4))
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

        mc_data[proc] = (scores4, scaled_weights, syst_scaled)
        in_win = scaled_weights.sum()
        print(f"  {proc:<15} {raw:>12.2f} {sumw_gen:>14.2f} {baseline_eff:>13.4%} {exp:>10.4f} {in_win:>14.4f}")

    return mc_data


def load_jecshifts_scored_parquets(jecshifts_dir, sumw_by_proc):
    """Same directory convention as create_datacards_ctag2d_100150.py, but keeps
    all 4 class-probability columns per event (needed by kappa/3-ratio, not just
    plain's mva_score_Signal)."""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"scores4": [], "weights": []})))
    raw_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for era in ERAS:
        era_dir = os.path.join(jecshifts_dir, era)
        if not os.path.isdir(era_dir):
            print(f"  WARNING: jecshifts era dir not found: {era_dir}")
            continue
        year = era_year(era)

        for sample_name in os.listdir(era_dir):
            proc = get_process_from_path(sample_name)
            if proc is None or proc not in JECSHIFTS_USABLE_PROCS:
                continue

            for syst in JECSHIFTS_SYSTEMATICS:
                for direction in ("Up", "Down"):
                    cat_dir = os.path.join(era_dir, sample_name, "base", f"{syst}_{year}{direction}")
                    if not os.path.isdir(cat_dir):
                        continue

                    for pq_file in glob.glob(os.path.join(cat_dir, "*.parquet")):
                        cols = ["zz_mass_inclusive", "weight_nominal"] + SCORE_COLS
                        try:
                            df = pq.read_table(pq_file, columns=cols).to_pandas()
                        except Exception as e:
                            print(f"  WARNING: {pq_file}: {e}")
                            continue

                        raw_sum[proc][syst][direction] += df["weight_nominal"].fillna(0).values.sum()

                        mask = (df["zz_mass_inclusive"] >= MASS_WINDOW[0]) & \
                               (df["zz_mass_inclusive"] < MASS_WINDOW[1])
                        df_win = df[mask]
                        df_win = df_win[df_win["mva_score_Signal"] >= 0]
                        if len(df_win) == 0:
                            continue

                        raw[proc][syst][direction]["scores4"].append(
                            df_win[SCORE_COLS].values.astype(np.float64)
                        )
                        raw[proc][syst][direction]["weights"].append(df_win["weight_nominal"].fillna(0).values)

    jec_data = defaultdict(lambda: defaultdict(dict))
    print(f"\n  {'Process':<15} {'Systematic':<16} {'Dir':<6} {'n_files_sum':>12} {'in_win_yield':>14}")
    print("  " + "-" * 66)
    for proc in JECSHIFTS_USABLE_PROCS:
        sumw_gen = sumw_by_proc.get(proc, 0.0)
        exp = EXPECTED_YIELDS.get(proc, 0.0)
        scale = exp / sumw_gen if sumw_gen > 0 and exp > 0 else 1.0

        for syst in JECSHIFTS_SYSTEMATICS:
            for direction in ("Up", "Down"):
                s4_list = raw[proc][syst][direction]["scores4"]
                w_list = raw[proc][syst][direction]["weights"]
                if not s4_list:
                    print(f"  WARNING: no jecshifts events for {proc}/{syst}{direction}")
                    jec_data[proc][syst][direction] = (np.zeros((0, 4)), np.array([]))
                    continue
                scores4 = np.concatenate(s4_list, axis=0)
                weights = np.concatenate(w_list) * scale
                jec_data[proc][syst][direction] = (scores4, weights)
                print(f"  {proc:<15} {syst:<16} {direction:<6} "
                      f"{raw_sum[proc][syst][direction]:>12.2f} {weights.sum():>14.4f}")

    return jec_data


# ---------------------------------------------------------------------------
# Discriminant construction
# ---------------------------------------------------------------------------

def plain_score(scores4):
    return scores4[:, SIG_IDX]


def kappa_score(scores4, kappas):
    """kappas ordered [k_qq, k_gg, k_oh] (Signal has no self-kappa)."""
    k_qq, k_gg, k_oh = kappas
    ps, pqq, pgg, poh = scores4[:, 2], scores4[:, 0], scores4[:, 1], scores4[:, 3]
    denom = ps + k_qq * pqq + k_gg * pgg + k_oh * poh
    return np.where(denom > 0, ps / np.where(denom > 0, denom, 1.0), 0.0)


def compute_ratios(scores4):
    ps, pqq, pgg, poh = scores4[:, 2], scores4[:, 0], scores4[:, 1], scores4[:, 3]

    def ratio(num, extra):
        denom = num + extra
        return np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)

    return np.stack([ratio(ps, pqq), ratio(ps, pgg), ratio(ps, poh)], axis=1)


# ---------------------------------------------------------------------------
# Kappa optimization -- fresh re-derivation against THIS run's real in-window
# yields (mirrors kappa_saranya.ipynb's DiscriminantOptimizer/S/sqrt(B) scan;
# reusing any previously-documented kappa value would repeat the exact
# stale-weights bug this project already found once).
# ---------------------------------------------------------------------------

def compute_s_over_sqrtb_fast(is_sig, weights, discriminant, eps=1e-12, min_eff=0.1):
    is_bkg = ~is_sig
    weights_sig = weights * is_sig
    weights_bkg = weights * is_bkg
    total_sig = np.sum(weights_sig)
    thresholds = np.arange(0.1, 1.01, 0.01)
    order = np.argsort(discriminant)[::-1]
    sorted_wsig = weights_sig[order]
    sorted_wbkg = weights_bkg[order]
    sorted_disc = discriminant[order]
    cumsum_sig = np.cumsum(sorted_wsig)
    cumsum_bkg = np.cumsum(sorted_wbkg)

    sb = np.zeros(len(thresholds))
    eff = np.zeros(len(thresholds))
    for i, th in enumerate(thresholds):
        n_above = np.searchsorted(-sorted_disc, -th, side="right")
        if n_above == 0:
            S, B = 0.0, 0.0
        else:
            S = cumsum_sig[n_above - 1]
            B = cumsum_bkg[n_above - 1]
        eff[i] = S / total_sig if total_sig > 0 else 0.0
        sb[i] = S / np.sqrt(B + eps) if B > 0 else 0.0

    valid = eff >= min_eff
    if not np.any(valid):
        best_idx = np.argmax(sb)
    else:
        best_idx = np.argmax(np.where(valid, sb, -np.inf))
    return sb[best_idx]


def optimize_kappa(mc_data, verbose=True):
    """Pools nominal (scores4, weight) across all 4 processes with class-balance
    weights = real_in_window_yield[c] / n_events[c] (so each class contributes
    its TRUE expected yield in aggregate, independent of raw MC statistics),
    then differential_evolution over (k_qq, k_gg) with k_oh = 1 - k_qq - k_gg
    (>=0 constrained), maximizing Signal-class S/sqrt(B)."""
    all_scores4, all_truth, all_cbweight = [], [], []
    real_yield = {}
    for proc in CLASS_NAMES:
        scores4, weights, _ = mc_data[proc]
        real_yield[proc] = weights.sum()

    if verbose:
        print("\n  Real in-window yields used as class-balance targets:")
        for proc in CLASS_NAMES:
            print(f"    {proc:<15} {real_yield[proc]:.4f}")

    for proc in CLASS_NAMES:
        scores4, weights, _ = mc_data[proc]
        n = len(scores4)
        if n == 0:
            continue
        cbw = np.full(n, real_yield[proc] / n)
        all_scores4.append(scores4)
        all_truth.append(np.full(n, CLASS_IDX[proc], dtype=int))
        all_cbweight.append(cbw)

    scores4 = np.concatenate(all_scores4, axis=0)
    truth = np.concatenate(all_truth, axis=0)
    cbweight = np.concatenate(all_cbweight, axis=0)
    is_sig = (truth == SIG_IDX)

    default_sb = compute_s_over_sqrtb_fast(is_sig, cbweight, scores4[:, SIG_IDX])

    def objective(x):
        k_qq, k_gg = x
        k_oh = max(0.0, 1.0 - k_qq - k_gg)
        disc = kappa_score(scores4, (k_qq, k_gg, k_oh))
        return -compute_s_over_sqrtb_fast(is_sig, cbweight, disc)

    bounds = [(0.0, 1.0), (0.0, 1.0)]
    constraint = NonlinearConstraint(lambda x: x[0] + x[1], -np.inf, 1.0)
    result = differential_evolution(
        objective, bounds=bounds, maxiter=50, popsize=15, tol=1e-6,
        seed=42, disp=verbose, constraints=[constraint],
    )
    k_qq, k_gg = result.x
    k_oh = max(0.0, 1.0 - k_qq - k_gg)
    best_sb = -result.fun

    if verbose:
        print(f"\n  Optimized kappa (qqZZ, ggZZ, Other_Higgs) = ({k_qq:.4f}, {k_gg:.4f}, {k_oh:.4f})")
        print(f"  S/sqrt(B): default(plain)={default_sb:.4f}  kappa-optimized={best_sb:.4f}"
              f"  improvement={100*(best_sb-default_sb)/default_sb:.2f}%")

    return (k_qq, k_gg, k_oh)


# ---------------------------------------------------------------------------
# 1D quantile binning (plain / kappa) -- copied from create_datacards_ctag2d_100150.py
# ---------------------------------------------------------------------------

def compute_quantile_bins_1d(scores, weights, n_bins=20):
    w = np.abs(weights)
    order = np.argsort(scores)
    scores_sorted, w_sorted = scores[order], w[order]
    cumw = np.cumsum(w_sorted)
    cumw /= cumw[-1]
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.interp(quantiles, cumw, scores_sorted)
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    if len(edges) - 1 < n_bins:
        print(f"  NOTE: requested {n_bins} quantile bins, got {len(edges)-1} after deduping.")
    return edges


# ---------------------------------------------------------------------------
# Nested/conditional quantile-tree binning (3-ratio) -- copied from
# create_datacards_ctag2d_3ratio.py
# ---------------------------------------------------------------------------

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


def build_nested_tree(ratios_ref, weights_ref, n1, n2, n3):
    r1, r2, r3 = ratios_ref[:, 0], ratios_ref[:, 1], ratios_ref[:, 2]
    w = weights_ref
    r1_edges = quantile_edges_fixed(r1, w, n1)
    r2_edges_by_i, r3_edges_by_ij = [], []
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


# ---------------------------------------------------------------------------
# Histogram building -- generic over "variant" (plain/kappa: 1D score + edges;
# ratio: 3D ratios + tree)
# ---------------------------------------------------------------------------

def build_histograms_1d(mc_data, jec_data, discriminant_fn, edges):
    n_bins = len(edges) - 1
    histograms = {}
    print(f"\n  {'Process':<15} {'n_events':>10} {'hist_integral':>14}")
    print("  " + "-" * 42)
    for proc in CLASS_NAMES:
        scores4, weights, syst_scaled = mc_data[proc]
        if len(weights) == 0:
            histograms[proc] = np.zeros(n_bins)
            print(f"  {proc:<15} {'(empty)':>10}")
            continue
        disc = discriminant_fn(scores4)
        h, _ = np.histogram(disc, bins=edges, weights=weights)
        histograms[proc] = h.astype(np.float64)
        print(f"  {proc:<15} {len(weights):>10} {h.sum():>14.4f}")

        for syst, by_dir in syst_scaled.items():
            for direction, w in by_dir.items():
                if len(w) != len(disc):
                    print(f"  WARNING: {proc}/{syst}{direction} length mismatch -- skipping")
                    continue
                hs, _ = np.histogram(disc, bins=edges, weights=w)
                histograms[f"{proc}_{syst}{direction}"] = hs.astype(np.float64)

        if jec_data is not None:
            for syst in JECSHIFTS_SYSTEMATICS:
                for direction in ("Up", "Down"):
                    js4, jw = jec_data.get(proc, {}).get(syst, {}).get(direction, (np.zeros((0, 4)), np.array([])))
                    if len(jw) == 0:
                        continue
                    jdisc = discriminant_fn(js4)
                    hs, _ = np.histogram(jdisc, bins=edges, weights=jw)
                    histograms[f"{proc}_{syst}{direction}"] = hs.astype(np.float64)
    return histograms


def build_histograms_ratio(mc_data, jec_data, tree, n1, n2, n3):
    n_bins = n1 * n2 * n3
    histograms = {}
    print(f"\n  {'Process':<15} {'n_events':>10} {'hist_integral':>14}")
    print("  " + "-" * 42)
    for proc in CLASS_NAMES:
        scores4, weights, syst_scaled = mc_data[proc]
        if len(weights) == 0:
            histograms[proc] = np.zeros(n_bins)
            print(f"  {proc:<15} {'(empty)':>10}")
            continue
        ratios = compute_ratios(scores4)
        flat_idx = assign_nested_bins(ratios, tree, n1, n2, n3)
        h = np.bincount(flat_idx, weights=weights, minlength=n_bins).astype(np.float64)
        histograms[proc] = h
        print(f"  {proc:<15} {len(weights):>10} {h.sum():>14.4f}")

        for syst, by_dir in syst_scaled.items():
            for direction, w in by_dir.items():
                if len(w) != len(weights):
                    print(f"  WARNING: {proc}/{syst}{direction} length mismatch -- skipping")
                    continue
                hs = np.bincount(flat_idx, weights=w, minlength=n_bins).astype(np.float64)
                histograms[f"{proc}_{syst}{direction}"] = hs

        if jec_data is not None:
            for syst in JECSHIFTS_SYSTEMATICS:
                for direction in ("Up", "Down"):
                    js4, jw = jec_data.get(proc, {}).get(syst, {}).get(direction, (np.zeros((0, 4)), np.array([])))
                    if len(jw) == 0:
                        continue
                    jratios = compute_ratios(js4)
                    jflat_idx = assign_nested_bins(jratios, tree, n1, n2, n3)
                    hs = np.bincount(jflat_idx, weights=jw, minlength=n_bins).astype(np.float64)
                    histograms[f"{proc}_{syst}{direction}"] = hs
    return histograms, n_bins


# ---------------------------------------------------------------------------
# Negative-bin scan -- copied from create_datacards_ctag2d_100150.py
# ---------------------------------------------------------------------------

def scan_negative_bins(histograms, label):
    print(f"\n[Negative-bin scan: {label}]")
    n_checked, flagged = 0, []
    for name, h in histograms.items():
        n_checked += 1
        neg = np.where(h < 0)[0]
        if len(neg):
            flagged.append((name, [(int(i), float(h[i])) for i in neg], float(h.sum())))
    print(f"  Histograms checked: {n_checked}  |  with negative bins: {len(flagged)}")
    for name, bins, total in flagged:
        print(f"    h_{name}: total={total:.4e}  negative_bins={bins}")
    if not flagged:
        print("  Clean -- no negative bins.")
    return flagged


# ---------------------------------------------------------------------------
# ROOT / datacard writers
# ---------------------------------------------------------------------------

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


def write_datacard(histograms, root_filename, output_dir, label, discriminant_desc,
                    stat_only, jec_included):
    dc_path = os.path.join(output_dir, f"datacard_{label}{'_statonly' if stat_only else ''}.txt")
    procs    = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"]
    proc_idx = [0, 1, 2, 3]
    yields   = {p: histograms[p].sum() for p in procs}
    col_w = 16

    def row(lbl, values):
        return lbl + "".join(f"{v:<{col_w}}" for v in values) + "\n"

    def syst_row(name, stype, vals):
        line = f"{name:<25}{stype:<8}"
        for p in procs:
            line += f"{vals.get(p, '-'):<{col_w}}"
        return line + "\n"

    with open(dc_path, "w") as dc:
        dc.write(f"# H+c -> ZZ -> 4l: {discriminant_desc}, ctag2d SF, NO Z+X"
                  f"{' (stat-only)' if stat_only else ''}\n")
        dc.write(f"# Mass window: {MASS_WINDOW[0]}-{MASS_WINDOW[1]} GeV  |  Lumi: {TOTAL_LUMI_PB:.1f} pb^-1\n")
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
            dc.write(syst_row("lumi_Run3",     "lnN", {p: "1.014" for p in procs}))
            dc.write(syst_row("QCDscale_gg",   "lnN", {"ggZZ": "1.039", "Signal": "1.039"}))
            dc.write(syst_row("pdf_gg",        "lnN", {"ggZZ": "1.032", "Signal": "1.032"}))
            dc.write(syst_row("kfactor_ggZZ",  "lnN", {"ggZZ": "1.10"}))
            dc.write(syst_row("BR_HZZ4l",      "lnN", {"Signal": "1.02", "Other_Higgs": "1.02"}))
            dc.write(syst_row("QCDscale_qqZZ", "lnN", {"qqZZ": "1.04"}))

            dc.write(f"\n# Shape systematics from per-event weight variations\n")
            for syst in SYSTEMATICS:
                vals = {p: "1" for p in procs if syst in USABLE_SYST.get(p, set())}
                if not vals:
                    continue
                dc.write(syst_row(syst, "shape", vals))

            if jec_included:
                jec_rows_written = False
                for syst in JECSHIFTS_SYSTEMATICS:
                    vals = {
                        p: "1" for p in procs
                        if f"{p}_{syst}Up" in histograms and f"{p}_{syst}Down" in histograms
                    }
                    if not vals:
                        continue
                    if not jec_rows_written:
                        dc.write(f"\n# JES/JER object-shift shape systematics\n")
                        jec_rows_written = True
                    dc.write(syst_row(syst, "shape", vals))

    print(f"  Datacard:  {dc_path}")
    return dc_path


def print_summary(histograms, label):
    procs = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"]
    print(f"\n[{label}] Process yields in [{MASS_WINDOW[0]},{MASS_WINDOW[1]}] GeV")
    total_bkg = 0
    for p in procs:
        y = histograms[p].sum()
        if p != "Signal":
            total_bkg += y
        print(f"  {p:<13} {y:>12.6f}")
    sig = histograms["Signal"].sum()
    if total_bkg > 0:
        print(f"  S/sqrt(B) = {sig/total_bkg**0.5:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scored-dir",
        default="/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored_v8",
    )
    parser.add_argument(
        "--sumw-dir",
        default="/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d",
    )
    parser.add_argument(
        "--jecshifts-dir",
        default=None,
        help="Off by default (2026-09-09 memory-constrained pass): the jecshifts "
             "tree is 19GB compressed (vs 3.2GB nominal) and reading 4 score "
             "columns from it OOM'd this shared host. Pass explicitly to include "
             "JES/JER once memory allows; without it this matches the older "
             "no-JES/JER reference (r=411.0/kappa_c=73.93), not the r=412.0 "
             "headline.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-bins", type=int, default=20, help="plain/kappa 1D quantile bins")
    parser.add_argument("--ratio-bins-per-axis", type=str, default="2,2,5")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    n1, n2, n3 = [int(x) for x in args.ratio_bins_per_axis.split(",")]

    print("=" * 70)
    print("H+c -> ZZ -> 4l  |  plain vs kappa vs 3-ratio comparison (ctag2d + JES/JER)")
    print(f"Mass window : {MASS_WINDOW[0]}-{MASS_WINDOW[1]} GeV")
    print(f"Bins        : plain/kappa={args.n_bins}, ratio={n1}x{n2}x{n3}={n1*n2*n3}")
    print("=" * 70)

    print(f"\n[1] sumw\n    Dir: {args.sumw_dir}")
    sumw_by_proc = load_sumw(args.sumw_dir)

    print(f"\n[2] MC scored parquets\n    Dir: {args.scored_dir}")
    mc_data = load_mc_scored_parquets(args.scored_dir, sumw_by_proc)

    if args.jecshifts_dir:
        print(f"\n[3] JES/JER object-shift systematics\n    Dir: {args.jecshifts_dir}")
        jec_data = load_jecshifts_scored_parquets(args.jecshifts_dir, sumw_by_proc)
    else:
        print("\n[3] JES/JER object-shift systematics: SKIPPED (--jecshifts-dir not set, "
              "memory-constrained pass -- matches r=411.0/kappa_c=73.93 reference, not r=412.0)")
        jec_data = None

    print("\n[4] Kappa optimization (fresh, this run's real yields)")
    kappas = optimize_kappa(mc_data)

    # --- plain ---
    ref_scores, ref_weights, _ = mc_data["qqZZ"]
    plain_edges = compute_quantile_bins_1d(plain_score(ref_scores), ref_weights, n_bins=args.n_bins)
    print(f"\n[5a] Plain edges ({len(plain_edges)-1} bins): {np.round(plain_edges, 4)}")
    hist_plain = build_histograms_1d(mc_data, jec_data, plain_score, plain_edges)

    # --- kappa ---
    kappa_fn = lambda s4: kappa_score(s4, kappas)
    kappa_edges = compute_quantile_bins_1d(kappa_fn(ref_scores), ref_weights, n_bins=args.n_bins)
    print(f"\n[5b] Kappa edges ({len(kappa_edges)-1} bins): {np.round(kappa_edges, 4)}")
    hist_kappa = build_histograms_1d(mc_data, jec_data, kappa_fn, kappa_edges)

    # --- 3-ratio ---
    ref_ratios = compute_ratios(ref_scores)
    tree = build_nested_tree(ref_ratios, ref_weights, n1, n2, n3)
    hist_ratio, n_bins_ratio = build_histograms_ratio(mc_data, jec_data, tree, n1, n2, n3)

    variants = {
        "plain": (hist_plain, len(plain_edges) - 1, "plain mva_score_Signal, quantile-binned"),
        "kappa": (hist_kappa, len(kappa_edges) - 1, "kappa-weighted Ds, quantile-binned"),
        "ratio": (hist_ratio, n_bins_ratio, "3-ratio joint (r1,r2,r3), nested-tree-binned"),
    }

    print("\n[6] Writing outputs + running negative-bin scan")
    for label, (hist, nb, desc) in variants.items():
        print_summary(hist, label)
        scan_negative_bins(hist, label)
        root_file = write_root_file(hist, nb, args.output, f"histograms_{label}.root")
        write_datacard(hist, os.path.basename(root_file), args.output, label, desc, stat_only=False, jec_included=(jec_data is not None))
        write_datacard(hist, os.path.basename(root_file), args.output, label, desc, stat_only=True, jec_included=(jec_data is not None))

    print(f"\n[Done] Kappa=({kappas[0]:.4f},{kappas[1]:.4f},{kappas[2]:.4f})  All files in: {args.output}")


if __name__ == "__main__":
    main()
