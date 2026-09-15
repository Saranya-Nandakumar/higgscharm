#!/usr/bin/env python3
"""
Rebuild the [100,150] ctag2d datacards using the OFFICIAL flavTaggingSF
up_Total/down_Total values AS-IS -- i.e. WITHOUT the 2026-09-02
swap-correction for WP=L0 (now reverted in analysis/corrections/ctag2d.py,
2026-09-15, pending POG/BTV confirmation -- see the revert writeup and
second-brain memory hczz_ctag2d_l0_updown_swap_fix).

The existing hplusc_mva_4class_ctag2d_scored_v8 production parquets still
carry the SWAP-CORRECTED weight_CMS_ctag2d_<year>Up/Down baked in (from when
the corrector's fix was live) -- reprocessing all 4 eras through condor to
get "clean" parquets would take hours to days. Instead, this script
RECOMPUTES the CMS_ctag2d Up/Down weight directly from the real jet-level
branches already stored in the same scored parquets (jet_btagPNetCvL,
jet_btagPNetCvB, jet_hadronFlavour, jet_pt -- present since the ctag2d
migration), using the exact same 3-leading-jet cap and category logic as
CTag2DCorrector, but reading up_Total/down_Total raw (no swap). Every other
weight/systematic/process is untouched -- this only overrides the single
CMS_ctag2d column.

Imports the real create_datacards_ctag2d_100150.py module and monkey-patches
only load_mc_scored_parquets(); everything else (ZX loading, JES/JER
loading, binning, datacard/ROOT writing) is the real, unmodified pipeline.

Usage: identical CLI to create_datacards_ctag2d_100150.py.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict

sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/scripts")
sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm")
import create_datacards_ctag2d_100150 as dc
from analysis.corrections.ctag2d import (
    _category_np, CID, CAT_TO_WP, BOUNDARIES_BY_ERA, CORRECTION_NAME, PT_BINS,
)
import correctionlib
from analysis.corrections.utils import correction_files

ERAS = dc.ERAS
CLASS_NAMES = dc.CLASS_NAMES
MASS_WINDOW = dc.MASS_WINDOW
USABLE_SYST = dc.USABLE_SYST
syst_col = dc.syst_col
get_process_from_path = dc.get_process_from_path
EXPECTED_YIELDS = dc.EXPECTED_YIELDS

_CORR_CACHE = {}


def _get_corr(year_dir_era):
    if year_dir_era not in _CORR_CACHE:
        _CORR_CACHE[year_dir_era] = correctionlib.CorrectionSet.from_file(
            correction_files["ctagging_2d"][year_dir_era]
        )[CORRECTION_NAME]
    return _CORR_CACHE[year_dir_era]


def _raw_event_sf(cvl, cvb, flav, pt, era, syst):
    """Per-jet raw (unswapped) SF for one syst ('central'/'up_Total'/'down_Total'),
    identical to CTag2DCorrector._eval_flat but with no correction applied."""
    corr = _get_corr(era)
    cat = _category_np(cvl, cvb, era)
    wp = np.where(cat >= 0, CAT_TO_WP[np.clip(cat, 0, 10)], -1)
    ok = (wp >= 0) & np.isfinite(pt)
    sf = np.ones(len(flav), dtype=np.float64)
    if ok.any():
        sf[ok] = corr.evaluate(
            syst, flav[ok], wp[ok],
            np.zeros(ok.sum(), dtype=np.float64),
            np.clip(pt[ok], *PT_BINS),
        )
    return sf


def _recompute_ctag2d_raw(df, era):
    """Returns (weight_up_raw, weight_dn_raw) full-event-weight arrays (same
    convention as the stored weight_CMS_ctag2d_<year>Up/Down columns:
    weight_nominal already has the central ctag2d SF folded in, and the
    Up/Down columns are the FULL alternate weight, not just a ratio)."""
    n = len(df)
    nj_cap = 3
    cvl_list = df["jet_btagPNetCvL"].values
    cvb_list = df["jet_btagPNetCvB"].values
    flav_list = df["jet_hadronFlavour"].values
    pt_list = df["jet_pt"].values

    nj = np.array([len(x[:nj_cap]) for x in cvl_list])
    cvl = np.concatenate([np.asarray(x[:nj_cap], dtype=np.float64) for x in cvl_list]) if n else np.array([])
    cvb = np.concatenate([np.asarray(x[:nj_cap], dtype=np.float64) for x in cvb_list]) if n else np.array([])
    flav = np.concatenate([np.asarray(x[:nj_cap], dtype=np.int64) for x in flav_list]) if n else np.array([])
    pt = np.concatenate([np.asarray(x[:nj_cap], dtype=np.float64) for x in pt_list]) if n else np.array([])

    sf_c_flat = _raw_event_sf(cvl, cvb, flav, pt, era, "central")
    sf_up_flat = _raw_event_sf(cvl, cvb, flav, pt, era, "up_Total")
    sf_dn_flat = _raw_event_sf(cvl, cvb, flav, pt, era, "down_Total")

    def to_event(flat):
        out = np.ones(n, dtype=np.float64)
        idx = 0
        for i, k in enumerate(nj):
            if k > 0:
                out[i] = np.prod(flat[idx:idx + k])
            idx += k
        return out

    sf_c = to_event(sf_c_flat)
    sf_up = to_event(sf_up_flat)
    sf_dn = to_event(sf_dn_flat)

    weight_nominal = df["weight_nominal"].fillna(0).values
    r_up = np.where(sf_c != 0, sf_up / sf_c, 1.0)
    r_dn = np.where(sf_c != 0, sf_dn / sf_c, 1.0)
    return weight_nominal * r_up, weight_nominal * r_dn


def load_mc_scored_parquets_rawsf(scored_dir, sumw_by_proc):
    """Copy of dc.load_mc_scored_parquets, patched to recompute CMS_ctag2d
    Up/Down from raw jet branches instead of trusting the swap-corrected
    baked-in parquet columns."""
    raw_weight = defaultdict(float)
    proc_scores = defaultdict(list)
    proc_weights = defaultdict(list)
    proc_syst_weights = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

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

            need_jet_cols = "CMS_ctag2d" in usable
            jet_cols = ["jet_btagPNetCvL", "jet_btagPNetCvB", "jet_hadronFlavour", "jet_pt"] if need_jet_cols else []

            for pq_file in glob.glob(os.path.join(sample_dir, "*.parquet")):
                base_cols = ["zz_mass_inclusive", "mva_score_Signal", "weight_nominal"]
                all_cols = base_cols + syst_cols_wanted + jet_cols
                try:
                    df = pq.read_table(pq_file, columns=all_cols).to_pandas()
                except Exception:
                    try:
                        available = pq.ParquetFile(pq_file).schema.names
                        cols = base_cols + [c for c in syst_cols_wanted if c in available] + \
                               [c for c in jet_cols if c in available]
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

                proc_scores[proc].append(df_win["mva_score_Signal"].values)
                nominal_win = df_win["weight_nominal"].fillna(0).values
                proc_weights[proc].append(nominal_win)

                ctag2d_up_raw = ctag2d_dn_raw = None
                if need_jet_cols and all(c in df_win.columns for c in jet_cols):
                    ctag2d_up_raw, ctag2d_dn_raw = _recompute_ctag2d_raw(df_win, era)

                for syst in usable:
                    for direction in ("Up", "Down"):
                        col = syst_col(syst, direction, era)
                        if syst == "CMS_ctag2d" and ctag2d_up_raw is not None:
                            arr = ctag2d_up_raw if direction == "Up" else ctag2d_dn_raw
                            proc_syst_weights[proc][syst][direction].append(arr)
                        elif col in df_win.columns:
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

        scores = np.concatenate(proc_scores[proc]) if proc_scores[proc] else np.array([])
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

        mc_data[proc] = (scores, scaled_weights, syst_scaled)
        in_win = scaled_weights.sum()
        print(f"  {proc:<15} {raw:>12.2f} {sumw_gen:>14.2f} {baseline_eff:>13.4%} {exp:>10.4f} {in_win:>14.4f}")

    return mc_data


# monkey-patch: main() calls load_mc_scored_parquets via the module's own
# global namespace, so overriding this attribute redirects that call.
dc.load_mc_scored_parquets = load_mc_scored_parquets_rawsf


if __name__ == "__main__":
    print("=" * 70)
    print("RAW-SF REBUILD: CMS_ctag2d Up/Down read from the official file")
    print("as-is (NO swap-correction) -- recomputed from jet branches in the")
    print("existing v8 scored parquets, everything else unchanged.")
    print("=" * 70)
    dc.main()
