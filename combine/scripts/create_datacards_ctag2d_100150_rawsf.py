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
import eos_io as E   # server-side EOS listings, xrootd read fallback

sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/scripts")
sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm")
import create_datacards_ctag2d_100150 as dc
from analysis.corrections.ctag2d import (
    _category_np, CID, CAT_TO_WP, BOUNDARIES_BY_ERA, CORRECTION_NAME, PT_BINS,
    correction_name, UPART_ERAS,
)
import correctionlib
from analysis.corrections.utils import correction_files

ERAS = dc.ERAS
CLASS_NAMES = dc.CLASS_NAMES
MASS_WINDOW = dc.MASS_WINDOW
USABLE_SYST = dc.USABLE_SYST
HPLUSBOTTOM_BUGGY_SYST = dc.HPLUSBOTTOM_BUGGY_SYST
syst_col = dc.syst_col
get_process_from_path = dc.get_process_from_path
EXPECTED_YIELDS = dc.EXPECTED_YIELDS

_CORR_CACHE = {}

# Per-jet c-tag inputs read from the scored parquets (2026-10-25): PNet for 2022/2023,
# UParTAK4 for 2024/2025 (jet_btagUParTAK4* columns, workflows since 2026-10-25).
PNET_JET_COLS = ["jet_btagPNetCvL", "jet_btagPNetCvB", "jet_hadronFlavour", "jet_pt"]
UPART_JET_COLS = ["jet_btagUParTAK4CvL", "jet_btagUParTAK4CvB", "jet_hadronFlavour", "jet_pt"]


def jet_score_cols(era):
    return UPART_JET_COLS if era in UPART_ERAS else PNET_JET_COLS

# --- capped-cell fix (2026-10-15) ---------------------------------------------------------
# 2022postEE, light jets (hadronFlavour 0), 20 <= pT < 35, all |eta|: up_Total sits at the
# file's [0.3, 3.0] clamp (C0: central 1.034, up 3.000, down 1.029; L0: central 0.980, up 0.300,
# down 0.996) while every named source of the cell is < 1% (quadrature 0.8% C0, 1.1% L0). It is
# the only such cell in the four files (scan: second-brain
# Notes/hczz-ctag2d-placeholder-qqzz-check-2026-10-14.md sec. 6). --ctag2d-capfix replaces that
# up_Total only, keeping the file's sign convention (C0 up, L0 down):
#   none   : file as is (default; reproduces every earlier card)
#   mirror : up = central +- |central - down_Total|   (the measured half of the same cell)
#   <x>    : up = central +- x * central                (e.g. 0.05: conservative cross-check)
CAPFIX = "none"
for _i, _a in enumerate(sys.argv):
    if _a == "--ctag2d-capfix":
        CAPFIX = sys.argv[_i + 1]
        del sys.argv[_i:_i + 2]
        break
if CAPFIX not in ("none", "mirror"):
    float(CAPFIX)   # anything else must be a relative width
CAPFIX_CELL = {"era": "2022postEE", "flav": 0, "wp_sign": {0: -1.0, 40: +1.0}, "pt": (20.0, 35.0)}
CAPFIX_LOG = defaultdict(int)   # (era, wp) -> number of jet evaluations whose up_Total was replaced
if CAPFIX != "none":
    import atexit
    print(f"--ctag2d-capfix {CAPFIX}: {CAPFIX_CELL}")
    atexit.register(lambda: print(f"--ctag2d-capfix {CAPFIX}: up_Total replaced for "
                                  f"{dict(CAPFIX_LOG) or 'NO jets (check!)'}", flush=True))


def _get_corr(year_dir_era):
    if year_dir_era not in _CORR_CACHE:
        _CORR_CACHE[year_dir_era] = correctionlib.CorrectionSet.from_file(
            correction_files["ctagging_2d"][year_dir_era]
        )[correction_name(year_dir_era)]
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
    if CAPFIX != "none" and syst == "up_Total" and era == CAPFIX_CELL["era"]:
        hi = CAPFIX_CELL["pt"][1]
        for wp_code, sign in CAPFIX_CELL["wp_sign"].items():
            m = ok & (flav == CAPFIX_CELL["flav"]) & (wp == wp_code) & (pt < hi)
            if not m.any():
                continue
            args = (flav[m], wp[m], np.zeros(m.sum(), dtype=np.float64), np.clip(pt[m], *PT_BINS))
            c = corr.evaluate("central", *args)
            if not np.allclose(sf[m], 3.0 if sign > 0 else 0.3):
                raise RuntimeError(f"--ctag2d-capfix: {era} light wp {wp_code} pT<{hi} up_Total is not at the "
                                   f"clamp ({np.unique(sf[m])[:5]}): the SF file changed, re-check the cell")
            if CAPFIX == "mirror":
                delta = np.abs(c - corr.evaluate("down_Total", *args))
            else:
                delta = float(CAPFIX) * c
            sf[m] = c + sign * delta
            CAPFIX_LOG[(era, wp_code)] += int(m.sum())
    return sf


def _recompute_ctag2d_raw(df, era):
    """Returns (weight_up_raw, weight_dn_raw) full-event-weight arrays (same
    convention as the stored weight_CMS_ctag2d_<year>Up/Down columns:
    weight_nominal already has the central ctag2d SF folded in, and the
    Up/Down columns are the FULL alternate weight, not just a ratio)."""
    n = len(df)
    nj_cap = 3
    cvl_list = df[jet_score_cols(era)[0]].values
    cvb_list = df[jet_score_cols(era)[1]].values
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


def load_mc_scored_parquets_rawsf(scored_dir, sumw_by_group):
    """Copy of dc.load_mc_scored_parquets, patched to recompute CMS_ctag2d
    Up/Down from raw jet branches instead of trusting the swap-corrected
    baked-in parquet columns."""
    raw_weight = defaultdict(float)
    proc_scores = defaultdict(list)
    proc_weights = defaultdict(list)
    proc_syst_weights = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for era in ERAS:
        era_dir = os.path.join(scored_dir, era)
        if not E.isdir(era_dir):
            print(f"  WARNING: era dir not found: {era_dir}")
            continue

        for sample_name in E.listdir(era_dir):
            proc = get_process_from_path(sample_name)
            if proc is None:
                continue

            sample_dir = os.path.join(era_dir, sample_name, "base")
            if not E.isdir(sample_dir):
                continue

            scale = dc.event_scale(era, sample_name, sumw_by_group)
            usable = USABLE_SYST.get(proc, set())
            # HPlusBottom-specific override -- see dc.HPLUSBOTTOM_BUGGY_SYST.
            # Mirrors the same fix in create_datacards_ctag2d_100150.py's
            # load_mc_scored_parquets (this function is a hand-duplicated
            # copy of it, patched only for CMS_ctag2d recomputation).
            is_hplusbottom = (proc != "Signal" and "HPlusBottom" in sample_name)
            requestable = usable - HPLUSBOTTOM_BUGGY_SYST if is_hplusbottom else usable
            syst_cols_wanted = []
            for syst in requestable:
                for direction in ("Up", "Down"):
                    syst_cols_wanted.append(syst_col(syst, direction, era))

            need_jet_cols = "CMS_ctag2d" in usable
            jet_cols = jet_score_cols(era) if need_jet_cols else []

            for pq_file in E.files(sample_dir, ".parquet"):
                base_cols = ["zz_mass_inclusive", "mva_score_Signal", "weight_nominal"]
                all_cols = base_cols + syst_cols_wanted + jet_cols
                df = E.read_parquet(pq_file, all_cols).to_pandas()   # missing columns dropped; failures raise

                w_all = df["weight_nominal"].fillna(0).values
                raw_weight[proc] += w_all.sum() * scale

                mask = (df["zz_mass_inclusive"] >= MASS_WINDOW[0]) & \
                       (df["zz_mass_inclusive"] < MASS_WINDOW[1])
                df_win = df[mask]
                valid = df_win["mva_score_Signal"] >= 0
                df_win = df_win[valid]
                if len(df_win) == 0:
                    continue

                proc_scores[proc].append(df_win["mva_score_Signal"].values)
                nominal_win = df_win["weight_nominal"].fillna(0).values * scale
                proc_weights[proc].append(nominal_win)

                ctag2d_up_raw = ctag2d_dn_raw = None
                if need_jet_cols and all(c in df_win.columns for c in jet_cols):
                    ctag2d_up_raw, ctag2d_dn_raw = _recompute_ctag2d_raw(df_win, era)
                    ctag2d_up_raw, ctag2d_dn_raw = ctag2d_up_raw * scale, ctag2d_dn_raw * scale

                for syst in usable:
                    for direction in ("Up", "Down"):
                        col = syst_col(syst, direction, era)
                        if syst == "CMS_ctag2d" and ctag2d_up_raw is not None:
                            arr = ctag2d_up_raw if direction == "Up" else ctag2d_dn_raw
                            proc_syst_weights[proc][syst][direction].append(arr)
                        elif is_hplusbottom and syst in HPLUSBOTTOM_BUGGY_SYST:
                            proc_syst_weights[proc][syst][direction].append(nominal_win)
                        elif col in df_win.columns:
                            proc_syst_weights[proc][syst][direction].append(
                                df_win[col].fillna(0).values * scale
                            )
                        else:
                            proc_syst_weights[proc][syst][direction].append(nominal_win)

    return dc.finalize_mc_data(proc_scores, proc_weights, proc_syst_weights, raw_weight)


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
