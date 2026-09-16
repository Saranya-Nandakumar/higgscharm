#!/usr/bin/env python3
"""
Z+c CR (ztomumu/ztoee, CR_Zplusc category) 2D ctag SF validation check.

Purpose: an independent data/MC check of whether the official PNet 2D
pseudo-continuous c-tag SF (CTag2DCorrector, analysis/corrections/ctag2d.py)
improves agreement in a genuine control region -- orthogonal to both the SR
and the MVA-score fit discriminant (second-brain memory
hczz_cr_sideband_strategy). No new condor production: reuses the existing
CR_Zplusc parquets.

SCOPE CAVEAT (deliberate, documented): the SR's CTag2DCorrector evaluates a
2D WP-code category (L0/C0-C4/B0-B4) per jet across the up-to-3-leading jets
of ANY flavor in `events.selected_jets`, then takes ak.prod across them for
one per-event multiplicative SF. ztomumu.yaml/ztoee.yaml's CR_Zplusc category
does NOT store per-jet CvL/CvB/hadronFlavour/pt for the full `jets` collection
(only for the `cjets` subset -- jets already passing `jet_ctagging(loose)` --
and separately a single leading-jet scalar). Confirmed directly by reading
the real parquet schema (2026-09-16): jet_pt/jet_eta/jet_phi exist as full
arrays but no jet_cvsl/jet_cvsb/jet_flavour array columns exist, so the full
SR scope cannot be reconstructed without a new condor campaign. This script
instead evaluates the SF on the `cjets` (already-tagged) subset only --
per user direction 2026-09-16 ("use the c-jets category and apply the SFs")
-- and flattens per-jet (not per-event ak.prod), since the point here is a
per-WP-category diagnostic, not a final per-event analysis yield.

For every cjets entry (flattened over events, all 4 eras, both channels):
  1. category(CvL, CvB) -> one of {L0,C0-C4,B0-B4} via the real
     analysis.corrections.ctag2d._category_np (identical function used in
     production).
  2. Data: unweighted (weight=1) counts per category -- real data carries no
     MC weight of any kind.
  3. MC, no SF: weight_nominal per jet entry (weight_nominal already excludes
     ctag2d -- confirmed: ztomumu.yaml/ztoee.yaml have no `ctagging_2d` key
     under corrections.event_weights).
  4. MC, with SF: weight_nominal * central per-jet SF (CTag2DCorrector's own
     `_eval_flat(central)`, called on this jet's own (CvL,CvB,flavour,pt) --
     NOT the full-event product).

Two SEPARATE histograms per channel (as requested): Data vs MC(no SF), and
Data vs MC(with SF), same category binning -- to see by eye whether applying
the SF moves each bin's MC central value closer to or further from data.

MC normalization: per-individual-dataset xs*lumi/sumw (dataset xsec/era from
analysis/filesets/<era>_nanov12.yaml via get_dataset_config, sumw aggregated
from the real per-job-split sumw/*.json sidecars already on disk) --
identical convention to create_datacards_ctag2d.py's load_sumw, but at
per-dataset (not per-group) granularity since e.g. DY 10to50/50-inf or
WW/WZ/ZZ each have their own xsec.

Usage:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    python3 create_cr_zplusc_ctag2d_check.py
"""
import os
import re
import sys
import glob
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from collections import defaultdict

HIGGSCHARM_DIR = "/afs/cern.ch/user/s/snandaku/Higgscharmnew/higgscharm"
sys.path.insert(0, HIGGSCHARM_DIR)
os.chdir(HIGGSCHARM_DIR)  # get_dataset_config() resolves fileset yamls via Path.cwd()

from analysis.corrections.ctag2d import _category_np, CAT_TO_WP, CORRECTION_NAME, PT_BINS
from analysis.corrections.utils import correction_files
from analysis.filesets.utils import get_dataset_config
import correctionlib

ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
CHANNELS = ["ztomumu", "ztoee"]
CATEGORY = "CR_Zplusc"
OUTPUT_BASE = "/eos/user/s/snandaku/higgscharm/outputs"

CATS = ["L0", "C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2", "B3", "B4"]
CAT_IDX = {n: i for i, n in enumerate(CATS)}

with open(os.path.join(HIGGSCHARM_DIR, "analysis/postprocess/luminosity.yaml")) as f:
    LUMI = yaml.safe_load(f)

OUT_JSON = "/tmp/snandaku/claude-178308/-eos-home-s-snandaku-second-brain/02880980-8f56-408d-9125-9511cc3bb4ee/scratchpad/cr_zplusc_ctag2d_check.json"

_corr_cache = {}
_dataset_config_cache = {}


def get_corr(era):
    if era not in _corr_cache:
        _corr_cache[era] = correctionlib.CorrectionSet.from_file(
            correction_files["ctagging_2d"][era]
        )[CORRECTION_NAME]
    return _corr_cache[era]


def get_dscfg(era):
    if era not in _dataset_config_cache:
        _dataset_config_cache[era] = get_dataset_config(era)
    return _dataset_config_cache[era]


def resolve_dataset(sample_name, dscfg):
    """sample_name is a directory like 'DYJetsToLL_10to50_13' or 'MuonE_7' --
    strip the trailing job-split '_<int>' and match against the era's real
    fileset registry. Returns (dataset_key, entry) or (None, None)."""
    stripped = re.sub(r"_\d+$", "", sample_name)
    for candidate in (sample_name, stripped):
        if candidate in dscfg:
            return candidate, dscfg[candidate]
    return None, None


def eval_sf_flat(cvl, cvb, flav, pt, era, syst="central"):
    corr = get_corr(era)
    cat = _category_np(cvl, cvb, era)
    wp = np.where(cat >= 0, CAT_TO_WP[np.clip(cat, 0, 10)], -1)
    ok = (wp >= 0) & np.isfinite(pt)
    sf = np.ones(len(flav), dtype=np.float64)
    if ok.any():
        sf[ok] = corr.evaluate(
            syst,
            flav[ok].astype(np.int64),
            wp[ok],
            np.zeros(ok.sum(), dtype=np.float64),
            np.clip(pt[ok], *PT_BINS),
        )
    return sf, cat


def load_sumw_for_dataset(era_dir, sample_name):
    sumw_glob = os.path.join(era_dir, sample_name, "sumw", "*.json")
    total = 0.0
    n = 0
    for jf in glob.glob(sumw_glob):
        try:
            with open(jf) as f:
                rec = json.load(f)
            total += rec["sumw"]
            n += 1
        except Exception as e:
            print(f"    WARNING: bad sumw file {jf}: {e}")
    return total, n


def process_channel(channel):
    era_dir_base = os.path.join(OUTPUT_BASE, channel)
    cat_hist_data = np.zeros(len(CATS))
    cat_hist_mc_nosf = defaultdict(lambda: np.zeros(len(CATS)))   # keyed by group ("dy_nlo","tt",...)
    cat_hist_mc_withsf = defaultdict(lambda: np.zeros(len(CATS)))
    n_data_files = n_mc_files = n_read_err = n_no_dataset_match = 0

    # ------------------------------------------------------------------
    # Pass 1: resolve every sample dir to its base dataset + accumulate
    # sumw across ALL job-split dirs sharing that dataset, per era.
    # ------------------------------------------------------------------
    sample_meta = {}  # (era, sample_name) -> (dataset_key, entry)
    sumw_by_era_dataset = defaultdict(float)  # (era, dataset_key) -> sumw

    for era in ERAS:
        era_dir = os.path.join(era_dir_base, era)
        if not os.path.isdir(era_dir):
            print(f"  WARNING: era dir not found: {era_dir}")
            continue
        dscfg = get_dscfg(era)

        for sample_name in sorted(os.listdir(era_dir)):
            sample_cat_dir = os.path.join(era_dir, sample_name, CATEGORY)
            if not os.path.isdir(sample_cat_dir):
                continue
            dataset_key, entry = resolve_dataset(sample_name, dscfg)
            if entry is None:
                n_no_dataset_match += 1
                continue
            sample_meta[(era, sample_name)] = (dataset_key, entry)
            if entry.get("process") != "Data":
                sumw, _ = load_sumw_for_dataset(era_dir, sample_name)
                sumw_by_era_dataset[(era, dataset_key)] += sumw

    print(f"  [{channel}] resolved {len(sample_meta)} sample dirs, "
          f"{n_no_dataset_match} unresolved, "
          f"{len(sumw_by_era_dataset)} (era,dataset) sumw entries")

    # ------------------------------------------------------------------
    # Pass 2: read parquets, scale MC by xsec*lumi/sumw, fill histograms.
    # ------------------------------------------------------------------
    for era in ERAS:
        era_dir = os.path.join(era_dir_base, era)
        if not os.path.isdir(era_dir):
            continue
        lumi = LUMI[era]

        for sample_name in sorted(os.listdir(era_dir)):
            meta = sample_meta.get((era, sample_name))
            if meta is None:
                continue
            dataset_key, entry = meta
            is_data = entry.get("process") == "Data"
            group = entry.get("key")
            if isinstance(group, list):  # e.g. ZZ: key: [diboson, zz] -- use the primary group
                group = group[0] if group else "other"

            scale = 1.0
            if not is_data:
                sumw = sumw_by_era_dataset.get((era, dataset_key), 0.0)
                xsec = entry.get("xsec")
                if not sumw or xsec is None:
                    print(f"    WARNING: no usable sumw/xsec for {era}/{dataset_key} "
                          f"(sumw={sumw}, xsec={xsec}) -- skipping {sample_name}")
                    continue
                scale = xsec * lumi / sumw

            sample_cat_dir = os.path.join(era_dir, sample_name, CATEGORY)
            pq_files = glob.glob(os.path.join(sample_cat_dir, "*.parquet"))
            for pq_file in pq_files:
                cols = ["cjets_pt", "cjets_cvsl", "cjets_cvsb", "cjets_flavour"]
                if not is_data:
                    cols.append("weight_nominal")
                try:
                    df = pq.read_table(pq_file, columns=cols).to_pandas()
                except Exception as e:
                    n_read_err += 1
                    continue

                if is_data:
                    n_data_files += 1
                else:
                    n_mc_files += 1

                pt_list = df["cjets_pt"].values
                cvl_list = df["cjets_cvsl"].values
                cvb_list = df["cjets_cvsb"].values
                flav_list = df["cjets_flavour"].values
                nj = np.array([len(x) for x in pt_list])
                if nj.sum() == 0:
                    continue

                pt = np.concatenate([np.asarray(x, dtype=np.float64) for x in pt_list])
                cvl = np.concatenate([np.asarray(x, dtype=np.float64) for x in cvl_list])
                cvb = np.concatenate([np.asarray(x, dtype=np.float64) for x in cvb_list])
                flav = np.concatenate([np.asarray(x, dtype=np.int64) for x in flav_list])

                sf_c, cat = eval_sf_flat(cvl, cvb, flav, pt, era, "central")

                if is_data:
                    for c in cat:
                        if c >= 0:
                            cat_hist_data[c] += 1
                else:
                    wgt_event = df["weight_nominal"].fillna(0).to_numpy() * scale
                    wgt_flat = np.repeat(wgt_event, nj)
                    for c, w, s in zip(cat, wgt_flat, sf_c):
                        if c >= 0:
                            cat_hist_mc_nosf[group][c] += w
                            cat_hist_mc_withsf[group][c] += w * s

    return {
        "data": cat_hist_data,
        "mc_nosf": {g: h.tolist() for g, h in cat_hist_mc_nosf.items()},
        "mc_withsf": {g: h.tolist() for g, h in cat_hist_mc_withsf.items()},
        "diag": {
            "n_data_files": n_data_files, "n_mc_files": n_mc_files,
            "n_read_err": n_read_err, "n_no_dataset_match": n_no_dataset_match,
        },
    }


def main():
    results = {}
    for channel in CHANNELS:
        print(f"\n{'='*90}\nChannel: {channel}\n{'='*90}")
        results[channel] = process_channel(channel)

    print(f"\n{'='*100}\nSUMMARY: cjets WP-category yields, Data vs MC (no SF) vs MC (with SF)\n{'='*100}")
    for channel, res in results.items():
        print(f"\n--- {channel} ---")
        mc_total_nosf = np.zeros(len(CATS))
        mc_total_withsf = np.zeros(len(CATS))
        for g in res["mc_nosf"]:
            mc_total_nosf += np.array(res["mc_nosf"][g])
        for g in res["mc_withsf"]:
            mc_total_withsf += np.array(res["mc_withsf"][g])
        data = res["data"]

        header = f"{'Category':<10}{'Data':>10}{'MC noSF':>12}{'D/MC noSF':>12}{'MC withSF':>12}{'D/MC withSF':>14}"
        print(header)
        print("-" * len(header))
        for i, c in enumerate(CATS):
            d = data[i]
            m0 = mc_total_nosf[i]
            m1 = mc_total_withsf[i]
            r0 = d / m0 if m0 > 0 else float("nan")
            r1 = d / m1 if m1 > 0 else float("nan")
            print(f"{c:<10}{d:>10.0f}{m0:>12.3f}{r0:>12.3f}{m1:>12.3f}{r1:>14.3f}")
        d_tot = data.sum()
        m0_tot = mc_total_nosf.sum()
        m1_tot = mc_total_withsf.sum()
        print("-" * len(header))
        print(f"{'TOTAL':<10}{d_tot:>10.0f}{m0_tot:>12.3f}{d_tot/m0_tot:>12.3f}{m1_tot:>12.3f}{d_tot/m1_tot:>14.3f}")
        print(f"  diag: {res['diag']}")

    out = {"categories": CATS}
    for channel in CHANNELS:
        out[channel] = {
            "data": results[channel]["data"].tolist(),
            "mc_nosf": results[channel]["mc_nosf"],
            "mc_withsf": results[channel]["mc_withsf"],
            "diag": results[channel]["diag"],
        }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote JSON sidecar: {OUT_JSON}")


if __name__ == "__main__":
    main()
