#!/usr/bin/env python3
"""
[100,150] ctag2d datacards with CMS_ctag2d DECORRELATED into its sources
(2026-09-24, following Livio Redard-Jacot's suggestion to use the 33
per-flavour/WP statistical uncertainties).

Replaces the single CMS_ctag2d (official up/down_Total) shape nuisance with:
  - CMS_ctag2d_stat_<flavX>_<WP>_<era>: the 33 `Stat_flav{L,C,B}_{L0,C0..C4,B0..B4}`
    sources of flavTaggingSF_<era>.json.gz, one nuisance per source and era
    (independent calibrations -> decorrelated across eras). Each source is
    evaluated on every jet, so its propagation into L0 (flavour-normalisation
    constraint, opposite direction) stays inside the same nuisance.
  - CMS_ctag2d_resid: the part of Total not covered by the 33 stat sources,
    per jet r = sign(Total shift) * sqrt(max(Total_shift^2 - sum_g stat_g^2, 0)),
    separately for up and down; correlated across eras (it is dominated by the
    systematic sources: JES, XSec_*, PU, ...).
In the file: each grouped Stat_flav*_* key == linear sum of its per-pT-bin keys,
aggregate up/down_Stat == quadrature of the 33, and Total is on median ~25%
larger than the quadrature of all exposed sources, hence the residual rather
than the named sources.

Nuisances whose Up and Down change no process's template by more than
--prune-threshold (relative, per bin, of the process's nominal yield) are dropped.

Everything else is create_datacards_ctag2d_100150_rawsf.py unchanged (it imports
the real create_datacards_ctag2d_100150.py and patches only the MC loader).
Usage: identical CLI to create_datacards_ctag2d_100150.py plus --prune-threshold.
"""
import os
import re
import sys
import glob
import gzip
import json
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
import eos_io as E   # server-side EOS listings, xrootd read fallback

sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/scripts")
sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm")
import create_datacards_ctag2d_100150 as dc
import create_datacards_ctag2d_100150_rawsf as rs
from analysis.corrections.ctag2d import _category_np, CAT_TO_WP, CORRECTION_NAME, PT_BINS, correction_name
from analysis.corrections.utils import correction_files

ERAS = dc.ERAS
JET_COLS = ["jet_btagPNetCvL", "jet_btagPNetCvB", "jet_hadronFlavour", "jet_pt"]
RESID = "CMS_ctag2d_resid"

PRUNE_THRESHOLD = 1e-3
for i, a in enumerate(sys.argv):
    if a == "--prune-threshold":
        PRUNE_THRESHOLD = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        break


def _stat_groups(era):
    with gzip.open(correction_files["ctagging_2d"][era]) as fh:
        corr = [c for c in json.load(fh)["corrections"] if c["name"] == correction_name(era)][0]
    keys = [k["key"] for k in corr["data"]["content"]]
    return sorted(k[3:] for k in keys if re.fullmatch(r"up_Stat_flav[LCB]_[LCB]\d", k))


STAT_GROUPS = {era: _stat_groups(era) for era in ERAS}


def stat_name(group, era):
    return f"CMS_ctag2d_stat_{group[5:]}_{era}"   # Stat_flavC_C0 -> CMS_ctag2d_stat_flavC_C0_<era>


NEW_SYST = [stat_name(g, era) for era in ERAS for g in STAT_GROUPS[era]] + [RESID]


def _jet_sfs(cvl, cvb, flav, pt, era):
    """Per-jet central SF and, per new nuisance, the (up, down) per-jet SF."""
    ev = lambda s: rs._raw_event_sf(cvl, cvb, flav, pt, era, s)
    c = ev("central")
    tu, td = ev("up_Total") - c, ev("down_Total") - c
    out = {}
    su2 = np.zeros_like(c)
    sd2 = np.zeros_like(c)
    for g in STAT_GROUPS[era]:
        gu, gd = ev("up_" + g) - c, ev("down_" + g) - c
        su2 += gu ** 2
        sd2 += gd ** 2
        out[stat_name(g, era)] = (c + gu, c + gd)
    ru = np.sign(tu) * np.sqrt(np.clip(tu ** 2 - su2, 0, None))
    rd = np.sign(td) * np.sqrt(np.clip(td ** 2 - sd2, 0, None))
    out[RESID] = (c + ru, c + rd)
    return c, out


def _event_ratios(df, era):
    """{nuisance: (ratio_up, ratio_down)} per event, product over the 3 leading jets."""
    n = len(df)
    cols = rs.jet_score_cols(era)   # PNet 2022/2023, UParTAK4 2024/2025
    lists = {col: [np.asarray(x[:3]) for x in df[col].values] for col in cols}
    nj = np.array([len(x) for x in lists["jet_pt"]])
    ev = np.repeat(np.arange(n), nj)
    cat = lambda col, dt: np.concatenate(lists[col]).astype(dt) if nj.sum() else np.array([], dt)
    cvl, cvb = cat(cols[0], np.float64), cat(cols[1], np.float64)
    flav, pt = cat("jet_hadronFlavour", np.int64), cat("jet_pt", np.float64)
    c, sfs = _jet_sfs(cvl, cvb, flav, pt, era)
    safe_c = np.where(c > 0, c, 1.0)

    def to_event(sf):
        lr = np.log(np.clip(np.where(c > 0, sf / safe_c, 1.0), 1e-6, None))
        return np.exp(np.bincount(ev, lr, n))

    return {name: (to_event(u), to_event(d)) for name, (u, d) in sfs.items()}


def load_mc_scored_parquets_decorr(scored_dir, sumw_by_group):
    """rawsf loader with CMS_ctag2d replaced by NEW_SYST (other systs untouched)."""
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
            proc = dc.get_process_from_path(sample_name)
            if proc is None:
                continue
            sample_dir = os.path.join(era_dir, sample_name, "base")
            if not E.isdir(sample_dir):
                continue
            scale = dc.event_scale(era, sample_name, sumw_by_group)
            usable = dc.USABLE_SYST.get(proc, set())
            has_ctag = RESID in usable   # CMS_ctag2d itself was swapped for NEW_SYST at import
            old_syst = [s for s in usable if s != "CMS_ctag2d" and s not in NEW_SYST]
            is_hb = (proc != "Signal" and "HPlusBottom" in sample_name)
            requestable = [s for s in old_syst if not (is_hb and s in dc.HPLUSBOTTOM_BUGGY_SYST)]
            syst_cols = [dc.syst_col(s, d, era) for s in requestable for d in ("Up", "Down")]
            jet_cols = rs.jet_score_cols(era) if has_ctag else []

            for pq_file in E.files(sample_dir, ".parquet"):
                base_cols = ["zz_mass_inclusive", "mva_score_Signal", "weight_nominal"]
                df = E.read_parquet(pq_file, base_cols + syst_cols + jet_cols).to_pandas()   # failures raise

                raw_weight[proc] += df["weight_nominal"].fillna(0).values.sum() * scale
                mask = (df["zz_mass_inclusive"] >= dc.MASS_WINDOW[0]) & \
                       (df["zz_mass_inclusive"] < dc.MASS_WINDOW[1]) & \
                       (df["mva_score_Signal"] >= 0)
                df_win = df[mask]
                if len(df_win) == 0:
                    continue
                proc_scores[proc].append(df_win["mva_score_Signal"].values)
                nominal = df_win["weight_nominal"].fillna(0).values * scale
                proc_weights[proc].append(nominal)

                for syst in old_syst:
                    for d in ("Up", "Down"):
                        col = dc.syst_col(syst, d, era)
                        if is_hb and syst in dc.HPLUSBOTTOM_BUGGY_SYST:
                            arr = nominal
                        elif col in df_win.columns:
                            arr = df_win[col].fillna(0).values * scale
                        else:
                            arr = nominal
                        proc_syst_weights[proc][syst][d].append(arr)

                if has_ctag:
                    if all(c in df_win.columns for c in jet_cols):
                        ratios = _event_ratios(df_win, era)
                    else:
                        print(f"  WARNING: {pq_file}: no jet columns, ctag2d variations = nominal")
                        ratios = {}
                    for name in NEW_SYST:
                        ru, rd = ratios.get(name, (None, None))
                        proc_syst_weights[proc][name]["Up"].append(nominal if ru is None else nominal * ru)
                        proc_syst_weights[proc][name]["Down"].append(nominal if rd is None else nominal * rd)
            print(f"  loaded {era} {sample_name} -> {proc}", flush=True)

    return dc.finalize_mc_data(proc_scores, proc_weights, proc_syst_weights, raw_weight)


# --- register the new nuisances, drop the single CMS_ctag2d ------------------
dc.load_mc_scored_parquets = load_mc_scored_parquets_decorr
del dc.SYSTEMATICS["CMS_ctag2d"]
for name in NEW_SYST:
    dc.SYSTEMATICS[name] = {"col": None, "era_dep": False}   # never read from parquet
for proc, usable in dc.USABLE_SYST.items():
    if "CMS_ctag2d" in usable:
        dc.USABLE_SYST[proc] = (usable - {"CMS_ctag2d"}) | set(NEW_SYST)

# --- pruning: wrap build_histograms, drop negligible nuisances ----------------
_orig_build = dc.build_histograms


def build_histograms_pruned(mc_data, zx_scores, zx_weights, jec_data=None):
    hists, hvar = _orig_build(mc_data, zx_scores, zx_weights, jec_data)
    procs = [p for p in dc.CLASS_NAMES if p in hists]
    kept, dropped = [], []
    report = []
    for name in NEW_SYST:
        worst, worst_int = 0.0, {}
        for p in procs:
            nom = hists[p]
            tot = nom.sum()
            if f"{p}_{name}Up" not in hists or tot <= 0:
                continue
            up, dn = hists[f"{p}_{name}Up"], hists[f"{p}_{name}Down"]
            worst = max(worst, np.abs(up - nom).max() / tot, np.abs(dn - nom).max() / tot)
            worst_int[p] = (up.sum() / tot - 1, dn.sum() / tot - 1)
        if not worst_int:
            raise RuntimeError(f"{name}: no Up/Down templates were built -- loader bug")
        report.append((worst, name, worst_int))
        if worst > PRUNE_THRESHOLD:
            kept.append(name)
        else:
            dropped.append(name)
            for p in procs:
                for d in ("Up", "Down"):
                    hists.pop(f"{p}_{name}{d}", None)
                    hvar.pop(f"{p}_{name}{d}", None)
            del dc.SYSTEMATICS[name]
            for p in dc.USABLE_SYST:
                dc.USABLE_SYST[p] = dc.USABLE_SYST[p] - {name}
    print(f"\n  ctag2d decorrelation: {len(kept)} nuisances kept, {len(dropped)} pruned "
          f"(max per-bin |shift|/yield <= {PRUNE_THRESHOLD})")
    print(f"  {'nuisance':<44} {'max bin shift':>13}   integral Up/Down per process")
    for worst, name, wi in sorted(report, reverse=True)[:40]:
        s = "  ".join(f"{p[:5]} {u:+.3f}/{d:+.3f}" for p, (u, d) in wi.items())
        print(f"  {name:<44} {worst:>13.4f}   {s}")
    return hists, hvar


dc.build_histograms = build_histograms_pruned

_write_datacard_orig = dc.write_datacard


def write_datacard_fixed(*a, **k):
    """dc.write_datacard pads names to 25 chars with no separator, so the longer
    CMS_ctag2d_stat_* names ran into the 'shape' column (text2workspace:
    'Unsupported pdf 1'). Re-insert the space after writing.
    (The JES/JER yield check now lives in dc.write_datacard itself.)"""
    path = _write_datacard_orig(*a, **k)
    txt = open(path).read()
    fixed = re.sub(r"^(CMS_ctag2d_\S+?)(shape)(\s)", r"\1 \2\3", txt, flags=re.M)
    open(path, "w").write(fixed)
    return path


dc.write_datacard = write_datacard_fixed


if __name__ == "__main__":
    print("=" * 70)
    print("CTAG2D DECORRELATED BUILD: 33 stat sources per era + residual,")
    print(f"prune threshold {PRUNE_THRESHOLD}. Everything else as rawsf.")
    print("=" * 70)
    dc.main()
