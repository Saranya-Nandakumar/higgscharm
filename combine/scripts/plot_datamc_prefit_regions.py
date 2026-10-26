#!/usr/bin/env python3
"""
Data/MC prefit plots with the full systematic band, per region (2026-09-24).
Styled on the H->WW+c deck: backgrounds stacked, H+c overlaid (scaled),
hatched prefit total uncertainty, data/MC ratio panel.

Regions (all from the v8 scored `base` category, i.e. SR object selection incl.
>=1 loose c-jet, mva_score_Signal >= 0, same as the datacards):
  onz      : |m4l - 91.1876| < 7 GeV                      -> m4l, MVA score
  qqzz     : m4l > 180 GeV                                -> m4l, MVA score
  sideband : 100 <= m4l < 150, 115 <= m4l < 135 BLINDED   -> m4l (data blind 115-135), MVA score (sidebands only)
  sr       : 100 <= m4l < 150, MVA score in the production quantile bins, data
             shown only outside the top --n-blind bins

Predictions are the production [100,150] card's inputs (create_datacards_ctag2d_100150_rawsf.py):
same processes, per-(dataset, era) xs*lumi/sumw normalisation (dc.event_scale), same USABLE_SYST / HPlusBottom override,
CMS_ctag2d recomputed from the official up/down_Total (rs._recompute_ctag2d_raw),
qqZZ scalevar_muR/muF normalisation-decoupled (rescaled to the nominal integral of
the plotted histogram), JES/JER object shifts from the jecshifts tree, lnN values
read from the reference datacard, Z+X from the CR_merged fake-rate parquets
(m4l-inclusive; windowed here).

Band = prefit: per bin, for every nuisance the shift of the TOTAL prediction
(summed over processes, i.e. correlated across processes), added in quadrature
separately for the upward/downward side of each bin, plus MC stat (sum w^2).

Two stages:
  --load : read all parquets (slow), write <out>/cache.pkl
  --plot : make plots + yields json from the cache
"""
import os
import re
import sys
import glob
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/scripts")
sys.path.insert(0, "/eos/user/s/snandaku/Higgscharmnew/higgscharm")
import create_datacards_ctag2d_100150 as dc
import create_datacards_ctag2d_100150_rawsf as rs
import eos_io as E

ERAS = dc.ERAS
MC_PROCS = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"]
DATA_PREFIXES = ("EGamma", "Muon")      # includes MuonEG; PD overlap removed in zzto4l_trigger
M4L_MIN = 70.0
JET_COLS = ["jet_btagPNetCvL", "jet_btagPNetCvB", "jet_hadronFlavour", "jet_pt"]
REF_CARD = ("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/"
            "combine_run3_100150_ctag2d_rawsf_withzxCRmerged_noZXrate_jecshifts_automcstats_2026_09_23")


def _flat_first(values):
    return np.array([x[0] if x is not None and len(x) else np.nan for x in values], dtype=float)


# ---------------------------------------------------------------------------
# Stage 1: load
# ---------------------------------------------------------------------------

def _read_mc_file(args):
    pq_file, proc, era, is_hb, scale = args
    usable = dc.USABLE_SYST.get(proc, set())
    requestable = usable - dc.HPLUSBOTTOM_BUGGY_SYST if is_hb else usable
    syst_cols = [dc.syst_col(s, d, era) for s in requestable for d in ("Up", "Down")]
    jet_cols = JET_COLS if "CMS_ctag2d" in usable else []
    base = ["zz_mass_inclusive", "mva_score_Signal", "weight_nominal"]
    try:
        df = E.read_parquet(pq_file, base + syst_cols + jet_cols).to_pandas()
    except Exception as e:
        return ("err", f"{pq_file}: {e}")
    m4l = _flat_first(df["zz_mass_inclusive"].values)
    keep = (m4l >= M4L_MIN) & (df["mva_score_Signal"].values >= 0)
    df = df[keep].reset_index(drop=True)
    out = {"m4l": m4l[keep], "score": df["mva_score_Signal"].values.astype(float),
           "w": df["weight_nominal"].fillna(0).values.astype(float) * scale, "syst": {}}
    if len(df) == 0:
        return ("mc", proc, out)
    nominal = out["w"]
    ct_up = ct_dn = None
    if jet_cols and all(c in df.columns for c in jet_cols):
        ct_up, ct_dn = rs._recompute_ctag2d_raw(df, era)
        ct_up, ct_dn = ct_up * scale, ct_dn * scale
    for s in usable:
        pair = []
        for d in ("Up", "Down"):
            col = dc.syst_col(s, d, era)
            if s == "CMS_ctag2d" and ct_up is not None:
                pair.append(ct_up if d == "Up" else ct_dn)
            elif is_hb and s in dc.HPLUSBOTTOM_BUGGY_SYST:
                pair.append(nominal)
            elif col in df.columns:
                pair.append(df[col].fillna(0).values.astype(float) * scale)
            else:
                pair.append(nominal)
        out["syst"][s] = tuple(pair)
    return ("mc", proc, out)


def _read_jec_file(args):
    pq_file, proc, syst, d, scale = args
    try:
        df = E.read_parquet(pq_file, ["zz_mass_inclusive", "mva_score_Signal", "weight_nominal"]).to_pandas()
    except Exception as e:
        return ("err", f"{pq_file}: {e}")
    m4l = _flat_first(df["zz_mass_inclusive"].values)
    keep = (m4l >= M4L_MIN) & (df["mva_score_Signal"].values >= 0)
    return ("jec", (proc, syst, d), {"m4l": m4l[keep],
                                      "score": df["mva_score_Signal"].values[keep].astype(float),
                                      "w": df["weight_nominal"].fillna(0).values[keep].astype(float) * scale})


def _read_data_file(args):
    pq_file, era = args
    try:
        df = E.read_parquet(pq_file, ["zz_mass_inclusive", "mva_score_Signal"]).to_pandas()
    except Exception as e:
        return ("err", f"{pq_file}: {e}")
    m4l = _flat_first(df["zz_mass_inclusive"].values)
    keep = (m4l >= M4L_MIN) & (df["mva_score_Signal"].values >= 0)
    return ("data", era, {"m4l": m4l[keep], "score": df["mva_score_Signal"].values[keep].astype(float)})


def _cat(parts, key):
    return np.concatenate([p[key] for p in parts]) if parts else np.array([])


def load(args):
    sumw = dc.load_sumw_by_group(args.sumw_dir)   # per-(dataset, era) normalisation, dc.event_scale

    mc_jobs, data_jobs, jec_jobs = [], [], []
    data_basenames = defaultdict(list)
    for era in ERAS:
        era_dir = os.path.join(args.scored_dir, era)
        for sample in E.listdir(era_dir):
            files = E.files(os.path.join(era_dir, sample, "base"), ".parquet")
            proc = dc.get_process_from_path(sample)
            if proc is not None:
                is_hb = proc == "Other_Higgs" and "HPlusBottom" in sample
                sc = dc.event_scale(era, sample, sumw)
                mc_jobs += [(f, proc, era, is_hb, sc) for f in files]
            elif sample.startswith(DATA_PREFIXES):
                data_jobs += [(f, era) for f in files]
                for f in files:
                    data_basenames[(era, re.sub(r"_\d+$", "", sample), os.path.basename(f))].append(f)
        jera = os.path.join(args.jecshifts_dir, era)
        year = dc.era_year(era)
        for sample in E.listdir(jera):
            proc = dc.get_process_from_path(sample)
            if proc not in dc.JECSHIFTS_USABLE_PROCS:
                continue
            sc = dc.event_scale(era, sample, sumw)
            for s in dc.JECSHIFTS_SYSTEMATICS:
                for d in ("Up", "Down"):
                    jec_jobs += [(f, proc, s, d, sc) for f in
                                 E.files(os.path.join(jera, sample, "base", f"{s}_{year}{d}"), ".parquet")]

    dups = {k: v for k, v in data_basenames.items() if len(v) > 1}
    print(f"  jobs: {len(mc_jobs)} MC, {len(data_jobs)} data, {len(jec_jobs)} jecshifts files; "
          f"duplicate data basenames across partitions: {len(dups)}")
    if dups:
        raise RuntimeError(f"duplicate data files across partitions, e.g. {list(dups.items())[:3]}")

    mc_parts, data_parts, jec_parts, errors = defaultdict(list), defaultdict(list), defaultdict(list), []
    with ProcessPoolExecutor(args.workers) as ex:
        for fn, jobs in ((_read_mc_file, mc_jobs), (_read_data_file, data_jobs), (_read_jec_file, jec_jobs)):
            for i, res in enumerate(ex.map(fn, jobs, chunksize=8)):
                if res[0] == "err":
                    errors.append(res[1])
                elif res[0] == "mc":
                    mc_parts[res[1]].append(res[2])
                elif res[0] == "data":
                    data_parts[res[1]].append(res[2])
                else:
                    jec_parts[res[1]].append(res[2])
                if (i + 1) % 500 == 0:
                    print(f"    {fn.__name__}: {i + 1}/{len(jobs)}", flush=True)
    print(f"  read errors: {len(errors)}")
    for e in errors[:20]:
        print("   ", e)
    if errors:
        raise RuntimeError(f"{len(errors)} parquet read errors -- refusing to write an incomplete cache")

    mc = {}
    for p in MC_PROCS:
        parts = mc_parts[p]
        systs = sorted(dc.USABLE_SYST.get(p, set()))
        mc[p] = {"m4l": _cat(parts, "m4l"), "score": _cat(parts, "score"), "w": _cat(parts, "w"),
                 "syst": {s: tuple(np.concatenate([q["syst"][s][k] for q in parts if len(q["w"])])
                                   for k in (0, 1)) for s in systs}}
        print(f"  {p:<12} n={len(mc[p]['w']):>8}  yield(m4l>={M4L_MIN:.0f})={mc[p]['w'].sum():.3f}")
    jec = {}
    for (p, s, d), parts in jec_parts.items():
        jec[(p, s, d)] = {"m4l": _cat(parts, "m4l"), "score": _cat(parts, "score"), "w": _cat(parts, "w")}
    data = {"m4l": np.concatenate([_cat(v, "m4l") for v in data_parts.values()]),
            "score": np.concatenate([_cat(v, "score") for v in data_parts.values()])}
    print(f"  data n(m4l>={M4L_MIN:.0f}) = {len(data['m4l'])}")

    zx = {"m4l": [], "score": [], "w": []}
    for era in ERAS:
        for cr in ("3p1f", "2p2f"):
            df = E.read_parquet(os.path.join(args.zx_dir, era, f"zx_{cr}_{era}_mva.parquet")).to_pandas()
            m = np.array([x[0] if hasattr(x, "__len__") and len(x) else (np.nan if hasattr(x, "__len__") else x)
                          for x in df["m4l"].values], dtype=float)
            zx["m4l"].append(m)
            zx["score"].append(df["mva_score_Signal"].values.astype(float))
            zx["w"].append(df["zx_weight"].values.astype(float))
    zx = {k: np.concatenate(v) for k, v in zx.items()}
    print(f"  ZX total {zx['w'].sum():.2f}, m4l range {np.nanmin(zx['m4l']):.1f}-{np.nanmax(zx['m4l']):.1f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cache.pkl"), "wb") as fh:
        pickle.dump({"mc": mc, "jec": jec, "data": data, "zx": zx, "errors": errors,
                     "inputs": vars(args)}, fh, protocol=4)
    print(f"  cache written: {args.out}/cache.pkl")


# ---------------------------------------------------------------------------
# Stage 2: histograms, band, plots
# ---------------------------------------------------------------------------

def read_ref_card(ref_card=REF_CARD):
    """lnN rows, SR quantile edges and rates from the reference card."""
    import uproot
    lnn, procs = {}, None
    for line in open(os.path.join(ref_card, "datacard_mva_with_zx.txt")):
        t = line.split()
        if t and t[0] == "process" and procs is None:
            procs = t[1:]
        if len(t) > 2 and t[1] == "lnN":
            lnn[t[0]] = {p: float(v) for p, v in zip(procs, t[2:]) if v != "-"}
    with uproot.open(os.path.join(ref_card, "histograms_mva_with_zx.root")) as f:
        edges = f["h_qqZZ"].axis().edges()
        ref_rates = {p: float(f[f"h_{p}"].values().sum()) for p in procs}
    return lnn, edges, ref_rates


class Region:
    def __init__(self, name, lo, hi, blind=None):
        self.name, self.lo, self.hi, self.blind = name, lo, hi, blind

    def mask(self, m4l):
        m = (m4l >= self.lo) & (m4l < self.hi)
        if self.blind is not None:
            m &= ~((m4l >= self.blind[0]) & (m4l < self.blind[1]))
        return m


def predict(cache, region, var, edges, lnn):
    """Per-process nominal + variance, and per-nuisance (up, down) total-prediction shifts."""
    mc, jec, zx = cache["mc"], cache["jec"], cache["zx"]
    nb = len(edges) - 1
    clip = lambda x: np.clip(x, edges[0], edges[-1] - 1e-9 * (edges[-1] - edges[0]))  # over/underflow into edge bins
    H = lambda x, w: np.histogram(clip(x), bins=edges, weights=w)[0]
    nom, var2 = {}, {}
    shifts = defaultdict(lambda: [np.zeros(nb), np.zeros(nb)])
    for p in MC_PROCS:
        d = mc[p]
        m = region.mask(d["m4l"])
        x = d[var][m]
        nom[p] = H(x, d["w"][m])
        var2[p] = H(x, d["w"][m] ** 2)
        for s, (wu, wd) in d["syst"].items():
            hu, hd = H(x, wu[m]), H(x, wd[m])
            if (p, s) in dc.NORM_DECOUPLED:
                if hu.sum() > 0:
                    hu = hu * nom[p].sum() / hu.sum()
                if hd.sum() > 0:
                    hd = hd * nom[p].sum() / hd.sum()
            shifts[s][0] += hu - nom[p]
            shifts[s][1] += hd - nom[p]
        for s in dc.JECSHIFTS_SYSTEMATICS:
            for k, dname in enumerate(("Up", "Down")):
                j = jec.get((p, s, dname))
                if j is None or len(j["w"]) == 0:
                    continue
                mj = region.mask(j["m4l"])
                shifts[s][k] += H(j[var][mj], j["w"][mj]) - nom[p]
    mz = region.mask(zx["m4l"])
    nom["ZX"] = H(zx[var][mz], zx["w"][mz])
    var2["ZX"] = H(zx[var][mz], zx["w"][mz] ** 2)
    for s, per in lnn.items():
        for p, kappa in per.items():
            if p in nom:
                shifts[s][0] += (kappa - 1) * nom[p]
                shifts[s][1] += (1 / kappa - 1) * nom[p]
    return nom, var2, dict(shifts)


def band(nom, var2, shifts, include_signal=False):
    procs = [p for p in nom if include_signal or p != "Signal"]
    tot = sum(nom[p] for p in procs)
    stat2 = sum(var2[p] for p in procs)
    up2, dn2 = stat2.copy(), stat2.copy()
    for s, (du, dd) in shifts.items():
        hi = np.maximum(np.maximum(du, dd), 0)
        lo = np.minimum(np.minimum(du, dd), 0)
        up2 += hi ** 2
        dn2 += lo ** 2
    return tot, np.sqrt(up2), np.sqrt(dn2), np.sqrt(stat2)


def garwood(n):
    from scipy.stats import chi2
    n = np.asarray(n, dtype=float)
    lo = np.where(n > 0, n - chi2.ppf(0.15865, 2 * n) / 2, 0)
    hi = chi2.ppf(0.84135, 2 * (n + 1)) / 2 - n
    return lo, hi


COLORS = {"qqZZ": "#5790fc", "ggZZ": "#f89c20", "Other_Higgs": "#e42536", "ZX": "#964a8b"}
LABELS = {"qqZZ": r"$q\bar{q}\to ZZ$", "ggZZ": r"$gg\to ZZ$", "Other_Higgs": "Other H", "ZX": "Z+X"}
STACK = ["ZX", "ggZZ", "qqZZ", "Other_Higgs"]


def plot(cache, region, var, edges, lnn, out_pdf, title, xlabel, sig_scale, data_blind_bins=(),
         bin_index_axis=False, logy=False, blind_x=None, subtitle=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    plt.style.use(hep.style.CMS)

    nom, var2, shifts = predict(cache, region, var, edges, lnn)
    tot, up, dn, stat = band(nom, var2, shifts)

    d = cache["data"]
    md = region.mask(d["m4l"])
    x = np.clip(d[var][md], edges[0], edges[-1] - 1e-9 * (edges[-1] - edges[0]))
    ndata = np.histogram(x, bins=edges)[0].astype(float)
    shown = np.ones(len(ndata), dtype=bool)
    shown[list(data_blind_bins)] = False
    if blind_x is not None:
        c = 0.5 * (edges[1:] + edges[:-1])
        shown &= ~((c >= blind_x[0]) & (c < blind_x[1]))
    ndata_shown = np.where(shown, ndata, np.nan)

    xe = np.arange(len(edges)) if bin_index_axis else edges
    c = 0.5 * (xe[1:] + xe[:-1])
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    fig.subplots_adjust(hspace=0.06)
    hep.histplot([nom[p] for p in STACK], xe, stack=True, histtype="fill", ax=ax,
                 color=[COLORS[p] for p in STACK], label=[LABELS[p] for p in STACK], edgecolor="k", linewidth=0.4)
    ax.stairs(tot + up, xe, baseline=tot - dn, fill=True, hatch="////", facecolor="none", edgecolor="0.3",
              linewidth=0, label="Total unc. (prefit)")
    hep.histplot(nom["Signal"] * sig_scale, xe, ax=ax, color="k", linestyle="--", linewidth=2,
                 label=rf"H+c $\times${sig_scale:g}")
    lo, hi = garwood(ndata)
    ax.errorbar(c[shown], ndata[shown], yerr=[lo[shown], hi[shown]], fmt="o", color="k", ms=6, label="Data")
    blocked = ~shown
    for i in np.where(blocked)[0]:
        for a in (ax, rax):
            a.axvspan(xe[i], xe[i + 1], color="0.85", zorder=0, lw=0)
    if blocked.any():
        ax.text(0.5 * (xe[np.where(blocked)[0][0]] + xe[np.where(blocked)[0][-1] + 1]), 0.02, "blinded",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=14, color="0.4")

    with np.errstate(divide="ignore", invalid="ignore"):
        r = ndata / tot
        rax.stairs(1 + up / tot, xe, baseline=1 - dn / tot, fill=True, hatch="////", facecolor="none",
                   edgecolor="0.3", linewidth=0)
        rax.errorbar(c[shown], r[shown], yerr=[lo[shown] / tot[shown], hi[shown] / tot[shown]],
                     fmt="o", color="k", ms=6)
    rax.axhline(1, color="k", lw=0.8)
    rax.set_ylim(0, 2.2)
    rax.set_ylabel("Data / MC")
    rax.set_xlabel(xlabel)
    ax.set_ylabel("Events / bin")
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(max(1e-2, 0.3 * np.min(tot[tot > 0])), 50 * max(np.nanmax(tot + up), np.nanmax(np.nan_to_num(ndata_shown))))
    else:
        ax.set_ylim(0, 1.6 * max(np.nanmax(tot + up), np.nanmax(np.nan_to_num(ndata_shown)) + 1,
                                 np.max(nom["Signal"] * sig_scale)))
    ax.set_xlim(xe[0], xe[-1])
    ax.legend(ncol=2, fontsize=15, loc="upper right")
    hep.cms.label("Work in progress", data=True, lumi=61.9, com=13.6, ax=ax, fontsize=18)
    ax.text(0.03, 0.93, title, transform=ax.transAxes, fontsize=16, va="top")
    if subtitle:
        ax.text(0.03, 0.86, subtitle, transform=ax.transAxes, fontsize=13, va="top", color="0.3")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=110)
    plt.close(fig)

    ranking = sorted(((s, float(np.sqrt(np.sum(np.maximum(np.abs(u), np.abs(v)) ** 2)))) for s, (u, v) in shifts.items()),
                     key=lambda t: -t[1])
    summary = {
        "region": region.name, "var": var, "edges": list(map(float, edges)),
        "yields": {p: float(h.sum()) for p, h in nom.items()},
        "yields_shown_bins": {p: float(h[shown].sum()) for p, h in nom.items()},
        "bkg_total_shown": float(tot[shown].sum()),
        "data_shown": int(np.nansum(ndata_shown)),
        "per_bin": {"bkg": tot.tolist(), "up": up.tolist(), "dn": dn.tolist(), "stat": stat.tolist(),
                    "data": [None if not s else int(n) for s, n in zip(shown, ndata)]},
        "top_nuisances_abs_events": ranking[:10],
    }
    return summary


def make_plots(args):
    with open(os.path.join(args.out, "cache.pkl"), "rb") as fh:
        cache = pickle.load(fh)
    lnn, sr_edges, ref_rates = read_ref_card(args.ref_card)
    print(f"  lnN from reference card: {lnn}")
    print(f"  SR quantile edges ({len(sr_edges) - 1} bins): {np.round(sr_edges, 4)}")

    # Cross-check: the SR MVA histogram must reproduce the reference card's rates.
    sr = Region("sr", 100, 150)
    nom, _, _ = predict(cache, sr, "score", sr_edges, lnn)
    print("  SR rate cross-check vs reference card:")
    ok = True
    for p, ref in ref_rates.items():
        mine = float(nom[p].sum())
        rel = abs(mine - ref) / ref if ref else 0
        ok &= rel < 1e-3
        print(f"    {p:<12} mine {mine:10.4f}  card {ref:10.4f}  rel.diff {rel:.2e}")
    if not ok:
        raise RuntimeError("SR rates do not reproduce the reference card -- not plotting")

    S = args.sig_scale
    nb = len(sr_edges) - 1
    onz = Region("onz", 91.1876 - 7, 91.1876 + 7)
    qqzz = Region("qqzz", 180, np.inf)
    sbm = Region("sideband_m4l", 100, 150)
    sbs = Region("sideband", 100, 150, blind=(115, 135))
    jobs = [
        (onz, "m4l", np.arange(84, 100, 2.0), r"$m_{4\ell}$ [GeV]", r"On-Z CR: $|m_{4\ell}-m_Z|<7$ GeV", {}),
        (onz, "score", sr_edges, "MVA discriminant bin (SR quantile binning)", "On-Z CR", {"bin_index_axis": True, "logy": True}),
        (qqzz, "m4l", np.array([180, 200, 220, 240, 260, 280, 300, 350, 400, 500, 600, 800, 1000.]),
         r"$m_{4\ell}$ [GeV] (last bin incl. overflow)", r"qqZZ CR: $m_{4\ell}>180$ GeV", {"logy": True}),
        (qqzz, "score", sr_edges, "MVA discriminant bin (SR quantile binning)", r"qqZZ CR: $m_{4\ell}>180$ GeV",
         {"bin_index_axis": True, "logy": True}),
        (sbm, "m4l", np.arange(100, 152.5, 2.5), r"$m_{4\ell}$ [GeV]", "SR m4l, data blind 115-135 GeV",
         {"blind_x": (115, 135)}),
        (sbs, "score", sr_edges, "MVA discriminant bin (SR quantile binning)",
         "SR sidebands: 100-115, 135-150 GeV", {"bin_index_axis": True}),
        (sr, "score", sr_edges, "MVA discriminant bin (SR quantile binning)",
         f"SR 100-150 GeV, top {args.n_blind} bins blind",
         {"bin_index_axis": True, "data_blind_bins": range(nb - args.n_blind, nb)}),
    ]
    summaries = []
    for region, var, edges, xl, title, kw in jobs:
        tag = f"{region.name}_{'m4l' if var == 'm4l' else 'mva'}"
        pdf = os.path.join(args.out, f"hczz_datamc_{tag}.pdf")
        s = plot(cache, region, var, edges, lnn, pdf, title, xl, S, **kw)
        summaries.append(s)
        b = s["bkg_total_shown"]
        print(f"  {tag:<22} data {s['data_shown']:>5}  bkg {b:9.2f}  data/bkg {s['data_shown'] / b if b else float('nan'):.3f}"
              f"  top: {', '.join(f'{n}({v:.2f})' for n, v in s['top_nuisances_abs_events'][:4])}")
    with open(os.path.join(args.out, "datamc_summary.json"), "w") as fh:
        json.dump({"sig_scale": S, "lnN": lnn, "regions": summaries}, fh, indent=1)
    print(f"  wrote {args.out}/datamc_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--scored-dir", default="/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored_v8")
    ap.add_argument("--jecshifts-dir", default="/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_jecshifts_scored_v2")
    ap.add_argument("--sumw-dir", default="/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d")
    ap.add_argument("--zx-dir", default="/eos/user/s/snandaku/Analysis/zx_background_mva_CRmerged_2026_09_23")
    ap.add_argument("--out", default="/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/datamc_prefit_2026_09_24")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ref-card", default=REF_CARD, help="card whose lnN rows, SR edges and rates are used/cross-checked")
    ap.add_argument("--sig-scale", type=float, default=350.0, help="H+c overlay scale (~ expected limit 352.5)")
    ap.add_argument("--n-blind", type=int, default=5, help="top MVA bins kept blind in the 'sr' plot")
    a = ap.parse_args()
    if a.load:
        load(a)
    if a.plot:
        make_plots(a)
