#!/usr/bin/env python3
"""
Bookkeeping and plots for the scored HcZZ productions (2026-10-26).

Replaces, for the hplusc_mva_4class* workflows, what `run_postprocess.py --postprocess --plot`
gives the other workflows, but works on the per-job outputs (no merging) and normalises
exactly as the datacard builder does (combine/scripts/create_datacards_ctag2d_100150.py):
every MC event gets weight_nominal * xs_group * lumi_era / sumw(era, group) (dc.event_scale).

Inputs
  --raw-dir     unscored workflow output: <era>/<dataset>_<N>/sumw/*.json (sumw sidecars) and
                <era>/<dataset>/<dataset>_<N>.coffea (per-job metadata with the cutflow)
  --scored-dir  run_mva_postprocess.py output: <era>/<dataset>_<N>/base/*.parquet (shift subdirs
                are not read)

Stages (run_mva_postprocess.py --stages ..., or this script directly)
  bookkeep  per era, per year (2022/2023 if both halves are given) and for all eras together:
            cutflow_<tag>.csv          weighted cutflow per process (pre-veto sumw -> ... -> card input)
            cutflow_eff_<tag>.csv      the same relative to the generated row
            yields_<tag>.csv / .tex    selected / mass window / card input, stat. unc., raw events
            sumw_check_<tag>.csv       sumw of the .coffea files vs the sumw used for the
                                       normalisation (flags missing or duplicated jobs)
            histograms_<tag>.coffea    {variable: hist.Hist(region, process, variable)}
            bookkeeping_<tag>.json     all numbers plus the inputs used
  plot      stack plots from histograms_<tag>.coffea (MC stat band, data if present, blinded)

Normalisation is the builder's: --sumw-table (the pre-veto table of the 2022/23 cards, "nom"
entries) or the sidecars with --sumw-key (default sumw_preveto, as the 2024 builder). Other_Higgs
is split per production mode (dc.split_other_higgs) unless --no-split-higgs.

Not reproduced here: the stitched 4FS+3FS signal of the 2022/23 cards (the Signal column is the
plain 4FS sample(s), normalised to 4FS sigma x BR), its flat muon ID SF, and Z+X.

Data (Muon*/EGamma*/DoubleMuon*/SingleMuon* directories): weight 1. Blinded unless --unblind:
no data in the mass-window region and in the yield table's window columns; the m4l histograms
have the data bins inside --blind-m4l (default 115-135 GeV) set to zero.

Usage
  python3 analysis/postprocess/bookkeeping.py --raw-dir $O/<wf> --scored-dir $O/<wf>_scored_vN \
      --eras 2022preEE 2022postEE 2023preBPix 2023postBPix --output-dir <dir> \
      [--sumw-table sumw_table.json] [--mass-window 105 160] [--stages bookkeep,plot]
"""
import os
import re
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
dc = None   # combine/scripts/create_datacards_ctag2d_100150.py (normalisation, process mapping), see _init_dc
E = None    # its eos_io (server-side EOS listings, xrootd read fallback)


def _init_dc():
    """Import the datacard builder only when bookkeeping runs (scoring-only jobs do not need it)."""
    global dc, E
    if dc is None:
        sys.path.insert(0, str(REPO / "combine" / "scripts"))
        sys.path.insert(0, str(REPO))
        import create_datacards_ctag2d_100150 as _dc
        dc, E = _dc, _dc.E
logger = logging.getLogger("bookkeeping")

DATA_PREFIXES = ("Muon", "EGamma", "DoubleMuon", "SingleMuon", "MuonEG")
CUTFLOW_CATEGORY = "base"

# variable -> (column, edges). Scalar parquet columns only.
HIST_VARS = {
    "m4l":             ("zz_mass_inclusive", np.arange(70, 172, 2.0)),
    "m4l_wide":        ("zz_mass_inclusive", np.arange(70, 510, 10.0)),
    "mva_score_Signal":      ("mva_score_Signal", np.linspace(0, 1, 21)),
    "mva_score_qqZZ":        ("mva_score_qqZZ", np.linspace(0, 1, 21)),
    "mva_score_ggZZ":        ("mva_score_ggZZ", np.linspace(0, 1, 21)),
    "mva_score_Other_Higgs": ("mva_score_Other_Higgs", np.linspace(0, 1, 21)),
    "zz_pt":           ("zz_pt_inclusive", np.arange(0, 310, 10.0)),
    "z1_mass":         ("z1_mass", np.arange(40, 122, 2.0)),
    "z2_mass":         ("z2_mass", np.arange(0, 122, 4.0)),
    "deltaR_ZZ":       ("deltaR_ZZ", np.linspace(0, 6, 31)),
    "n_cjet":          ("n_cjet", np.arange(-0.5, 6.5, 1.0)),
    "jet_multiplicity": ("jet_multiplicity", np.arange(-0.5, 10.5, 1.0)),
    "cjet_HT":         ("cjet_HT", np.arange(0, 510, 20.0)),
    "jet_ht":          ("jet_ht", np.arange(0, 810, 30.0)),
    "leadingjet_cvsl": ("leadingjet_cvsl", np.linspace(0, 1, 21)),
    "leadingjet_cvsb": ("leadingjet_cvsb", np.linspace(0, 1, 21)),
}
REGIONS = ("selected", "window")   # all selected events / mass window and mva_score_Signal >= 0
MASS_COL, SCORE_COL, WEIGHT_COL = "zz_mass_inclusive", "mva_score_Signal", "weight_nominal"


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------
def configure(eras, split_higgs=True, sumw_key="sumw_preveto"):
    """Point the builder module at these eras (it loops over dc.ERAS) and its conventions."""
    _init_dc()
    dc.ERAS[:] = list(eras)
    for era in eras:
        if era not in dc.LUMINOSITY_PB:
            if era == "2024":
                dc.LUMINOSITY_PB[era] = 108960.0      # analysis/postprocess/luminosity.yaml
            else:
                raise SystemExit(f"no luminosity for era {era} in the datacard builder")
    dc.SUMW_KEY = sumw_key
    if split_higgs and "Other_Higgs" in dc.CLASS_NAMES:
        dc.split_other_higgs()
    return [p for p in dc.CLASS_NAMES]


def is_data(name):
    return name.startswith(DATA_PREFIXES)


def process_of(name, exclude):
    # parquets_<sample>/: merged copies written by run_postprocess.py --postprocess (would double count)
    if name.startswith("parquets_") or (exclude and any(re.search(x, name) for x in exclude)):
        return None
    if is_data(name):
        return "Data"
    return dc.get_process_from_path(name)


def load_sumw_table(path):
    tab = json.load(open(path))
    out = {}
    for era, groups in tab["nom"].items():
        for g, rec in groups.items():
            if rec.get("n_missing"):
                raise RuntimeError(f"sumw table: {rec['n_missing']} chunks without sumw for {era} {g}")
            out[(era, g)] = rec["sumw"]
    return out


def load_sidecar_sumw(raw_dir):
    """dc.load_sumw_by_group, skipping loose files (<sample>.parquet/.coffea) and the merged
    parquets_<sample>/ dirs that an old run_postprocess.py --postprocess left in the era dir."""
    def key(era, part):
        if part.startswith("parquets_") or "." in part:
            return None
        g = dc.norm_group(part)
        return (era, g) if g else None
    return dict(dc.load_sumw(raw_dir, key_fn=key))


def year_of(era):
    return dc.era_year(era)


def era_groups(eras):
    """tag -> eras: each era, each complete year, and all eras."""
    groups = {e: [e] for e in eras}
    by_year = defaultdict(list)
    for e in eras:
        by_year[year_of(e)].append(e)
    for y, es in by_year.items():
        if len(es) > 1:
            groups[y] = es
    if len(eras) > 1:
        groups["all"] = list(eras)
    return groups


# ---------------------------------------------------------------------------
# per-job .coffea: cutflow
# ---------------------------------------------------------------------------
def _load_coffea(path):
    import lz4.frame
    import cloudpickle
    return cloudpickle.loads(lz4.frame.decompress(E.read_bytes(path)))


def collect_cutflows(raw_dir, era, exclude, workers):
    """(process, group) -> {"steps": {step: sumw}, "sumw": .., "sumw_preveto": .., "n_jobs": ..}."""
    era_dir = os.path.join(raw_dir, era)
    jobs = []
    for d in E.listdir(era_dir):
        sub = os.path.join(era_dir, d)
        if not E.isdir(sub):
            continue
        for f in E.files(sub, ".coffea"):
            name = Path(f).stem               # <dataset>_<N>, same as the partition dir name
            if not re.fullmatch(re.escape(d) + r"_\d+", name):
                continue                      # not a per-job output (e.g. old base/cutflow_base_*.coffea)
            proc = process_of(name, exclude)
            if proc is not None:
                jobs.append((f, name, proc))

    def one(job):
        f, name, proc = job
        meta = _load_coffea(f)["metadata"]
        cat = meta.get(CUTFLOW_CATEGORY, {})
        return name, proc, meta, cat.get("cutflow", {})

    out = defaultdict(lambda: {"steps": defaultdict(float), "sumw": 0.0, "sumw_preveto": 0.0, "n_jobs": 0})
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, proc, meta, cutflow in ex.map(one, jobs):
            group = "Data" if proc == "Data" else dc.norm_group(name)
            rec = out[(proc, group)]
            rec["n_jobs"] += 1
            sw = float(meta.get("sumw", 0.0))
            rec["sumw"] += sw
            rec["sumw_preveto"] += float(meta.get("sumw_preveto", sw) or sw)
            for step, v in cutflow.items():
                rec["steps"][step] += float(v)
    logger.info(f"  {era}: {len(jobs)} .coffea files")
    return out


# ---------------------------------------------------------------------------
# scored parquets: yields and histograms
# ---------------------------------------------------------------------------
def _empty_acc():
    return {
        "n_raw_sel": 0, "w_sel": 0.0, "w2_sel": 0.0,
        "n_raw_win": 0, "w_win": 0.0, "w2_win": 0.0,
        "n_raw_m4l": 0, "w_m4l": 0.0, "w2_m4l": 0.0,
        # per variable: [sum w, sum w^2] with underflow and overflow (len(edges) + 1 entries)
        "hist": {r: {v: [np.zeros(len(e) + 1), np.zeros(len(e) + 1)] for v, (_, e) in HIST_VARS.items()}
                 for r in REGIONS},
    }


def _add_acc(a, b):
    for k in a:
        if k == "hist":
            for r in a["hist"]:
                for v in a["hist"][r]:
                    a["hist"][r][v][0] += b["hist"][r][v][0]
                    a["hist"][r][v][1] += b["hist"][r][v][1]
        else:
            a[k] += b[k]


def _read_file(path, columns):
    have = set(E.parquet_columns(path))
    cols = [c for c in columns if c in have]
    return E.read_parquet(path, cols).to_pandas()


def _file_acc(path, scale, window, data):
    columns = sorted({MASS_COL, SCORE_COL, WEIGHT_COL} | {c for c, _ in HIST_VARS.values()})
    df = _read_file(path, columns)
    acc = _empty_acc()
    if len(df) == 0:
        return acc
    w = np.ones(len(df)) if data else df[WEIGHT_COL].fillna(0).values * scale
    m4l = df[MASS_COL].values
    in_m4l = (m4l >= window[0]) & (m4l < window[1])
    score_ok = df[SCORE_COL].values >= 0 if SCORE_COL in df else np.ones(len(df), bool)
    in_win = in_m4l & score_ok
    acc["n_raw_sel"], acc["w_sel"], acc["w2_sel"] = len(df), w.sum(), (w ** 2).sum()
    acc["n_raw_m4l"], acc["w_m4l"], acc["w2_m4l"] = int(in_m4l.sum()), w[in_m4l].sum(), (w[in_m4l] ** 2).sum()
    acc["n_raw_win"], acc["w_win"], acc["w2_win"] = int(in_win.sum()), w[in_win].sum(), (w[in_win] ** 2).sum()
    for region, mask in (("selected", slice(None)), ("window", in_win)):
        for v, (col, edges) in HIST_VARS.items():
            if col not in df:
                continue
            x = df[col].values[mask].astype(float)
            ww = w[mask]
            ok = np.isfinite(x)
            fe = np.concatenate(([-np.inf], edges, [np.inf]))      # underflow / overflow bins
            acc["hist"][region][v][0] += np.histogram(x[ok], fe, weights=ww[ok])[0]
            acc["hist"][region][v][1] += np.histogram(x[ok], fe, weights=ww[ok] ** 2)[0]
    return acc


def collect_yields(scored_dir, era, sumw_by_group, window, exclude, workers):
    """process -> accumulated yields/histograms for this era."""
    era_dir = os.path.join(scored_dir, era)
    tasks = []
    for name in E.listdir(era_dir):
        proc = process_of(name, exclude)
        if proc is None:
            continue
        base = os.path.join(era_dir, name, "base")
        if not E.isdir(base):
            continue
        data = proc == "Data"
        scale = 1.0 if data else dc.event_scale(era, name, sumw_by_group)
        for f in E.files(base, ".parquet"):
            tasks.append((proc, f, scale, data))
    logger.info(f"  {era}: {len(tasks)} scored parquet files")
    out = defaultdict(_empty_acc)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_file_acc, f, s, window, d): p for p, f, s, d in tasks}
        for fut in as_completed(futs):
            _add_acc(out[futs[fut]], fut.result())
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _ordered(procs, mc_order):
    head = [p for p in mc_order if p in procs and p != "Signal"]
    tail = [p for p in procs if p not in head and p not in ("Signal", "Data")]
    sig = ["Signal"] if "Signal" in procs else []
    data = ["Data"] if "Data" in procs else []
    return head + tail + sig + data


def make_tables(eras, cutflows, yields, sumw_by_group, window, mc_order, blind):
    """Cutflow, efficiency, yields and sumw check summed over `eras`."""
    steps_seen = []
    cf = defaultdict(lambda: defaultdict(float))     # proc -> row -> value
    check = []
    for era in eras:
        for (proc, group), rec in cutflows[era].items():
            if proc == "Data":
                scale = 1.0
            else:
                used = sumw_by_group.get((era, group), float("nan"))
                # = dc.event_scale for every sample of this normalisation group
                scale = dc.group_xs_pb(era, group) * dc.LUMINOSITY_PB[era] / used
                check.append({"era": era, "process": proc, "group": group, "n_jobs": rec["n_jobs"],
                              "coffea_sumw_preveto": rec["sumw_preveto"], "coffea_sumw": rec["sumw"],
                              "sumw_used": used, "ratio_coffea_preveto/used": rec["sumw_preveto"] / used})
                cf[proc]["generated (pre jet veto)"] += rec["sumw_preveto"] * scale
                cf[proc]["jet veto map"] += rec["sumw"] * scale
            for step, v in rec["steps"].items():
                if step == "initial":
                    if proc == "Data":
                        cf[proc]["jet veto map"] += v
                    continue
                if step not in steps_seen:
                    steps_seen.append(step)
                cf[proc][step] += v * scale
    win = f"m4l in [{window[0]:g}, {window[1]:g})"
    for era in eras:
        for proc, acc in yields[era].items():
            cf[proc]["selected (scored parquets)"] += acc["w_sel"]
            cf[proc][win] += acc["w_m4l"]
            cf[proc][win + ", mva score >= 0 (card input)"] += acc["w_win"]
    rows = ["generated (pre jet veto)", "jet veto map"] + steps_seen + \
        ["selected (scored parquets)", win, win + ", mva score >= 0 (card input)"]
    procs = _ordered(cf.keys(), mc_order)
    cutflow = pd.DataFrame({p: [cf[p].get(r, np.nan) for r in rows] for p in procs}, index=rows)
    bkg = [p for p in procs if p not in ("Signal", "Data")]
    cutflow.insert(len(bkg), "Total background", cutflow[bkg].sum(axis=1, min_count=1))
    if blind and "Data" in cutflow:
        cutflow.loc[rows[-2:], "Data"] = np.nan
    eff = cutflow / cutflow.iloc[0]
    if "Data" in eff:
        eff["Data"] = cutflow["Data"] / cutflow["Data"].loc["jet veto map"]

    yrows = []
    for proc in procs:
        a = _empty_acc()
        for era in eras:
            if proc in yields[era]:
                _add_acc(a, yields[era][proc])
        hide = blind and proc == "Data"
        yrows.append({
            "process": proc,
            "selected": a["w_sel"], "selected_stat": np.sqrt(a["w2_sel"]), "selected_raw": a["n_raw_sel"],
            "window": np.nan if hide else a["w_m4l"], "window_stat": np.nan if hide else np.sqrt(a["w2_m4l"]),
            "card_input": np.nan if hide else a["w_win"], "card_input_stat": np.nan if hide else np.sqrt(a["w2_win"]),
            "card_input_raw": -1 if hide else a["n_raw_win"],
        })
    ydf = pd.DataFrame(yrows).set_index("process")
    tot = ydf.loc[[p for p in bkg if p in ydf.index]]
    ydf.loc["Total background"] = [tot["selected"].sum(), np.sqrt((tot["selected_stat"] ** 2).sum()),
                                   tot["selected_raw"].sum(), tot["window"].sum(),
                                   np.sqrt((tot["window_stat"] ** 2).sum()), tot["card_input"].sum(),
                                   np.sqrt((tot["card_input_stat"] ** 2).sum()), tot["card_input_raw"].sum()]
    return cutflow, eff, ydf, pd.DataFrame(check)


def yields_latex(ydf, window, tag):
    lines = [r"\begin{tabular}{lrrr}", r"\hline",
             rf"Process & Selected & $m_{{4\ell}}\in[{window[0]:g},{window[1]:g})$ & Card input \\", r"\hline"]
    for p, r in ydf.iterrows():
        def f(v, s):
            return "--" if not np.isfinite(v) else (f"{v:.2f} $\\pm$ {s:.2f}" if v < 1000 else f"{v:.0f} $\\pm$ {s:.0f}")
        if p == "Total background":
            lines.append(r"\hline")
        lines.append(f"{p.replace('_', ' ')} & {f(r['selected'], r['selected_stat'])} & "
                     f"{f(r['window'], r['window_stat'])} & {f(r['card_input'], r['card_input_stat'])} \\\\")
    lines += [r"\hline", r"\end{tabular}", f"% {tag}, stat. uncertainties only"]
    return "\n".join(lines) + "\n"


def make_hists(eras, yields, blind, blind_m4l):
    import hist
    procs = sorted({p for e in eras for p in yields[e]})
    out = {}
    for v, (col, edges) in HIST_VARS.items():
        h = hist.Hist(hist.axis.StrCategory(list(REGIONS), name="region"),
                      hist.axis.StrCategory(procs, name="process"),
                      hist.axis.Variable(edges, name=v, label=col), storage=hist.storage.Weight())
        view = h.view(flow=True)   # category axes: overflow slot only; variable axis: under + overflow
        for ir, r in enumerate(REGIONS):
            for ip, p in enumerate(procs):
                val = sum((yields[e][p]["hist"][r][v][0] for e in eras if p in yields[e]), np.zeros(len(edges) + 1))
                var = sum((yields[e][p]["hist"][r][v][1] for e in eras if p in yields[e]), np.zeros(len(edges) + 1))
                if p == "Data" and blind:
                    if r == "window":
                        val, var = val * 0, var * 0
                    elif col == MASS_COL:
                        lo = np.concatenate(([-np.inf], edges))
                        hi = np.concatenate((edges, [np.inf]))
                        m = (hi > blind_m4l[0]) & (lo < blind_m4l[1])    # every bin touching the blinded range
                        val, var = np.where(m, 0, val), np.where(m, 0, var)
                view.value[ir, ip, :] = val
                view.variance[ir, ip, :] = var
        out[v] = h
    return out


def run_bookkeep(a):
    from coffea.util import save
    eras = a.eras
    mc_order = configure(eras, split_higgs=not a.no_split_higgs, sumw_key=a.sumw_key)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.sumw_table:
        sumw_by_group = load_sumw_table(a.sumw_table)
        sumw_src = f"table {a.sumw_table} ('nom')"
    else:
        sumw_by_group = load_sidecar_sumw(a.raw_dir)
        sumw_src = f"sidecars {a.raw_dir} key {a.sumw_key}"
    logger.info(f"sumw: {sumw_src}")
    cutflows, yields = {}, {}
    for era in eras:
        logger.info(f"[{era}] cutflows from {a.raw_dir}")
        cutflows[era] = collect_cutflows(a.raw_dir, era, a.exclude, a.workers)
        logger.info(f"[{era}] yields from {a.scored_dir}")
        yields[era] = collect_yields(a.scored_dir, era, sumw_by_group, a.mass_window, a.exclude, a.workers)
    blind = not a.unblind
    for tag, es in era_groups(eras).items():
        cutflow, eff, ydf, check = make_tables(es, cutflows, yields, sumw_by_group, a.mass_window, mc_order, blind)
        cutflow.to_csv(out_dir / f"cutflow_{tag}.csv")
        eff.to_csv(out_dir / f"cutflow_eff_{tag}.csv")
        ydf.to_csv(out_dir / f"yields_{tag}.csv")
        (out_dir / f"yields_{tag}.tex").write_text(yields_latex(ydf, a.mass_window, tag))
        check.to_csv(out_dir / f"sumw_check_{tag}.csv", index=False)
        save(make_hists(es, yields, blind, a.blind_m4l), str(out_dir / f"histograms_{tag}.coffea"))
        bad = check[(check["ratio_coffea_preveto/used"] - 1).abs() > 1e-3] if len(check) else check
        meta = {
            "tag": tag, "eras": es, "lumi_pb": sum(dc.LUMINOSITY_PB[e] for e in es),
            "raw_dir": a.raw_dir, "scored_dir": a.scored_dir, "sumw": sumw_src,
            "mass_window": list(a.mass_window), "split_higgs": not a.no_split_higgs, "blind": blind,
            "exclude": a.exclude, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cutflow": json.loads(cutflow.to_json()), "yields": json.loads(ydf.to_json()),
            "sumw_check_mismatches": json.loads(bad.to_json(orient="records")),
        }
        (out_dir / f"bookkeeping_{tag}.json").write_text(json.dumps(meta, indent=1))
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
            logger.info(f"\n===== {tag} ({', '.join(es)}) =====\n{cutflow}\n\n{ydf}")
        if len(bad):
            logger.warning(f"{tag}: {len(bad)} (era, group) with .coffea pre-veto sumw != normalisation sumw "
                           f"(missing/duplicated jobs, or a table over a different chunk set); see sumw_check_{tag}.csv")
    logger.info(f"bookkeeping done in {time.time() - t0:.0f} s -> {out_dir}")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
LABELS = {"m4l": r"$m_{4\ell}$ [GeV]", "m4l_wide": r"$m_{4\ell}$ [GeV]", "zz_pt": r"$p_T^{4\ell}$ [GeV]",
          "z1_mass": r"$m_{Z_1}$ [GeV]", "z2_mass": r"$m_{Z_2}$ [GeV]", "deltaR_ZZ": r"$\Delta R(Z_1, Z_2)$",
          "n_cjet": "Number of c-tagged jets", "jet_multiplicity": "Number of jets", "cjet_HT": r"c-jet $H_T$ [GeV]",
          "jet_ht": r"Jet $H_T$ [GeV]", "leadingjet_cvsl": "Leading jet CvL", "leadingjet_cvsb": "Leading jet CvB",
          "mva_score_Signal": "MVA score (H+c)", "mva_score_qqZZ": "MVA score (qqZZ)",
          "mva_score_ggZZ": "MVA score (ggZZ)", "mva_score_Other_Higgs": "MVA score (other Higgs)"}
LOG_Y = {"m4l_wide"}
COLORS = {"qqZZ": "#5790fc", "ggZZ": "#f89c20", "ggH": "#e42536", "VBF": "#964a8b", "WminusH": "#9c9ca1",
          "WplusH": "#7a21dd", "ZH": "#a96b59", "ttH": "#92dadd", "bbH": "#b9ac70", "HPlusB": "#717581",
          "Other_Higgs": "#e42536"}


def run_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from coffea.util import load
    hep.style.use("CMS")
    out_dir = Path(a.output_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for hf in sorted(out_dir.glob("histograms_*.coffea")):
        tag = hf.stem.replace("histograms_", "")
        meta = json.loads((out_dir / f"bookkeeping_{tag}.json").read_text())
        lumi_fb = meta["lumi_pb"] / 1000.0
        hists = load(str(hf))
        for v, h in hists.items():
            procs = list(h.axes["process"])
            edges = h.axes[v].edges
            for region in REGIONS:
                hr = h[{"region": region}]
                bkg = [p for p in procs if p not in ("Signal", "Data")]
                order = sorted(bkg, key=lambda p: hr[{"process": p}].values().sum())
                vals = [hr[{"process": p}].values() for p in order]
                tot = np.sum(vals, axis=0) if vals else np.zeros(len(edges) - 1)
                var = np.sum([hr[{"process": p}].variances() for p in order], axis=0) if vals else tot * 0
                has_data = "Data" in procs and hr[{"process": "Data"}].values().sum() > 0
                if has_data:
                    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
                else:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    rax = None
                if vals:
                    hep.histplot(vals, edges, ax=ax, stack=True, histtype="fill", label=order,
                                 color=[COLORS.get(p) for p in order])
                    err = np.sqrt(var)
                    ax.stairs(tot + err, edges, baseline=tot - err, fill=True, hatch="////", facecolor="none",
                              label="MC stat.", edgecolor="grey", linewidth=0)
                if "Signal" in procs:
                    s = hr[{"process": "Signal"}].values()
                    k = a.signal_scale
                    if s.sum() > 0:
                        hep.histplot(s * k, edges, ax=ax, color="black", ls="--", label=f"H+c x {k:g}")
                if has_data:
                    d = hr[{"process": "Data"}].values()
                    c = 0.5 * (edges[1:] + edges[:-1])
                    show = d > 0
                    ax.errorbar(c[show], d[show], yerr=np.sqrt(d[show]), fmt="ko", label="Data")
                    with np.errstate(divide="ignore", invalid="ignore"):
                        r = np.where(tot > 0, d / tot, np.nan)
                        re_ = np.where(tot > 0, np.sqrt(d) / tot, np.nan)
                        band = np.where(tot > 0, np.sqrt(var) / tot, 0)
                    rax.errorbar(c[show], r[show], yerr=re_[show], fmt="ko")
                    rax.stairs(1 + band, edges, baseline=1 - band, fill=True, color="grey", alpha=0.3)
                    rax.axhline(1, color="grey", ls=":")
                    rax.set_ylim(0, 2)
                    rax.set_ylabel("Data / MC")
                    rax.set_xlabel(LABELS.get(v, h.axes[v].label))
                else:
                    ax.set_xlabel(LABELS.get(v, h.axes[v].label))
                ax.set_ylabel("Events / bin")
                ax.set_xlim(edges[0], edges[-1])
                if v in LOG_Y:
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=max(1e-3, tot[tot > 0].min() * 0.5) if (tot > 0).any() else 1e-3)
                else:
                    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
                ax.legend(ncol=2, fontsize=12, loc="upper right")
                hep.cms.label("Work in progress", data=has_data, lumi=round(lumi_fb, 1), com=13.6, ax=ax, fontsize=17)
                win = meta["mass_window"]
                txt = "all selected events" if region == "selected" else f"$m_{{4\\ell}}\\in[{win[0]:g},{win[1]:g})$ GeV"
                ax.text(0.03, 0.95, f"{tag}\n{txt}", transform=ax.transAxes, ha="left", va="top", fontsize=14)
                for ext in ("png", "pdf"):
                    fig.savefig(plot_dir / f"{v}_{region}_{tag}.{ext}", bbox_inches="tight")
                plt.close(fig)
        logger.info(f"plots for {tag} -> {plot_dir}")


# ---------------------------------------------------------------------------
def add_arguments(p, standalone=True):
    if standalone:
        p.add_argument("--raw-dir", required=True, help="unscored workflow output (sumw sidecars, .coffea)")
        p.add_argument("--scored-dir", required=True, help="run_mva_postprocess.py output")
        p.add_argument("--eras", nargs="+", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--stages", default="bookkeep,plot", help="comma-separated: bookkeep, plot")
    p.add_argument("--sumw-table", default=None,
                   help="pre-veto sumw table of the 2022/23 cards ('nom' entries); default: sidecars")
    p.add_argument("--sumw-key", default="sumw_preveto", help="sidecar key without --sumw-table")
    p.add_argument("--mass-window", nargs=2, type=float, default=(105.0, 160.0), metavar=("LO", "HI"))
    p.add_argument("--no-split-higgs", action="store_true", help="keep one pooled Other_Higgs")
    p.add_argument("--exclude", nargs="*", default=[], help="regexes of sample directories to skip")
    p.add_argument("--unblind", action="store_true")
    p.add_argument("--blind-m4l", nargs=2, type=float, default=(115.0, 135.0))
    p.add_argument("--signal-scale", type=float, default=100.0, help="H+c overlay scale in the plots")
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = add_arguments(p).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    if "bookkeep" in stages:
        run_bookkeep(a)
    if "plot" in stages:
        run_plot(a)


if __name__ == "__main__":
    main()
