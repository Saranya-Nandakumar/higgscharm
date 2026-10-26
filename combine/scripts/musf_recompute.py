"""Per-event muon ID SF for background parquets that lack it (2026-10-22).

The JES/JER shift production (hplusc_mva_4class_ctag2d_jecshifts*, JEC V4/V3 via correctionlib)
was copied from the nominal workflow before the muon ID SF block was added there, so its
parquets carry neither the central muon ID SF nor weight_CMS_eff_m_id_<year>{Up,Down}. To take
the background nominal templates from that production (one JEC version, V4/V3, for nominal,
JES/JER shifts and signal), install() wraps eos_io.read_parquet: for every background file under
one of the given directories that has no weight_CMS_eff_m_id_<year> column, the loose-ID SF
(muon_Z.json NUM_LooseID_DEN_TrackerMuons, as analysis/corrections/muon.py) is recomputed from
the four stored leptons (muons: |lepton_mass - m_mu| < 5 MeV; SF = 1 outside 15 < pT < 500 GeV,
|eta| < 2.399) and
  weight_nominal and every requested weight_*Up/Down column (full weights)  *= SF,
  weight_CMS_eff_m_id_<year>{Up,Down} = weight_nominal(before) * SF_{systup,systdown}.
Checked against scored_v8 (which has the processor's SF): per-event agreement except for events
with an additional selected muon that is not stored (8.5% of qqZZ 2022postEE events, up to 5%),
yield 0.05%.
Signal samples (HPlusCharm, Stitch3FS) are never touched.
"""
import os
import re
import numpy as np
import pyarrow as pa
import correctionlib

MUON_JSON = {
    "2022preEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22CDSep23-Summer22-NanoAODv12/2025-08-14/muon_Z.json.gz",
    "2022postEE": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22EFGSep23-Summer22EE-NanoAODv12/2025-08-14/muon_Z.json.gz",
    "2023preBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23CSep23-Summer23-NanoAODv12/2025-08-14/muon_Z.json.gz",
    "2023postBPix": "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-23DSep23-Summer23BPix-NanoAODv12/2025-08-14/muon_Z.json.gz",
}
LEP = ["lepton_mass", "lepton_pt", "lepton_eta"]
SIGNAL_TAGS = ("HPlusCharm", "Stitch3FS")
_CS = {}
STATS = {"files": 0, "events": 0}


def _sf(era, tab):
    if era not in _CS:
        _CS[era] = correctionlib.CorrectionSet.from_file(MUON_JSON[era])["NUM_LooseID_DEN_TrackerMuons"]
    c = _CS[era]
    m, pt, eta = (tab.column(k).combine_chunks() for k in LEP)
    off = np.asarray(m.offsets)
    mf, pf, ef = (np.asarray(x.flatten()) for x in (m, pt, eta))
    ef = np.abs(ef)
    use = (np.abs(mf - 0.10566) < 0.005) & (pf > 15) & (pf < 500) & (ef < 2.399)
    out = {}
    for var in ("nominal", "systup", "systdown"):
        v = np.ones_like(pf)
        if use.any():
            v[use] = c.evaluate(ef[use], pf[use], var)
        out[var] = np.multiply.reduceat(np.append(v, 1.0), off[:-1]) if len(off) > 1 else np.array([])
        out[var][np.diff(off) == 0] = 1.0
    return out


def install(E, dirs):
    dirs = [d.rstrip("/") + "/" for d in dirs]
    orig = E.read_parquet

    def read_parquet(path, columns=None):
        p = str(path)
        if columns is None or not any(p.startswith(d) or d in p for d in dirs) or any(t in p for t in SIGNAL_TAGS):
            return orig(path, columns)
        era = next((e for e in MUON_JSON if f"/{e}/" in p), None)
        yr = era[:4] if era else None
        if era is None or "weight_nominal" not in columns:
            return orig(path, columns)
        avail = set(E.parquet_columns(path))
        if f"weight_CMS_eff_m_id_{yr}" in avail:
            return orig(path, columns)
        want = list(dict.fromkeys(list(columns) + LEP))
        tab = orig(path, want)
        sf = _sf(era, tab)
        w0 = np.asarray(tab.column("weight_nominal").to_numpy(zero_copy_only=False), dtype=float)
        cols = {}
        for name in tab.column_names:
            if name in LEP and name not in columns:
                continue
            arr = tab.column(name)
            if name == "weight_nominal" or (name.startswith("weight_") and re.search(r"(Up|Down)$", name)):
                arr = pa.array(np.asarray(arr.to_numpy(zero_copy_only=False), dtype=float) * sf["nominal"])
            cols[name] = arr
        for d, var in (("Up", "systup"), ("Down", "systdown")):
            k = f"weight_CMS_eff_m_id_{yr}{d}"
            if k in columns:
                cols[k] = pa.array(w0 * sf[var])
        STATS["files"] += 1
        STATS["events"] += len(w0)
        return pa.table(cols)

    E.read_parquet = read_parquet
    print(f"  [musf] per-event muon ID SF recomputed for background files under: {dirs}")
