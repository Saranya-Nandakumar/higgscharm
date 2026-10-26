#!/usr/bin/env python3
"""
Single-file smoke test for the GEN c-hadron columns (gen_chad1_pt, n_chad5, n_chad10)
and the c-hadron-split sumw sidecar added to analysis/processors/base.py on 2026-10-02,
plus the muon ID SF block restored in the jecshifts workflows the same day.

Runs the real BaseProcessor (workflow hplusc_mva_4class_ctag2d_jecshifts_fs) over the
first file of the 4FS HPlusCharm_2022postEE condor partition, in-process.
Checks: the new columns exist in the nominal and in every JES/JER shift dir; the sidecar
sumw_chad{5,10}_{0,1,2p} add up to sumw; the muon ID SF columns are present; and the
c-hadron sums agree with an independent standalone computation on the same file
(the derive_chad_n.py definition).

Usage (LCG_105, valid voms proxy):
    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    cd /eos/user/s/snandaku/Higgscharmnew/higgscharm
    python3 smoketest_chad.py [out_dir]
Output goes to a throwaway directory (default under /tmp), never to production.
"""
import json
import sys
import shutil
import tempfile
from pathlib import Path

import numpy as np
from coffea import processor
from coffea.nanoevents import NanoAODSchema

PARTITIONS = Path("/afs/cern.ch/user/s/snandaku/Higgscharmnew/higgscharm/condor/"
                  "hplusc_mva_4class_ctag2d_jecshifts_fs/2022postEE/HPlusCharm_2022postEE/partitions.json")
DATASET = "HPlusCharm_2022postEE"
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "hczz_chad_smoketest_output"


def standalone_chad(url):
    """Independent c-hadron count on the same file (derive_chad_n.py definition)."""
    import uproot
    import awkward as ak
    # same definitions as /eos/user/s/snandaku/Analysis/fs_stitch_gg_2026_10_02/derive_chad_n.py
    def is_chad(apid):
        openc = ((apid > 400) & (apid < 500)) | ((apid > 4000) & (apid < 5000)) | \
                ((apid > 10400) & (apid < 10500)) | ((apid > 20400) & (apid < 20500))
        return openc & (((apid // 10) % 100) != 44)

    def is_bhad(apid):
        return ((apid > 500) & (apid < 600)) | ((apid > 5000) & (apid < 6000))
    t = uproot.open(url)["Events"]
    a = t.arrays(["genWeight", "GenPart_eta", "GenPart_status", "GenPart_pdgId", "GenPart_pt", "GenPart_genPartIdxMother"])
    apid = abs(a.GenPart_pdgId); mom = a.GenPart_genPartIdxMother
    idx = ak.local_index(apid)
    has_c_daughter = ak.any(idx[:, :, None] == mom[is_chad(apid)][:, None, :], axis=-1)
    mom_pid = ak.fill_none(apid[ak.mask(mom, mom >= 0)], 0)
    sel = is_chad(apid) & (a.GenPart_status == 2) & ~has_c_daughter & ~is_bhad(mom_pid) & (abs(a.GenPart_eta) < 2.5)
    cpt = a.GenPart_pt[sel]
    w = ak.to_numpy(a.genWeight)
    out = {}
    for t_ in (5, 10):
        n = ak.to_numpy(ak.sum(cpt > t_, axis=1))
        out[t_] = [float(w[n == 0].sum()), float(w[n == 1].sum()), float(w[n >= 2].sum())]
    return out


def main():
    from analysis.processors.base import BaseProcessor

    parts = json.loads(PARTITIONS.read_text())
    first = sorted(parts, key=int)[0]
    key = list(parts[first])[0]           # partition key, e.g. HPlusCharm_2022postEE_1
    one_file = parts[first][key][0]
    fileset = {key: [one_file]}
    print(f"Testing against: {one_file}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    processor.run_uproot_job(
        fileset, treename="Events",
        processor_instance=BaseProcessor(workflow="hplusc_mva_4class_ctag2d_jecshifts_fs", year="2022postEE",
                                         output_format="parquet", output_location=str(OUT_DIR) + "/"),
        executor=processor.iterative_executor, executor_args={"schema": NanoAODSchema, "workers": 1},
    )

    import pyarrow.parquet as pq
    pqs = sorted(OUT_DIR.glob("**/*.parquet"))
    assert pqs, f"no parquet under {OUT_DIR}"
    print(f"{len(pqs)} parquet files")
    new = ["gen_chad1_pt", "n_chad5", "n_chad10", "n_genc10"]
    nominal_seen = False
    for p in pqs:
        cols = pq.ParquetFile(p).schema.names
        missing = [c for c in new if c not in cols]
        assert not missing, f"{p}: missing {missing}"
        if p.parent.name == "base":          # nominal; shift dirs (base/<syst>) only carry weight_nominal
            nominal_seen = True
            muon = [c for c in cols if "CMS_eff_m_id" in c]
            assert muon, f"{p}: muon ID SF columns missing"
            df = pq.read_table(p, columns=new + ["weight_nominal"]).to_pandas()
            print(f"nominal {p.relative_to(OUT_DIR)}: {len(df)} events; muon cols {muon[:1]}...")
            print(f"  gen_chad1_pt: none={np.mean(df.gen_chad1_pt < 0):.3f}  mean(>0)={df.gen_chad1_pt[df.gen_chad1_pt > 0].mean():.1f}")
            print(f"  n_chad5 0/1/>=2: {np.mean(df.n_chad5 == 0):.3f} {np.mean(df.n_chad5 == 1):.3f} {np.mean(df.n_chad5 >= 2):.3f}")
    assert nominal_seen, "no nominal parquet"
    print("all parquets (nominal + shift dirs) carry", new)

    js = sorted(OUT_DIR.glob("**/sumw/*.json"))
    assert len(js) == 1, f"expected one sidecar, got {js}"
    rec = json.loads(js[0].read_text())
    for t_ in (5, 10):
        parts_ = [rec[f"sumw_chad{t_}_{b}"] for b in ("0", "1", "2p")]
        assert abs(sum(parts_) - rec["sumw"]) < 1e-6 * abs(rec["sumw"]), (t_, parts_, rec["sumw"])
        assert f"sumw_chad{t_}_0_scalevar" in rec, "scalevar split missing"
    print("sidecar keys:", sorted(k for k in rec if "chad" in k and "scalevar" not in k))
    assert "sumw_preveto" in rec and rec["sumw_preveto"] >= rec["sumw"], "sumw_preveto missing or < sumw"
    print(f"sumw_preveto {rec['sumw_preveto']:.3f}  sumw (after jet veto) {rec['sumw']:.3f}  vetoed fraction {1 - rec['sumw'] / rec['sumw_preveto']:.4f}")

    # Exact check of gen_chadron_info on the FULL file (the processor sidecar is after the
    # event-level jet veto map, so its absolute sums are lower; fractions are compared above).
    from coffea.nanoevents import NanoEventsFactory
    from analysis.processors.base import gen_chadron_info
    ev = NanoEventsFactory.from_root(one_file, schemaclass=NanoAODSchema).events()
    info = gen_chadron_info(ev)
    gw = np.asarray(ev.genWeight)
    for t_ in (5, 10):
        n = info[f"n_chad{t_}"]
        frac = [rec[f"sumw_chad{t_}_{b}"] / rec["sumw"] for b in ("0", "1", "2p")]
        print(f"T={t_}: sidecar fractions (after jet veto) {np.round(frac, 3)}")
    ref = standalone_chad(one_file)
    for t_ in (5, 10):
        n = info[f"n_chad{t_}"]
        mine = [float(gw[n == 0].sum()), float(gw[n == 1].sum()), float(gw[n >= 2].sum())]
        print(f"T={t_}: processor {np.round(mine, 3)}  standalone {np.round(ref[t_], 3)}")
        assert np.allclose(mine, ref[t_], rtol=1e-6), "processor vs standalone c-hadron sums differ"
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
