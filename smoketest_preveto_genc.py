#!/usr/bin/env python3
"""
Smoke test for the pre-jet-veto GEN c-jet sumw split added to analysis/processors/base.py on
2026-10-05 (sidecar keys sumw_preveto_genc{0,1,2p} and *_scalevar), a test for the direct
3FS+4FS stitching. Documented in second-brain
Notes/hczz-preveto-genc-sumw-and-jec-pershift-sumw-2026-10-05.md.

Runs the OLD processor (base.py.bak_20261005_preveto_genc) and the NEW one over the same
single file (first file of the HPlusCharm_2022postEE jecshifts_fschad partition) and checks:
  1. every parquet (nominal + all shift dirs) has the same schema, row count and GEN-level
     columns (reco columns vary run to run: unseeded muon smearing, see check 1 in main());
  2. every sidecar key of the old processor is unchanged in the new one;
  3. the new keys exist, sumw_preveto_genc{0,1,2p} add up to sumw_preveto exactly,
     and the 9 scale-variation sums are present;
  4. the new split equals an independent uproot computation on the full file
     (GenJet_hadronFlavour == 4, GenJet_pt > 10, sum of genWeight per 0 / 1 / >= 2).

Usage (LCG_105, valid voms proxy):
    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    cd /eos/user/s/snandaku/Higgscharmnew/higgscharm
    python3 smoketest_preveto_genc.py [out_dir]
Output goes to a throwaway directory, never to production.
"""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from coffea import processor
from coffea.nanoevents import NanoAODSchema

HERE = Path(__file__).resolve().parent
PARTITIONS = Path("/afs/cern.ch/user/s/snandaku/Higgscharmnew/higgscharm/condor/"
                  "hplusc_mva_4class_ctag2d_jecshifts_fschad/2022postEE/HPlusCharm_2022postEE/partitions.json")
WORKFLOW = "hplusc_mva_4class_ctag2d_jecshifts_fschad"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "hczz_preveto_genc_smoketest"


def load_processor(path, name):
    from importlib.machinery import SourceFileLoader
    spec = importlib.util.spec_from_file_location(name, path, loader=SourceFileLoader(name, str(path)))
    mod = importlib.util.module_from_spec(spec)
    # coffea cloudpickles the processor; a module that is not in sys.modules would be pickled
    # by value and lose the names the selection strings use through eval()
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.BaseProcessor


def run(proc_cls, fileset, out_dir):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    processor.run_uproot_job(
        fileset, treename="Events",
        processor_instance=proc_cls(workflow=WORKFLOW, year="2022postEE",
                                    output_format="parquet", output_location=str(out_dir) + "/"),
        executor=processor.iterative_executor, executor_args={"schema": NanoAODSchema, "workers": 1},
    )


def standalone_genc(url):
    import uproot
    import awkward as ak
    a = uproot.open(url)["Events"].arrays(["genWeight", "GenJet_pt", "GenJet_hadronFlavour"])
    n = ak.to_numpy(ak.sum((a.GenJet_hadronFlavour == 4) & (a.GenJet_pt > 10.0), axis=1))
    w = ak.to_numpy(a.genWeight).astype(np.float64)
    return [float(w[n == 0].sum()), float(w[n == 1].sum()), float(w[n >= 2].sum())], float(w.sum())


def main():
    import pyarrow.parquet as pq
    parts = json.loads(PARTITIONS.read_text())
    first = sorted(parts, key=int)[0]
    key = list(parts[first])[0]
    one_file = parts[first][key][0]
    fileset = {key: [one_file]}
    print(f"Testing against: {one_file}")

    old_cls = load_processor(HERE / "analysis/processors/base.py.bak_20261005_preveto_genc", "base_old")
    new_cls = load_processor(HERE / "analysis/processors/base.py", "base_new")
    run(old_cls, fileset, OUT / "old")
    run(new_cls, fileset, OUT / "new")

    # 1. parquets identical
    old_pq = {p.relative_to(OUT / "old"): p for p in (OUT / "old").glob("**/*.parquet")}
    new_pq = {p.relative_to(OUT / "new"): p for p in (OUT / "new").glob("**/*.parquet")}
    assert old_pq and set(old_pq) == set(new_pq), "parquet file sets differ"
    # Reco-level columns are NOT reproducible run to run, even old vs old: the muon scale and
    # smearing (analysis/corrections/muon_ss.py:125 random(), :453 np.random.rand) is unseeded
    # (found 2026-10-05: 297/862 columns differ between two runs of the unchanged processor).
    # So compare schemas, row counts and the GEN-level columns, which must be identical.
    gen_cols = ["n_genc10", "n_chad5", "n_chad10", "gen_chad1_pt"]
    for rel in old_pq:
        a, b = pq.read_table(old_pq[rel]), pq.read_table(new_pq[rel])
        assert a.schema.equals(b.schema), f"schema differs: {rel}"
        assert a.num_rows == b.num_rows, f"row count differs: {rel}"
        for c in gen_cols:
            if c in a.schema.names:
                assert a.column(c).equals(b.column(c)), f"GEN column {c} differs: {rel}"
    print(f"1. {len(old_pq)} parquets: same schema, row counts and GEN columns between old and new processor")

    # 2. old sidecar keys unchanged
    (js_old,) = (OUT / "old").glob("**/sumw/*.json")
    (js_new,) = (OUT / "new").glob("**/sumw/*.json")
    r_old, r_new = json.loads(js_old.read_text()), json.loads(js_new.read_text())
    for k, v in r_old.items():
        assert r_new.get(k) == v, f"sidecar key changed: {k}"
    added = sorted(set(r_new) - set(r_old))
    print(f"2. all {len(r_old)} old sidecar keys unchanged; added: {added}")

    # 3. new keys consistent
    pre = [r_new[f"sumw_preveto_genc{b}"] for b in ("0", "1", "2p")]
    assert abs(sum(pre) - r_new["sumw_preveto"]) < 1e-6 * abs(r_new["sumw_preveto"]), (pre, r_new["sumw_preveto"])
    for b in ("0", "1", "2p"):
        assert len(r_new[f"sumw_preveto_genc{b}_scalevar"]) == 9
    post = [r_new[f"sumw_genc{b}"] for b in ("0", "1", "2p")]
    print(f"3. preveto genc 0/1/2p {np.round(pre, 3)} (sum {sum(pre):.3f} = sumw_preveto {r_new['sumw_preveto']:.3f})")
    print(f"   postveto genc 0/1/2p {np.round(post, 3)} (sum {sum(post):.3f} = sumw {r_new['sumw']:.3f})")
    print(f"   fractions pre {np.round(np.array(pre) / sum(pre), 4)}  post {np.round(np.array(post) / sum(post), 4)}")

    # 4. independent computation
    ref, tot = standalone_genc(one_file)
    # the processor sums float32 genWeight (as for the existing sumw key); standalone is float64
    assert np.allclose(pre, ref, rtol=1e-6), (pre, ref)
    assert abs(tot - r_new["sumw_preveto"]) < 1e-6 * abs(tot)
    print(f"4. standalone uproot genc sums {np.round(ref, 3)} match (rtol 1e-6, float32 vs float64)")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
