#!/usr/bin/env python3
"""
Real single-file smoke test for the muon efficiency SF wiring added
2026-08-21 (see README_HcZZ.md's 2026-08-21 dated update and second-brain
memory hczz_systematics_completeness_audit §8 for the full writeup).

Runs the ACTUAL BaseProcessor (same class submit.py uses on a condor worker)
over one real file via coffea's iterative_executor, in-process -- no condor,
no apptainer needed (coffea imports directly in LCG_105). This exercises the
real NanoAOD-level correction code path, unlike rebuilding a datacard from
already-scored parquets (which never calls analysis/corrections/*.py at all
for a column that isn't already sitting in the parquet). This is exactly the
kind of check that caught two real regressions during the muon-SF work
(lhepdf.py's lhe_alphaS NameError, analysis/data/nnlo_ps's missing
__init__.py) that a datacard-only check could not have found.

Usage (LCG_105 env, needs a valid voms proxy for the xrootd read):
    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    voms-proxy-init --voms cms --valid 192:00   # if not already valid
    cd /eos/user/s/snandaku/Higgscharmnew/higgscharm
    python3 smoketest_muon_sf.py

Expected output (as of 2026-08-21, GluGluHtoZZto4L/2023postBPix, first file
in condor/hplusc_mva_4class_ctag2d/2023postBPix/GluGluHtoZZto4L/partitions.json):
    DONE, output keys: ['metadata']
    191 events selected
    weight_CMS_eff_m_id_2023 columns present: True
    weight_CMS_eff_m_id_2023        mean=0.999...  (near 1, sane)
    weight_CMS_eff_m_id_2023Up      mean=~weight_nominal, +small spread
    weight_CMS_eff_m_id_2023Down    mean=~weight_nominal, -small spread

Output parquet is written to /tmp (see OUT_DIR below) -- this is a throwaway
validation run, NOT a production output. It does not touch any real
production output tree and does not change any existing combine number.
"""

import json
import shutil
import tempfile
from pathlib import Path

from coffea import processor
from coffea.nanoevents import NanoAODSchema

REPO = Path(__file__).parent
PARTITION_SOURCE = (
    REPO / "condor" / "hplusc_mva_4class_ctag2d" / "2023postBPix"
    / "GluGluHtoZZto4L" / "partitions.json"
)
OUT_DIR = Path(tempfile.gettempdir()) / "hczz_muon_sf_smoketest_output"


def main():
    from analysis.processors.base import BaseProcessor

    # Build a single-file fileset from the first file of an existing, real
    # condor partition (avoids needing DAS/Rucio access just for a smoke test).
    partitions = json.loads(PARTITION_SOURCE.read_text())
    one_file = partitions["1"]["GluGluHtoZZto4L"][0]
    fileset = {"GluGluHtoZZto4L": [one_file]}
    print(f"Testing against: {one_file}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    out = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=BaseProcessor(
            workflow="hplusc_mva_4class_ctag2d",
            year="2023postBPix",
            output_format="parquet",
            output_location=str(OUT_DIR) + "/",
        ),
        executor=processor.iterative_executor,
        executor_args={"schema": NanoAODSchema, "workers": 1},
    )
    print("DONE, output keys:", list(out.keys()) if hasattr(out, "keys") else out)

    # Inspect the written parquet for the new muon-SF columns.
    import pyarrow.parquet as pq
    import pandas as pd

    parquet_files = list(OUT_DIR.glob("**/*.parquet"))
    assert parquet_files, f"No parquet written under {OUT_DIR}"
    pf = parquet_files[0]
    cols = pq.ParquetFile(pf).schema.names
    muon_cols = [c for c in cols if "CMS_eff_m" in c]
    print(f"weight_CMS_eff_m_id_2023 columns present: {bool(muon_cols)} -> {muon_cols}")
    assert muon_cols, "muon SF columns missing -- wiring regressed!"

    df = pd.read_parquet(pf, columns=["weight_nominal"] + muon_cols)
    print(f"{len(df)} events selected")
    print(df.describe())

    # Sanity bounds -- a real per-jet/per-lepton efficiency SF should sit
    # close to 1 with a small spread, not blow up or degenerate to exactly 1
    # for every event (that would mean the correction silently no-op'd for
    # everyone, e.g. every muon out of the calibration's pT/eta range).
    sf_col = [c for c in muon_cols if c.endswith("2023")][0]
    mean_sf = df[sf_col].mean()
    assert 0.9 < mean_sf < 1.1, f"muon ID SF mean {mean_sf} outside sane [0.9,1.1] range"
    assert df[sf_col].std() > 0, "muon ID SF is degenerate (zero spread) -- check wiring"
    print(f"\nSanity checks passed: mean SF={mean_sf:.4f}, std={df[sf_col].std():.4f}")


if __name__ == "__main__":
    main()
