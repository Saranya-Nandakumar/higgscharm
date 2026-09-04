import os, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCORED_DIR = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored"
ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
MASS_WINDOW = (90, 160)
BIN9_LO, BIN9_HI = 0.004268, 0.005078

rows = []
n_files = 0
for era in ERAS:
    era_dir = os.path.join(SCORED_DIR, era)
    for sample_name in os.listdir(era_dir):
        if not ("HPlusCharm" in sample_name or "SomeSMSignal" in sample_name):
            continue
        sample_dir = os.path.join(era_dir, sample_name, "base")
        if not os.path.isdir(sample_dir):
            continue
        for pqf in glob.glob(os.path.join(sample_dir, "*.parquet")):
            n_files += 1
            try:
                df = pq.read_table(pqf, columns=["zz_mass_inclusive", "mva_score_Signal", "weight_nominal"]).to_pandas()
            except Exception as e:
                print("SKIP", pqf, e)
                continue
            mask = (df["zz_mass_inclusive"] >= MASS_WINDOW[0]) & (df["zz_mass_inclusive"] < MASS_WINDOW[1])
            df = df[mask]
            df = df[df["mva_score_Signal"] >= 0]
            if len(df) == 0:
                continue
            in_bin = df[(df["mva_score_Signal"] >= BIN9_LO) & (df["mva_score_Signal"] < BIN9_HI)]
            if len(in_bin) > 0:
                for _, r in in_bin.iterrows():
                    rows.append((era, sample_name, os.path.basename(pqf), r["mva_score_Signal"], r["weight_nominal"]))

print(f"Scanned {n_files} parquet files across {len(ERAS)} eras (Signal-matching samples only)")
print(f"Events landing in bin9 [{BIN9_LO},{BIN9_HI}): {len(rows)}")
print()
total_raw = 0.0
for era, sample, fn, score, w in rows:
    total_raw += w
    flag = "  <<<< NEGATIVE WEIGHT" if w < 0 else ""
    print(f"{era:14s} {sample:28s} {fn:30s} score={score:.6f}  weight_nominal={w:.6e}{flag}")
print()
print(f"Sum of raw weight_nominal in bin9: {total_raw:.6e}")
