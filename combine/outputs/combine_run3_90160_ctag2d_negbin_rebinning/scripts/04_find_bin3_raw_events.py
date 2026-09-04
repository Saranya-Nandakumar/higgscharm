import os, glob
import numpy as np
import pyarrow.parquet as pq

SCORED_DIR = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored"
ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
MASS_WINDOW = (90, 160)
BIN3_LO, BIN3_HI = 0.001406, 0.001688

cols_wanted = ["zz_mass_inclusive", "mva_score_Signal", "weight_nominal",
               "weight_CMS_ctag2d_2022Up", "weight_CMS_ctag2d_2023Up",
               "weight_CMS_pileup_2022Up", "weight_CMS_pileup_2023Up"]

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
                avail = pq.ParquetFile(pqf).schema.names
                cols = ["zz_mass_inclusive","mva_score_Signal","weight_nominal"] + [c for c in cols_wanted[3:] if c in avail]
                df = pq.read_table(pqf, columns=cols).to_pandas()
            except Exception as e:
                print("SKIP", pqf, e); continue
            mask = (df["zz_mass_inclusive"] >= MASS_WINDOW[0]) & (df["zz_mass_inclusive"] < MASS_WINDOW[1])
            df = df[mask]
            df = df[df["mva_score_Signal"] >= 0]
            if len(df) == 0:
                continue
            in_bin = df[(df["mva_score_Signal"] >= BIN3_LO) & (df["mva_score_Signal"] < BIN3_HI)]
            if len(in_bin) > 0:
                for _, r in in_bin.iterrows():
                    ctag2d = r.get("weight_CMS_ctag2d_2022Up", r.get("weight_CMS_ctag2d_2023Up", np.nan))
                    pileup = r.get("weight_CMS_pileup_2022Up", r.get("weight_CMS_pileup_2023Up", np.nan))
                    rows.append((era, sample_name, r["mva_score_Signal"], r["weight_nominal"], ctag2d, pileup))

print(f"Scanned {n_files} files; events in bin3 [{BIN3_LO},{BIN3_HI}): {len(rows)}")
print()
tot_nom = tot_ctag = tot_pu = 0.0
for era, sample, score, wnom, wctag, wpu in rows:
    tot_nom += wnom
    tot_ctag += wctag if not np.isnan(wctag) else wnom
    tot_pu += wpu if not np.isnan(wpu) else wnom
    print(f"{era:14s} {sample:26s} score={score:.6f} nominal={wnom:+.4e} ctag2dUp={wctag:+.4e} pileupUp={wpu:+.4e}")
print()
print(f"Sum nominal={tot_nom:.4e}  Sum ctag2dUp={tot_ctag:.4e}  Sum pileupUp={tot_pu:.4e}")
