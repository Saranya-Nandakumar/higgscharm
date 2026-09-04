import os, glob
import numpy as np
import pyarrow.parquet as pq

SCORED_DIR = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored"
ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
MASS_WINDOW = (90, 160)
SCORE_MAX = 0.01  # covers old bins 0-12

cols_wanted = ["zz_mass_inclusive", "mva_score_Signal", "weight_nominal",
               "weight_CMS_ctag2d_2022Up", "weight_CMS_ctag2d_2023Up",
               "weight_CMS_pileup_2022Up", "weight_CMS_pileup_2023Up"]

scores, wnom, wctag, wpu = [], [], [], []
for era in ERAS:
    era_dir = os.path.join(SCORED_DIR, era)
    for sample_name in os.listdir(era_dir):
        if not ("HPlusCharm" in sample_name or "SomeSMSignal" in sample_name):
            continue
        sample_dir = os.path.join(era_dir, sample_name, "base")
        if not os.path.isdir(sample_dir):
            continue
        for pqf in glob.glob(os.path.join(sample_dir, "*.parquet")):
            avail = pq.ParquetFile(pqf).schema.names
            cols = ["zz_mass_inclusive", "mva_score_Signal", "weight_nominal"] + \
                   [c for c in cols_wanted[3:] if c in avail]
            df = pq.read_table(pqf, columns=cols).to_pandas()
            mask = (df["zz_mass_inclusive"] >= MASS_WINDOW[0]) & (df["zz_mass_inclusive"] < MASS_WINDOW[1])
            df = df[mask]
            df = df[df["mva_score_Signal"] >= 0]
            df = df[df["mva_score_Signal"] < SCORE_MAX]
            if len(df) == 0:
                continue
            scores.append(df["mva_score_Signal"].values)
            wnom.append(df["weight_nominal"].values)
            # combine era-specific columns (only one populated per era's files)
            c = df.get("weight_CMS_ctag2d_2022Up")
            if c is None or c.isna().all():
                c = df.get("weight_CMS_ctag2d_2023Up")
            p = df.get("weight_CMS_pileup_2022Up")
            if p is None or p.isna().all():
                p = df.get("weight_CMS_pileup_2023Up")
            wctag.append(c.fillna(df["weight_nominal"]).values if c is not None else df["weight_nominal"].values)
            wpu.append(p.fillna(df["weight_nominal"]).values if p is not None else df["weight_nominal"].values)

scores = np.concatenate(scores)
wnom = np.concatenate(wnom)
wctag = np.concatenate(wctag)
wpu = np.concatenate(wpu)
print(f"Total events with score<{SCORE_MAX}: {len(scores)}")

# original 20-bin edges up to bin 12
EDGES = [0.000000, 0.000957, 0.001176, 0.001406, 0.001688, 0.002030, 0.002450,
         0.002956, 0.003558, 0.004268, 0.005078, 0.006105, 0.007523, 0.009719]

print("\nPer-original-bin sums (nominal / ctag2dUp / pileupUp):")
for i in range(len(EDGES)-1):
    lo, hi = EDGES[i], EDGES[i+1]
    m = (scores>=lo)&(scores<hi)
    print(f"  bin{i:2d} [{lo:.6f},{hi:.6f}): n={m.sum():3d} nom={wnom[m].sum():+9.4f} ctag2dUp={wctag[m].sum():+9.4f} pileupUp={wpu[m].sum():+9.4f}")

print("\nCandidate merge: bins 1-4 (covers old bin3's neighborhood)")
lo, hi = EDGES[1], EDGES[5]
m = (scores>=lo)&(scores<hi)
print(f"  [{lo:.6f},{hi:.6f}): n={m.sum():3d} nom={wnom[m].sum():+9.4f} ctag2dUp={wctag[m].sum():+9.4f} pileupUp={wpu[m].sum():+9.4f}")

print("\nCandidate merge: bins 2-4")
lo, hi = EDGES[2], EDGES[5]
m = (scores>=lo)&(scores<hi)
print(f"  [{lo:.6f},{hi:.6f}): n={m.sum():3d} nom={wnom[m].sum():+9.4f} ctag2dUp={wctag[m].sum():+9.4f} pileupUp={wpu[m].sum():+9.4f}")

print("\nCandidate merge: bins 1-6 (wider)")
lo, hi = EDGES[1], EDGES[7]
m = (scores>=lo)&(scores<hi)
print(f"  [{lo:.6f},{hi:.6f}): n={m.sum():3d} nom={wnom[m].sum():+9.4f} ctag2dUp={wctag[m].sum():+9.4f} pileupUp={wpu[m].sum():+9.4f}")
