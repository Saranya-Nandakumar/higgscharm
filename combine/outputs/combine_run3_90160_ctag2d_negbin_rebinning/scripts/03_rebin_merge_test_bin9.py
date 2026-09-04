import os, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCORED_DIR = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored"
ERAS = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]
MASS_WINDOW = (90, 160)

# original 20 quantile edges from the ROOT file (bins 0..19)
EDGES = [0.000000, 0.000957, 0.001176, 0.001406, 0.001688, 0.002030, 0.002450,
         0.002956, 0.003558, 0.004268, 0.005078, 0.006105, 0.007523, 0.009719,
         0.013643, 0.022087, 0.043934, 0.100734, 0.237676, 0.495754, 1.000000]

all_scores = []
all_weights = []
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
            except Exception:
                continue
            mask = (df["zz_mass_inclusive"] >= MASS_WINDOW[0]) & (df["zz_mass_inclusive"] < MASS_WINDOW[1])
            df = df[mask]
            df = df[df["mva_score_Signal"] >= 0]
            if len(df) == 0:
                continue
            all_scores.append(df["mva_score_Signal"].values)
            all_weights.append(df["weight_nominal"].values)

scores = np.concatenate(all_scores)
weights = np.concatenate(all_weights)
print(f"Total raw Signal events (mass window, valid score): {len(scores)}, files={n_files}")
print(f"Overall negative-weight fraction: {(weights<0).mean():.4%}")
print(f"Raw sum of weights (whole window): {weights.sum():.6f}")

# reproduce scale factor from the known scaled total (0.0518 from datacard rate line)
raw_total = weights.sum()
scaled_total_target = 0.0518  # from datacard "rate" row for Signal
scale = scaled_total_target / raw_total
print(f"Inferred scale factor: {scale:.6e}")

h_raw, _ = np.histogram(scores, bins=EDGES, weights=weights)
h_scaled = h_raw * scale
print("\nOriginal 20-bin (raw / scaled) around bin 9:")
for i in range(6, 13):
    lo, hi = EDGES[i], EDGES[i+1]
    print(f"  bin {i:2d} [{lo:.6f},{hi:.6f}): raw={h_raw[i]:.4f}  scaled={h_scaled[i]:.6e}")

# Now test coarser binning: merge bin9 into neighbors various ways
print("\n--- Rebinning tests ---")
def merged_value(lo_idx, hi_idx):
    lo, hi = EDGES[lo_idx], EDGES[hi_idx]
    mask = (scores >= lo) & (scores < hi)
    raw = weights[mask].sum()
    n = mask.sum()
    return raw, n, raw*scale

for (a,b,label) in [(8,10,"merge bin8+9"), (9,11,"merge bin9+10"), (8,11,"merge bin8+9+10"), (7,11,"merge bin7+8+9+10")]:
    raw, n, scaled = merged_value(a,b)
    flag = "  <<<< STILL NEGATIVE" if scaled < 0 else "  (positive)"
    print(f"  {label:22s} [{EDGES[a]:.6f},{EDGES[b]:.6f}): n_events={n:3d} raw={raw:.4f} scaled={scaled:.6e}{flag}")

# Test a fully independent coarser quantile-like binning: just split into 10 bins total (halve resolution)
# by pairing up bins 0-1, 2-3, ... 18-19 from the original 20 edges (crude, but shows the effect of "half the bins")
print("\n--- Halved bin-count test (merge consecutive original-edge pairs) ---")
for i in range(0, 20, 2):
    lo, hi = EDGES[i], EDGES[i+2]
    mask = (scores >= lo) & (scores < hi)
    raw = weights[mask].sum()
    n = mask.sum()
    scaled = raw*scale
    flag = "  <<<< NEGATIVE" if scaled < 0 else ""
    print(f"  merged-bin {i//2:2d} [{lo:.6f},{hi:.6f}): n={n:3d} raw={raw:.4f} scaled={scaled:.6e}{flag}")
