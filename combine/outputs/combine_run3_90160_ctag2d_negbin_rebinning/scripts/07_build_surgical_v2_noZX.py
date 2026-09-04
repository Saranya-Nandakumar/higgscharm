import sys, os
sys.path.insert(0, "/eos/home-s/snandaku/b-hive_ttcc/combine")
import create_datacards_ctag2d as cd
import numpy as np
import uproot

SCORED_DIR = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d_scored"
SUMW_DIR   = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_ctag2d"
OUTPUT     = "/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_negbin_rebinning/surgical_v2_noZX"

print("[1a] sumw")
sumw_by_proc = cd.load_sumw(SUMW_DIR)

print("\n[1b] MC scored parquets")
mc_data = cd.load_mc_scored_parquets(SCORED_DIR, sumw_by_proc)

print("\n[1c] Original 20-bin quantile edges")
orig_edges = cd.compute_quantile_bins(mc_data, n_bins=20)
print(np.round(orig_edges, 6))

# Surgical merge: collapse the sparse low-score bins 7,8,9,10 (the ones that
# showed negative content in Signal's nominal + systematic-varied histograms)
# into one wide bin. Drop the interior edges at index 8,9,10; keep everything
# else untouched -- full 20-bin resolution everywhere Signal/qqZZ actually
# have statistics.
drop_idx = {3, 4, 8, 9, 10}
surgical_edges = np.array([e for i, e in enumerate(orig_edges) if i not in drop_idx])
print(f"\nSurgical edges ({len(surgical_edges)-1} bins), merged region "
      f"[{orig_edges[7]:.6f},{orig_edges[11]:.6f}) replacing old bins 7-10:")
print(np.round(surgical_edges, 6))

cd.MVA_BINS = surgical_edges

print("\n[3] Building histograms (MC only, no ZX)")
histograms = cd.build_histograms(mc_data, np.array([]), np.array([]))
histograms.pop("ZX", None)  # no reducible background in this test

os.makedirs(OUTPUT, exist_ok=True)

# --- write ROOT file (MC processes + their systematic variants only) ---
root_path = os.path.join(OUTPUT, "histograms_surgical_v2_noZX.root")
nominal_procs = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"]
with uproot.recreate(root_path, compression=None) as f:
    for proc, h in histograms.items():
        f[f"h_{proc}"] = (h, cd.MVA_BINS)
    data_obs = sum(histograms[p] for p in nominal_procs if p in histograms)
    f["h_data_obs"] = (data_obs, cd.MVA_BINS)
print(f"\nROOT file: {root_path}")

# --- write no-ZX datacard (same USABLE_SYST wiring as production, no ZX row) ---
dc_path = os.path.join(OUTPUT, "datacard_surgical_v2_noZX.txt")
procs = ["Signal", "ggZZ", "qqZZ", "Other_Higgs"]
proc_idx = [0, 1, 2, 3]
yields = {p: histograms[p].sum() for p in procs}
col_w = 16

def row(label, values):
    return label + "".join(f"{v:<{col_w}}" for v in values) + "\n"

with open(dc_path, "w") as dc:
    dc.write("# H+c -> ZZ -> 4l: MVA shape analysis, ctag2d, SURGICAL REBIN, NO Z+X\n")
    dc.write(f"# Mass window: {cd.MASS_WINDOW[0]}-{cd.MASS_WINDOW[1]} GeV | Lumi: {cd.TOTAL_LUMI_PB:.1f} pb^-1\n")
    dc.write(f"# Bins: merged old-20-bin edges 7-10 into one ({len(cd.MVA_BINS)-1} bins total)\n")
    dc.write("imax 1\njmax %d\nkmax *\n" % (len(procs) - 1))
    dc.write("-" * 80 + "\n")
    dc.write(f"shapes * * histograms_surgical_v2_noZX.root h_$PROCESS h_$PROCESS_$SYSTEMATIC\n")
    dc.write("-" * 80 + "\n")
    dc.write("bin         hczz\n")
    dc.write("observation -1\n")
    dc.write("-" * 80 + "\n")
    dc.write(row("bin         ", ["hczz"] * len(procs)))
    dc.write(row("process     ", procs))
    dc.write(row("process     ", [str(i) for i in proc_idx]))
    dc.write(row("rate        ", [f"{yields[p]:.6f}" for p in procs]))
    dc.write("-" * 80 + "\n")

    def syst_row(name, stype, vals):
        line = f"{name:<25}{stype:<8}"
        for p in procs:
            line += f"{vals.get(p, '-'):<{col_w}}"
        return line + "\n"

    dc.write(syst_row("lumi_Run3", "lnN", {p: "1.014" for p in procs}))
    dc.write(syst_row("pdf_gg", "lnN", {"ggZZ": "1.05", "Signal": "1.05"}))
    dc.write(syst_row("QCDscale_ggZZ", "lnN", {"ggZZ": "1.10"}))
    dc.write("\n# Shape systematics\n")
    for syst in cd.SYSTEMATICS:
        vals = {p: "1" for p in procs if syst in cd.USABLE_SYST.get(p, set())}
        if not vals:
            continue
        dc.write(syst_row(syst, "shape", vals))

print(f"Datacard: {dc_path}")

print("\n[Summary]")
total_bkg = sum(yields[p] for p in procs if p != "Signal")
print(f"Signal={yields['Signal']:.6f}  ggZZ={yields['ggZZ']:.6f}  qqZZ={yields['qqZZ']:.6f}  Other_Higgs={yields['Other_Higgs']:.6f}")
print(f"Total bkg={total_bkg:.4f}  S/sqrt(B)={yields['Signal']/total_bkg**0.5:.4f}")

# --- negative-bin scan ---
print("\n[Negative-bin scan]")
f = uproot.open(root_path)
keys = sorted(set(k.split(';')[0] for k in f.keys()))
neg = []
n_th1 = 0
for k in keys:
    obj = f[k]
    if not getattr(obj, "classname", "").startswith("TH1"):
        continue
    n_th1 += 1
    vals = obj.values()
    if np.any(vals < 0):
        neg.append((k, [(int(i), float(vals[i])) for i in np.where(vals < 0)[0]], float(vals.sum())))
print(f"Total TH1: {n_th1}, with negative bins: {len(neg)}")
for name, bins, tot in neg:
    print(f"  {name}: total={tot:.4e} negative_bins={bins}")
