import uproot
import numpy as np

def scan(path, label):
    print(f"\n=== {label} : {path} ===")
    f = uproot.open(path)
    keys = sorted(set(k.split(';')[0] for k in f.keys()))
    total_hists = 0
    neg_hists = []
    for k in keys:
        obj = f[k]
        cname = getattr(obj, "classname", "")
        if not cname.startswith("TH1"):
            continue
        total_hists += 1
        vals = obj.values()
        if np.any(vals < 0):
            neg_bins = np.where(vals < 0)[0]
            neg_hists.append((k, [(int(i), float(vals[i])) for i in neg_bins], float(vals.sum())))
    print(f"Total TH1 histograms: {total_hists}")
    print(f"Histograms with >=1 negative bin: {len(neg_hists)}")
    for name, bins, total in neg_hists:
        print(f"  {name}: total_integral={total:.4f}  negative_bins={bins}")
    return neg_hists

paths = [
    ("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d/histograms_mva_with_zx.root", "ctag2d, with-ZX"),
    ("/eos/user/s/snandaku/Analysis/combine_run3_90160_v5/histograms_mva_with_zx.root", "1D-SF production, with-ZX"),
]

for path, label in paths:
    try:
        scan(path, label)
    except Exception as e:
        print(f"\n=== {label} : {path} ===")
        print(f"ERROR: {e}")
