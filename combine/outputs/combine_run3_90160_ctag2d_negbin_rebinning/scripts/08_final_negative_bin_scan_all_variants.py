import uproot
import numpy as np

def scan(path, label, exclude=()):
    f = uproot.open(path)
    keys = sorted(set(k.split(';')[0] for k in f.keys()))
    neg = []
    n_th1 = 0
    for k in keys:
        if k in exclude:
            continue
        obj = f[k]
        if not getattr(obj, "classname", "").startswith("TH1"):
            continue
        n_th1 += 1
        vals = obj.values()
        if np.any(vals < 0):
            neg.append((k, [(int(i), round(float(vals[i]), 8)) for i in np.where(vals < 0)[0]], float(vals.sum())))
    print(f"=== {label} ===")
    print(f"  TH1 checked: {n_th1}  |  negative-bin histograms: {len(neg)}")
    for name, bins, tot in neg:
        print(f"    {name}: total={tot:.4e}  negative_bins={bins}")
    print()

scan("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d/histograms_mva_with_zx.root",
     "Original 20-bin, WITH ZX (all processes incl. h_ZX)")

scan("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d/histograms_mva_with_zx.root",
     "Original 20-bin, NO ZX (h_ZX/h_data_obs excluded)", exclude=("h_ZX", "h_data_obs"))

scan("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_negbin_rebinning/rebinning_tests/10bins/histograms_mva_with_zx.root",
     "10-bin uniform, WITH ZX")

scan("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_negbin_rebinning/rebinning_tests/8bins/histograms_mva_with_zx.root",
     "8-bin uniform, WITH ZX")

scan("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_negbin_rebinning/rebinning_tests/surgical_v1_noZX/histograms_surgical_noZX.root",
     "Surgical v1 (bins 7-10 merged only), NO ZX")

scan("/eos/user/s/snandaku/Higgscharmnew/higgscharm/combine/outputs/combine_run3_90160_ctag2d_negbin_rebinning/surgical_v2_noZX/histograms_surgical_v2_noZX.root",
     "Surgical v2 (bins 2-4 AND 7-10 merged), NO ZX  <-- final candidate")
