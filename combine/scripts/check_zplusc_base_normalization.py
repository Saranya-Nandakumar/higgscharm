#!/usr/bin/env python3
"""Quick sanity check: is the ~0.5 Data/MC deficit seen in CR_Zplusc also
present in the `base` (no c-jet requirement) category of the same
ztomumu/ztoee workflows, same 2022postEE production? Isolates whether the
deficit is c-tag-related or a pre-existing whole-workflow normalization gap.
Reuses sumw/xsec machinery from create_cr_zplusc_ctag2d_check.py as-is."""
import sys
sys.path.insert(0, "/tmp/snandaku/claude-178308/-eos-home-s-snandaku-second-brain/02880980-8f56-408d-9125-9511cc3bb4ee/scratchpad")
import os, glob
import numpy as np
import pyarrow.parquet as pq
import create_cr_zplusc_ctag2d_check as m

OUTPUT_BASE = m.OUTPUT_BASE
ERAS = ["2022postEE"]

for channel in ["ztomumu"]:
    era_dir_base = os.path.join(OUTPUT_BASE, channel)
    sample_meta = {}
    sumw_by_era_dataset = {}
    for era in ERAS:
        era_dir = os.path.join(era_dir_base, era)
        dscfg = m.get_dscfg(era)
        for sample_name in sorted(os.listdir(era_dir)):
            sample_cat_dir = os.path.join(era_dir, sample_name, "base")
            if not os.path.isdir(sample_cat_dir):
                continue
            dataset_key, entry = m.resolve_dataset(sample_name, dscfg)
            if entry is None:
                continue
            sample_meta[(era, sample_name)] = (dataset_key, entry)
            if entry.get("process") != "Data":
                sumw, _ = m.load_sumw_for_dataset(era_dir, sample_name)
                sumw_by_era_dataset[(era, dataset_key)] = sumw_by_era_dataset.get((era, dataset_key), 0.0) + sumw

    data_n = 0
    mc_yield = 0.0
    mc_by_group = {}
    for era in ERAS:
        era_dir = os.path.join(era_dir_base, era)
        lumi = m.LUMI[era]
        for sample_name in sorted(os.listdir(era_dir)):
            meta = sample_meta.get((era, sample_name))
            if meta is None:
                continue
            dataset_key, entry = meta
            is_data = entry.get("process") == "Data"
            group = entry.get("key")
            if isinstance(group, list):
                group = group[0] if group else "other"
            scale = 1.0
            if not is_data:
                sumw = sumw_by_era_dataset.get((era, dataset_key), 0.0)
                xsec = entry.get("xsec")
                if not sumw or xsec is None:
                    continue
                scale = xsec * lumi / sumw
            mass_col = "dimuon_mass" if channel == "ztomumu" else "dielectron_mass"
            sample_cat_dir = os.path.join(era_dir, sample_name, "base")
            for pq_file in glob.glob(os.path.join(sample_cat_dir, "*.parquet")):
                read_cols = [mass_col] + ([] if is_data else ["weight_nominal"])
                try:
                    df = pq.read_table(pq_file, columns=read_cols).to_pandas()
                except Exception as e:
                    print(f"    WARNING: {pq_file}: {e}")
                    continue
                n = len(df)
                if is_data:
                    data_n += n
                else:
                    w = df["weight_nominal"].fillna(0).to_numpy().sum() * scale
                    mc_yield += w
                    mc_by_group[group] = mc_by_group.get(group, 0.0) + w

    print(f"\n=== {channel} base category (2022postEE) ===")
    print(f"  Data: {data_n}")
    print(f"  MC total: {mc_yield:.2f}")
    print(f"  Data/MC: {data_n/mc_yield:.3f}" if mc_yield else "  MC=0")
    for g, y in sorted(mc_by_group.items(), key=lambda x: -x[1]):
        print(f"    {g:<15}{y:>14.2f}")
