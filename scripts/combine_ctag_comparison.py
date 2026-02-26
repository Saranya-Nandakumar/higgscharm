#!/usr/bin/env python3
"""
Combine Comparison: C-tagging Categories vs MVA Score

Reads parquet files from higgscharm framework output with pre-computed
c-tag categories (leadjet_ctag_category) and creates datacards for combine.

Usage:
    # Compare c-tag categories (from hplusc_mva) vs MVA (from hplusc)
    python scripts/combine_ctag_comparison.py \
        --ctag-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva \
        --mva-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc \
        --output-dir /eos/user/s/snandaku/Analysis/combine/ctag_vs_mva

    # Or use same directory for both (if workflow has both)
    python scripts/combine_ctag_comparison.py \
        --ctag-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc \
        --output-dir /eos/user/s/snandaku/Analysis/combine/comparison
"""

import os
import sys
import argparse
import numpy as np
import glob
from collections import defaultdict

try:
    import pyarrow.parquet as pq
    import uproot
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyarrow uproot")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

# Cross-sections in pb
PROCESS_XS_PB = {
    "signal": 0.000026,
    "ggH": 0.01434,
    "qqZZ": 1.39 + 0.027,
    "hplusb": 0.000173,
    "other_higgs": 0.00112 + 0.000156 + 0.000244 + 0.000775 + 0.00312 + 0.000144,
}

# Luminosity per era in pb^-1
LUMINOSITY_PB = {
    "2022preEE": 7980.4,
    "2022postEE": 26671.7,
    "2023preBPix": 17794.0,
    "2023postBPix": 9450.0,
}

# Process mapping
PROCESS_MAPPING = {
    "SomeSMSignal": "signal",
    "2023postBPixHB": "hplusb",
    "2023preBPixHB": "hplusb",
    "2022postEEHB": "hplusb",
    "2022preEEHB": "hplusb",
    "GluGluHtoZZto4L": "ggH",
    "ZZto4L": "qqZZ",
    "GluGlutoContinto2Zto4E": "qqZZ",
    "GluGlutoContinto2Zto4Mu": "qqZZ",
    "GluGlutoContinto2Zto4Tau": "qqZZ",
    "GluGluToContinto2Zto2E2Mu": "qqZZ",
    "GluGluToContinto2Zto2E2Tau": "qqZZ",
    "GluGluToContinto2Zto2Mu2Tau": "qqZZ",
    "VBFHto2Zto4L": "other_higgs",
    "WminusH_Hto2Zto4L": "other_higgs",
    "WplusH_Hto2Zto4L": "other_higgs",
    "ZHto2Zto4L": "other_higgs",
    "TTH_Hto2Z": "other_higgs",
    "bbH_Hto2Zto4L": "other_higgs",
}

# Category configuration
CATEGORY_ORDER = ["L0", "C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2", "B3", "B4"]
COMBINE_PROCESSES = ["signal", "ggH", "hplusb", "qqZZ", "other_higgs"]

# MVA binning
MVA_BINS = np.linspace(0, 1, 21)

# Mass window
MASS_WINDOW = (100, 150)


def get_process_name(sample_name):
    """Map sample directory name to combine process name."""
    base_name = sample_name.rstrip('0123456789').rstrip('_')
    for key, proc in PROCESS_MAPPING.items():
        if key in sample_name or key == base_name:
            return proc
    return None


def load_ctag_data(input_dir, eras, verbose=True):
    """Load c-tag category data from parquet files."""
    if verbose:
        print(f"\n  Loading c-tag data from: {input_dir}")

    data_by_process = defaultdict(lambda: {
        'categories': [],
        'weights': [],
        'n_events': 0,
    })
    raw_yields = defaultdict(float)

    for era in eras:
        era_path = os.path.join(input_dir, era)
        if not os.path.exists(era_path):
            continue

        samples = [d for d in os.listdir(era_path)
                   if os.path.isdir(os.path.join(era_path, d))]

        for sample in samples:
            proc_name = get_process_name(sample)
            if proc_name is None:
                continue

            base_path = os.path.join(era_path, sample, 'base')
            if not os.path.exists(base_path):
                continue

            parquet_files = glob.glob(os.path.join(base_path, '*.parquet'))

            for pq_file in parquet_files:
                try:
                    df = pq.read_table(pq_file).to_pandas()

                    # Mass window cut
                    if 'm4l' in df.columns:
                        mask = (df['m4l'] >= MASS_WINDOW[0]) & (df['m4l'] <= MASS_WINDOW[1])
                        df = df[mask]

                    if len(df) == 0:
                        continue

                    weights = df['weight_nominal'].values if 'weight_nominal' in df.columns else np.ones(len(df))
                    raw_yields[proc_name] += weights.sum()

                    # Use pre-computed category
                    if 'leadjet_ctag_category' in df.columns:
                        categories = df['leadjet_ctag_category'].values
                        valid_mask = categories >= 0
                        categories = categories[valid_mask]
                        valid_weights = weights[valid_mask]

                        if len(categories) > 0:
                            data_by_process[proc_name]['categories'].extend(categories.tolist())
                            data_by_process[proc_name]['weights'].extend(valid_weights.tolist())
                            data_by_process[proc_name]['n_events'] += len(categories)

                except Exception as e:
                    if verbose:
                        print(f"    Error: {pq_file}: {e}")

    # Convert to arrays
    for proc in data_by_process:
        data_by_process[proc]['categories'] = np.array(data_by_process[proc]['categories'], dtype=np.int32)
        data_by_process[proc]['weights'] = np.array(data_by_process[proc]['weights'], dtype=np.float32)

    return data_by_process, raw_yields


def load_mva_data(input_dir, eras, verbose=True):
    """Load MVA score data from parquet files."""
    if verbose:
        print(f"\n  Loading MVA data from: {input_dir}")

    data_by_process = defaultdict(lambda: {
        'mva_scores': [],
        'weights': [],
        'n_events': 0,
    })
    raw_yields = defaultdict(float)

    for era in eras:
        era_path = os.path.join(input_dir, era)
        if not os.path.exists(era_path):
            continue

        samples = [d for d in os.listdir(era_path)
                   if os.path.isdir(os.path.join(era_path, d))]

        for sample in samples:
            proc_name = get_process_name(sample)
            if proc_name is None:
                continue

            base_path = os.path.join(era_path, sample, 'base')
            if not os.path.exists(base_path):
                continue

            parquet_files = glob.glob(os.path.join(base_path, '*.parquet'))

            for pq_file in parquet_files:
                try:
                    df = pq.read_table(pq_file).to_pandas()

                    # Mass window cut
                    if 'm4l' in df.columns:
                        mask = (df['m4l'] >= MASS_WINDOW[0]) & (df['m4l'] <= MASS_WINDOW[1])
                        df = df[mask]

                    # Require at least 1 jet
                    if 'n_jet' in df.columns:
                        df = df[df['n_jet'] >= 1]

                    if len(df) == 0:
                        continue

                    weights = df['weight_nominal'].values if 'weight_nominal' in df.columns else np.ones(len(df))
                    raw_yields[proc_name] += weights.sum()

                    # Get MVA scores
                    if 'mva_signal_score' in df.columns:
                        mva_scores = df['mva_signal_score'].values
                        valid_mask = ~np.isnan(mva_scores)
                        mva_scores = mva_scores[valid_mask]
                        valid_weights = weights[valid_mask]

                        if len(mva_scores) > 0:
                            data_by_process[proc_name]['mva_scores'].extend(mva_scores.tolist())
                            data_by_process[proc_name]['weights'].extend(valid_weights.tolist())
                            data_by_process[proc_name]['n_events'] += len(mva_scores)

                except Exception as e:
                    if verbose:
                        print(f"    Error: {pq_file}: {e}")

    # Convert to arrays
    for proc in data_by_process:
        data_by_process[proc]['mva_scores'] = np.array(data_by_process[proc]['mva_scores'], dtype=np.float32)
        data_by_process[proc]['weights'] = np.array(data_by_process[proc]['weights'], dtype=np.float32)

    return data_by_process, raw_yields


def apply_normalization(data_by_process, raw_yields, eras, verbose=True):
    """Apply cross-section normalization."""
    total_lumi = sum(LUMINOSITY_PB[era] for era in eras)
    expected_yields = {proc: xs * total_lumi for proc, xs in PROCESS_XS_PB.items()}

    for proc in COMBINE_PROCESSES:
        raw = raw_yields.get(proc, 0)
        expected = expected_yields.get(proc, 0)
        if raw > 0 and expected > 0:
            scale = expected / raw
            if proc in data_by_process:
                data_by_process[proc]['weights'] *= scale
            if verbose:
                print(f"    {proc}: scale = {scale:.4f} (raw={raw:.4f}, expected={expected:.4f})")

    return data_by_process


def create_ctag_histograms(data_by_process, output_dir, verbose=True):
    """Create c-tagging category histograms."""
    n_bins = len(CATEGORY_ORDER)
    histograms = {}

    if verbose:
        print(f"\n  C-tag histograms ({n_bins} categories):")
        print(f"  {'Process':<15} {'Events':>10} {'Weighted':>15}")
        print("  " + "-" * 45)

    for proc_name in COMBINE_PROCESSES:
        if proc_name not in data_by_process or data_by_process[proc_name]['n_events'] == 0:
            histograms[f"h_{proc_name}"] = np.zeros(n_bins, dtype=np.float64)
            if verbose:
                print(f"  {proc_name:<15} {'0':>10} {'0.0000':>15}")
            continue

        data = data_by_process[proc_name]
        categories = data['categories']
        weights = data['weights']

        hist, _ = np.histogram(categories, bins=n_bins, range=(0, n_bins), weights=weights)
        histograms[f"h_{proc_name}"] = hist.astype(np.float64)

        if verbose:
            print(f"  {proc_name:<15} {len(categories):>10} {weights.sum():>15.4f}")

    data_obs = sum(histograms.values())

    output_file = os.path.join(output_dir, "histograms_ctag.root")
    edges = np.arange(n_bins + 1, dtype=np.float64)

    with uproot.recreate(output_file) as f:
        for name, hist in histograms.items():
            f[name] = (hist, edges)
        f["h_data_obs"] = (data_obs, edges)

    if verbose:
        print(f"\n  Saved: {output_file}")

    return histograms


def create_mva_histograms(data_by_process, output_dir, verbose=True):
    """Create MVA signal score histograms."""
    n_bins = len(MVA_BINS) - 1
    histograms = {}

    if verbose:
        print(f"\n  MVA histograms ({n_bins} bins):")
        print(f"  {'Process':<15} {'Events':>10} {'Weighted':>15} {'Mean MVA':>10}")
        print("  " + "-" * 55)

    for proc_name in COMBINE_PROCESSES:
        if proc_name not in data_by_process or data_by_process[proc_name]['n_events'] == 0:
            histograms[f"h_{proc_name}"] = np.zeros(n_bins, dtype=np.float64)
            if verbose:
                print(f"  {proc_name:<15} {'0':>10} {'0.0000':>15} {'N/A':>10}")
            continue

        data = data_by_process[proc_name]
        mva_scores = data['mva_scores']
        weights = data['weights']

        hist, _ = np.histogram(mva_scores, bins=MVA_BINS, weights=weights)
        histograms[f"h_{proc_name}"] = hist.astype(np.float64)

        if verbose:
            mean_mva = np.average(mva_scores, weights=weights) if len(mva_scores) > 0 else 0
            print(f"  {proc_name:<15} {len(mva_scores):>10} {weights.sum():>15.4f} {mean_mva:>10.4f}")

    data_obs = sum(histograms.values())

    output_file = os.path.join(output_dir, "histograms_mva.root")

    with uproot.recreate(output_file) as f:
        for name, hist in histograms.items():
            f[name] = (hist, MVA_BINS)
        f["h_data_obs"] = (data_obs, MVA_BINS)

    if verbose:
        print(f"\n  Saved: {output_file}")

    return histograms


def create_datacard(output_dir, hist_file, histograms, method, verbose=True):
    """Create combine datacard."""
    datacard_path = os.path.join(output_dir, f"datacard_{method}.txt")
    yields = {proc: histograms[f"h_{proc}"].sum() for proc in COMBINE_PROCESSES}

    with open(datacard_path, 'w') as dc:
        dc.write(f"# H+c -> ZZ -> 4l: {method.upper()} method\n")
        dc.write(f"imax 1\njmax {len(COMBINE_PROCESSES)-1}\nkmax *\n")
        dc.write("-" * 80 + "\n")
        dc.write(f"shapes * * {hist_file} h_$PROCESS\n")
        dc.write("-" * 80 + "\n")
        dc.write("bin         hczz\nobservation -1\n")
        dc.write("-" * 80 + "\n")
        dc.write("bin         " + "".join([f"{'hczz':<15}" for _ in COMBINE_PROCESSES]) + "\n")
        dc.write("process     " + "".join([f"{p:<15}" for p in COMBINE_PROCESSES]) + "\n")
        dc.write("process     " + "".join([f"{i:<15}" for i in range(len(COMBINE_PROCESSES))]) + "\n")
        dc.write("rate        " + "".join([f"{yields[p]:<15.6f}" for p in COMBINE_PROCESSES]) + "\n")
        dc.write("-" * 80 + "\n")
        dc.write("lumi_13TeV      lnN    " + "1.025         " * len(COMBINE_PROCESSES) + "\n")
        dc.write("pdf_gg          lnN    " + "".join([
            "1.05          " if p in ["signal", "ggH", "hplusb"] else "-             "
            for p in COMBINE_PROCESSES
        ]) + "\n")
        dc.write("pdf_qq          lnN    " + "".join([
            "1.03          " if p == "qqZZ" else "-             "
            for p in COMBINE_PROCESSES
        ]) + "\n")

    if verbose:
        print(f"  Datacard: {datacard_path}")

    return datacard_path


def main():
    parser = argparse.ArgumentParser(description='Compare c-tag categories vs MVA for combine')
    parser.add_argument('--ctag-dir', '-c',
                        default='/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva',
                        help='Directory with c-tag category parquet files')
    parser.add_argument('--mva-dir', '-m',
                        default='/eos/user/s/snandaku/higgscharm/outputs/hplusc',
                        help='Directory with MVA score parquet files')
    parser.add_argument('--output-dir', '-o',
                        default='/eos/user/s/snandaku/Analysis/combine/ctag_vs_mva',
                        help='Output directory for datacards')
    parser.add_argument('--eras', '-e', nargs='+',
                        default=['2022preEE', '2022postEE', '2023preBPix', '2023postBPix'],
                        help='Eras to include')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()
    verbose = not args.quiet

    print("=" * 80)
    print("Combine Comparison: C-tag Categories vs MVA Score")
    print("=" * 80)
    print(f"\nC-tag source: {args.ctag_dir}")
    print(f"MVA source:   {args.mva_dir}")
    print(f"Output:       {args.output_dir}")
    print(f"Eras:         {', '.join(args.eras)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load c-tag data
    print("\n" + "=" * 80)
    print("Loading C-tag data...")
    print("=" * 80)
    ctag_data, ctag_raw = load_ctag_data(args.ctag_dir, args.eras, verbose)

    if sum(d['n_events'] for d in ctag_data.values()) == 0:
        print("\nWARNING: No c-tag events loaded!")
        print("Make sure parquet files have 'leadjet_ctag_category' column.")

    # Load MVA data
    print("\n" + "=" * 80)
    print("Loading MVA data...")
    print("=" * 80)
    mva_data, mva_raw = load_mva_data(args.mva_dir, args.eras, verbose)

    if sum(d['n_events'] for d in mva_data.values()) == 0:
        print("\nWARNING: No MVA events loaded!")
        print("Make sure parquet files have 'mva_signal_score' column.")

    # Normalize
    print("\n" + "=" * 80)
    print("Applying normalization...")
    print("=" * 80)
    print("\n  C-tag normalization:")
    ctag_data = apply_normalization(ctag_data, ctag_raw, args.eras, verbose)
    print("\n  MVA normalization:")
    mva_data = apply_normalization(mva_data, mva_raw, args.eras, verbose)

    # Create histograms
    print("\n" + "=" * 80)
    print("Creating histograms...")
    print("=" * 80)
    ctag_hist = create_ctag_histograms(ctag_data, args.output_dir, verbose)
    mva_hist = create_mva_histograms(mva_data, args.output_dir, verbose)

    # Create datacards
    print("\n" + "=" * 80)
    print("Creating datacards...")
    print("=" * 80)
    create_datacard(args.output_dir, "histograms_ctag.root", ctag_hist, "ctag", verbose)
    create_datacard(args.output_dir, "histograms_mva.root", mva_hist, "mva", verbose)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  {'Process':<15} {'C-tag':>15} {'MVA':>15}")
    print("  " + "-" * 50)
    for proc in COMBINE_PROCESSES:
        ctag_yield = ctag_hist[f"h_{proc}"].sum()
        mva_yield = mva_hist[f"h_{proc}"].sum()
        print(f"  {proc:<15} {ctag_yield:>15.4f} {mva_yield:>15.4f}")

    print(f"\n" + "=" * 80)
    print("Run Combine:")
    print("=" * 80)
    print(f"""
  cd {args.output_dir}

  # Asymptotic limits
  combine -M AsymptoticLimits datacard_ctag.txt -n _ctag --run expected
  combine -M AsymptoticLimits datacard_mva.txt -n _mva --run expected

  # Significance
  combine -M Significance datacard_ctag.txt -n _ctag -t -1 --expectSignal=1
  combine -M Significance datacard_mva.txt -n _mva -t -1 --expectSignal=1
""")


if __name__ == "__main__":
    main()
