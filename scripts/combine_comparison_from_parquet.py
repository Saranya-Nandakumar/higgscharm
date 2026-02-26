#!/usr/bin/env python3
"""
Combine Comparison: C-tagging Categories vs MVA Score

Reads parquet files directly from the higgscharm framework output
and creates datacards for combine to compare sensitivity.

Usage:
    python scripts/combine_comparison_from_parquet.py --output-dir /path/to/output

    # Or with custom input directory:
    python scripts/combine_comparison_from_parquet.py \
        --input-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc \
        --output-dir /eos/user/s/snandaku/Analysis/combine/comparison

Author: Claude Code
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
    "signal": 0.000026,        # H+c signal
    "ggH": 0.01434,            # gg->H->ZZ
    "qqZZ": 1.39 + 0.027,      # qq->ZZ + gg->ZZ continuum
    "hplusb": 0.000173,        # H+b
    "other_higgs": 0.00112 + 0.000156 + 0.000244 + 0.000775 + 0.00312 + 0.000144,  # VBF+VH+ttH+bbH
}

# Luminosity per era in pb^-1
LUMINOSITY_PB = {
    "2022preEE": 7980.4,
    "2022postEE": 26671.7,
    "2023preBPix": 17794.0,
    "2023postBPix": 9450.0,
}

# Process name mapping from sample names to combine process names
PROCESS_MAPPING = {
    # Signal
    "SomeSMSignal": "signal",
    # H+b (era-specific)
    "2023postBPixHB": "hplusb",
    "2023preBPixHB": "hplusb",
    "2022postEEHB": "hplusb",
    "2022preEEHB": "hplusb",
    # ggH
    "GluGluHtoZZto4L": "ggH",
    # ZZ backgrounds (qq->ZZ and gg->ZZ continuum)
    "ZZto4L": "qqZZ",
    "GluGlutoContinto2Zto4E": "qqZZ",
    "GluGlutoContinto2Zto4Mu": "qqZZ",
    "GluGlutoContinto2Zto4Tau": "qqZZ",
    "GluGluToContinto2Zto2E2Mu": "qqZZ",
    "GluGluToContinto2Zto2E2Tau": "qqZZ",
    "GluGluToContinto2Zto2Mu2Tau": "qqZZ",
    # Other Higgs production modes
    "VBFHto2Zto4L": "other_higgs",
    "WminusH_Hto2Zto4L": "other_higgs",
    "WplusH_Hto2Zto4L": "other_higgs",
    "ZHto2Zto4L": "other_higgs",
    "TTH_Hto2Z": "other_higgs",
    "bbH_Hto2Zto4L": "other_higgs",
}

# C-tagging category boundaries (PNet-based)
MVA_JET_TAG_BOUNDARIES = {
    "C4": {"x": (0.7339, 1.0000), "y": (0.0000, 0.0382)},  # Tightest c-tag
    "C3": {"x": (0.7339, 1.0000), "y": (0.0382, 0.1851)},
    "C2": {"x": (0.7339, 1.0000), "y": (0.1851, 0.2688)},
    "C1": {"x": (0.4, 0.7339), "y": (0.0000, 0.9000)},
    "C0": {"x": (0.15, 0.4), "y": (0.0000, 0.9000)},
    "L0": {"x": (0.0000, 0.15), "y": (0.0000, 0.9000)},     # Light jets
    "B0": {"x": (0.7339, 1.0000), "y": (0.2688, 0.4057)},
    "B1": {"x": (0.7339, 1.0000), "y": (0.4057, 0.4068)},
    "B2": {"x": (0.7339, 1.0000), "y": (0.4068, 0.6788)},
    "B3": {"x": (0.7339, 1.0000), "y": (0.6788, 0.8406)},
    "B4": {"x": (0.7339, 1.0000), "y": (0.8406, 1.0000)},   # Tightest b-tag
}

CATEGORY_ORDER = ["L0", "C0", "C1", "C2", "C3", "C4", "B0", "B1", "B2", "B3", "B4"]
COMBINE_PROCESSES = ["signal", "ggH", "hplusb", "qqZZ", "other_higgs"]

# MVA binning
MVA_BINS = np.linspace(0, 1, 21)  # 20 bins from 0 to 1

# Mass window for selection
MASS_WINDOW = (100, 150)

# Whether to use pre-computed categories from parquet (if available)
USE_PRECOMPUTED_CATEGORIES = True


# ============================================================================
# Helper Functions
# ============================================================================

def get_process_name(sample_name):
    """Map sample directory name to combine process name."""
    # Remove era suffixes like _1, _2, etc.
    base_name = sample_name.rstrip('0123456789').rstrip('_')

    for key, proc in PROCESS_MAPPING.items():
        if key in sample_name or key == base_name:
            return proc
    return None


def compute_jet_category(B, CvB, CvL):
    """
    Compute jet tagging category from PNet scores.

    Args:
        B: btagPNetB score
        CvB: btagPNetCvB score
        CvL: btagPNetCvL score

    Returns:
        Category index (0-10) or -1 if no category matches
    """
    pBvsC = 1.0 - CvB
    pBplusC = B + (1.0 - B) * CvL
    pBplusC = np.clip(pBplusC, 0, 1)
    pBvsC = np.clip(pBvsC, 0, 1)

    for idx, cat_name in enumerate(CATEGORY_ORDER):
        bounds = MVA_JET_TAG_BOUNDARIES[cat_name]
        x_min, x_max = bounds["x"]
        y_min, y_max = bounds["y"]

        if (pBplusC >= x_min and pBplusC < x_max and
            pBvsC >= y_min and pBvsC < y_max):
            return idx

    return -1


def compute_jet_categories_vectorized(B_arr, CvB_arr, CvL_arr):
    """Vectorized version of jet category computation."""
    pBvsC = 1.0 - CvB_arr
    pBplusC = B_arr + (1.0 - B_arr) * CvL_arr
    pBplusC = np.clip(pBplusC, 0, 1)
    pBvsC = np.clip(pBvsC, 0, 1)

    categories = np.full(len(B_arr), -1, dtype=np.int32)

    for idx, cat_name in enumerate(CATEGORY_ORDER):
        bounds = MVA_JET_TAG_BOUNDARIES[cat_name]
        x_min, x_max = bounds["x"]
        y_min, y_max = bounds["y"]

        mask = (
            (pBplusC >= x_min) & (pBplusC < x_max) &
            (pBvsC >= y_min) & (pBvsC < y_max) &
            (categories == -1)  # Only assign if not already assigned
        )
        categories[mask] = idx

    return categories


# ============================================================================
# Data Loading
# ============================================================================

def load_parquet_files(input_dir, eras=None, verbose=True):
    """
    Load all parquet files from framework output directory.

    Args:
        input_dir: Base directory containing era subdirectories
        eras: List of eras to include (default: all 4)
        verbose: Print progress

    Returns:
        Dictionary with data organized by process
    """
    if eras is None:
        eras = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]

    data_by_process = defaultdict(lambda: {
        'mva_signal_score': [],
        'jet_categories': [],
        'weights': [],
        'n_events': 0,
    })

    raw_yields = defaultdict(float)
    total_files = 0
    total_events = 0

    for era in eras:
        era_path = os.path.join(input_dir, era)
        if not os.path.exists(era_path):
            if verbose:
                print(f"  Warning: Era directory not found: {era_path}")
            continue

        # List all sample directories
        samples = [d for d in os.listdir(era_path)
                   if os.path.isdir(os.path.join(era_path, d))]

        for sample in samples:
            proc_name = get_process_name(sample)
            if proc_name is None:
                continue

            # Find parquet files in base/ subdirectory
            base_path = os.path.join(era_path, sample, 'base')
            if not os.path.exists(base_path):
                continue

            parquet_files = glob.glob(os.path.join(base_path, '*.parquet'))

            for pq_file in parquet_files:
                try:
                    table = pq.read_table(pq_file)
                    df = table.to_pandas()

                    # Apply mass window selection
                    if 'm4l' in df.columns:
                        mask = (df['m4l'] >= MASS_WINDOW[0]) & (df['m4l'] <= MASS_WINDOW[1])
                        df = df[mask]

                    # Require at least 1 jet for c-tagging
                    if 'n_jet' in df.columns:
                        df = df[df['n_jet'] >= 1]

                    if len(df) == 0:
                        continue

                    total_files += 1
                    total_events += len(df)

                    # Get weights
                    weights = df['weight_nominal'].values if 'weight_nominal' in df.columns else np.ones(len(df))

                    # Track raw weighted yields for normalization
                    raw_yields[proc_name] += weights.sum()

                    # Get MVA signal score
                    if 'mva_signal_score' in df.columns:
                        mva_scores = df['mva_signal_score'].values
                    else:
                        # If MVA not available, use zeros
                        mva_scores = np.zeros(len(df))

                    # Compute jet categories from leading jet
                    if all(col in df.columns for col in ['jet_btagPNetB', 'jet_btagPNetCvB', 'jet_btagPNetCvL']):
                        B_col = df['jet_btagPNetB'].values
                        CvB_col = df['jet_btagPNetCvB'].values
                        CvL_col = df['jet_btagPNetCvL'].values

                        # Extract leading jet values (handle array columns)
                        categories = []
                        valid_indices = []

                        for i in range(len(df)):
                            b_val = B_col[i]
                            cvb_val = CvB_col[i]
                            cvl_val = CvL_col[i]

                            # Handle array-type columns (take first element = leading jet)
                            if hasattr(b_val, '__len__'):
                                if len(b_val) == 0:
                                    continue
                                b_val = b_val[0]
                                cvb_val = cvb_val[0] if len(cvb_val) > 0 else 0.0
                                cvl_val = cvl_val[0] if len(cvl_val) > 0 else 0.0

                            if b_val is None or np.isnan(b_val):
                                continue
                            if cvb_val is None or np.isnan(cvb_val):
                                continue
                            if cvl_val is None or np.isnan(cvl_val):
                                continue

                            cat = compute_jet_category(b_val, cvb_val, cvl_val)
                            if cat >= 0:
                                categories.append(cat)
                                valid_indices.append(i)

                        if len(categories) > 0:
                            categories = np.array(categories, dtype=np.int32)
                            valid_weights = weights[valid_indices]
                            valid_mva = mva_scores[valid_indices]

                            data_by_process[proc_name]['jet_categories'].extend(categories.tolist())
                            data_by_process[proc_name]['mva_signal_score'].extend(valid_mva.tolist())
                            data_by_process[proc_name]['weights'].extend(valid_weights.tolist())
                            data_by_process[proc_name]['n_events'] += len(categories)

                except Exception as e:
                    if verbose:
                        print(f"  Error reading {pq_file}: {e}")
                    continue

    # Convert lists to arrays
    for proc in data_by_process:
        data_by_process[proc]['jet_categories'] = np.array(data_by_process[proc]['jet_categories'], dtype=np.int32)
        data_by_process[proc]['mva_signal_score'] = np.array(data_by_process[proc]['mva_signal_score'], dtype=np.float32)
        data_by_process[proc]['weights'] = np.array(data_by_process[proc]['weights'], dtype=np.float32)

    if verbose:
        print(f"\n  Loaded {total_files} parquet files, {total_events} total events")
        print(f"\n  Raw weighted yields:")
        for proc in COMBINE_PROCESSES:
            print(f"    {proc}: {raw_yields[proc]:.4f}")

    return data_by_process, raw_yields


def apply_xsec_normalization(data_by_process, raw_yields, eras=None, verbose=True):
    """
    Apply cross-section based normalization.

    Scales weights so that total yield = xs * lumi for each process.
    """
    if eras is None:
        eras = ["2022preEE", "2022postEE", "2023preBPix", "2023postBPix"]

    total_lumi = sum(LUMINOSITY_PB[era] for era in eras)
    expected_yields = {proc: xs * total_lumi for proc, xs in PROCESS_XS_PB.items()}

    if verbose:
        print(f"\n  Total luminosity: {total_lumi:.1f} pb^-1")
        print(f"\n  Expected yields (xs * lumi):")
        for proc in COMBINE_PROCESSES:
            print(f"    {proc}: {expected_yields.get(proc, 0):.4f}")

    # Compute and apply scale factors
    scale_factors = {}
    for proc in COMBINE_PROCESSES:
        raw = raw_yields.get(proc, 0)
        expected = expected_yields.get(proc, 0)

        if raw > 0 and expected > 0:
            scale = expected / raw
            scale_factors[proc] = scale
            if proc in data_by_process:
                data_by_process[proc]['weights'] *= scale
        else:
            scale_factors[proc] = 1.0

    if verbose:
        print(f"\n  Scale factors applied:")
        for proc in COMBINE_PROCESSES:
            print(f"    {proc}: {scale_factors[proc]:.4f}")

    return data_by_process, scale_factors


# ============================================================================
# Histogram Creation
# ============================================================================

def create_ctag_histograms(data_by_process, output_dir, verbose=True):
    """
    Create c-tagging category histograms.

    Returns:
        Dictionary of histogram arrays
    """
    n_bins = len(CATEGORY_ORDER)
    histograms = {}

    if verbose:
        print(f"\n  C-tagging histograms ({n_bins} categories):")
        print(f"  {'Process':<15} {'Events':>10} {'Weighted':>15}")
        print("  " + "-" * 45)

    for proc_name in COMBINE_PROCESSES:
        if proc_name not in data_by_process or data_by_process[proc_name]['n_events'] == 0:
            histograms[f"h_{proc_name}"] = np.zeros(n_bins, dtype=np.float64)
            if verbose:
                print(f"  {proc_name:<15} {'0':>10} {'0.0000':>15}")
            continue

        data = data_by_process[proc_name]
        categories = data['jet_categories']
        weights = data['weights']

        hist, _ = np.histogram(categories, bins=n_bins, range=(0, n_bins), weights=weights)
        histograms[f"h_{proc_name}"] = hist.astype(np.float64)

        if verbose:
            print(f"  {proc_name:<15} {len(categories):>10} {weights.sum():>15.4f}")

    # Create data_obs (sum of all MC for Asimov dataset)
    data_obs = sum(histograms.values())

    # Save to ROOT file
    output_file = os.path.join(output_dir, "histograms_ctag.root")
    edges = np.arange(n_bins + 1, dtype=np.float64)

    with uproot.recreate(output_file) as f:
        for name, hist in histograms.items():
            f[name] = (hist, edges)
        f["h_data_obs"] = (data_obs, edges)

    if verbose:
        print(f"\n  Saved to {output_file}")

    return histograms


def create_mva_histograms(data_by_process, output_dir, verbose=True):
    """
    Create MVA signal score histograms.

    Returns:
        Dictionary of histogram arrays
    """
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
        mva_scores = data['mva_signal_score']
        weights = data['weights']

        hist, _ = np.histogram(mva_scores, bins=MVA_BINS, weights=weights)
        histograms[f"h_{proc_name}"] = hist.astype(np.float64)

        if verbose:
            mean_mva = np.average(mva_scores, weights=weights) if len(mva_scores) > 0 else 0
            print(f"  {proc_name:<15} {len(mva_scores):>10} {weights.sum():>15.4f} {mean_mva:>10.4f}")

    # Create data_obs (sum of all MC for Asimov dataset)
    data_obs = sum(histograms.values())

    # Save to ROOT file
    output_file = os.path.join(output_dir, "histograms_mva.root")

    with uproot.recreate(output_file) as f:
        for name, hist in histograms.items():
            f[name] = (hist, MVA_BINS)
        f["h_data_obs"] = (data_obs, MVA_BINS)

    if verbose:
        print(f"\n  Saved to {output_file}")

    return histograms


# ============================================================================
# Datacard Creation
# ============================================================================

def create_datacard(output_dir, hist_file, histograms, method, verbose=True):
    """
    Create combine datacard for shape-based analysis.

    Args:
        output_dir: Output directory
        hist_file: Name of histogram ROOT file
        histograms: Dictionary of histogram arrays
        method: 'ctag' or 'mva'
    """
    datacard_path = os.path.join(output_dir, f"datacard_{method}.txt")

    # Get yields
    yields = {proc: histograms[f"h_{proc}"].sum() for proc in COMBINE_PROCESSES}

    with open(datacard_path, 'w') as dc:
        dc.write(f"# H+c -> ZZ -> 4l analysis: {method.upper()} method\n")
        dc.write(f"# Generated by combine_comparison_from_parquet.py\n")
        dc.write(f"#\n")
        dc.write(f"imax 1  # number of bins\n")
        dc.write(f"jmax {len(COMBINE_PROCESSES)-1}  # number of backgrounds\n")
        dc.write(f"kmax *  # number of nuisances\n")
        dc.write("-" * 80 + "\n")

        # Shapes
        dc.write(f"shapes * * {hist_file} h_$PROCESS\n")
        dc.write("-" * 80 + "\n")

        # Observation
        dc.write("bin         hczz\n")
        dc.write("observation -1\n")  # -1 means use data_obs from shapes
        dc.write("-" * 80 + "\n")

        # Rates
        dc.write("bin         " + "".join([f"{'hczz':<15}" for _ in COMBINE_PROCESSES]) + "\n")
        dc.write("process     " + "".join([f"{p:<15}" for p in COMBINE_PROCESSES]) + "\n")
        dc.write("process     " + "".join([f"{i:<15}" for i in range(len(COMBINE_PROCESSES))]) + "\n")
        dc.write("rate        " + "".join([f"{yields[p]:<15.6f}" for p in COMBINE_PROCESSES]) + "\n")
        dc.write("-" * 80 + "\n")

        # Systematics
        dc.write("# Systematic uncertainties\n")
        dc.write("lumi_13TeV      lnN    " + "".join(["1.025         " for _ in COMBINE_PROCESSES]) + "\n")

        # Add more systematics as needed
        # Theory uncertainties (example)
        dc.write("pdf_gg          lnN    " + "".join([
            "1.05          " if p in ["signal", "ggH", "hplusb"] else "-             "
            for p in COMBINE_PROCESSES
        ]) + "\n")

        dc.write("pdf_qq          lnN    " + "".join([
            "1.03          " if p == "qqZZ" else "-             "
            for p in COMBINE_PROCESSES
        ]) + "\n")

        dc.write("QCDscale_ggH    lnN    " + "".join([
            "1.10          " if p == "ggH" else "-             "
            for p in COMBINE_PROCESSES
        ]) + "\n")

    if verbose:
        print(f"  Datacard: {datacard_path}")

    return datacard_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create combine datacards comparing c-tagging vs MVA'
    )
    parser.add_argument(
        '--input-dir', '-i',
        default='/eos/user/s/snandaku/higgscharm/outputs/hplusc',
        help='Input directory with parquet files (default: hplusc output)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='/eos/user/s/snandaku/Analysis/combine/ctag_vs_mva',
        help='Output directory for datacards and histograms'
    )
    parser.add_argument(
        '--eras', '-e',
        nargs='+',
        default=['2022preEE', '2022postEE', '2023preBPix', '2023postBPix'],
        help='Eras to include'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip cross-section normalization'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    print("=" * 80)
    print("Combine Comparison: C-tagging Categories vs MVA Score")
    print("=" * 80)
    print(f"\nInput directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Eras: {', '.join(args.eras)}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("\n" + "=" * 80)
    print("Loading parquet files...")
    print("=" * 80)

    data_by_process, raw_yields = load_parquet_files(
        args.input_dir,
        eras=args.eras,
        verbose=verbose
    )

    if sum(d['n_events'] for d in data_by_process.values()) == 0:
        print("\nERROR: No events loaded! Check input directory and file structure.")
        print("Expected structure: {input_dir}/{era}/{sample}/base/*.parquet")
        sys.exit(1)

    # Apply normalization
    if not args.no_normalize:
        print("\n" + "=" * 80)
        print("Applying cross-section normalization...")
        print("=" * 80)
        data_by_process, scale_factors = apply_xsec_normalization(
            data_by_process,
            raw_yields,
            eras=args.eras,
            verbose=verbose
        )

    # Create histograms
    print("\n" + "=" * 80)
    print("Creating histograms...")
    print("=" * 80)

    ctag_histograms = create_ctag_histograms(data_by_process, args.output_dir, verbose=verbose)
    mva_histograms = create_mva_histograms(data_by_process, args.output_dir, verbose=verbose)

    # Create datacards
    print("\n" + "=" * 80)
    print("Creating datacards...")
    print("=" * 80)

    ctag_datacard = create_datacard(
        args.output_dir,
        "histograms_ctag.root",
        ctag_histograms,
        "ctag",
        verbose=verbose
    )

    mva_datacard = create_datacard(
        args.output_dir,
        "histograms_mva.root",
        mva_histograms,
        "mva",
        verbose=verbose
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n  {'Process':<15} {'C-tag yield':>15} {'MVA yield':>15}")
    print("  " + "-" * 50)
    for proc in COMBINE_PROCESSES:
        ctag_yield = ctag_histograms[f"h_{proc}"].sum()
        mva_yield = mva_histograms[f"h_{proc}"].sum()
        print(f"  {proc:<15} {ctag_yield:>15.4f} {mva_yield:>15.4f}")

    print(f"\n  Total signal: C-tag = {ctag_histograms['h_signal'].sum():.4f}, MVA = {mva_histograms['h_signal'].sum():.4f}")

    # Print combine commands
    print("\n" + "=" * 80)
    print("Run Combine:")
    print("=" * 80)
    print(f"""
  cd {args.output_dir}

  # Run asymptotic limits
  combine -M AsymptoticLimits datacard_ctag.txt -n _ctag --run expected
  combine -M AsymptoticLimits datacard_mva.txt -n _mva --run expected

  # Or run significance
  combine -M Significance datacard_ctag.txt -n _ctag -t -1 --expectSignal=1
  combine -M Significance datacard_mva.txt -n _mva -t -1 --expectSignal=1

  # Compare results
  cat higgsCombine_ctag.AsymptoticLimits.mH120.root
  cat higgsCombine_mva.AsymptoticLimits.mH120.root
""")


if __name__ == "__main__":
    main()
