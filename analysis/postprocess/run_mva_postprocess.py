#!/usr/bin/env python3
"""
Run MVA post-processing on parquet files.

This script:
1. Runs MVA inference using a b-hive model and config
2. Adds MVA score columns to parquet files
3. Creates histograms for the MVA scores

Usage:
    python run_mva_postprocess.py \
        --input-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class \
        --config /eos/home-s/snandaku/b-hive_ttcc/config/hc_zzto4l_mw_training_4class.yml \
        --model /eos/home-s/snandaku/b-hive_ttcc/output/TrainingTask/.../best_model.pt \
        --output-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_scored \
        --era 2022preEE

    # Or process all eras:
    python run_mva_postprocess.py \
        --input-dir /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class \
        --config /eos/home-s/snandaku/b-hive_ttcc/config/hc_zzto4l_mw_training_4class.yml \
        --model /eos/home-s/snandaku/b-hive_ttcc/output/TrainingTask/.../best_model.pt \
        --all-eras
"""

import os
import sys
import argparse
import yaml
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hist
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from coffea.util import save as coffea_save
    HAS_COFFEA = True
except ImportError:
    HAS_COFFEA = False
    def coffea_save(obj, path): pass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.postprocess.mva_inference import MVAPostProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Histogram Configuration for MVA Scores
# =============================================================================
MVA_HISTOGRAM_CONFIG = {
    'mva_signal_score': {
        'bins': 50,
        'range': (0, 1),
        'label': 'MVA Signal Score',
        'log_scale': False,
    },
    'mva_score_qqZZ': {
        'bins': 50,
        'range': (0, 1),
        'label': 'MVA qqZZ Score',
        'log_scale': False,
    },
    'mva_score_ggZZ': {
        'bins': 50,
        'range': (0, 1),
        'label': 'MVA ggZZ Score',
        'log_scale': False,
    },
    'mva_score_Signal': {
        'bins': 50,
        'range': (0, 1),
        'label': 'MVA Signal Score',
        'log_scale': False,
    },
    'mva_score_Other_Higgs': {
        'bins': 50,
        'range': (0, 1),
        'label': 'MVA Other Higgs Score',
        'log_scale': False,
    },
    'mva_class_prediction': {
        'bins': 5,
        'range': (-0.5, 4.5),
        'label': 'MVA Predicted Class',
        'log_scale': False,
    },
}


def build_mva_coffea_histograms(class_names: List[str]) -> Dict[str, hist.Hist]:
    """Build coffea histograms for MVA scores (compatible with existing plotter)."""
    histograms = {}

    # MVA signal score histogram
    histograms['mva_signal_score'] = hist.Hist(
        hist.axis.Regular(50, 0, 1, name='mva_signal_score', label='MVA Signal Score'),
        hist.axis.StrCategory(name='process', categories=[], growth=True),
        hist.axis.StrCategory(name='variation', categories=[], growth=True),
        hist.storage.Weight(),
    )

    # Per-class score histograms
    for cls in class_names:
        histograms[f'mva_score_{cls}'] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name=f'mva_score_{cls}', label=f'MVA {cls} Score'),
            hist.axis.StrCategory(name='process', categories=[], growth=True),
            hist.axis.StrCategory(name='variation', categories=[], growth=True),
            hist.storage.Weight(),
        )

    # Class prediction histogram
    histograms['mva_class_prediction'] = hist.Hist(
        hist.axis.Integer(0, len(class_names), name='mva_class_prediction', label='MVA Predicted Class'),
        hist.axis.StrCategory(name='process', categories=[], growth=True),
        hist.axis.StrCategory(name='variation', categories=[], growth=True),
        hist.storage.Weight(),
    )

    return histograms


def fill_mva_coffea_histograms(
    histograms: Dict[str, hist.Hist],
    df: pd.DataFrame,
    process_name: str,
    class_names: List[str],
    mass_window: tuple = (90, 160),
):
    """Fill coffea histograms from DataFrame with MVA scores."""
    # Apply mass window cut
    m4l_col = None
    for col in ['zz_mass_inclusive', 'm4l']:
        if col in df.columns:
            m4l_col = col
            break

    if m4l_col:
        mask = (df[m4l_col] >= mass_window[0]) & (df[m4l_col] < mass_window[1])
        df_mw = df[mask]
    else:
        df_mw = df

    # Only use valid MVA scores (>= 0)
    if 'mva_signal_score' in df_mw.columns:
        valid_mask = df_mw['mva_signal_score'] >= 0
        df_valid = df_mw[valid_mask]
    else:
        return histograms

    if len(df_valid) == 0:
        return histograms

    # Get weights if available
    weights = np.ones(len(df_valid))
    for weight_col in ['weight', 'weight_nominal', 'genWeight']:
        if weight_col in df_valid.columns:
            weights = df_valid[weight_col].fillna(1.0).values
            break

    # Fill signal score
    if 'mva_signal_score' in df_valid.columns:
        histograms['mva_signal_score'].fill(
            mva_signal_score=df_valid['mva_signal_score'].values,
            process=process_name,
            variation='nominal',
            weight=weights,
        )

    # Fill per-class scores
    for cls in class_names:
        col = f'mva_score_{cls}'
        if col in df_valid.columns:
            histograms[col].fill(
                **{col: df_valid[col].values},
                process=process_name,
                variation='nominal',
                weight=weights,
            )

    # Fill class prediction
    if 'mva_class_prediction' in df_valid.columns:
        pred = df_valid['mva_class_prediction'].values.astype(int)
        histograms['mva_class_prediction'].fill(
            mva_class_prediction=pred,
            process=process_name,
            variation='nominal',
            weight=weights,
        )

    return histograms


def create_mva_histograms(
    df: pd.DataFrame,
    output_dir: Path,
    sample_name: str,
    class_names: List[str],
    mass_window: tuple = (90, 160),
):
    """Create histograms for MVA scores."""
    hist_dir = output_dir / 'histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Apply mass window cut for histograms
    m4l_col = None
    for col in ['zz_mass_inclusive', 'm4l']:
        if col in df.columns:
            m4l_col = col
            break

    if m4l_col:
        mask = (df[m4l_col] >= mass_window[0]) & (df[m4l_col] < mass_window[1])
        df_mw = df[mask]
    else:
        df_mw = df

    # Create histograms for each MVA score column
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    mva_columns = [
        'mva_signal_score',
        'mva_class_prediction',
    ] + [f'mva_score_{cls}' for cls in class_names]

    for i, col in enumerate(mva_columns[:6]):
        ax = axes[i]
        if col in df_mw.columns:
            data = df_mw[col].dropna()
            if len(data) > 0:
                config = MVA_HISTOGRAM_CONFIG.get(col, {'bins': 50, 'range': (0, 1)})
                ax.hist(data, bins=config['bins'], range=config['range'],
                       histtype='step', linewidth=2, label=sample_name)
                ax.set_xlabel(config.get('label', col))
                ax.set_ylabel('Events')
                ax.set_title(f'{col}')
                ax.legend()
                if config.get('log_scale', False):
                    ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, f'{col}\nnot found', ha='center', va='center')
            ax.set_title(col)

    plt.tight_layout()
    plt.savefig(hist_dir / f'{sample_name}_mva_histograms.png', dpi=150)
    plt.savefig(hist_dir / f'{sample_name}_mva_histograms.pdf')
    plt.close()

    logger.info(f"Saved histograms to {hist_dir / f'{sample_name}_mva_histograms.png'}")

    # Also save individual histograms as numpy arrays for later combination
    hist_data = {}
    for col in mva_columns:
        if col in df_mw.columns:
            data = df_mw[col].dropna().values
            if len(data) > 0:
                config = MVA_HISTOGRAM_CONFIG.get(col, {'bins': 50, 'range': (0, 1)})
                counts, edges = np.histogram(data, bins=config['bins'], range=config['range'])
                hist_data[col] = {'counts': counts, 'edges': edges}

    np.savez(hist_dir / f'{sample_name}_mva_histograms.npz', **{
        k: v['counts'] for k, v in hist_data.items()
    })

    return hist_data


def process_single_file(
    input_file: str,
    output_file: str,
    mva_processor: MVAPostProcessor,
    create_histograms: bool = True,
) -> Dict:
    """Process a single parquet file."""
    try:
        df = mva_processor.process_parquet(input_file, output_file)
        return {'status': 'success', 'file': input_file, 'n_events': len(df)}
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")
        return {'status': 'error', 'file': input_file, 'error': str(e)}


def process_era(
    input_dir: Path,
    output_dir: Path,
    era: str,
    mva_processor: MVAPostProcessor,
    create_hists: bool = True,
    n_workers: int = 8,
):
    """Process all parquet files for a given era."""
    era_input = input_dir / era
    era_output = output_dir / era

    if not era_input.exists():
        logger.warning(f"Era directory not found: {era_input}")
        return

    # Find all parquet files
    parquet_files = list(era_input.glob('**/*.parquet'))
    logger.info(f"Found {len(parquet_files)} parquet files for {era}")

    if len(parquet_files) == 0:
        return

    # Prepare file pairs (input, output)
    file_pairs = []
    for pq_file in parquet_files:
        rel_path = pq_file.relative_to(era_input)
        out_file = era_output / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        file_pairs.append((str(pq_file), str(out_file)))

    # Process files in parallel using ThreadPoolExecutor
    results = []
    logger.info(f"Processing with {n_workers} workers...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_single_file, inp, out, mva_processor): inp
            for inp, out in file_pairs
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(parquet_files)} files processed")

    # Summary
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_error = sum(1 for r in results if r['status'] == 'error')
    total_events = sum(r.get('n_events', 0) for r in results if r['status'] == 'success')

    logger.info(f"Era {era}: {n_success} files processed, {n_error} errors, {total_events:,} total events")

    # Create combined histograms per process
    if create_hists:
        create_combined_histograms(era_output, era, mva_processor.class_names)


def create_combined_histograms(
    output_dir: Path,
    era: str,
    class_names: List[str],
    mass_window: tuple = (90, 160),
):
    """Create combined histograms for all processes in an era.

    Saves both:
    - PNG/PDF plots (matplotlib)
    - .coffea files (compatible with existing plotter)
    """
    hist_dir = output_dir / 'histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Find all process directories
    process_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != 'histograms']

    all_histograms = {}

    # Build coffea histograms (shared across all processes)
    coffea_histograms = build_mva_coffea_histograms(class_names)

    for proc_dir in process_dirs:
        process_name = proc_dir.name
        parquet_files = list(proc_dir.glob('**/*.parquet'))

        if len(parquet_files) == 0:
            continue

        logger.info(f"Creating histograms for {process_name} ({len(parquet_files)} files)")

        # Combine all parquet files for this process
        dfs = []
        for pq in parquet_files[:100]:  # Limit to avoid memory issues
            try:
                df = pd.read_parquet(pq)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Could not read {pq}: {e}")

        if len(dfs) == 0:
            continue

        combined_df = pd.concat(dfs, ignore_index=True)

        # Create matplotlib histograms (PNG/PDF)
        hist_data = create_mva_histograms(
            combined_df,
            output_dir,
            process_name,
            class_names,
            mass_window
        )
        all_histograms[process_name] = hist_data

        # Fill coffea histograms
        fill_mva_coffea_histograms(
            coffea_histograms,
            combined_df,
            process_name,
            class_names,
            mass_window
        )

    # Create stacked histogram plot (matplotlib)
    if len(all_histograms) > 0:
        create_stacked_mva_plot(all_histograms, hist_dir, era, class_names)

    # Save coffea histograms
    coffea_file = output_dir / f'{era}_mva_histograms.coffea'
    coffea_save(coffea_histograms, coffea_file)
    logger.info(f"Saved coffea histograms to {coffea_file}")


def create_stacked_mva_plot(
    all_histograms: Dict,
    output_dir: Path,
    era: str,
    class_names: List[str],
):
    """Create stacked histogram plots for all processes."""
    mva_columns = ['mva_signal_score'] + [f'mva_score_{cls}' for cls in class_names]

    # Color map for processes
    colors = {
        'ZZto4L': 'blue',
        'GluGluHtoZZto4L': 'orange',
        'SomeSMSignal': 'red',
        'HPlusCharm': 'red',
        'HPlusBottom': 'green',
        'VBFHto2Zto4L': 'purple',
        'default': 'gray',
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(mva_columns[:6]):
        ax = axes[i]

        for proc_name, hist_data in all_histograms.items():
            if col in hist_data:
                counts = hist_data[col]['counts']
                edges = hist_data[col]['edges']
                centers = (edges[:-1] + edges[1:]) / 2

                color = colors.get(proc_name, colors['default'])
                # Shorten process name for legend
                short_name = proc_name[:15] + '...' if len(proc_name) > 15 else proc_name
                ax.step(centers, counts, where='mid', label=short_name, linewidth=1.5)

        config = MVA_HISTOGRAM_CONFIG.get(col, {'label': col})
        ax.set_xlabel(config.get('label', col))
        ax.set_ylabel('Events')
        ax.set_title(f'{col}')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.1)

    plt.suptitle(f'MVA Score Distributions - {era}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'mva_scores_stacked_{era}.png', dpi=150)
    plt.savefig(output_dir / f'mva_scores_stacked_{era}.pdf')
    plt.close()

    logger.info(f"Saved stacked plots to {output_dir / f'mva_scores_stacked_{era}.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Run MVA post-processing on parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Input directory containing parquet files (organized by era/process)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: same as input, files updated in place)'
    )
    parser.add_argument(
        '--config', '-c',
        required=True,
        help='Path to b-hive config YAML file'
    )
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to trained model file (.pt)'
    )
    parser.add_argument(
        '--era', '-e',
        default=None,
        help='Process specific era (e.g., 2022preEE). If not specified, process all eras.'
    )
    parser.add_argument(
        '--all-eras',
        action='store_true',
        help='Process all eras found in input directory'
    )
    parser.add_argument(
        '--no-mass-window',
        action='store_true',
        help='Do not apply mass window filter during inference'
    )
    parser.add_argument(
        '--no-histograms',
        action='store_true',
        help='Skip histogram creation'
    )
    parser.add_argument(
        '--mass-window-min',
        type=float,
        default=90,
        help='Mass window minimum (default: 90)'
    )
    parser.add_argument(
        '--mass-window-max',
        type=float,
        default=160,
        help='Mass window maximum (default: 160)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)

    # Initialize MVA processor
    logger.info(f"Loading config: {args.config}")
    logger.info(f"Loading model: {args.model}")

    mva_processor = MVAPostProcessor(
        bhive_config_path=args.config,
        bhive_model_path=args.model,
        apply_mass_window=not args.no_mass_window,
    )

    # Override mass window if specified
    if args.mass_window_min:
        mva_processor.mass_window_min = args.mass_window_min
    if args.mass_window_max:
        mva_processor.mass_window_max = args.mass_window_max

    # Determine which eras to process
    if args.era:
        eras = [args.era]
    elif args.all_eras:
        eras = [d.name for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('20')]
    else:
        # If no era specified and not all-eras, process all directories
        eras = [d.name for d in input_dir.iterdir() if d.is_dir()]

    logger.info(f"Processing eras: {eras}")

    # Process each era
    for era in eras:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing era: {era}")
        logger.info(f"{'='*60}")

        # Real per-era jet_is_* category boundaries -- see MVAPostProcessor.__init__'s
        # `era` docstring / JET_TAG_BOUNDARIES_BY_ERA. mva_processor is created once
        # above and reused across eras, so the era must be updated per iteration
        # (a bug found 2026-08-14: it was never set at all before this fix, so every
        # era silently used the wrong legacy boundary set).
        mva_processor.era = era

        process_era(
            input_dir=input_dir,
            output_dir=output_dir,
            era=era,
            mva_processor=mva_processor,
            create_hists=not args.no_histograms,
            n_workers=args.workers,
        )

    logger.info("\nMVA post-processing complete!")


if __name__ == '__main__':
    main()
