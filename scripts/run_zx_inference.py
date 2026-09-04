#!/usr/bin/env python3
"""
Run MVA inference on Z+X background parquet files.

Takes the weighted Z+X parquets from estimate_zx_background.py and scores each
event with the trained MVA, producing mva_score_* columns for combine histograms.

Usage:
    python scripts/run_zx_inference.py \
        --input /eos/user/s/snandaku/Analysis/zx_background \
        --config /eos/home-s/snandaku/b-hive_ttcc/config/hc_zzto4l_mw_training_4class_nomass.yml \
        --model /eos/home-s/snandaku/b-hive_ttcc/output/TrainingTask/hc_zzto4l_mw_training_4class_nomass/hcZZ_big4classjetyesmasslooseoldcbal/train_loose_oldcwithmassbal/MLP_HcZZ_MW_Deep_4class/epochs_40/nominal/best_model.pt \
        --output /eos/user/s/snandaku/Analysis/zx_background_mva
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add higgscharm to path for MVAPostProcessor
HIGGSCHARM = '/eos/user/s/snandaku/Higgscharmnew/higgscharm'
if HIGGSCHARM not in sys.path:
    sys.path.insert(0, HIGGSCHARM)

from analysis.postprocess.mva_inference import MVAPostProcessor

ERAS = ['2022postEE', '2022preEE', '2023preBPix', '2023postBPix']

# Column name mapping: Z+X parquet name -> MVA expected name
# 3P1F columns
COL_MAP_3P1F = {
    'm4l_3p1f':          'm4l',
    'pT_4l_3p1f':        'pT_4l',
    'eta_4l_3p1f':       'eta_4l',
    'phi_4l_3p1f':       'phi_4l',
    'z1_mass_3p1f':      'z1_mass',
    'z2_mass_3p1f':      'z2_mass',
    'z1_pt_3p1f':        'z1_pt',
    'z1_eta_3p1f':       'z1_eta',
    'z1_phi_3p1f':       'z1_phi',
    'z2_pt_3p1f':        'z2_pt',
    'z2_eta_3p1f':       'z2_eta',
    'z2_phi_3p1f':       'z2_phi',
    'deltaR_ZZ_3p1f':    'deltaR_ZZ',
    'z1_l1_pt_3p1f':     'z1_l1_pt',
    'z1_l1_eta_3p1f':    'z1_l1_eta',
    'z1_l1_phi_3p1f':    'z1_l1_phi',
    'z1_l2_pt_3p1f':     'z1_l2_pt',
    'z1_l2_eta_3p1f':    'z1_l2_eta',
    'z1_l2_phi_3p1f':    'z1_l2_phi',
    'l3_pt_3p1f':        'z2_l1_pt',
    'l3_eta_3p1f':       'z2_l1_eta',
    'l4_pt_3p1f':        'z2_l2_pt',
    'l4_eta_3p1f':       'z2_l2_eta',
    'cjets_h_dphi_3p1f': 'cjets_h_dphi_inclusive',
}

# 2P2F columns
COL_MAP_2P2F = {
    'm4l_2p2f':          'm4l',
    'pT_4l_2p2f':        'pT_4l',
    'eta_4l_2p2f':       'eta_4l',
    'phi_4l_2p2f':       'phi_4l',
    'z1_mass_2p2f':      'z1_mass',
    'z2_mass_2p2f':      'z2_mass',
    'z1_pt_2p2f':        'z1_pt',
    'z1_eta_2p2f':       'z1_eta',
    'z1_phi_2p2f':       'z1_phi',
    'z2_pt_2p2f':        'z2_pt',
    'z2_eta_2p2f':       'z2_eta',
    'z2_phi_2p2f':       'z2_phi',
    'deltaR_ZZ_2p2f':    'deltaR_ZZ',
    'z1_l1_pt_2p2f':     'z1_l1_pt',
    'z1_l1_eta_2p2f':    'z1_l1_eta',
    'z1_l1_phi_2p2f':    'z1_l1_phi',
    'z1_l2_pt_2p2f':     'z1_l2_pt',
    'z1_l2_eta_2p2f':    'z1_l2_eta',
    'z1_l2_phi_2p2f':    'z1_l2_phi',
    'l3_pt_2p2f':        'z2_l1_pt',
    'l3_eta_2p2f':       'z2_l1_eta',
    'l4_pt_2p2f':        'z2_l2_pt',
    'l4_eta_2p2f':       'z2_l2_eta',
    'cjets_h_dphi_2p2f': 'cjets_h_dphi_inclusive',
}

# Columns that are the same name in both CR and MVA inputs
PASSTHROUGH = [
    'jet_multiplicity', 'n_cjet', 'jet_ht', 'cjet_HT', 'nSV',
    'jet_pt', 'jet_eta', 'jet_btagPNetB', 'jet_btagPNetCvL', 'jet_btagPNetCvB',
    'cjets_pt', 'cjets_eta', 'cjets_btagPNetB', 'cjets_btagPNetCvL', 'cjets_btagPNetCvB',
    'weight_nominal', 'weight_zx', 'zx_weight', 'zx_sign',
]


def prepare_df_for_inference(df, col_map):
    """Rename CR columns to MVA-expected names and pass through common columns."""
    out = {}
    # Rename category-specific columns
    for src, dst in col_map.items():
        if src in df.columns:
            out[dst] = df[src]
        else:
            logging.debug(f'  Missing column {src} -> {dst}, will default to 0 in MVA')
    # Pass through common columns
    for col in PASSTHROUGH:
        if col in df.columns:
            out[col] = df[col]
    return pd.DataFrame(out)


def run_inference_on_era(era, input_dir, output_dir, processor):
    """Run MVA inference on 3P1F and 2P2F parquets for one era."""
    os.makedirs(os.path.join(output_dir, era), exist_ok=True)

    for cr_type, col_map in [('3p1f', COL_MAP_3P1F), ('2p2f', COL_MAP_2P2F)]:
        in_file = os.path.join(input_dir, era, f'zx_{cr_type}_{era}.parquet')
        out_file = os.path.join(output_dir, era, f'zx_{cr_type}_{era}_mva.parquet')

        if not os.path.exists(in_file):
            logging.warning(f'Not found: {in_file}, skipping')
            continue

        logging.info(f'\n--- {era} {cr_type.upper()} ---')
        df_orig = pd.read_parquet(in_file)
        logging.info(f'  Loaded {len(df_orig)} events')

        # Rename columns for MVA
        df_mva = prepare_df_for_inference(df_orig, col_map)

        # Load model and get class names
        processor._load_model()

        # Prepare features
        features = processor.prepare_features(df_mva)
        logging.info(f'  Feature matrix shape: {features.shape}')

        # Run inference (no mass window filter for Z+X — apply mass window separately)
        result = processor.predict(features)

        # Add MVA scores back to original dataframe (preserving all original columns)
        for i, cls in enumerate(processor.class_names):
            df_orig[f'mva_score_{cls}'] = result['scores'][:, i]
        df_orig['mva_signal_score'] = result['signal_score']
        df_orig['mva_class_prediction'] = result['class_prediction']

        # Save
        df_orig.to_parquet(out_file, index=False)
        logging.info(f'  Saved: {out_file}')
        logging.info(f'  Mean signal score: {result["signal_score"].mean():.4f}')
        logging.info(f'  Class distribution: {np.bincount(result["class_prediction"])}')


def main():
    parser = argparse.ArgumentParser(description='Run MVA inference on Z+X parquets')
    parser.add_argument('--input', required=True,
                        help='Directory with zx_3p1f/zx_2p2f parquets (from estimate_zx_background.py)')
    parser.add_argument('--config', required=True,
                        help='b-hive config YAML')
    parser.add_argument('--model', required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--output', required=True,
                        help='Output directory for MVA-scored parquets')
    parser.add_argument('--eras', nargs='+', default=ERAS,
                        help='Eras to process (default: all 4)')
    parser.add_argument('--no-mass-window', action='store_true',
                        help='Skip mass window filter (score all events)')
    args = parser.parse_args()

    logging.info(f'Config:  {args.config}')
    logging.info(f'Model:   {args.model}')
    logging.info(f'Input:   {args.input}')
    logging.info(f'Output:  {args.output}')

    processor = MVAPostProcessor(
        bhive_config_path=args.config,
        bhive_model_path=args.model,
        apply_mass_window=not args.no_mass_window,
    )

    for era in args.eras:
        logging.info(f'\n========== {era} ==========')
        # era must be set per-loop (not just at construction) so
        # _compute_jet_tags() uses the real per-era jet-tag boundaries
        # instead of silently falling back to the legacy wrong ones.
        processor.era = era
        run_inference_on_era(era, args.input, args.output, processor)

    logging.info('\nDone!')


if __name__ == '__main__':
    main()
