#!/usr/bin/env python3
"""
Merge parquet chunks by sample, run MVA inference, save scored parquets.

Usage:
    python test_mva_inference.py \
        --input  /eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class/2022preEE \
        --output /eos/user/s/snandaku/higgscharm/outputs/hplusc_mvascores/2022preEE \
        --config /eos/home-s/snandaku/b-hive_ttcc/config/hc_zzto4l_mw_training_4class.yml \
        --model  /eos/home-s/snandaku/b-hive_ttcc/output/TrainingTask/hc_zzto4l_mw_training_4class/hcZZ_big4class/train_11lossreweight4class/MLP_HcZZ_MW_Deep_4class/epochs_40/nominal/best_model.pt
"""

import os
import re
import sys
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Add higgscharm to path
HIGGSCHARM = str(Path(__file__).parent)
if HIGGSCHARM not in sys.path:
    sys.path.insert(0, HIGGSCHARM)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "mva_inference",
    Path(__file__).parent / "analysis" / "postprocess" / "mva_inference.py",
)
_mod = importlib.util.load_from_spec(_spec)
_spec.loader.exec_module(_mod)
MVAPostProcessor = _mod.MVAPostProcessor


def group_chunks_by_sample(input_dir: Path) -> dict:
    """
    Group chunk subdirectories by base sample name.
    e.g. HPlusBottom_2022preEE_1, HPlusBottom_2022preEE_2 → HPlusBottom_2022preEE
    """
    groups = defaultdict(list)
    for chunk_dir in sorted(input_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        name = chunk_dir.name
        # Strip trailing _N chunk index
        base = re.sub(r"_\d+$", "", name)
        groups[base].append(chunk_dir)
    return dict(groups)


def merge_sample(chunk_dirs: list, category: str) -> pd.DataFrame:
    """Read and concatenate all parquets from chunk_dirs/category/."""
    dfs = []
    for chunk_dir in chunk_dirs:
        cat_dir = chunk_dir / category
        if not cat_dir.exists():
            continue
        parquets = sorted(cat_dir.glob("*.parquet"))
        if not parquets:
            continue
        for pq in parquets:
            try:
                dfs.append(pd.read_parquet(pq))
            except Exception as e:
                logging.warning(f"  Could not read {pq}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def run_inference_on_df(df: pd.DataFrame, processor: MVAPostProcessor) -> pd.DataFrame:
    """Run MVA inference and add score columns to df."""
    if df.empty:
        return df

    processor._load_model()
    features = processor.prepare_features(df)
    result = processor.predict(features)

    for i, cls in enumerate(processor.class_names):
        df[f"mva_score_{cls}"] = result["scores"][:, i]
    df["mva_signal_score"] = result["signal_score"]
    df["mva_class_prediction"] = result["class_prediction"]
    return df


def main():
    parser = argparse.ArgumentParser(description="Merge + MVA inference test")
    parser.add_argument("--input",  required=True, help="Input era dir (e.g. .../hplusc_mva_4class/2022preEE)")
    parser.add_argument("--output", required=True, help="Output dir for scored parquets")
    parser.add_argument("--config", required=True, help="b-hive config YAML")
    parser.add_argument("--model",  required=True, help="Path to best_model.pt")
    parser.add_argument("--category", default="base", help="Category subdirectory (default: base)")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="Only process these samples (default: all)")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        logging.error(f"Input not found: {input_dir}")
        sys.exit(1)

    logging.info(f"Config: {args.config}")
    logging.info(f"Model:  {args.model}")

    processor = MVAPostProcessor(
        bhive_config_path=args.config,
        bhive_model_path=args.model,
        apply_mass_window=False,  # score all events, apply window downstream
    )
    # Load model once up front
    processor._load_model()
    logging.info(f"Classes: {processor.class_names}")

    groups = group_chunks_by_sample(input_dir)
    logging.info(f"Found {len(groups)} samples, {sum(len(v) for v in groups.values())} chunk dirs")

    if args.samples:
        groups = {k: v for k, v in groups.items() if k in args.samples}
        logging.info(f"Filtered to {len(groups)} samples: {list(groups.keys())}")

    n_ok, n_skip = 0, 0
    for sample_name, chunk_dirs in sorted(groups.items()):
        out_file = output_dir / f"{sample_name}.parquet"

        # --- Step 1: merge ---
        logging.info(f"\n[{sample_name}] Merging {len(chunk_dirs)} chunk(s)...")
        df = merge_sample(chunk_dirs, args.category)
        if df.empty:
            logging.warning(f"  No events found, skipping")
            n_skip += 1
            continue
        logging.info(f"  Merged: {len(df)} events, {len(df.columns)} columns")

        # --- Step 2: MVA inference ---
        logging.info(f"  Running MVA inference...")
        df = run_inference_on_df(df, processor)

        n_in_window = ((df["mva_signal_score"] >= 0) &
                       (df.get("m4l", df.get("zz_mass_inclusive", pd.Series(dtype=float))).between(100, 150))
                       ).sum() if "m4l" in df.columns or "zz_mass_inclusive" in df.columns else len(df)
        logging.info(f"  Mean signal score: {df['mva_signal_score'].mean():.4f}")
        logging.info(f"  Events in mass window [100,150]: {n_in_window}")

        # --- Step 3: save ---
        df.to_parquet(out_file, index=False)
        logging.info(f"  Saved: {out_file}")
        n_ok += 1

    logging.info(f"\nDone: {n_ok} samples scored, {n_skip} skipped (empty).")
    logging.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
