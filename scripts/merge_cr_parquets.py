#!/usr/bin/env python3
"""
Merge CR parquet files per era and category into single files.
Uses pyarrow dataset API for efficiency.

Usage:
    python merge_cr_parquets.py --year 2022postEE
    python merge_cr_parquets.py --year all
    python merge_cr_parquets.py --workflow zzto4l_CR --year all
    python merge_cr_parquets.py --workflow hplusc_mva_4class_CR --year all --force
"""
import argparse
import glob
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path

ERAS = ["2022postEE", "2022preEE", "2023preBPix", "2023postBPix"]
CATEGORIES = ["CR_3P1F", "CR_2P2F"]


def merge_era_category(era, category, cr_base, merge_out, force=False):
    files = glob.glob(f"{cr_base}/{era}/*/{category}/*.parquet")
    if not files:
        print(f"  No files found for {era}/{category}")
        return

    out_path = Path(merge_out) / era / f"{category}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"  Already exists, skipping: {out_path}")
        return

    if out_path.exists() and force:
        print(f"  Force-overwriting: {out_path}")

    # Pre-filter corrupted files
    good_files = []
    bad_files = []
    for f in files:
        try:
            pq.read_metadata(f)
            good_files.append(f)
        except Exception as e:
            bad_files.append(f)
            print(f"  Skipping corrupted file: {f} ({e})")
    if bad_files:
        print(f"  Skipped {len(bad_files)} corrupted file(s), using {len(good_files)} good files")
    if not good_files:
        print(f"  No valid files found for {era}/{category}")
        return

    print(f"  Merging {len(good_files)} files -> {out_path}")
    dataset = ds.dataset(good_files, format="parquet")
    table = dataset.to_table()
    print(f"  Rows: {table.num_rows:,}  Columns: {table.num_columns}")
    # Allow nulls in all fields to avoid schema conflicts across files
    new_schema = pa.schema([f.with_nullable(True) for f in table.schema])
    table = table.cast(new_schema)
    pq.write_table(table, out_path, compression="snappy")
    print(f"  Done: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="all", help="Era or 'all'")
    parser.add_argument("--workflow", default="hplusc_mva_4class_CR",
                        help="CR workflow name (default: hplusc_mva_4class_CR)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing merged files")
    args = parser.parse_args()

    cr_base   = f"/eos/user/s/snandaku/higgscharm/outputs/{args.workflow}"
    merge_out = f"/eos/user/s/snandaku/higgscharm/outputs/{args.workflow}_merged"

    print(f"Workflow:  {args.workflow}")
    print(f"CR base:   {cr_base}")
    print(f"Merge out: {merge_out}")

    eras = ERAS if args.year == "all" else [args.year]

    for era in eras:
        print(f"\n=== {era} ===")
        for cat in CATEGORIES:
            merge_era_category(era, cat, cr_base, merge_out, force=args.force)

    print("\nAll done!")


if __name__ == "__main__":
    main()
