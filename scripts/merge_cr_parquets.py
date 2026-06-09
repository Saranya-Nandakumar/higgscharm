#!/usr/bin/env python3
"""
Merge hplusc_mva_4class_CR parquet files per era and category into single files.
Uses pyarrow dataset API for efficiency.

Usage:
    python merge_cr_parquets.py --year 2022postEE
    python merge_cr_parquets.py --year all
"""
import argparse
import glob
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path

CR_BASE = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR"
MERGE_OUT = "/eos/user/s/snandaku/higgscharm/outputs/hplusc_mva_4class_CR_merged"
ERAS = ["2022postEE", "2022preEE", "2023preBPix", "2023postBPix"]
CATEGORIES = ["CR_3P1F", "CR_2P2F"]


def merge_era_category(era, category):
    files = glob.glob(f"{CR_BASE}/{era}/*/{category}/*.parquet")
    if not files:
        print(f"  No files found for {era}/{category}")
        return

    out_path = Path(MERGE_OUT) / era / f"{category}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  Already exists, skipping: {out_path}")
        return

    print(f"  Merging {len(files)} files -> {out_path}")
    dataset = ds.dataset(files, format="parquet")
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
    args = parser.parse_args()

    eras = ERAS if args.year == "all" else [args.year]

    for era in eras:
        print(f"\n=== {era} ===")
        for cat in CATEGORIES:
            merge_era_category(era, cat)

    print("\nAll done!")


if __name__ == "__main__":
    main()
