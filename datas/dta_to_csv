#!/usr/bin/env python3
"""Convert all .dta files in a directory to .csv using pandas.

Usage:
    python make_csv.py --dir C:\path\to\dir [--overwrite]

This script reads every .dta file in the given directory and writes a
corresponding .csv file next to it.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def convert_dta_to_csv(dta_path: Path, overwrite: bool = False) -> bool:
    csv_path = dta_path.with_suffix('.csv')
    if csv_path.exists() and not overwrite:
        print(f"Skipping {dta_path.name} because {csv_path.name} already exists.")
        return True

    try:
        df = pd.read_stata(dta_path)
    except Exception as exc:
        print(f"ERROR: failed to read {dta_path.name}: {exc}")
        return False

    try:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    except Exception as exc:
        print(f"ERROR: failed to write {csv_path.name}: {exc}")
        return False

    print(f"Converted {dta_path.name} -> {csv_path.name} (rows={len(df):,}, cols={len(df.columns):,})")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description='Convert all .dta files in a folder to .csv')
    parser.add_argument('--dir', '-d', default='.', help='Directory containing .dta files')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Overwrite existing CSV files')
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: directory does not exist: {root}", file=sys.stderr)
        return 1

    dta_files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == '.dta')
    if not dta_files:
        print(f"No .dta files found in {root}.")
        return 0

    print(f"Found {len(dta_files)} .dta file(s) in {root}.")
    success_count = 0
    failure_count = 0

    for dta_file in dta_files:
        if convert_dta_to_csv(dta_file, overwrite=args.overwrite):
            success_count += 1
        else:
            failure_count += 1

    print(f"\nDone. Converted {success_count} file(s), failed {failure_count} file(s).")
    return 0 if failure_count == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
