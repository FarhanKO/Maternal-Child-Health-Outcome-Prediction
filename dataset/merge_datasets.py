#!/usr/bin/env python3
"""Merge DHS-derived CSV files into a single non-redundant maternal/child dataset.

Usage:
    python merge_dhs_datasets.py --dir C:/path/to/Datas --out final_maternal_child_dataset.csv

The script is designed to be upload-ready for GitHub as a single merge tool.
It reads the base pregnancy questionnaire and merges available CSVs using the
following rules:
- Add coverage flags for childrens_recode and pregnancy_postnatal
- Merge baseline births_recode fields incrementally into the base record
- Merge non-suffixed individual_recode variables by caseid
- Merge household_recode using hhid or v001/v002 if available
- Merge verbal_autopsy on caseid or alternate join keys when possible

Missing files are tolerated and logged.
"""

from pathlib import Path
import argparse
import logging
import re

import pandas as pd

LOG = logging.getLogger("merge_dhs")


def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    LOG.addHandler(handler)
    LOG.setLevel(level)


def read_csv_columns(path: Path):
    if not path.exists():
        LOG.info('File not found when reading header: %s', path)
        return []
    try:
        return list(pd.read_csv(path, nrows=0, low_memory=False).columns)
    except Exception:
        LOG.exception('Failed to read header for %s', path)
        return []


def load_selected(path: Path, usecols=None, dtype=None):
    if not path.exists():
        LOG.info('File not found when loading: %s', path)
        return pd.DataFrame(columns=(usecols or []))

    available = read_csv_columns(path)
    effective_usecols = None
    if usecols:
        effective_usecols = [c for c in usecols if c in available]
    try:
        return pd.read_csv(path, usecols=effective_usecols, dtype=dtype, low_memory=False)
    except Exception:
        LOG.warning('Primary read failed for %s; retrying with python engine', path)
        try:
            return pd.read_csv(path, usecols=effective_usecols, dtype=dtype, engine='python', on_bad_lines='skip')
        except Exception:
            LOG.exception('Failed loading %s', path)
            return pd.DataFrame(columns=(effective_usecols or []))


def normalize_keys(df, keys):
    for key in keys:
        if key in df.columns:
            df[key] = df[key].astype(str).str.strip()


def build_flag_series(df_base, keys):
    if not keys:
        return pd.Series(0, index=df_base.index)
    keyset = set(keys)
    return df_base.apply(
        lambda row: 1 if (row.get('caseid'), row.get('pidx')) in keyset else 0,
        axis=1,
    ).astype(int)


def filter_non_suffixed_columns(cols):
    pattern = re.compile(r"_\d+$")
    return [c for c in cols if not pattern.search(c)]


def construct_hhid_from_hr(hr_df):
    if 'hhid' in hr_df.columns:
        return hr_df
    if 'v001' in hr_df.columns and 'v002' in hr_df.columns:
        hr_df = hr_df.copy()
        hr_df['hhid'] = hr_df['v001'].astype(str).str.strip() + '_' + hr_df['v002'].astype(str).str.strip()
        return hr_df
    LOG.warning('No hhid or v001/v002 found in household_recode; proceeding without household join.')
    return hr_df


def merge_births_incremental(df_base, births_path: Path):
    if not births_path.exists():
        LOG.warning('births_recode.csv missing; skipping births merge.')
        return df_base

    births_cols = read_csv_columns(births_path)
    if not births_cols:
        return df_base

    key_col = 'pidx' if 'pidx' in births_cols else 'bidx' if 'bidx' in births_cols else None
    if not key_col:
        LOG.warning('No pidx/bidx key found in births_recode; skipping births merge.')
        return df_base

    incremental = [c for c in births_cols if c not in df_base.columns and c not in ('caseid', 'pidx', 'bidx')]
    LOG.info('Found %d incremental columns in births_recode', len(incremental))
    if not incremental:
        LOG.info('No incremental births columns to merge')
        return df_base

    if 'pidx' not in df_base.columns:
        LOG.warning('Base file has no pidx column; skipping births merge.')
        return df_base

    normalize_keys(df_base, ['caseid', 'pidx'])
    base_keyset = set(zip(df_base['caseid'], df_base['pidx']))
    cols_to_read = ['caseid', key_col] + incremental
    available = [c for c in cols_to_read if c in births_cols]
    LOG.info('Reading births_recode in chunks, selected %d columns', len(available))

    chunks = []
    try:
        reader = pd.read_csv(
            births_path,
            usecols=available,
            dtype=str,
            chunksize=20000,
            low_memory=False,
            on_bad_lines='skip',
        )
        for chunk in reader:
            chunk = chunk.rename(columns={'bidx': 'pidx'})
            normalize_keys(chunk, ['caseid', 'pidx'])
            if 'caseid' not in chunk.columns or 'pidx' not in chunk.columns:
                continue
            chunk = chunk.loc[chunk[['caseid', 'pidx']].notna().all(axis=1)]
            tuples = list(zip(chunk['caseid'], chunk['pidx']))
            mask = [t in base_keyset for t in tuples]
            filtered = chunk.loc[mask, ['caseid', 'pidx'] + [c for c in incremental if c in chunk.columns]]
            if not filtered.empty:
                chunks.append(filtered)
    except Exception:
        LOG.exception('Error reading births_recode in chunks')
        return df_base

    if not chunks:
        LOG.info('No matching births rows found for base keys; skipping births merge.')
        return df_base

    df_births = pd.concat(chunks, ignore_index=True)
    df_births = df_births.drop_duplicates(subset=['caseid', 'pidx'], keep='first')
    df_base = df_base.merge(df_births, on=['caseid', 'pidx'], how='left', copy=False)
    LOG.info('After births merge shape: %s', df_base.shape)
    return df_base


def merge_individual_recode(df_base, individual_path: Path):
    if not individual_path.exists():
        LOG.warning('individual_recode.csv missing; skipping IR merge.')
        return df_base

    ir_cols = read_csv_columns(individual_path)
    if not ir_cols:
        return df_base

    ir_keep = [c for c in filter_non_suffixed_columns(ir_cols) if c]
    if 'caseid' not in ir_keep:
        ir_keep.insert(0, 'caseid')

    LOG.info('IR will keep %d columns', len(ir_keep))
    df_ir = load_selected(individual_path, usecols=ir_keep)
    if 'caseid' not in df_ir.columns:
        LOG.warning('individual_recode has no caseid; skipping IR merge.')
        return df_base

    normalize_keys(df_ir, ['caseid'])
    df_ir = df_ir.drop_duplicates(subset=['caseid'], keep='first')
    df_base = df_base.merge(df_ir, on='caseid', how='left', copy=False)
    LOG.info('After individual_recode merge shape: %s', df_base.shape)
    return df_base


def merge_household_recode(df_base, household_path: Path):
    if not household_path.exists():
        LOG.warning('household_recode.csv missing; skipping HR merge.')
        return df_base

    hr_cols = read_csv_columns(household_path)
    if not hr_cols:
        return df_base

    hr_keep = [c for c in filter_non_suffixed_columns(hr_cols) if c]
    if not hr_keep:
        LOG.info('No usable household columns found; skipping HR merge.')
        return df_base

    LOG.info('HR will keep %d columns', len(hr_keep))
    df_hr = load_selected(household_path, usecols=hr_keep)
    df_hr = construct_hhid_from_hr(df_hr)
    normalize_keys(df_hr, ['hhid', 'v001', 'v002'])

    if 'hhid' in df_base.columns:
        normalize_keys(df_base, ['hhid'])
        df_base = df_base.merge(df_hr, on='hhid', how='left', copy=False)
        LOG.info('After household_recode merge on hhid shape: %s', df_base.shape)
        return df_base

    if 'v001' in df_base.columns and 'v002' in df_base.columns:
        df_base = df_base.copy()
        normalize_keys(df_base, ['v001', 'v002'])
        df_base['hhid'] = df_base['v001'] + '_' + df_base['v002']
        df_base = df_base.merge(df_hr, on='hhid', how='left', copy=False)
        LOG.info('After household_recode merge via v001/v002 shape: %s', df_base.shape)
        return df_base

    LOG.warning('Base has no household identifier (hhid or v001/v002); skipping HR merge.')
    return df_base


def merge_verbal_autopsy(df_base, verbal_path: Path):
    if not verbal_path.exists():
        LOG.warning('verbal_autopsy.csv missing; skipping verbal_autopsy merge.')
        return df_base

    v_cols = read_csv_columns(verbal_path)
    if not v_cols:
        return df_base

    if 'caseid' in v_cols and 'caseid' in df_base.columns:
        df_v = load_selected(verbal_path)
        normalize_keys(df_v, ['caseid'])
        df_base = df_base.merge(df_v, on='caseid', how='left', copy=False)
        LOG.info('After verbal_autopsy merge on caseid shape: %s', df_base.shape)
        return df_base

    possible_keys = ['v001', 'v002', 'v003', 'hv001', 'hv002', 'hhid']
    join_keys = [k for k in possible_keys if k in v_cols and k in df_base.columns]
    if not join_keys:
        LOG.warning('No suitable join keys found for verbal_autopsy; skipping merge.')
        return df_base

    df_v = load_selected(verbal_path, usecols=join_keys + [c for c in v_cols if c not in join_keys])
    normalize_keys(df_v, join_keys)
    normalize_keys(df_base, join_keys)
    df_base = df_base.merge(df_v, on=join_keys, how='left', copy=False)
    LOG.info('After verbal_autopsy merge on %s shape: %s', join_keys, df_base.shape)
    return df_base


def main():
    parser = argparse.ArgumentParser(description='Merge DHS CSV files into one maternal/child dataset.')
    parser.add_argument('--dir', '-d', default='.', help='Directory containing CSV files')
    parser.add_argument('--out', '-o', default='final_maternal_child_dataset.csv', help='Output CSV path')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file')
    parser.add_argument('--verbose', action='store_true', help='Show debug logging')
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    root = Path(args.dir)
    if not root.exists() or not root.is_dir():
        LOG.error('Directory does not exist: %s', root)
        return 2

    paths = {
        'base': root / 'pregnancy_questionnaire.csv',
        'births': root / 'births_recode.csv',
        'children': root / 'childrens_recode.csv',
        'household': root / 'household_recode.csv',
        'individual': root / 'individual_recode.csv',
        'verbal': root / 'verbal_autopsy.csv',
        'postnatal': root / 'pregnancy_postnatal.csv',
    }

    for label, path in paths.items():
        if not path.exists():
            LOG.warning('Expected file %s missing: %s', label, path)

    if not paths['base'].exists():
        LOG.error('Base file pregnancy_questionnaire.csv is required.')
        return 3

    try:
        LOG.info('Loading primary base table: %s', paths['base'])
        df_base = pd.read_csv(paths['base'], low_memory=False)
        normalize_keys(df_base, ['caseid', 'pidx'])
        LOG.info('Base shape: %s', df_base.shape)
    except Exception:
        LOG.exception('Failed to load base table')
        return 4

    base_rows = len(df_base)

    try:
        LOG.info('Creating coverage flags for childrens_recode and pregnancy_postnatal')
        child_keys = []
        if paths['children'].exists():
            child_usecols = [c for c in ('caseid', 'pidx') if c in read_csv_columns(paths['children'])]
            if child_usecols:
                df_children = load_selected(paths['children'], usecols=child_usecols, dtype=str)
                normalize_keys(df_children, ['caseid', 'pidx'])
                child_keys = list(df_children[['caseid', 'pidx']].dropna().drop_duplicates().itertuples(index=False, name=None))

        post_keys = []
        if paths['postnatal'].exists():
            post_usecols = [c for c in ('caseid', 'pidx') if c in read_csv_columns(paths['postnatal'])]
            if post_usecols:
                df_post = load_selected(paths['postnatal'], usecols=post_usecols, dtype=str)
                normalize_keys(df_post, ['caseid', 'pidx'])
                post_keys = list(df_post[['caseid', 'pidx']].dropna().drop_duplicates().itertuples(index=False, name=None))

        flag_frame = pd.DataFrame({
            'is_in_children_recode': build_flag_series(df_base, child_keys),
            'is_in_postnatal': build_flag_series(df_base, post_keys),
        }, index=df_base.index)
        df_base = pd.concat([df_base, flag_frame], axis=1)
        LOG.info('Flags added; current shape: %s', df_base.shape)
    except Exception:
        LOG.exception('Failed to create coverage flags')

    df_base = merge_births_incremental(df_base, paths['births'])
    df_base = merge_individual_recode(df_base, paths['individual'])
    df_base = merge_household_recode(df_base, paths['household'])
    df_base = merge_verbal_autopsy(df_base, paths['verbal'])

    LOG.info('Final shape: %s', df_base.shape)
    if len(df_base) != base_rows:
        LOG.warning('Row count changed during merge: %s -> %s', base_rows, len(df_base))

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        LOG.error('Output exists and --overwrite not set: %s', out_path)
        return 5

    try:
        LOG.info('Saving final dataset to %s', out_path)
        df_base.to_csv(out_path, index=False)
    except Exception:
        LOG.exception('Failed to save final dataset')
        return 6

    LOG.info('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
