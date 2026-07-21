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


