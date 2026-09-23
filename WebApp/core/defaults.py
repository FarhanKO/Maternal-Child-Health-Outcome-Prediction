"""Filling the fields the caller did not supply.

A form that asks 33 questions still has to hand the models ~117 features. How
the other 84 are chosen matters more than it looks.

The flat `BASE_TEMPLATE` holds each variable's *marginal* median, so the
record it describes belongs to nobody: pick "richest" and the wealth index
score still arrives at the national median (-31,590, a middle-quintile
household) while the quintile says richest. The model then sees a
contradiction it never saw in training, and the anomaly detector is right to
call the combination unusual.

So a missing field is resolved in four steps, first hit wins:

  1. the value the caller supplied;
  2. a functional rule — anything an interviewer would never ask twice
     (BMI from height and weight, age band from age, years since marriage
     from age minus age at marriage);
  3. the cohort-typical value *within the caller's own wealth-quintile x
     residence cell*, from group_defaults.json;
  4. the flat template, for anything the first three cannot reach.

Step 3 is what keeps the unentered fields in the same neighbourhood as the
entered ones. It is still a population value, not a measurement: a report
lists every field it touched, and the more a caller fills in, the less of
this there is to do.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.presets import BASE_TEMPLATE

TABLE_PATH = Path(__file__).resolve().parent / "group_defaults.json"

# Which features step 3 is allowed to fill from the caller's cell.
#
# Measured, not assumed. Filling *everything* from the cell makes the record
# more plausible as a whole but injects a wealth-shaped signal that is not
# this subject's, and on a held-out sample that cost neonatal_death 0.02
# ROC-AUC while gaining nothing elsewhere. The entries below are the ones
# the caller's own answers nearly determine:
#
#   v191  the wealth index *score* that v190 is binned from. The flat
#         template hands the model -31,590 (a middle-quintile household) even
#         when the caller said "richest", whose cell median is +158,777.
#         Leaving that contradiction in place is strictly worse than
#         inverting the binning.
#   v190  the reverse, for a caller who gave the score but not the quintile.
#
# Widen this list only with a measurement in hand.
GROUP_FILL = {"v191", "v190"}

# DHS scales a rule needs to speak. Derived from the pooled cohorts.
EDU_ATTAINMENT = {"no education": "no education", "primary": "incomplete primary",
                  "secondary": "incomplete secondary", "higher": "higher"}
EDU_SINGLE_YEARS = {"no education": 0.0, "primary": 4.0, "secondary": 8.0,
                    "higher": 12.0}
EDU_YEAR_AT_LEVEL = {"no education": 0.0, "primary": 4.0, "secondary": 3.0,
                     "higher": 2.0}
AGE_BANDS = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
PARTNER_AGE_GAP = 7.0          # median (partner age - mother age)


@lru_cache(maxsize=1)
def _table() -> dict:
    if not TABLE_PATH.is_file():
        return {"by_wealth_residence": {}, "by_wealth": {}}
    return json.loads(TABLE_PATH.read_text(encoding="utf-8"))


def _age_band(age):
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return np.nan
    for band in AGE_BANDS:
        lo, hi = (int(x) for x in band.split("-"))
        if lo <= age <= hi:
            return band
    return np.nan


def _bmi(height_cm, weight_kg):
    h = height_cm / 100.0
    return np.where((h > 0) & np.isfinite(weight_kg), weight_kg / (h * h), np.nan)


# (target, (required inputs...), fn(frame) -> Series)
# Each rule fires only where the target is missing and every input is present.
RULES: list[tuple[str, tuple[str, ...], callable]] = [
    ("bmi", ("height_cm", "weight_kg"),
     lambda d: pd.Series(_bmi(d["height_cm"].astype(float),
                              d["weight_kg"].astype(float)), index=d.index).round(2)),
    ("v013", ("v012",), lambda d: d["v012"].map(_age_band)),
    # education: attainment, single years and year-at-level all follow the level
    ("v149", ("v106",), lambda d: d["v106"].map(EDU_ATTAINMENT)),
    ("v133", ("v106",), lambda d: d["v106"].map(EDU_SINGLE_YEARS)),
    ("v107", ("v106",), lambda d: d["v106"].map(EDU_YEAR_AT_LEVEL)),
    ("v729", ("v701",), lambda d: d["v701"].map(EDU_ATTAINMENT)),
    ("v715", ("v701",), lambda d: d["v701"].map(EDU_SINGLE_YEARS)),
    ("v702", ("v701",), lambda d: d["v701"].map(EDU_YEAR_AT_LEVEL)),
    # marriage and partner age track the mother's age
    ("v512", ("v012", "v511"),
     lambda d: (d["v012"].astype(float) - d["v511"].astype(float)).clip(lower=0)),
    ("v730", ("v012",), lambda d: d["v012"].astype(float) + PARTNER_AGE_GAP),
    ("v525", ("v511",), lambda d: d["v511"].astype(float)),
    # living children cannot exceed children ever born
    ("v218", ("v201",), lambda d: d["v201"].astype(float)),
    ("v219", ("v201",), lambda d: d["v201"].astype(float)),
    ("v220", ("v201",), lambda d: d["v201"].astype(float).clip(upper=6)),
    # geography and the child's record are mirrored across DHS recodes
    ("v139", ("v024",), lambda d: d["v024"]),
    ("v140", ("v025",), lambda d: d["v025"]),
    ("v141", ("v026",), lambda d: d["v026"]),
    ("p4", ("b4",), lambda d: d["b4"]),
    ("p0", ("b0",), lambda d: d["b0"]),
    ("p19", ("b19",), lambda d: d["b19"].astype(float)),
    ("b19", ("p19",), lambda d: d["p19"].astype(float)),
]


def apply_rules(frame: pd.DataFrame, known: pd.DataFrame) -> list[str]:
    """Fill what can be computed from what the caller gave. Mutates `frame`.

    `known` is a boolean frame, True where the caller supplied a real value.
    A rule may consume a value an earlier rule produced (v013 from v012), but
    never overwrites something the caller sent.
    """
    fired = []
    for target, inputs, fn in RULES:
        if target not in frame.columns or any(i not in frame.columns for i in inputs):
            continue
        need = ~known[target] if target in known.columns else pd.Series(True, index=frame.index)
        have = pd.Series(True, index=frame.index)
        for i in inputs:
            have &= frame[i].notna()
        mask = need & have & frame[target].isna()
        if not mask.any():
            continue
        values = fn(frame)
        frame.loc[mask, target] = values[mask]
        if frame.loc[mask, target].notna().any():
            fired.append(target)
    return fired


def _assign(frame: pd.DataFrame, index, column: str, value) -> None:
    """Write a scalar into part of a column, widening the dtype first.

    A column the caller never sent arrives all-NaN and therefore float64;
    pandas 3 raises instead of upcasting when a string default is written
    into it, so the cast is made explicit here.
    """
    if isinstance(value, str) and frame[column].dtype != object:
        frame[column] = frame[column].astype(object)
    frame.loc[index, column] = value


def _cell_key(wealth, residence) -> tuple[str, str]:
    w = str(wealth).strip().lower() if isinstance(wealth, str) else ""
    r = str(residence).strip().lower() if isinstance(residence, str) else ""
    return w, r


def apply_group_defaults(frame: pd.DataFrame, features: list[str],
                         known: pd.DataFrame) -> list[str]:
    """Fill remaining gaps from the caller's wealth x residence cell.

    Only cells the caller never supplied are touched. A value sent as null is
    a statement that the field is unknown for this subject — the survey is
    modular and a blank `m14` means the maternity module was not asked, which
    the boosted trees learn a split direction for. Imputing it would erase
    that, so `known` is honoured here exactly as in `apply_rules`.

    Rows are grouped by cell so one lookup serves a whole batch. A row whose
    quintile is itself unknown falls back to the wealth-only table and then to
    the flat template, which is what step 4 handles.
    """
    table = _table()
    by_wr = table.get("by_wealth_residence", {})
    by_w = table.get("by_wealth", {})
    if not by_wr and not by_w:
        return []

    wealth = frame["v190"] if "v190" in frame.columns else pd.Series(np.nan, index=frame.index)
    resid = frame["v025"] if "v025" in frame.columns else pd.Series(np.nan, index=frame.index)
    # A caller who gave neither still gets the template's own cell, which is a
    # real place in the data rather than the marginal median of everywhere.
    wealth = wealth.fillna(BASE_TEMPLATE.get("v190"))
    resid = resid.fillna(BASE_TEMPLATE.get("v025"))

    touched: set[str] = set()
    keys = pd.Series([_cell_key(w, r) for w, r in zip(wealth, resid)], index=frame.index)
    for (w, r), idx in keys.groupby(keys).groups.items():
        cell = by_wr.get(f"{w}|{r}") or by_w.get(w) or {}
        if not cell:
            continue
        block = frame.loc[idx]
        for feat in features:
            if feat not in GROUP_FILL or feat not in cell or feat not in block.columns:
                continue
            fillable = block[feat].isna() & ~known.loc[idx, feat]
            if not fillable.any():
                continue
            _assign(frame, block.index[fillable], feat, cell[feat])
            touched.add(feat)
    return sorted(touched)


def apply_template(frame: pd.DataFrame, features: list[str],
                   known: pd.DataFrame) -> list[str]:
    """Last resort: the flat marginal template, for anything the cell table
    does not carry. Honours `known` for the same reason as above."""
    touched = []
    for feat in features:
        if feat not in BASE_TEMPLATE:
            continue
        if feat not in frame.columns:
            frame[feat] = BASE_TEMPLATE[feat]
            touched.append(feat)
            continue
        fillable = frame[feat].isna() & ~known[feat]
        if fillable.any():
            _assign(frame, frame.index[fillable], feat, BASE_TEMPLATE[feat])
            touched.append(feat)
    return touched
