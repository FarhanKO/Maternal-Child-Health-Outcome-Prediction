"""Cohort loading, grouped splitting, and the two preprocessing paths.

Two preprocessors are built per target, not one, because the eight algorithms
do not want the same input. The dense path imputes, scales and one-hot encodes
for the linear model and the neural networks, which cannot accept a missing
value at all. The native path leaves numeric missingness untouched for the
boosted trees, which learn their own default split direction for NaN -- and a
blank m14 in a modular survey means "the maternity module was not asked",
which is informative in itself.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from src.feature_engineering import ORDERED_LEVELS, features_for
from src.utils import (COHORT_DIR, RANDOM_STATE, SMOTE_THRESHOLD, TEST_SIZE,
                       denullify)


def load_cohort(name, cohort_dir=COHORT_DIR):
    """One cohort by name. Parquet is preferred because CSV cannot round-trip
    a nullable integer -- the nulls come back as NaN and the column comes back
    as float."""
    pq = os.path.join(cohort_dir, f"cohort_{name}.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    csv = os.path.join(cohort_dir, f"cohort_{name}.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(
            f"cohort '{name}' not found in {cohort_dir}. See data/README.md "
            f"for how to build the cohorts.")
    return pd.read_csv(csv, low_memory=False)


def load_cohorts(registry, cohort_dir=COHORT_DIR, extra=("mother",)):
    """Every cohort the registry needs, plus the mother-level view.

    The cohort list is derived from the registry rather than hardcoded, so a
    new target cannot be added without its cohort being loaded here.
    """
    needed = sorted({registry[t]["cohort"] for t in registry} | set(extra))
    return {n: load_cohort(n, cohort_dir) for n in needed}


def make_xy(target, registry, cohorts):
    """Design matrix, label and grouping key for one target.

    Rows are the target's *universe* -- never complete cases. A child with a
    missing height is outside the measured-child universe; a child with a
    missing wealth quintile is inside it with one feature absent, and the
    preprocessor deals with that.
    """
    spec = registry[target]
    d = cohorts[spec["cohort"]]
    d = d[d[target].notna()]

    feats = features_for(target, registry, set(d.columns))
    feats = [f for f in feats if d[f].nunique(dropna=True) > 1]

    X = denullify(d[feats])
    y = pd.to_numeric(d[target], errors="coerce").astype(int)
    groups = d["mother_id"].astype(str)
    return X, y, groups


def grouped_split(X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Grouped hold-out -- a mother never appears in both train and test.

    This matters most on the births cohort, where 90% of rows belong to a
    mother who appears more than once. Under a random split, siblings sharing
    a mother, a household, a village and most of the feature vector would land
    on both sides, and every reported metric would be optimistic.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=random_state)
    tr, te = next(gss.split(X, y, groups))
    return (X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te],
            groups.iloc[tr], groups.iloc[te])


def build_preprocessor(X, native_nan=False):
    """Column transformer for one design matrix.

    native_nan=True leaves numeric missingness in place, for the boosted trees.
    Returns the transformer plus the three column groups, so the caller can
    report how wide each path is.
    """
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    ordered = [c for c in cat if c in ORDERED_LEVELS]
    nominal = [c for c in cat if c not in ORDERED_LEVELS]

    ordinal = OrdinalEncoder(
        categories=[ORDERED_LEVELS[c] for c in ordered],
        handle_unknown="use_encoded_value", unknown_value=np.nan,
        encoded_missing_value=np.nan)

    if native_nan:
        numeric_step = "passthrough"
        ordered_step = ordinal
    else:
        # add_indicator keeps "this was missing" as its own column, which in a
        # modular survey is often more informative than the imputed value.
        numeric_step = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", RobustScaler()),
        ])
        ordered_step = Pipeline([
            ("ordinal", ordinal),
            ("impute", SimpleImputer(strategy="median")),
        ])

    # min_frequency pools genuinely rare levels into one "infrequent" category,
    # which is stable across resamples. max_categories would instead keep the
    # N most common levels and silently drop the rest.
    nominal_step = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25,
                                 sparse_output=False)),
    ])

    transformer = ColumnTransformer(
        [("num", numeric_step, num),
         ("ord", ordered_step, ordered),
         ("nom", nominal_step, nominal)],
        remainder="drop")
    return transformer, num, ordered, nominal


def build_datasets(registry, cohorts, targets=None, test_size=TEST_SIZE):
    """Everything the training harness needs, per target.

    Asserts that no mother crosses the split -- the single guarantee the whole
    validation strategy rests on.
    """
    targets = targets or list(registry)
    datasets = {}
    for t in targets:
        X, y, g = make_xy(t, registry, cohorts)
        X_tr, X_te, y_tr, y_te, g_tr, g_te = grouped_split(X, y, g, test_size)

        prep_dense, num, ordc, nomc = build_preprocessor(X, native_nan=False)
        prep_native, *_ = build_preprocessor(X, native_nan=True)

        overlap = len(set(g_tr) & set(g_te))
        assert overlap == 0, f"{t}: {overlap} mothers in both train and test"

        prevalence = float(y.mean())
        datasets[t] = dict(
            X=X, y=y, groups=g,
            X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, g_tr=g_tr, g_te=g_te,
            prep=prep_dense, prep_native=prep_native,
            n_num=len(num), n_ord=len(ordc), n_nom=len(nomc),
            prevalence=prevalence,
            imbalanced=prevalence < SMOTE_THRESHOLD,
            scale_pos_weight=(1 - prevalence) / prevalence,
        )
    return datasets
