"""Shared constants, paths and artifact I/O.

Everything in `src/` is path-agnostic: functions take paths as parameters and
never hardcode a dataset location. The only exception is the artifact
directory, which is fixed relative to this package so `train.py` and `app/`
agree on where models live without passing it around.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

RANDOM_STATE = 42

# --- directories -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
DATA_DIR = os.path.join(BASE_DIR, "data")
COHORT_DIR = os.path.join(DATA_DIR, "cohorts")

# --- modelling policy ------------------------------------------------------
# Selection metric. PR-AUC, not ROC-AUC: four of the five outcomes are rare,
# and ROC-AUC is dominated by the true-negative mass on a 4% outcome.
SCORING = "average_precision"

TEST_SIZE = 0.20          # grouped hold-out, split on mother_id
ENSEMBLE_SIZE = 15        # greedy forward-selection steps
DEEP_EPOCHS = 150         # RealMLP / TabM
SMOTE_THRESHOLD = 0.20    # below this prevalence a target counts as imbalanced

# How many false alarms one missed case is worth. A policy choice, not a
# modelling one: harms cost more to miss, care-pathway descriptors are
# symmetric. Neonatal death is singled out because it is irreversible.
FN_COST_BY_TARGET = {"neonatal_death": 10}
FN_COST_ADVERSE = 5
FN_COST_PATHWAY = 1

# Operational ceiling. A screening tool that escalates most of the population
# cannot be acted on, however good its recall.
MAX_FLAG_RATE = 0.30

# Conformal prediction: 1 - ALPHA is the coverage guarantee.
ALPHA = 0.10


def fn_cost(target, kind):
    """False-negative weight for one target."""
    if target in FN_COST_BY_TARGET:
        return FN_COST_BY_TARGET[target]
    return FN_COST_ADVERSE if kind == "adverse" else FN_COST_PATHWAY


def usable_device():
    """torch.cuda.is_available() is not enough on a very new GPU.

    An RTX 5060 is compute capability 12.0 (sm_120) while torch 2.6.0+cu124
    ships kernels only up to sm_90. is_available() returns True, every
    allocation succeeds, and then the first real kernel raises "no kernel image
    is available for execution on the device". So probe with an actual matmul
    rather than trusting the flag.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        x = torch.randn(32, 32, device="cuda")
        _ = (x @ x).sum().item()
        return "cuda"
    except Exception:
        return "cpu"


def denullify(frame):
    """Nullable extension dtypes -> numpy-backed types.

    The cohort parquets preserve pandas' nullable dtypes (Int64, Float64,
    string, category), which hold pd.NA. scikit-learn evaluates missing values
    in a boolean context and raises "boolean value of NA is ambiguous" on
    every estimator, so the conversion happens once, on load.
    """
    out = frame.copy()
    for c in out.columns:
        dt = str(out[c].dtype)
        if dt in ("Int64", "Float64", "boolean"):
            out[c] = out[c].astype("float64")
        elif dt in ("string", "str") or isinstance(out[c].dtype,
                                                   pd.CategoricalDtype):
            out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
    return out


def get_scores(pipe, X):
    """Probabilities where available, scaled decision scores otherwise."""
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):
            pass
    raw = pipe.decision_function(X)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)


def save_artifact(obj, filename, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    joblib.dump(obj, path)
    return path


def load_artifact(filename, models_dir=MODELS_DIR):
    return joblib.load(os.path.join(models_dir, filename))


def save_manifest(manifest, models_dir=MODELS_DIR, filename="model_manifest.json"):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def load_registry(path):
    """The per-target feature list, blacklist, cohort and phase."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
