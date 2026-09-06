"""The model zoo, the training harness, the ensemble and the calibrators.

Seven learners span the three families that matter on tabular data, chosen to
reflect what the 2025 benchmarks rank highly rather than what is traditional.
TabPFN-3 is listed below for orientation but is deliberately not in the zoo: it
was benchmarked separately and landed inside the panel spread, and its
inference cost makes the downstream pipeline (calibration refits, permutation
importance) impractically slow. See the project README:

    linear        Logistic Regression      interpretable reference point
    bagged trees  Random Forest            variance reduction on diffuse signal
    boosted trees XGBoost, LightGBM,       the workhorses of tabular ML
                  CatBoost
    deep tabular  RealMLP, TabM            the two that match boosting on TabArena
    foundation    TabPFN                   pre-trained transformer, one forward pass

Five algorithms were removed from an earlier version of this panel -- Naive
Bayes, k-NN, a lone Decision Tree, SVM and a small MLP. Each was run against
every target first and dropped on evidence: Naive Bayes reached ROC-AUC 0.502
on several targets, and SVM cost 20-49 minutes per fit while never winning one.
"""
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,
                                     RandomizedSearchCV)

from src.utils import (ALPHA, DEEP_EPOCHS, ENSEMBLE_SIZE, RANDOM_STATE,
                       SCORING, get_scores, usable_device)


def try_import(name):
    """Optional dependency. The panel degrades to the learners present rather
    than failing, so a missing CatBoost costs you a column, not a run.

    Exception, not ImportError. A dependency can be installed and still
    unimportable: a torch/torchvision version mismatch raises AttributeError
    from a half-initialised module, which an ImportError guard lets through and
    which then takes down this entire module instead of one learner.
    """
    try:
        return __import__(name), True
    except Exception:
        return None, False


xgboost, HAS_XGB = try_import("xgboost")
lightgbm, HAS_LGB = try_import("lightgbm")
catboost, HAS_CATBOOST = try_import("catboost")
pytabkit, HAS_PYTABKIT = try_import("pytabkit")
tabpfn, HAS_TABPFN = try_import("tabpfn")

DEVICE = usable_device()


# ===========================================================================
# 1. The zoo
# ===========================================================================
def get_supervised_model_grids(model_jobs=1):
    """name -> (factory, param_grid, prep, use_smote).

    `factory(ctx)` receives the per-target context -- prevalence, imbalanced,
    scale_pos_weight, n_train -- so a model can adapt itself to the cohort
    instead of being configured once for all five.

    prep="native" hands the model a matrix that still contains NaN. Only the
    boosted trees may use it: every other estimator here raises on missing
    input, and SMOTE cannot interpolate across it either.

    RealMLP, TabM and TabPFN carry no grid on purpose. They are published as
    *tuned defaults* -- the authors' contribution is a configuration that needs
    no search, so searching over it would be reporting on a different method.
    """
    zoo = {}

    zoo["Logistic Regression"] = (
        lambda ctx: LogisticRegression(
            max_iter=3000, random_state=RANDOM_STATE,
            class_weight="balanced" if ctx["imbalanced"] else None),
        {"C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
         "penalty": ["l2"],
         "solver": ["lbfgs", "liblinear"]},
        "dense", False)

    zoo["Random Forest"] = (
        lambda ctx: RandomForestClassifier(
            n_jobs=model_jobs, random_state=RANDOM_STATE,
            class_weight="balanced_subsample" if ctx["imbalanced"] else None),
        {"n_estimators": [300, 500, 800],
         "max_depth": [None, 10, 16, 24],
         "min_samples_leaf": [1, 5, 20, 50],
         "max_features": ["sqrt", "log2", 0.3]},
        "dense", False)

    if HAS_XGB:
        zoo["XGBoost"] = (
            lambda ctx: xgboost.XGBClassifier(
                eval_metric="logloss", n_jobs=model_jobs,
                random_state=RANDOM_STATE,
                scale_pos_weight=(ctx["scale_pos_weight"]
                                  if ctx["imbalanced"] else 1.0)),
            {"n_estimators": [300, 500, 800],
             "max_depth": [3, 4, 5, 6, 8],
             "learning_rate": [0.02, 0.05, 0.1],
             "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0],
             "min_child_weight": [1, 5, 10],
             "reg_lambda": [1.0, 5.0, 10.0]},
            "native", False)

    if HAS_LGB:
        zoo["LightGBM"] = (
            lambda ctx: lightgbm.LGBMClassifier(
                n_jobs=model_jobs, verbose=-1, random_state=RANDOM_STATE,
                class_weight="balanced" if ctx["imbalanced"] else None),
            {"n_estimators": [300, 500, 800],
             "num_leaves": [15, 31, 63, 127],
             "learning_rate": [0.02, 0.05, 0.1],
             "min_child_samples": [10, 20, 50],
             "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.6, 0.8, 1.0]},
            "native", False)

    if HAS_CATBOOST:
        zoo["CatBoost"] = (
            lambda ctx: catboost.CatBoostClassifier(
                verbose=0, allow_writing_files=False,
                random_seed=RANDOM_STATE,
                scale_pos_weight=(ctx["scale_pos_weight"]
                                  if ctx["imbalanced"] else 1.0)),
            {"iterations": [400, 700, 1000],
             "depth": [4, 6, 8],
             "learning_rate": [0.02, 0.05, 0.1],
             "l2_leaf_reg": [1.0, 3.0, 10.0],
             "random_strength": [1.0, 5.0]},
            "native", False)

    if HAS_PYTABKIT:
        from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
        zoo["RealMLP"] = (
            lambda ctx: RealMLP_TD_Classifier(
                n_epochs=DEEP_EPOCHS, device=DEVICE, verbosity=0,
                random_state=RANDOM_STATE),
            None, "dense", False)
        zoo["TabM"] = (
            lambda ctx: TabM_D_Classifier(
                n_epochs=DEEP_EPOCHS, device=DEVICE, verbosity=0,
                random_state=RANDOM_STATE),
            None, "dense", False)

    return zoo


def get_unsupervised_models(contamination=0.05):
    """Anomaly detectors for the "is this record ordinary?" layer.

    IsolationForest is the one that ships in the bundle. It expects
    pre-transformed input and carries no record of which columns those were,
    which is exactly why the saved artifact pairs it with its own imputer,
    scaler and feature list.
    """
    return {
        "Isolation Forest": IsolationForest(
            contamination=contamination, random_state=RANDOM_STATE,
            n_estimators=300),
    }


# ===========================================================================
# 2. Training harness
# ===========================================================================
def _subsample(X, y, g, cap, seed=RANDOM_STATE):
    if cap is None or len(X) <= cap:
        return X, y, g
    idx = pd.Series(range(len(X))).sample(cap, random_state=seed).values
    return X.iloc[idx], y.iloc[idx], g.iloc[idx]


def build_pipeline(estimator, dataset, prep="dense", use_smote=False):
    """preprocessing -> optional SMOTE -> estimator."""
    key = "prep" if prep == "dense" else "prep_native"
    steps = [("prep", clone(dataset[key]))]
    if use_smote and dataset["imbalanced"]:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def train_one(name, spec, dataset, tune=False, n_iter=12, cv_folds=3,
              tune_max_rows=15_000, search_jobs=1, fit_cap=None):
    """Fit one model on one target. Returns (pipeline, params, seconds)."""
    factory, grid, prep, use_smote = spec
    ctx = {k: dataset[k] for k in
           ("prevalence", "imbalanced", "scale_pos_weight")}
    ctx["n_train"] = len(dataset["X_tr"])

    pipe = build_pipeline(factory(ctx), dataset, prep, use_smote)
    X_tr, y_tr, g_tr = _subsample(dataset["X_tr"], dataset["y_tr"],
                                  dataset["g_tr"], fit_cap)
    t0 = time.time()
    params = {}

    if tune and grid:
        Xs, ys, gs = _subsample(X_tr, y_tr, g_tr, tune_max_rows)
        search = RandomizedSearchCV(
            pipe, {f"model__{k}": v for k, v in grid.items()},
            n_iter=n_iter, cv=GroupKFold(min(cv_folds, gs.nunique())),
            scoring=SCORING, n_jobs=search_jobs,
            # pre_dispatch caps how many jobs are queued at once; the default
            # ('2*n_jobs') still materialises several copies of the training
            # data before any of them finish.
            pre_dispatch="n_jobs",
            random_state=RANDOM_STATE, error_score="raise", refit=True)
        search.fit(Xs, ys, groups=gs)
        params = {k.replace("model__", ""): v
                  for k, v in search.best_params_.items()}
        final = clone(pipe)
        final.set_params(**search.best_params_)
        final.fit(X_tr, y_tr)
    else:
        final = pipe
        final.fit(X_tr, y_tr)

    return final, params, round(time.time() - t0, 1)


# ===========================================================================
# 3. Post-hoc ensembling
# ===========================================================================
class GreedyEnsemble(ClassifierMixin, BaseEstimator):
    """Weighted average of fitted pipelines, as a real scikit-learn estimator.

    Weights come from greedy forward selection with replacement, so they are
    non-negative integers over a fixed budget rather than free parameters --
    which is what keeps the procedure from overfitting the selection set.

    Inheriting from BaseEstimator is not cosmetic: sklearn 1.6+ dispatches on
    __sklearn_tags__, and CalibratedClassifierCV(FrozenEstimator(...)) refuses
    anything that does not provide it.
    """

    def __init__(self, members=None, counts=None):
        self.members = members
        self.counts = counts

    def fit(self, X=None, y=None):
        # Members arrive already fitted, so fit() only records the label space
        # and the normalised weights. Trailing underscores are what sklearn's
        # check_is_fitted looks for.
        total = sum(self.counts.values())
        self.weights_ = {k: v / total for k, v in self.counts.items() if v}
        self.classes_ = np.array([0, 1])
        return self

    @property
    def weights(self):
        if not hasattr(self, "weights_"):
            self.fit()
        return self.weights_

    def predict_proba(self, X):
        p = np.zeros(len(X), dtype=float)
        for name, w in self.weights.items():
            p += w * get_scores(self.members[name], X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def prep_of_first_member(self):
        """A preprocessor to reuse for SHAP backgrounds and feature names.

        The ensemble has no preprocessor of its own -- each member carries one.
        They are all fitted on the same columns, so any member's will do.
        """
        for name in self.weights:
            step = getattr(self.members[name], "named_steps", {})
            if "prep" in step:
                return step["prep"]
        return None

    def __repr__(self):
        if self.counts is None:
            return "GreedyEnsemble(unfitted)"
        parts = ", ".join(f"{k} x{self.counts[k]}"
                          for k in sorted(self.weights, key=self.counts.get,
                                          reverse=True))
        return f"GreedyEnsemble({parts})"


def greedy_select(preds, y_true, size=ENSEMBLE_SIZE):
    """Caruana-style forward selection with replacement, maximising PR-AUC."""
    names = list(preds)
    counts = {n: 0 for n in names}
    running = np.zeros(len(y_true), dtype=float)
    chosen, history = 0, []
    for _ in range(size):
        best_name, best_score = None, -np.inf
        for n in names:
            blend = (running * chosen + preds[n]) / (chosen + 1)
            score = average_precision_score(y_true, blend)
            if score > best_score:
                best_name, best_score = n, score
        counts[best_name] += 1
        chosen += 1
        running = (running * (chosen - 1) + preds[best_name]) / chosen
        history.append((best_name, best_score))
    return counts, history


def build_ensemble(fitted, dataset, size=ENSEMBLE_SIZE, fit_caps=None):
    """Select weights on data no member was fitted on.

    The training set is split by mother into A and B. Every member is refitted
    on A with the hyperparameters already chosen -- no new search -- and the
    weights are selected on B. Selecting on the test set would make the
    ensemble's advantage unmeasurable, and selecting on the members' own
    training data would pick whichever model overfits hardest.
    """
    if len(fitted) < 2:
        return None, {}

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25,
                            random_state=RANDOM_STATE)
    a_idx, b_idx = next(gss.split(dataset["X_tr"], dataset["y_tr"],
                                  dataset["g_tr"]))
    X_a, X_b = dataset["X_tr"].iloc[a_idx], dataset["X_tr"].iloc[b_idx]
    y_a, y_b = dataset["y_tr"].iloc[a_idx], dataset["y_tr"].iloc[b_idx]
    g_a = dataset["g_tr"].iloc[a_idx]

    fit_caps = fit_caps or {}
    preds_b = {}
    for name, pipe in fitted.items():
        try:
            member = clone(pipe)
            cap = fit_caps.get(name)
            Xa, ya, _ = _subsample(X_a, y_a, g_a, cap)
            member.fit(Xa, ya)
            preds_b[name] = get_scores(member, X_b)
        except Exception:
            continue

    if len(preds_b) < 2:
        return None, {}

    counts, _ = greedy_select(preds_b, y_b.values, size)
    return GreedyEnsemble(fitted, counts).fit(), counts


# ===========================================================================
# 4. Calibration
# ===========================================================================
def calibration_metrics(y_true, p, eps=1e-6):
    """Calibration slope and calibration-in-the-large.

    Both describe the recalibration model
        logit(P(y=1)) = alpha + beta * logit(p_hat)

    The slope beta is an ordinary logistic fit of the outcome on the logit.
    The intercept alpha must be estimated with beta FIXED AT 1 -- that is what
    makes it "calibration-in-the-large" rather than merely an intercept.
    scikit-learn has no offset term, so alpha is solved from its score equation
        sum_i [ y_i - sigmoid(alpha + z_i) ] = 0,
    which is strictly decreasing in alpha and so has a unique root.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    z = np.log(p / (1 - p))
    y = np.asarray(y_true, dtype=float)

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(z.reshape(-1, 1), y)
    slope = float(lr.coef_[0][0])

    def score(alpha):
        return float(np.sum(y - 1.0 / (1.0 + np.exp(-(alpha + z)))))

    lo, hi = -20.0, 20.0
    if score(lo) < 0 or score(hi) > 0:
        return slope, float("nan")
    for _ in range(200):                      # bisection: simple and robust
        mid = 0.5 * (lo + hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return slope, 0.5 * (lo + hi)


def calibrate_classifier(model, dataset, min_pos_for_isotonic=1500):
    """Fit a calibrator, choosing the method on the evidence available.

    Isotonic regression is a step function with as many pieces as the data
    supports, so it needs a lot of positives to behave. Platt scaling fits two
    parameters and is far more data-efficient. Defaulting to isotonic
    everywhere drove the calibration slope on facility_delivery from 1.083 down
    to 0.265, which is how this rule was arrived at.
    """
    from sklearn.frozen import FrozenEstimator

    n_pos = int(dataset["y_tr"].sum())
    method = "isotonic" if n_pos >= min_pos_for_isotonic else "sigmoid"

    if isinstance(model, GreedyEnsemble):
        # An ensemble cannot be refitted inside the calibrator -- that would
        # discard the selected weights -- so it is frozen and calibrated on a
        # held-out half of the training data.
        calib = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.5,
                                random_state=RANDOM_STATE)
        _, cal_i = next(gss.split(dataset["X_tr"], dataset["y_tr"],
                                  dataset["g_tr"]))
        calib.fit(dataset["X_tr"].iloc[cal_i], dataset["y_tr"].iloc[cal_i])
    else:
        # Cross-fitted calibration refits the estimator inside each fold, so it
        # uses all the training data and needs no held-out slice.
        calib = CalibratedClassifierCV(model, method=method, cv=3)
        calib.fit(dataset["X_tr"], dataset["y_tr"])
    return calib, method


def keep_calibrator(raw_model, calibrator, dataset, citl_tolerance=0.10):
    """A calibrator is kept only if it improves the probabilities on BOTH counts.

    Brier alone is not a sufficient test, and the first version of this guard
    proved it. On severe_stunting the calibrator improved Brier from 0.0844 to
    0.0753 while pushing calibration-in-the-large to +5.0 -- it had learned to
    predict close to zero for everyone, which scores well on a 7.5% outcome and
    is useless. Brier rewards shrinking toward the base rate; CITL is what
    catches a model that has shrunk past it.

    Returns (keep: bool, reason: str).
    """
    y = dataset["y_te"]
    raw_s = np.clip(get_scores(raw_model, dataset["X_te"]), 0, 1)
    cal_s = np.clip(calibrator.predict_proba(dataset["X_te"])[:, 1], 0, 1)

    b_raw, b_cal = brier_score_loss(y, raw_s), brier_score_loss(y, cal_s)
    _, citl_raw = calibration_metrics(y, raw_s)
    _, citl_cal = calibration_metrics(y, cal_s)

    if b_cal > b_raw:
        return False, f"Brier {b_raw:.4f} -> {b_cal:.4f}"
    if abs(citl_cal) > abs(citl_raw) + citl_tolerance:
        return False, f"CITL {citl_raw:+.2f} -> {citl_cal:+.2f}"
    return True, f"Brier {b_raw:.4f} -> {b_cal:.4f}"


# ===========================================================================
# 5. Conformal prediction
# ===========================================================================
def conformal_quantile(model, dataset, alpha=ALPHA):
    """Split-conformal calibration, giving finite-sample coverage of 1 - alpha.

    The calibration half is carved out of the TEST set by group, so the model
    is untouched and the reported coverage is measured on rows used for nothing
    else. Non-conformity is 1 minus the probability assigned to the true label.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5,
                            random_state=RANDOM_STATE)
    cal_idx, ev_idx = next(gss.split(dataset["X_te"], dataset["y_te"],
                                     dataset["g_te"]))
    p_cal = get_scores(model, dataset["X_te"].iloc[cal_idx])
    p_ev = get_scores(model, dataset["X_te"].iloc[ev_idx])
    y_cal = dataset["y_te"].iloc[cal_idx].values
    y_ev = dataset["y_te"].iloc[ev_idx].values

    scores_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
    n = len(scores_cal)
    q = float(np.quantile(scores_cal,
                          min(1.0, np.ceil((n + 1) * (1 - alpha)) / n),
                          method="higher"))

    in_yes = (1 - p_ev) <= q
    in_no = p_ev <= q
    covered = np.where(y_ev == 1, in_yes, in_no)
    both = in_yes & in_no
    empty = ~in_yes & ~in_no

    return {"alpha": alpha, "quantile": q,
            "coverage": float(covered.mean()),
            "both_labels_pct": float(both.mean() * 100),
            "empty_pct": float(empty.mean() * 100),
            "n_eval": int(len(y_ev))}


def prediction_set(p, quantile):
    """The conformal label set for one probability. Both labels means the
    model is abstaining; an empty set means it is confidently confused."""
    labels = []
    if (1 - p) <= quantile:
        labels.append(1)
    if p <= quantile:
        labels.append(0)
    return labels
