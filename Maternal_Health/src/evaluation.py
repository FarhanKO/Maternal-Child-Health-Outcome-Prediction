"""Metrics, operating points, decision curves, subgroup audits and figures.

The reporting here goes past a single accuracy number on purpose. On a 4%
outcome a model that predicts "survives" every time is 96% accurate and finds
nothing, so accuracy is not reported as a headline anywhere; PR-AUC is the
selection metric and the operating point is chosen by expected cost.
"""
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src.utils import MAX_FLAG_RATE, RANDOM_STATE, get_scores


# ===========================================================================
# 1. Metrics
# ===========================================================================
def get_eval_metrics(y_true, y_pred, scores):
    """The row that goes into the model comparison table.

    Note that Recall here is measured at the DEFAULT 0.5 threshold, not at the
    cost-optimal operating point -- the two must never be read together. For
    facility_delivery that is 0.831 at 0.5 and 0.472 at its actual threshold
    of 0.794.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, scores),
        "PR-AUC": average_precision_score(y_true, scores),
        "Brier": brier_score_loss(y_true, np.clip(scores, 0, 1)),
    }


def evaluate_models(fitted, X_test, y_test):
    """Comparison table for every fitted model on one target, sorted by PR-AUC."""
    rows = {}
    for name, pipe in fitted.items():
        scores = get_scores(pipe, X_test)
        rows[name] = get_eval_metrics(y_test, pipe.predict(X_test), scores)
    return pd.DataFrame(rows).T.sort_values("PR-AUC", ascending=False)


# ===========================================================================
# 2. Operating point
# ===========================================================================
def choose_threshold(y_true, scores, fn_cost, max_flag_rate=MAX_FLAG_RATE,
                     grid=None):
    """Minimise expected cost SUBJECT TO an operational capacity cap.

    Cost alone drives common outcomes to absurd thresholds: stunting at 23.5%
    prevalence was flagging 60% of children, which no service can act on. The
    cap is the operational reality that a pure cost function ignores.

    Returns (threshold, flag_rate, curve_dataframe).
    """
    grid = np.linspace(0.02, 0.95, 150) if grid is None else grid
    y_true = np.asarray(y_true)
    rows = []
    for t in grid:
        pred = (scores >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        rows.append({
            "threshold": t,
            "cost": fn_cost * fn + fp,
            "recall": recall_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
            "flag_rate": float(pred.mean()),
        })
    curve = pd.DataFrame(rows)

    feasible = curve["flag_rate"] <= max_flag_rate
    if feasible.any():
        masked = curve["cost"].where(feasible, curve["cost"].max() + 1)
        best_i = int(masked.idxmin())
    else:
        best_i = int(curve["cost"].idxmin())
    return (float(curve.loc[best_i, "threshold"]),
            float(curve.loc[best_i, "flag_rate"]), curve)


def threshold_comparison(y_true, scores, threshold):
    """Default 0.5 against the chosen operating point, in cases rather than
    rates -- missed cases and false alarms are what a service actually feels."""
    rows = []
    for label, t in [("default 0.50", 0.5), ("cost-optimal", threshold)]:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({"threshold": label, "t": round(t, 3),
                     "recall": round(recall_score(y_true, pred, zero_division=0), 3),
                     "precision": round(precision_score(y_true, pred, zero_division=0), 3),
                     "missed (FN)": int(fn), "false alarms (FP)": int(fp)})
    return pd.DataFrame(rows)


# ===========================================================================
# 3. Decision curve analysis
# ===========================================================================
def net_benefit(y_true, p, thresholds):
    """Net benefit at each threshold probability.

    NB = TP/n - FP/n * pt/(1-pt). The weight pt/(1-pt) is the exchange rate the
    threshold implies: at pt = 0.326 you are declaring one true positive worth
    0.48 false positives.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    out = []
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        out.append(tp / n - (fp / n) * (pt / (1 - pt)))
    return np.array(out)


def decision_curve(y_true, p, operating_point, thresholds=None):
    """Model against treat-all and treat-none, plus the useful range.

    Net benefit is bounded above by prevalence, so a 3.8% outcome cannot return
    more than 38 net cases per 1,000 even with a perfect model. A rare outcome
    predicted from weak features yields a small absolute benefit however good
    the discrimination looks -- which is exactly what AUC conceals.
    """
    y_true = np.asarray(y_true)
    thresholds = (np.linspace(0.01, 0.60, 60) if thresholds is None
                  else thresholds)
    prev = float(y_true.mean())

    nb_model = net_benefit(y_true, p, thresholds)
    nb_all = np.array([prev - (1 - prev) * (pt / (1 - pt)) for pt in thresholds])

    useful = thresholds[(nb_model > nb_all) & (nb_model > 0)]
    nb_at_op = float(net_benefit(y_true, p, [operating_point])[0])
    return {
        "thresholds": thresholds,
        "nb_model": nb_model,
        "nb_treat_all": nb_all,
        "useful_from": float(useful.min()) if len(useful) else float("nan"),
        "useful_to": float(useful.max()) if len(useful) else float("nan"),
        "nb_at_operating_point": nb_at_op,
        "net_cases_per_1000": round(nb_at_op * 1000, 1),
    }


# ===========================================================================
# 4. Uncertainty and subgroups
# ===========================================================================
def bootstrap_ci(y_true, scores, threshold, n_boot=400, seed=RANDOM_STATE):
    """Percentile intervals on every headline number.

    Interval width tracks the positive count, not the point estimate:
    facility_delivery rests on 1,240 test positives and spans 0.031 on ROC-AUC;
    severe_stunting rests on 179 and spans 0.078 on a similar point estimate.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    stats = {"Recall": [], "Precision": [], "ROC-AUC": [], "PR-AUC": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        pred = (scores[idx] >= threshold).astype(int)
        stats["Recall"].append(recall_score(y_true[idx], pred, zero_division=0))
        stats["Precision"].append(precision_score(y_true[idx], pred, zero_division=0))
        stats["ROC-AUC"].append(roc_auc_score(y_true[idx], scores[idx]))
        stats["PR-AUC"].append(average_precision_score(y_true[idx], scores[idx]))
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)))
            for k, v in stats.items() if v}


def subgroup_report(frame, y_true, scores, threshold, subgroups, min_n=150):
    """Discrimination and sensitivity by wealth, residence, education, region.

    The two behave very differently, and the difference is the finding.
    Discrimination is comparatively stable across strata; sensitivity at one
    global threshold is not. For facility_delivery, recall ranges from 0.075 to
    0.803 across wealth quintiles -- an artefact of applying a single cut-point
    to a steeply graded risk distribution, not a decision anyone made.
    """
    rows = []
    for col, label in subgroups.items():
        if col not in frame.columns:
            continue
        for level, idx in frame.groupby(col, observed=True).groups.items():
            pos = y_true.loc[idx]
            if len(pos) < min_n or pos.nunique() < 2:
                continue
            sel = frame.index.get_indexer(idx)
            rows.append({
                "attribute": label, "level": str(level), "n": len(pos),
                "prevalence": round(float(pos.mean()), 4),
                "roc": round(roc_auc_score(pos, scores[sel]), 3),
                "recall @ thr": round(recall_score(
                    pos, (scores[sel] >= threshold).astype(int),
                    zero_division=0), 3),
                "flag rate": round(float((scores[sel] >= threshold).mean()), 3),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 5. Figures
# ===========================================================================
def _save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_roc_pr_curves(y_true, score_map, title="", save_path=None):
    """ROC beside precision-recall, with the no-skill line on both."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    prev = float(np.mean(y_true))
    for name, s in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        axes[0].plot(fpr, tpr, lw=1.6,
                     label=f"{name} ({roc_auc_score(y_true, s):.3f})")
        pr, rc, _ = precision_recall_curve(y_true, s)
        axes[1].plot(rc, pr, lw=1.6,
                     label=f"{name} ({average_precision_score(y_true, s):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.9)
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("recall")
    axes[0].set_title(f"ROC — {title}", fontsize=10)
    axes[1].axhline(prev, color="k", ls="--", lw=0.9)
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title(f"Precision-Recall — {title}", fontsize=10)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_threshold_analysis(curve, threshold, title="", save_path=None):
    """Recall, precision and scaled cost against the threshold."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(curve["threshold"], curve["recall"], lw=1.6, label="recall")
    ax.plot(curve["threshold"], curve["precision"], lw=1.6, label="precision")
    ax2 = ax.twinx()
    ax2.plot(curve["threshold"], curve["cost"] / curve["cost"].max(),
             color="grey", ls=":", lw=1.4)
    ax2.set_ylabel("cost (scaled)", fontsize=8)
    ax.axvline(threshold, color="crimson", ls="--", lw=1.4)
    ax.axvline(0.5, color="black", ls=":", lw=1)
    ax.set_xlabel("threshold"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_calibration(y_true, raw, calibrated, title="", save_path=None):
    """Reliability curves before and after, with Brier in the legend."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for s, label in [(raw, "raw"), (calibrated, "calibrated")]:
        if s is None:
            continue
        s = np.clip(s, 0, 1)
        frac, mean_pred = calibration_curve(y_true, s, n_bins=10,
                                            strategy="quantile")
        ax.plot(mean_pred, frac, "o-", lw=1.5, ms=4,
                label=f"{label} (Brier {brier_score_loss(y_true, s):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_decision_curve(dca, title="", save_path=None):
    """Net benefit against threshold probability, with both default policies."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(dca["thresholds"], dca["nb_model"], lw=1.8, label="model")
    ax.plot(dca["thresholds"], dca["nb_treat_all"], lw=1.2, ls="--",
            label="treat all")
    ax.axhline(0, color="black", lw=1.0, ls=":", label="treat none")
    ax.set_ylim(min(-0.05, float(np.min(dca["nb_model"])) * 1.1),
                max(0.05, float(np.max(dca["nb_model"])) * 1.2))
    ax.set_xlabel("threshold probability"); ax.set_ylabel("net benefit")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_model_comparison(tables, metric="PR-AUC", save_path=None):
    """One grouped bar per target, so the panel is read as a range rather than
    a leaderboard -- most of the differences sit inside the bootstrap noise."""
    import matplotlib.pyplot as plt
    wide = pd.DataFrame({t: tbl[metric] for t, tbl in tables.items()})
    fig, ax = plt.subplots(figsize=(1.9 * len(wide.columns) + 4, 4.6))
    wide.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylabel(metric); ax.set_xlabel("")
    ax.set_title(f"{metric} by model and target", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
