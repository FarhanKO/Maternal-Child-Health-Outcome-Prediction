"""Maternity Health — training entry point.

This is the ONLY place in the `Maternity Health/` codebase that reads the
prepared cohorts. Every src/ module is path-agnostic — they take `path` as a
parameter and never hardcode one — so this file is where a path belongs, not
inside src/.

Run from inside the `Maternity Health/` folder:

    python train.py
    python train.py --targets stunted severe_stunting
    python train.py --tune                      # randomised search, much slower
    python train.py --data /some/other/cohorts

Each target is trained and saved separately. One bundle per target lands in
models/, self-sufficient enough that app/app.py can load it and score a record
with nothing else from this file in memory:

    models/facility_delivery.joblib
    models/neonatal_death.joblib
    models/stunted.joblib
    models/severe_stunting.joblib
    models/underweight_child.joblib
    models/anomaly_bundle.joblib
    models/model_manifest.json

By default it looks for data/cohorts/ and data/target_registry.json — see
data/README.md for how to produce them, they are intentionally not committed.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
# joblib/loky start worker processes from a clean interpreter and do not
# inherit the filter above, so the environment carries it down to them.
import os
os.environ["PYTHONWARNINGS"] = "ignore"

from src.evaluation import (bootstrap_ci, choose_threshold, decision_curve,
                            evaluate_models, subgroup_report,
                            threshold_comparison)
from src.models import (DEVICE, GreedyEnsemble, build_ensemble,
                        calibrate_classifier, calibration_metrics,
                        conformal_quantile, get_supervised_model_grids,
                        get_unsupervised_models, keep_calibrator, train_one)
from src.preprocessing import build_datasets, load_cohorts
from src.utils import (ALPHA, MODELS_DIR, RESULTS_DIR, fn_cost, get_scores,
                       load_registry, save_artifact, save_manifest)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_COHORT_DIR = BASE_DIR / "data" / "cohorts"
DEFAULT_REGISTRY = BASE_DIR / "data" / "target_registry.json"

SUBGROUPS = {"v190": "wealth quintile", "v025": "residence",
             "v106": "education", "v024": "division"}

# Deep learners are capped so one cohort of 112,546 births does not dominate
# the runtime. Everything else sees the full training set.
FIT_CAP = {"RealMLP": 40_000, "TabM": 40_000}


def train_target(target, dataset, registry, zoo, args):
    """Train every model on one target, then ensemble, calibrate and save.

    Returns the manifest entry and the per-model comparison table.
    """
    print(f"\n{'=' * 78}\n{target}   "
          f"({len(dataset['X_tr']):,} train / {len(dataset['X_te']):,} test, "
          f"prevalence {dataset['prevalence'] * 100:.2f}%)\n{'=' * 78}")

    fitted, params = {}, {}
    for name, spec in zoo.items():
        try:
            pipe, chosen, secs = train_one(
                name, spec, dataset, tune=args.tune,
                fit_cap=FIT_CAP.get(name))
            scores = get_scores(pipe, dataset["X_te"])
            fitted[name] = pipe
            params[name] = chosen
            print(f"  {name:22s} PR-AUC={np.round(_ap(dataset, scores), 4):<8} "
                  f"ROC-AUC={np.round(_roc(dataset, scores), 4):<8} ({secs}s)")
        except Exception as exc:
            print(f"  {name:22s} FAILED: {type(exc).__name__}: {exc}")

    if not fitted:
        raise RuntimeError(f"no model trained for {target}")

    # --- post-hoc ensembling ------------------------------------------------
    if not args.no_ensemble and len(fitted) >= 2:
        print("  building greedy ensemble (members refitted on a nested "
              "grouped split of the training data)...")
        ens, counts = build_ensemble(fitted, dataset, fit_caps=FIT_CAP)
        if ens is not None:
            fitted["Ensemble"] = ens
            params["Ensemble"] = {k: v for k, v in counts.items() if v}
            print(f"    {ens}")

    table = evaluate_models(fitted, dataset["X_te"], dataset["y_te"])
    best_name = table["PR-AUC"].idxmax()
    best = fitted[best_name]
    print(f"  winner: {best_name}  "
          f"PR-AUC={table.loc[best_name, 'PR-AUC']:.4f}  "
          f"ROC-AUC={table.loc[best_name, 'ROC-AUC']:.4f}")

    scores = get_scores(best, dataset["X_te"])

    # --- operating point ----------------------------------------------------
    cost = fn_cost(target, registry[target]["kind"])
    threshold, flag_rate, curve = choose_threshold(dataset["y_te"], scores, cost)
    print(f"  threshold {threshold:.3f} (FN cost x{cost}), "
          f"flags {flag_rate * 100:.1f}% of records")

    # --- calibration, with a two-criterion guard ---------------------------
    calibrator, method = calibrate_classifier(best, dataset)
    keep, reason = keep_calibrator(best, calibrator, dataset)
    if keep:
        print(f"  calibrator kept ({method}): {reason}")
    else:
        calibrator = None
        print(f"  calibrator REJECTED ({method}): {reason}")
        print("    raw scores are used downstream -- thresholds, conformal "
              "sets and the saved bundle alike.")

    slope, citl = calibration_metrics(dataset["y_te"], np.clip(scores, 0, 1))
    conformal = conformal_quantile(best, dataset, alpha=ALPHA)
    print(f"  calibration slope {slope:.3f}, CITL {citl:+.3f} | "
          f"conformal q={conformal['quantile']:.3f}, "
          f"coverage {conformal['coverage']:.1%}")

    # --- decision curve -----------------------------------------------------
    dca = decision_curve(dataset["y_te"], scores, threshold)
    print(f"  net benefit at operating point: "
          f"{dca['net_cases_per_1000']} cases per 1,000 screened "
          f"(useful range {dca['useful_from']:.3f}-{dca['useful_to']:.3f})")

    # --- the bundle, one file per target -----------------------------------
    prep = (best.named_steps["prep"] if hasattr(best, "named_steps")
            else best.prep_of_first_member)
    bundle = {
        "target": target,
        "cohort": registry[target]["cohort"],
        "phase": registry[target]["phase"],
        "kind": registry[target]["kind"],
        "positive_label": registry[target].get("positive_label", ""),
        "model_name": best_name,
        "best_params": params.get(best_name, {}),
        "threshold": float(threshold),
        "features": list(dataset["X"].columns),
        "raw_pipeline": best,
        "calibrated": calibrator,
        "prevalence": dataset["prevalence"],
        "encoded_names": (list(prep.get_feature_names_out())
                          if prep is not None else []),
        "shap_background": (prep.transform(dataset["X_tr"].iloc[:200])
                            if prep is not None else None),
        "conformal": {"alpha": conformal["alpha"],
                      "quantile": conformal["quantile"]},
        "ensemble_members": (dict(best.counts)
                             if isinstance(best, GreedyEnsemble) else None),
    }
    save_artifact(bundle, f"{target}.joblib")

    n_members = (sum(1 for v in bundle["ensemble_members"].values() if v)
                 if bundle["ensemble_members"] else 1)
    entry = {
        "file": f"{target}.joblib",
        "phase": bundle["phase"],
        "kind": bundle["kind"],
        "model": best_name,
        "ensemble_members": n_members,
        "threshold": round(float(threshold), 4),
        "flag_rate": round(flag_rate, 4),
        "n_features": len(bundle["features"]),
        "prevalence": round(dataset["prevalence"], 4),
        # NOTE the key name. The comparison table thresholds at 0.5, whereas
        # "threshold" above is the cost-optimal cut-point. The two must not be
        # read together.
        "test_recall_at_0.5": round(float(table.loc[best_name, "Recall"]), 4),
        "test_pr_auc": round(float(table.loc[best_name, "PR-AUC"]), 4),
        "test_roc_auc": round(float(table.loc[best_name, "ROC-AUC"]), 4),
        "calibrated": calibrator is not None,
        "calibration_slope": round(slope, 3),
        "calibration_citl": round(citl, 3),
        "conformal_alpha": conformal["alpha"],
        "conformal_quantile": round(conformal["quantile"], 4),
        "conformal_coverage": round(conformal["coverage"], 4),
        "net_cases_per_1000": dca["net_cases_per_1000"],
    }
    return entry, table, scores, threshold, curve


def _ap(dataset, scores):
    from sklearn.metrics import average_precision_score
    return average_precision_score(dataset["y_te"], scores)


def _roc(dataset, scores):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(dataset["y_te"], scores)


def train_anomaly_detector(cohorts, registry, n_features=30):
    """The "is this record ordinary?" layer.

    An IsolationForest pickled on its own expects pre-transformed features and
    carries no record of which columns those were or how they were scaled, so
    a new record could not be prepared for it. The bundle pairs it with its
    imputer, scaler, feature list and a reference score distribution — which is
    what converts a raw isolation score into "more ordinary than X% of the
    cohort".
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    cohort = registry[list(registry)[0]]["cohort"]
    frame = cohorts[cohort]
    num = (frame.select_dtypes(include=[np.number])
           .dropna(axis=1, thresh=int(0.5 * len(frame))))
    feats = num.columns[:n_features].tolist()

    # The imputer and scaler are stored as separate objects rather than as one
    # Pipeline because that is the schema the rest of the project already
    # writes, and a bundle format that differs between producers is a bug
    # waiting to happen.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(frame[feats].astype(float)))

    detector = get_unsupervised_models()["Isolation Forest"]
    detector.fit(X)
    reference = detector.score_samples(X[:5000])

    bundle = {"features": feats, "imputer": imputer, "scaler": scaler,
              "model": detector, "reference_scores": reference,
              "contamination": 0.05, "cohort": cohort}
    save_artifact(bundle, "anomaly_bundle.joblib")
    print(f"\nanomaly detector: IsolationForest on {len(feats)} numeric "
          f"features, {len(reference):,} reference scores")
    return bundle


def main(args):
    t_start = time.time()
    print(f"device for deep-tabular learners: {DEVICE}")
    if DEVICE == "cpu":
        print("  (RealMLP and TabM run on CPU here; a CUDA build matching your "
              "GPU's compute capability makes them roughly 10x faster)")

    print(f"\nloading registry from {args.registry} ...")
    registry = load_registry(args.registry)
    targets = args.targets or list(registry)
    unknown = [t for t in targets if t not in registry]
    if unknown:
        raise SystemExit(f"unknown target(s): {', '.join(unknown)}")

    print(f"loading cohorts from {args.data} ...")
    cohorts = load_cohorts(registry, cohort_dir=str(args.data))
    for name, frame in cohorts.items():
        print(f"  {name:16s} {len(frame):7,} rows x {frame.shape[1]:3d} cols   "
              f"{frame['mother_id'].nunique():7,} mothers")

    print("\nbuilding design matrices and grouped splits "
          "(a mother never crosses the split)...")
    datasets = build_datasets(registry, cohorts, targets)

    zoo = get_supervised_model_grids(model_jobs=args.model_jobs)
    if args.models:
        unknown = [m for m in args.models if m not in zoo]
        if unknown:
            raise SystemExit(f"unknown model(s): {', '.join(unknown)}. "
                             f"available: {', '.join(zoo)}")
        zoo = {m: zoo[m] for m in args.models}
    print(f"model panel: {', '.join(zoo)}"
          f"{' + Ensemble' if not args.no_ensemble else ''}")
    if args.tune:
        print("randomised search ENABLED -- expect this to take several hours")

    RESULTS_PATH = Path(RESULTS_DIR)
    RESULTS_PATH.mkdir(exist_ok=True)

    manifest, all_tables, fairness_rows = {}, {}, []
    for target in targets:
        entry, table, scores, threshold, curve = train_target(
            target, datasets[target], registry, zoo, args)
        manifest[target] = entry
        all_tables[target] = table

        curve.to_csv(RESULTS_PATH / f"threshold_curve_{target}.csv", index=False)
        threshold_comparison(datasets[target]["y_te"], scores, threshold) \
            .to_csv(RESULTS_PATH / f"operating_point_{target}.csv", index=False)

        ci = bootstrap_ci(datasets[target]["y_te"], scores, threshold)
        for metric, (mean, lo, hi) in ci.items():
            manifest[target][f"{metric} 95% CI"] = \
                f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"

        frame = cohorts[registry[target]["cohort"]]
        te_frame = frame.loc[datasets[target]["X_te"].index]
        fair = subgroup_report(te_frame, datasets[target]["y_te"], scores,
                               threshold, SUBGROUPS)
        if len(fair):
            fair.insert(0, "target", target)
            fairness_rows.append(fair)

    # --- results tables -----------------------------------------------------
    metrics = pd.concat(
        {t: tbl for t, tbl in all_tables.items()}, names=["target", "model"])
    metrics.to_csv(RESULTS_PATH / "metrics.csv")

    pd.DataFrame(manifest).T.to_csv(RESULTS_PATH / "winners.csv")
    if fairness_rows:
        pd.concat(fairness_rows).to_csv(RESULTS_PATH / "fairness.csv",
                                        index=False)

    train_anomaly_detector(cohorts, registry)
    save_manifest(manifest)

    print(f"\n{'=' * 78}")
    print(f"done in {(time.time() - t_start) / 60:.1f} min")
    print(f"  models  -> {MODELS_DIR}")
    print(f"  results -> {RESULTS_DIR}")
    print(f"{'=' * 78}")
    print(pd.DataFrame(manifest).T[
        ["model", "test_roc_auc", "test_pr_auc", "threshold",
         "net_cases_per_1000"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the maternal and child health risk cascade.")
    parser.add_argument("--data", type=Path, default=DEFAULT_COHORT_DIR,
                        help="directory holding cohort_*.parquet")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                        help="target_registry.json")
    parser.add_argument("--targets", nargs="*", default=None,
                        help="subset of targets to train (default: all)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of learners to fit (default: the whole "
                             "panel). Names must match src.models exactly, "
                             'e.g. --models CatBoost "Logistic Regression"')
    parser.add_argument("--tune", action="store_true",
                        help="randomised hyperparameter search per model")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="skip greedy post-hoc ensembling")
    parser.add_argument("--model-jobs", type=int, default=1,
                        help="threads inside each estimator")
    main(parser.parse_args())
