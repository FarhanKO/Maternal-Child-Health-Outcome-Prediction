"""The cascade engine: load the bundles once, score records the same way
everywhere.

Everything a caller needs is read from the bundles train.py wrote — feature
lists, thresholds, calibrators, conformal quantiles, importances, and the
fitted preprocessors, from which the accepted vocabulary of every categorical
column is recovered. Nothing here imports train.py.

Input semantics, shared by the dashboard, the batch scorer and the API:

  * a field that is OMITTED is filled in four steps — a functional rule
    where one exists (BMI from height and weight), then the cohort-typical
    value within the caller's own wealth-quintile x residence cell, then the
    flat marginal template — and the response lists what happened to each
    under `inputs.defaulted` and `inputs.fill_strategy` (see core/defaults.py);
  * a field sent as null/NaN is treated as genuinely unknown and left missing,
    which the preprocessors were trained to handle;
  * a categorical value outside the training vocabulary is treated as missing
    and reported under `inputs.warnings`;
  * unknown field names are ignored and reported under `inputs.ignored`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from core import paths  # noqa: F401  (Maternal_Health/ on sys.path for src.*)
from core.defaults import (apply_group_defaults, apply_rules,
                           apply_template)
from core.labels import group as feature_group
from core.labels import label as feature_label
from core.model_store import (OPTIONAL_FILES, REQUIRED_BUNDLES, models_dir)
from core.presets import BASE_TEMPLATE

# The deep-tabular members log a Lightning banner on every predict call.
for _name in ("pytorch_lightning", "lightning", "lightning_fabric",
              "lightning_utilities"):
    logging.getLogger(_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

PHASE_ORDER = ["delivery", "newborn", "child"]
PHASE_LABEL = {"delivery": "Delivery", "newborn": "Newborn · 0–28 days",
               "child": "Child · 0–59 months"}
TARGET_ORDER = ["facility_delivery", "neonatal_death", "stunted",
                "severe_stunting", "underweight_child"]
TARGET_LABEL = {
    "facility_delivery": "Facility delivery",
    "neonatal_death": "Neonatal death",
    "stunted": "Stunting",
    "severe_stunting": "Severe stunting",
    "underweight_child": "Underweight",
}
TARGET_DESCRIPTION = {
    "facility_delivery": (
        "Probability that the birth takes place in a health facility. This "
        "is a care-pathway outcome, so a high value is favourable and it "
        "does not count towards the triage band."),
    "neonatal_death": (
        "Probability that the newborn dies within the first 28 days of life."),
    "stunted": (
        "Probability that the child's height-for-age is below −2 SD of the "
        "WHO reference median."),
    "severe_stunting": (
        "Probability that the child's height-for-age is below −3 SD of the "
        "WHO reference median."),
    "underweight_child": (
        "Probability that the child's weight-for-age is below −2 SD of the "
        "WHO reference median."),
}
TARGET_NOTE = {
    "severe_stunting": (
        "Negative net benefit at its operating point in decision-curve "
        "analysis. Reported for completeness; not a standalone screen."),
}
BAND_DESCRIPTION = {
    "ROUTINE": "No adverse outcome above its operating threshold.",
    "MONITOR": "One adverse outcome above threshold — schedule follow-up.",
    "PRIORITY": "Two or more adverse outcomes above threshold — prioritise "
                "contact time.",
}
DISCLAIMER = (
    "Population-level risk estimates from socio-demographic survey inputs, "
    "trained on BDHS 2017-18 and 2022. A probability of 0.31 means roughly a "
    "third of women with this profile had the outcome, not that this one "
    "will. Not a diagnosis, not calibrated for other settings, and not a "
    "substitute for clinical assessment.")

MISSING_TOKEN = "__missing__"

# Ranges for the numeric fields the assessment form exposes. The vocabulary
# of every categorical field comes from the fitted encoders instead.
NUMERIC_HINTS = {
    "v012": dict(min=15, max=49, step=1),
    "v201": dict(min=0, max=12, step=1),
    "v218": dict(min=0, max=12, step=1),
    "v212": dict(min=10, max=45, step=1),
    "v511": dict(min=10, max=40, step=1),
    "v133": dict(min=0, max=18, step=1),
    "v136": dict(min=1, max=25, step=1),
    "v137": dict(min=0, max=7, step=1),
    "height_cm": dict(min=120.0, max=190.0, step=0.5),
    "weight_kg": dict(min=25.0, max=130.0, step=0.5),
    "bmi": dict(min=12.0, max=50.0, step=0.1),
    "m14": dict(min=0, max=15, step=1),
    "m1": dict(min=0, max=7, step=1),
    "b11": dict(min=9, max=240, step=1),
    "b19": dict(min=0, max=59, step=1),
    "p20": dict(min=5, max=10, step=1),
    "risk_index": dict(min=0, max=6, step=1),
    "v730": dict(min=15, max=95, step=1),
    "v613": dict(min=0, max=12, step=1),
    "v238": dict(min=0, max=3, step=1),
}

def _register_pickle_aliases() -> None:
    """Bundles written from a notebook recorded the ensemble class under
    __main__; train.py's live in src.models. Both must resolve."""
    import __main__

    from src.models import GreedyEnsemble
    from src.utils import get_scores

    __main__.GreedyEnsemble = GreedyEnsemble
    __main__.get_scores = get_scores


def band_for(n_flagged: int) -> str:
    if n_flagged <= 0:
        return "ROUTINE"
    if n_flagged == 1:
        return "MONITOR"
    return "PRIORITY"


def _iter_preps(bundle: dict) -> Iterable[Any]:
    """Every fitted ColumnTransformer inside a bundle. An ensemble carries one
    per member, and only the dense ones hold imputer medians."""
    pipe = bundle["raw_pipeline"]
    members = (pipe.members.values() if hasattr(pipe, "members")
               else [pipe])
    for m in members:
        steps = getattr(m, "named_steps", {})
        if "prep" in steps:
            yield steps["prep"]


class CascadeEngine:
    """Five bundles, one anomaly detector, one way to score."""

    def __init__(self, directory: str | Path | None = None):
        _register_pickle_aliases()
        from src.models import prediction_set
        from src.utils import get_scores

        self._prediction_set = prediction_set
        self._get_scores = get_scores

        self.directory = Path(directory or models_dir())
        self.bundles: dict[str, dict] = {}
        for name in TARGET_ORDER:
            path = self.directory / f"{name}.joblib"
            if path.is_file():
                self.bundles[name] = joblib.load(path)
        missing = [f for f in REQUIRED_BUNDLES
                   if f[:-7] not in self.bundles]
        if missing:
            raise FileNotFoundError(
                f"missing bundles in {self.directory}: {', '.join(missing)}")

        anomaly_path = self.directory / "anomaly_bundle.joblib"
        self.anomaly = joblib.load(anomaly_path) if anomaly_path.is_file() else None

        manifest_path = self.directory / "model_manifest.json"
        self.manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                         if manifest_path.is_file() else {})

        self.version = self._fingerprint()
        self.schema = self._build_schema()
        self.features = list(self.schema)
        self.targets = list(self.bundles)
        self.adverse_targets = [t for t, b in self.bundles.items()
                                if b["kind"] == "adverse"]
        self.loaded_at = time.time()

    # ------------------------------------------------------------------ meta
    def _fingerprint(self) -> str:
        parts = []
        for f in REQUIRED_BUNDLES + OPTIONAL_FILES:
            p = self.directory / f
            if p.is_file():
                parts.append(f"{f}:{p.stat().st_size}")
        return hashlib.sha1(",".join(parts).encode()).hexdigest()[:10]

    def _build_schema(self) -> dict[str, dict]:
        schema: dict[str, dict] = {}
        for target, bundle in self.bundles.items():
            for feat in bundle["features"]:
                schema.setdefault(feat, {"used_by": []})["used_by"].append(target)
            for prep in _iter_preps(bundle):
                for name, trans, cols in prep.transformers_:
                    if name == "remainder":
                        continue
                    steps = getattr(trans, "named_steps", {})
                    for i, col in enumerate(cols):
                        spec = schema.setdefault(col, {"used_by": []})
                        if name == "num":
                            spec["type"] = "numeric"
                            if "impute" in steps and "median" not in spec:
                                spec["median"] = float(
                                    steps["impute"].statistics_[i])
                        elif name == "ord":
                            enc = steps.get("ordinal", trans)
                            spec["type"] = "ordinal"
                            spec["allowed"] = [str(v) for v in enc.categories_[i]]
                        elif name == "nom":
                            enc = steps["onehot"]
                            spec["type"] = "nominal"
                            cats = [str(v) for v in enc.categories_[i]
                                    if str(v) != MISSING_TOKEN]
                            spec["allowed"] = sorted(
                                set(spec.get("allowed", [])) | set(cats))
        for feat, spec in schema.items():
            spec.setdefault("type", "numeric")
            spec["label"] = feature_label(feat)
            spec["group"] = feature_group(feat)
            spec["default"] = BASE_TEMPLATE.get(feat)
            if spec["type"] == "numeric":
                spec.update(NUMERIC_HINTS.get(feat, {}))
                if "median" not in spec and isinstance(spec["default"], (int, float)):
                    spec["median"] = float(spec["default"])
        ordered = sorted(schema, key=lambda f: (
            ["Mother", "Anthropometry", "Household & wealth",
             "Reproductive history", "Care access", "Child & birth",
             "Partner & autonomy", "Other"].index(schema[f]["group"]), f))
        return {f: schema[f] for f in ordered}

    def model_summary(self) -> list[dict]:
        rows = []
        for target, b in self.bundles.items():
            m = self.manifest.get(target, {})
            members = {k: v for k, v in (b.get("ensemble_members") or {}).items() if v}
            rows.append({
                "target": target,
                "label": TARGET_LABEL.get(target, target),
                "phase": b["phase"],
                "phase_label": PHASE_LABEL.get(b["phase"], b["phase"]),
                "kind": b["kind"],
                "cohort": b.get("cohort"),
                "positive_label": b.get("positive_label", ""),
                "model": b.get("model_name"),
                "ensemble_members": members or None,
                "n_features": len(b["features"]),
                "prevalence": float(b["prevalence"]),
                "threshold": float(b["threshold"]),
                "calibrated": b.get("calibrated") is not None,
                "conformal_quantile": (b.get("conformal") or {}).get("quantile"),
                "test_roc_auc": m.get("test_roc_auc"),
                "test_pr_auc": m.get("test_pr_auc"),
                "description": TARGET_DESCRIPTION.get(target, ""),
                "note": TARGET_NOTE.get(target),
            })
        return rows

    def importance(self, target: str, top: int = 20) -> list[tuple[str, float]]:
        imp = self.bundles[target].get("importance") or {}
        return sorted(imp.items(), key=lambda kv: -kv[1])[:top]

    # ------------------------------------------------------------- prepare
    def prepare(self, records: list[dict] | pd.DataFrame,
                fill: str = "template") -> tuple[pd.DataFrame, dict]:
        """Coerce, validate, derive and fill. Returns the frame and a report
        of what happened to the inputs — one list per category, shared by
        every row because the batch path validates whole columns."""
        # A list of dicts is scored record by record: a key one record omits
        # is that record's default, not a null. A DataFrame (CSV upload) is
        # column-oriented, so an empty cell there means "unknown".
        absent: pd.DataFrame | None = None
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        else:
            records = [{str(k).strip().lower(): v for k, v in r.items()}
                       for r in records]
            frame = pd.DataFrame(records)
            absent = pd.DataFrame(
                [{c: c not in r for c in frame.columns} for r in records],
                index=frame.index)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        report = {"provided": [], "defaulted": [], "ignored": [], "warnings": []}

        known = [c for c in frame.columns if c in self.schema]
        report["ignored"] = [c for c in frame.columns if c not in self.schema]
        frame = frame[known]
        if absent is not None:
            absent = absent[known]

        for col in known:
            spec = self.schema[col]
            s = frame[col]
            if spec["type"] == "numeric":
                if s.dtype == object:
                    s = s.replace({True: 1, False: 0, "yes": 1, "no": 0})
                coerced = pd.to_numeric(s, errors="coerce")
                bad = coerced.isna() & s.notna()
                if bad.any():
                    report["warnings"].append(
                        f"{col}: {int(bad.sum())} non-numeric value(s) treated as missing")
                frame[col] = coerced.astype(float)
            else:
                text = s.astype(object).map(
                    lambda v: None if pd.isna(v) else str(v).strip().lower())
                allowed = set(spec.get("allowed", []))
                if allowed:
                    unknown = text.notna() & ~text.isin(allowed)
                    if unknown.any():
                        sample = sorted(set(text[unknown].astype(str)))[:3]
                        report["warnings"].append(
                            f"{col}: value(s) {sample} not in the training "
                            f"vocabulary; treated as missing")
                        text = text.where(~unknown, None)
                frame[col] = text.astype(object).where(text.notna(), np.nan)

        # What the caller actually supplied, before anything is filled in.
        # A key omitted from a record and a key sent as null are different:
        # only the first may be replaced by a default.
        supplied = [c for c in known
                    if absent is None or not bool(absent[c].all())]
        report["provided"] = sorted(supplied)
        if absent is not None and len(records) == 1:
            report["provided"] = sorted(c for c in known if not absent[c].iloc[0])

        # Create the absent columns with the dtype their values will need.
        # An all-NaN column is float64, and pandas 3 raises rather than
        # silently upcasting when a categorical default is written into it.
        for feat in self.features:
            if feat not in frame.columns:
                frame[feat] = (np.nan if self.schema[feat]["type"] == "numeric"
                               else pd.Series([None] * len(frame),
                                              index=frame.index, dtype=object))
        frame = frame[self.features].copy()

        if fill == "template":
            # `known` marks cells the caller filled, so a rule never
            # overwrites a real value — including a deliberate null.
            known_mask = pd.DataFrame(False, index=frame.index,
                                      columns=self.features)
            for col in known:
                known_mask[col] = (~absent[col].to_numpy(dtype=bool)
                                   if absent is not None else True)
            derived = apply_rules(frame, known_mask)
            grouped = apply_group_defaults(frame, self.features, known_mask)
            templated = apply_template(frame, self.features, known_mask)
            report["derived"] = sorted(set(derived))
            report["defaulted"] = sorted(
                (set(grouped) | set(templated) | set(derived))
                - set(report["provided"]))
            report["fill_strategy"] = {
                "derived": sorted(set(derived)),
                "from_wealth_residence_cell": sorted(
                    set(grouped) - set(derived) - set(report["provided"])),
                "from_flat_template": sorted(
                    set(templated) - set(grouped) - set(derived)
                    - set(report["provided"])),
            }
        else:
            report["derived"] = []
            report["defaulted"] = []

        # The pipelines were fitted on object-dtype categoricals holding
        # np.nan (see src.utils.denullify). pandas 3 infers its own `str`
        # dtype for a column built from a scalar, and the fitted encoders do
        # not treat the two identically, so the frame is normalised here.
        frame = frame.copy()
        for feat in self.features:
            if self.schema[feat]["type"] == "numeric":
                frame[feat] = pd.to_numeric(frame[feat], errors="coerce").astype("float64")
            else:
                col = frame[feat].astype(object)
                frame[feat] = col.where(col.notna(), np.nan)
        return frame, report

    # --------------------------------------------------------------- score
    def _probabilities(self, target: str, frame: pd.DataFrame) -> np.ndarray:
        b = self.bundles[target]
        X = frame.reindex(columns=b["features"])
        if b.get("calibrated") is not None:
            return np.asarray(b["calibrated"].predict_proba(X)[:, 1], dtype=float)
        return np.asarray(self._get_scores(b["raw_pipeline"], X), dtype=float)

    def _anomaly_percentiles(self, frame: pd.DataFrame) -> np.ndarray | None:
        if self.anomaly is None:
            return None
        a = self.anomaly
        X = frame.reindex(columns=a["features"]).astype(float)
        Xt = a["scaler"].transform(a["imputer"].transform(X))
        s = a["model"].score_samples(Xt)
        ref = np.asarray(a["reference_scores"])
        return np.array([(ref < v).mean() * 100 for v in s], dtype=float)

    def score_frame(self, frame: pd.DataFrame | list[dict],
                    fill: str = "template",
                    include_anomaly: bool = True) -> tuple[pd.DataFrame, dict]:
        """Batch path: one row in, one row out, with per-target columns.
        Accepts a DataFrame (CSV semantics) or a list of dicts (per-record
        semantics) — see `prepare`."""
        prepared, report = self.prepare(frame, fill=fill)
        out = pd.DataFrame(index=prepared.index)
        for target in self.targets:
            p = self._probabilities(target, prepared)
            out[f"{target}_risk"] = np.round(p, 4)
            out[f"{target}_flag"] = (p >= self.bundles[target]["threshold"]).astype(int)
        out["n_flags"] = out[[f"{t}_flag" for t in self.adverse_targets]].sum(axis=1)
        out["band"] = out["n_flags"].map(band_for)
        if include_anomaly:
            pct = self._anomaly_percentiles(prepared)
            if pct is not None:
                out["anomaly_percentile"] = np.round(pct, 1)
        return out, report

    def assess(self, record: dict, fill: str = "template",
               include_anomaly: bool = True,
               include_record: bool = False) -> dict:
        """Single record → the full cascade report, JSON-serialisable."""
        t0 = time.perf_counter()
        prepared, report = self.prepare([record], fill=fill)

        outcomes = []
        for target in self.targets:
            b = self.bundles[target]
            p = float(self._probabilities(target, prepared)[0])
            thr = float(b["threshold"])
            prev = float(b["prevalence"])
            q = (b.get("conformal") or {}).get("quantile")
            labels = self._prediction_set(p, q) if q is not None else None
            outcomes.append({
                "target": target,
                "label": TARGET_LABEL.get(target, target),
                "phase": b["phase"],
                "phase_label": PHASE_LABEL.get(b["phase"], b["phase"]),
                "kind": b["kind"],
                "description": TARGET_DESCRIPTION.get(target, ""),
                "positive_label": b.get("positive_label", ""),
                "model": b.get("model_name"),
                "probability": round(p, 4),
                "national_rate": round(prev, 4),
                "threshold": round(thr, 4),
                "flagged": bool(p >= thr),
                "delta_points": round((p - prev) * 100, 1),
                "lift": round(p / prev, 2) if prev else None,
                "prediction_set": labels,
                "committed": (len(labels) == 1) if labels is not None else None,
                "note": TARGET_NOTE.get(target),
            })

        n_flagged = sum(o["flagged"] for o in outcomes if o["kind"] == "adverse")
        band = band_for(n_flagged)
        anomaly_pct = None
        if include_anomaly:
            pct = self._anomaly_percentiles(prepared)
            if pct is not None:
                anomaly_pct = round(float(pct[0]), 1)

        phases = []
        for phase in PHASE_ORDER:
            block = [o for o in outcomes if o["phase"] == phase]
            if block:
                phases.append({"phase": phase,
                               "label": PHASE_LABEL[phase],
                               "outcomes": block})

        result = {
            "band": band,
            "band_description": BAND_DESCRIPTION[band],
            "n_flagged": int(n_flagged),
            "n_adverse": len(self.adverse_targets),
            "anomaly_percentile": anomaly_pct,
            "outcomes": outcomes,
            "phases": phases,
            "inputs": report,
            "meta": {
                "model_version": self.version,
                "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
            "disclaimer": DISCLAIMER,
        }
        if include_record:
            row = prepared.iloc[0]
            result["record_used"] = {
                k: (None if (isinstance(v, float) and math.isnan(v)) else
                    (v.item() if hasattr(v, "item") else v))
                for k, v in row.items()}
        return result

    def assess_many(self, records: list[dict], fill: str = "template",
                    include_anomaly: bool = True) -> tuple[list[dict], dict]:
        """Batch path for the API: compact per-record summaries."""
        scored, report = self.score_frame(list(records), fill=fill,
                                          include_anomaly=include_anomaly)
        rows = []
        for i, (_, r) in enumerate(scored.iterrows()):
            rows.append({
                "index": i,
                "band": r["band"],
                "n_flagged": int(r["n_flags"]),
                "anomaly_percentile": (float(r["anomaly_percentile"])
                                       if "anomaly_percentile" in r else None),
                "outcomes": {
                    t: {"probability": float(r[f"{t}_risk"]),
                        "flagged": bool(r[f"{t}_flag"])}
                    for t in self.targets},
            })
        return rows, report

    # ----------------------------------------------------------- schema API
    def schema_payload(self) -> dict:
        fields = []
        for feat, spec in self.schema.items():
            entry = {"name": feat, "label": spec["label"], "group": spec["group"],
                     "type": spec["type"], "default": spec["default"],
                     "used_by": spec["used_by"]}
            if spec["type"] == "numeric":
                for k in ("min", "max", "step", "median"):
                    if k in spec:
                        entry[k] = spec[k]
            else:
                entry["allowed"] = spec.get("allowed", [])
            fields.append(entry)
        return {"fields": fields, "n_fields": len(fields),
                "semantics": {
                    "omitted": "resolved in three steps and reported under "
                               "inputs.fill_strategy: a functional identity "
                               "where one exists (bmi from height_cm and "
                               "weight_kg), then the value typical of the "
                               "caller's wealth-quintile x residence cell, "
                               "then the flat cohort template",
                    "null": "treated as genuinely unknown",
                    "unknown_category": "treated as missing and reported "
                                        "under inputs.warnings",
                    "unknown_field": "ignored and reported under inputs.ignored"},
                "model_version": self.version}
