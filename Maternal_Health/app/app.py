"""Maternity Health — risk screening dashboard.

Loads the five saved bundles from models/ and runs the mother-child cascade on
a single record, a synthetic patient, or an uploaded CSV.

This app imports NOTHING from train.py. Every model, threshold, feature list,
importance score, conformal quantile and preprocessing step is read from disk —
which is the only way to know the artifacts are self-contained.

    streamlit run app/app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

# GreedyEnsemble is pickled by reference, so the class must be importable
# before joblib.load runs. It lives in src.models here rather than __main__,
# which is exactly why these bundles load outside the notebook that made them.
from src.models import GreedyEnsemble, prediction_set          # noqa: E402
from src.utils import get_scores                               # noqa: E402
from assets.sample_patients import SAMPLE_PATIENTS, BASE_TEMPLATE  # noqa: E402

# Backwards compatibility. Bundles written by the exploratory notebooks recorded
# the class as "__main__.GreedyEnsemble", because that is where it was defined
# when they ran. Aliasing it here lets those older files load unchanged
# alongside the ones train.py writes.
import __main__                                                # noqa: E402
__main__.GreedyEnsemble = GreedyEnsemble
__main__.get_scores = get_scores

st.set_page_config(page_title="Maternity Health — Risk Screening",
                   page_icon="🤰", layout="wide",
                   initial_sidebar_state="expanded")

CSS = APP_DIR / "assets" / "style.css"
if CSS.exists():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

PHASE_ORDER = ["delivery", "newborn", "child"]
PHASE_LABEL = {"delivery": "Delivery", "newborn": "Newborn (0-28 days)",
               "child": "Child (0-59 months)"}


# --------------------------------------------------------------------------
# LOADING
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model bundles...")
def load_bundles():
    """One bundle per target, plus the anomaly detector.

    Each was written by train.py and carries its own preprocessor, so nothing
    here needs to know how a feature was encoded.
    """
    bundles = {}
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        if path.stem == "anomaly_bundle":
            continue
        bundles[path.stem] = joblib.load(path)
    anomaly = None
    if (MODELS_DIR / "anomaly_bundle.joblib").exists():
        anomaly = joblib.load(MODELS_DIR / "anomaly_bundle.joblib")
    return bundles, anomaly


@st.cache_data(show_spinner=False)
def load_results():
    out = {}
    for name in ["metrics.csv", "winners.csv", "fairness.csv"]:
        path = RESULTS_DIR / name
        if path.exists():
            out[name] = pd.read_csv(path)
    return out


def score_record(bundle, row_df):
    """Calibrated probability where a calibrator survived the guard, raw
    otherwise. train.py drops any calibrator that made the probabilities worse
    on either Brier or calibration-in-the-large, and when it does the raw
    scores are what everything downstream uses."""
    X = row_df.reindex(columns=bundle["features"])
    if bundle.get("calibrated") is not None:
        return float(bundle["calibrated"].predict_proba(X)[:, 1][0])
    return float(get_scores(bundle["raw_pipeline"], X)[0])


def anomaly_percentile(anomaly, row_df):
    """Where this record sits in the reference score distribution.

    A raw isolation score means nothing on its own; "more ordinary than 82% of
    the cohort" does.
    """
    if anomaly is None:
        return None
    X = row_df.reindex(columns=anomaly["features"]).astype(float)
    Xt = anomaly["scaler"].transform(anomaly["imputer"].transform(X))
    s = anomaly["model"].score_samples(Xt)
    return float((anomaly["reference_scores"] < s[0]).mean() * 100)


def assess(bundles, anomaly, row_df):
    """The cascade: every target scored, banded, and set against its threshold."""
    rows = []
    for target, b in bundles.items():
        p = score_record(b, row_df)
        thr = b["threshold"]
        q = (b.get("conformal") or {}).get("quantile")
        labels = prediction_set(p, q) if q is not None else []
        rows.append({
            "target": target,
            "phase": b["phase"],
            "kind": b["kind"],
            "probability": p,
            "national rate": b["prevalence"],
            "threshold": thr,
            "flagged": p >= thr,
            "lift": p / b["prevalence"] if b["prevalence"] else np.nan,
            "committed": len(labels) == 1,
        })
    frame = pd.DataFrame(rows)
    frame["phase"] = pd.Categorical(frame["phase"], PHASE_ORDER, ordered=True)
    return frame.sort_values(["phase", "target"]), anomaly_percentile(anomaly, row_df)


def band(assessment):
    """Triage band from the adverse outcomes only.

    facility_delivery is deliberately excluded from escalation: a high
    probability of reaching a facility is a *favourable* sign, and counting it
    as risk put every wealthy urban mother in the top band.
    """
    adverse = assessment[assessment["kind"] == "adverse"]
    score = int(adverse["flagged"].sum())
    if score == 0:
        return "ROUTINE", score
    if score == 1:
        return "MONITOR", score
    return "PRIORITY", score


# --------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------
if not MODELS_DIR.exists() or not any(MODELS_DIR.glob("*.joblib")):
    st.error("No model bundles found in `models/`. Run `python train.py` first "
             "— see data/README.md for how to obtain the cohorts.")
    st.stop()

BUNDLES, ANOMALY = load_bundles()
RESULTS = load_results()

st.sidebar.title("Maternity Health")
st.sidebar.caption("BDHS 2017-18 + 2022 · 5 outcomes · 3 phases")
page = st.sidebar.radio(
    "",
    ["Risk Assessment", "Overview", "Model Performance", "Explainability",
     "Batch Scoring", "How It Works"],
    label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Population-level risk estimates from socio-demographic inputs. "
    "Not a diagnosis, and not a substitute for clinical assessment.")


# ---------------------------------------------------------- Risk Assessment
if page == "Risk Assessment":
    st.title("Risk Assessment")
    st.caption("Where she delivers → whether the baby survives → how the "
               "child grows.")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Subject")
        preset = st.selectbox("Start from", ["Custom"] + list(SAMPLE_PATIENTS))
        template = dict(BASE_TEMPLATE)
        if preset != "Custom":
            template.update(SAMPLE_PATIENTS[preset])

        age = st.slider("Mother's age", 15, 49, int(template.get("v012", 27)))
        parity = st.slider("Children ever born", 0, 12,
                           int(template.get("v201", 1)))
        educ = st.selectbox(
            "Education", ["no education", "primary", "secondary", "higher"],
            index=["no education", "primary", "secondary", "higher"].index(
                template.get("v106", "secondary")))
        wealth = st.selectbox(
            "Wealth quintile",
            ["poorest", "poorer", "middle", "richer", "richest"],
            index=["poorest", "poorer", "middle", "richer", "richest"].index(
                template.get("v190", "middle")))
        residence = st.selectbox("Residence", ["urban", "rural"],
                                 index=0 if template.get("v025") == "urban" else 1)
        bmi = st.slider("BMI", 12.0, 45.0, float(template.get("bmi", 21.5)), 0.1)
        height = st.slider("Height (cm)", 130.0, 185.0,
                           float(template.get("height_cm", 151.0)), 0.5)
        anc = st.slider("Antenatal visits", 0, 15, int(template.get("m14", 4)))

        record = dict(template)
        record.update({"v012": age, "v201": parity, "v106": educ,
                       "v190": wealth, "v025": residence, "bmi": bmi,
                       "height_cm": height, "m14": anc})
        row = pd.DataFrame([record])
        go = st.button("Assess", type="primary", use_container_width=True)

    with right:
        if go or preset != "Custom":
            assessment, ordinary = assess(BUNDLES, ANOMALY, row)
            label, n_flags = band(assessment)

            colour = {"ROUTINE": "#2a9d8f", "MONITOR": "#e9a03b",
                      "PRIORITY": "#d64545"}[label]
            st.markdown(
                f"<div style='background:{colour};color:white;padding:14px 18px;"
                f"border-radius:10px;font-size:1.35rem;font-weight:600'>"
                f"{label} &nbsp;·&nbsp; {n_flags} of 4 adverse outcomes flagged"
                f"</div>", unsafe_allow_html=True)

            if ordinary is not None:
                st.caption(
                    f"Anomaly check: more ordinary than {ordinary:.0f}% of the "
                    f"reference cohort. A low figure is expected here and is "
                    f"not a finding about the subject — the unedited fields "
                    f"come from a template that sits at the median of every "
                    f"variable at once, and that combination does not occur in "
                    f"the data. The check earns its keep on real uploaded "
                    f"records, in **Batch Scoring**, where it flags rows whose "
                    f"feature pattern the models were never trained on.")

            for phase in PHASE_ORDER:
                block = assessment[assessment["phase"] == phase]
                if block.empty:
                    continue
                st.markdown(f"**{PHASE_LABEL[phase]}**")
                for _, r in block.iterrows():
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                    c1.write(f"`{r['target']}`")
                    c2.metric("risk", f"{r['probability'] * 100:.1f}%",
                              f"{(r['probability'] - r['national rate']) * 100:+.1f} pt vs national",
                              delta_color="inverse" if r["kind"] == "adverse" else "normal")
                    c3.write(f"national {r['national rate'] * 100:.1f}%")
                    c4.write("🚩 flagged" if r["flagged"] else "— below threshold")
                    if not r["committed"]:
                        c4.caption("model abstains at 90% coverage")
        else:
            st.info("Set the profile on the left and press **Assess**, or pick "
                    "a sample subject to score immediately.")


# ------------------------------------------------------------------ Overview
elif page == "Overview":
    st.title("Overview")
    st.write(
        "Five outcomes across three phases of the maternal and child health "
        "cascade, modelled on two pooled rounds of the Bangladesh DHS — "
        "121,067 records from 45,458 mothers.")

    rows = [{"target": t, "phase": b["phase"], "kind": b["kind"],
             "cohort": b["cohort"], "model": b["model_name"],
             "national rate": f"{b['prevalence'] * 100:.2f}%",
             "threshold": round(b["threshold"], 3),
             "features": len(b["features"]),
             "calibrated": b.get("calibrated") is not None}
            for t, b in BUNDLES.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Class balance")
    prev = pd.Series({t: b["prevalence"] for t, b in BUNDLES.items()}
                     ).sort_values()
    st.bar_chart(prev)
    st.caption(
        "The spread is why one recipe cannot be applied uniformly. "
        "`facility_delivery` near 57% is genuinely balanced; `neonatal_death` "
        "at 4% means 23.8 survivors per death, and a model predicting "
        "\"survives\" every time scores 96% accuracy while finding nothing.")


# ---------------------------------------------------------- Model Performance
elif page == "Model Performance":
    st.title("Model Performance")
    if "winners.csv" in RESULTS:
        st.subheader("Winning model per target")
        st.dataframe(RESULTS["winners.csv"], use_container_width=True)
    if "metrics.csv" in RESULTS:
        st.subheader("Full model comparison")
        st.dataframe(RESULTS["metrics.csv"], use_container_width=True)
        st.caption(
            "Most differences between algorithms sit inside the bootstrap "
            "intervals. The panel is reported as a range, not a leaderboard.")

    for img in ["model_comparison.png", "roc_pr_curves.png",
                "threshold_analysis.png", "calibration.png",
                "decision_curves.png"]:
        path = IMAGES_DIR / img
        if path.exists():
            st.image(str(path), caption=img, use_container_width=True)


# ------------------------------------------------------------ Explainability
elif page == "Explainability":
    st.title("Explainability")
    target = st.selectbox("Target", list(BUNDLES))
    b = BUNDLES[target]

    imp = pd.Series(b.get("importance") or {}).sort_values(ascending=False)
    if len(imp):
        st.subheader("Permutation importance")
        st.bar_chart(imp.head(15))
        st.caption(
            "Measured on held-out data, so this is what the model relies on to "
            "*generalise*. Because it permutes the raw column, a categorical "
            "like `v024` is shuffled as a unit rather than split across its "
            "one-hot columns.")
    else:
        st.info("This bundle carries no stored importances.")

    if b.get("ensemble_members"):
        st.subheader("Ensemble composition")
        members = {k: v for k, v in b["ensemble_members"].items() if v}
        st.bar_chart(pd.Series(members).sort_values(ascending=False))
        st.caption(
            f"{len(members)} of 7 candidates carry weight, over 15 greedy "
            "steps. An ensemble has no exact SHAP — a weighted average of "
            "boosted trees and a neural network is not a tree — so any "
            "attribution shown for it approximates the heaviest tree member.")


# -------------------------------------------------------------- Batch Scoring
elif page == "Batch Scoring":
    st.title("Batch Scoring")
    st.write("Upload a CSV of records. Missing columns are filled from the "
             "template, and every model scores every row.")
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        frame = pd.read_csv(up)
        st.write(f"{len(frame):,} rows uploaded.")
        filled = frame.copy()
        for col, val in BASE_TEMPLATE.items():
            if col not in filled.columns:
                filled[col] = val

        out = frame.copy()
        for target, b in BUNDLES.items():
            X = filled.reindex(columns=b["features"])
            if b.get("calibrated") is not None:
                p = b["calibrated"].predict_proba(X)[:, 1]
            else:
                p = get_scores(b["raw_pipeline"], X)
            out[f"{target}_risk"] = np.round(p, 4)
            out[f"{target}_flag"] = (p >= b["threshold"]).astype(int)

        adverse = [t for t, b in BUNDLES.items() if b["kind"] == "adverse"]
        out["flags"] = out[[f"{t}_flag" for t in adverse]].sum(axis=1)
        out["band"] = pd.cut(out["flags"], [-1, 0, 1, 99],
                             labels=["ROUTINE", "MONITOR", "PRIORITY"])

        st.dataframe(out.head(200), use_container_width=True)
        st.download_button("Download scored CSV",
                           out.to_csv(index=False).encode("utf-8"),
                           "scored_records.csv", "text/csv")


# -------------------------------------------------------------- How It Works
else:
    st.title("How It Works")
    st.markdown("""
### The cascade

Five outcomes, ordered by when they happen, so the report reads as a sequence
rather than a list: **where she delivers → whether the baby survives → how the
child grows.** Each is a point at which an intervention exists.

### Why the split is grouped

90% of rows in the births cohort belong to a mother who appears more than once.
Under a random split, siblings — who share a mother, a household, a village, a
wealth quintile and most of the feature vector — would land on both sides, and
every reported metric would be optimistic. `GroupKFold` on `mother_id` puts an
entire family in one fold.

### Why thresholds are not 0.5

Each threshold minimises expected cost, where one missed case is worth several
false alarms, **subject to flagging no more than 30% of the population**. Cost
alone drives common outcomes to absurd operating points: stunting at 23.5%
prevalence was flagging 60% of children, which no service can act on.

### What the numbers are not

A probability of 0.31 for stunting means roughly a third of women with this
profile have a stunted child — not that this child will be. These are
population-level estimates built to prioritise limited contact time.

### Known limits

- Discrimination transports across survey rounds; **calibration does not.**
  Predicted risk exceeded observed by 1.4x to 5.9x when a model trained on
  2017-18 was applied to 2022, so absolute probabilities must be recalibrated
  on arrival in a new setting.
- Sensitivity at one global threshold varies sharply by wealth and education.
  For `facility_delivery`, recall ranges from 0.075 in the poorest quintile to
  0.803 in the richest — an artefact of a single cut-point on a steeply graded
  risk distribution, not a decision anyone made.
- `severe_stunting` has **negative net benefit** at its operating point in
  decision curve analysis. It is reported for completeness and should not be
  deployed as a standalone screen.
""")
