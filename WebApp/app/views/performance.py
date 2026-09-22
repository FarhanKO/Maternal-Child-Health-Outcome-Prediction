"""Model Performance: the numbers, then the figures the paper was built on."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import state
from app.ui.theme import (INDIGO, PLOTLY_CONFIG, PLOTLY_LAYOUT, ROSE,
                          TEAL, callout, footer, page_title, section)
from core.engine import TARGET_LABEL

FIGURE_TABS = {
    "Discrimination": [
        ("roc_pr_curves.png", "ROC and precision-recall curves per target. PR-AUC is the "
                              "selection metric because four outcomes are rare."),
        ("model_comparison.png", "Seven learners plus the greedy ensemble. Most differences sit "
                                 "inside the bootstrap intervals — a range, not a leaderboard."),
    ],
    "Operating points": [
        ("threshold_analysis.png", "Expected cost against threshold under a 30% flag-rate cap."),
        ("confusion_matrices.png", "Confusion matrices at the cost-optimal thresholds."),
        ("decision_curves.png", "Net benefit against treat-all and treat-none. Severe stunting "
                                "is negative at its operating point."),
    ],
    "Calibration & uncertainty": [
        ("calibration.png", "Reliability diagrams. Three of five calibrators were refused by "
                            "the two-criterion guard."),
        ("conformal_coverage.png", "Split-conformal coverage: 89.3–90.9% empirical at a nominal 90%."),
    ],
    "Transport & equity": [
        ("temporal_validation.png", "Train on 2017-18, test on 2022. Ranking transports; "
                                    "probabilities run 1.44× to 5.90× observed."),
        ("fairness_subgroups.png", "Recall at one global threshold by wealth, residence, "
                                   "education and division."),
        ("equity.png", "The direction of the subgroup gap reverses between stunting and "
                       "facility delivery."),
    ],
    "Data": [
        ("cohort_nesting.png", "A DHS survey is nested universes, not one table."),
        ("outcome_prevalence.png", "Survey-weighted prevalence per round."),
        ("missingness.png", "Structural versus random missingness."),
        ("geography.png", "Division-level variation."),
    ],
}


def _metric_chart(summary: list[dict]) -> go.Figure:
    labels = [TARGET_LABEL[s["target"]] for s in summary]
    fig = go.Figure()
    fig.add_bar(name="ROC-AUC", x=labels, y=[s["test_roc_auc"] for s in summary],
                marker=dict(color=INDIGO, cornerradius=6, opacity=0.9),
                text=[f"{s['test_roc_auc']:.3f}" for s in summary], textposition="outside")
    fig.add_bar(name="PR-AUC", x=labels, y=[s["test_pr_auc"] for s in summary],
                marker=dict(color=TEAL, cornerradius=6, opacity=0.9),
                text=[f"{s['test_pr_auc']:.3f}" for s in summary], textposition="outside")
    fig.add_scatter(name="Prevalence (PR-AUC baseline)", x=labels,
                    y=[s["prevalence"] for s in summary], mode="markers",
                    marker=dict(symbol="line-ew", size=28, line=dict(width=3, color=ROSE)))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, barmode="group",
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
                      yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="rgba(15,23,42,.06)", zeroline=False),
                      xaxis=dict(showgrid=False))
    return fig


def render() -> None:
    engine = state.get_engine()
    summary = engine.model_summary()

    page_title("Model Performance",
               "Every number below is measured on a grouped hold-out where no mother "
               "appears on both sides of the split. Intervals are 400 bootstrap "
               "resamples.", eyebrow="evidence")

    section("Winner per target")
    table = pd.DataFrame([{
        "target": s["label"], "phase": s["phase"], "kind": s["kind"],
        "cohort": s["cohort"], "model": s["model"],
        "members": (len(s["ensemble_members"]) if s["ensemble_members"] else 1),
        "ROC-AUC": s["test_roc_auc"], "PR-AUC": s["test_pr_auc"],
        "prevalence": s["prevalence"], "threshold": s["threshold"],
        "calibrated": s["calibrated"], "features": s["n_features"],
    } for s in summary])
    st.dataframe(table, hide_index=True, column_config={
        "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
        "prevalence": st.column_config.NumberColumn(format="percent"),
        "threshold": st.column_config.NumberColumn(format="%.3f"),
        "calibrated": st.column_config.CheckboxColumn(),
    })

    a, b = st.columns([1.5, 1], gap="medium")
    with a:
        with st.container(border=True):
            st.plotly_chart(_metric_chart(summary), config=PLOTLY_CONFIG)
    with b:
        callout("Why PR-AUC picks the winner",
                "ROC-AUC is dominated by the true-negative mass on a 4% outcome: "
                "a model that predicts 'survives' every time scores 96% accuracy "
                "while finding nothing. PR-AUC's baseline is the prevalence itself, "
                "so the lift over it is what the model actually adds.")
        st.markdown("")
        rows = pd.DataFrame([{
            "target": s["label"],
            "lift over prevalence": s["test_pr_auc"] / s["prevalence"],
        } for s in summary])
        st.dataframe(rows, hide_index=True, column_config={
            "lift over prevalence": st.column_config.NumberColumn(format="×%.2f")})

    # -------------------------------------------------------- results csvs
    metrics_path = state.RESULTS_DIR / "metrics.csv"
    if metrics_path.exists():
        with st.expander("Full comparison table from results/metrics.csv"):
            st.dataframe(pd.read_csv(metrics_path), hide_index=True)

    # ------------------------------------------------------------- figures
    section("Figures", "Generated by the modelling notebook; each figure is the "
                       "evidence behind one sentence of the paper.")
    tabs = st.tabs(list(FIGURE_TABS))
    for tab, (name, figs) in zip(tabs, FIGURE_TABS.items()):
        with tab:
            for fname, caption in figs:
                path = state.IMAGES_DIR / fname
                if not path.exists():
                    continue
                with st.container(border=True):
                    st.image(str(path), width="stretch")
                    st.caption(caption)
    footer()
