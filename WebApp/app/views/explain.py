"""Explainability: what each model leans on, and what it was forbidden to see."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import state
from app.ui.theme import (MUTED, PLOTLY_CONFIG, PLOTLY_LAYOUT, TEAL,
                          card, footer, page_title, section)
from core.engine import TARGET_LABEL
from core.labels import label as flabel
from src.feature_engineering import LEAKAGE_CLASSES

LEAKAGE_TITLE = {
    "post_outcome": "Post-outcome",
    "same_moment": "Same-moment",
    "definitional": "Definitional",
    "arithmetic_identity": "Arithmetic identity",
    "missingness_as_outcome": "Missingness-as-outcome",
}


def _importance_chart(items: list[tuple[str, float]]) -> go.Figure:
    names = [f"{flabel(c)}  <span style='color:{MUTED};font-size:11px'>{c}</span>" for c, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=vals, colorscale=[[0, "#c7f0ea"], [1, TEAL]], cornerradius=6),
        hovertemplate="%{y}<br>Δ PR-AUC when permuted: %{x:.4f}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=30 * len(items) + 60,
                      xaxis=dict(showgrid=True, gridcolor="rgba(15,23,42,.06)", zeroline=False,
                                 title="permutation importance (drop in PR-AUC on held-out data)",
                                 title_font=dict(size=11, color=MUTED)),
                      yaxis=dict(showgrid=False, tickfont=dict(size=12)))
    return fig


def _members_chart(members: dict) -> go.Figure:
    names, vals = list(members), list(members.values())
    fig = go.Figure(go.Pie(labels=names, values=vals, hole=0.6, sort=False,
                           marker=dict(line=dict(color="white", width=2)),
                           textinfo="label+value", textfont=dict(size=11),
                           hovertemplate="%{label}: %{value} of 15 greedy steps<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False,
                      annotations=[dict(text=f"<b>{len(members)}</b><br><span style='font-size:11px;color:{MUTED}'>members</span>",
                                        x=0.5, y=0.5, showarrow=False, font=dict(size=20))])
    return fig


def render() -> None:
    engine = state.get_engine()
    page_title("Explainability",
               "Permutation importance measured on held-out data, so this is what "
               "each model relies on to generalise — not what it memorised.",
               eyebrow="evidence")

    target = st.pills("Target", engine.targets, default=engine.targets[0],
                      format_func=lambda t: TARGET_LABEL[t], selection_mode="single")
    if target is None:
        target = engine.targets[0]
    summary = {s["target"]: s for s in engine.model_summary()}[target]

    top = st.slider("Features shown", 5, 30, 15, key="explain_top", width=320)
    items = engine.importance(target, top=top)

    left, right = st.columns([1.6, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f"**What `{target}` relies on**")
            if items:
                st.plotly_chart(_importance_chart(items), config=PLOTLY_CONFIG)
                st.caption("Because it permutes the raw column, a categorical like "
                           "`v024` is shuffled as a unit rather than split across its "
                           "one-hot columns.")
            else:
                st.info("This bundle carries no stored importances.")
    with right:
        card(summary["label"], summary["description"], kicker=f"{summary['phase_label']} · {summary['kind']}")
        st.markdown("")
        with st.container(border=True):
            if summary["ensemble_members"]:
                st.markdown("**Ensemble composition**")
                st.plotly_chart(_members_chart(summary["ensemble_members"]), config=PLOTLY_CONFIG)
                st.caption("Greedy forward selection with replacement over 15 steps, "
                           "weights chosen on data no member was fitted on. An "
                           "ensemble has no exact SHAP — a weighted average of boosted "
                           "trees and a neural network is not a tree.")
            else:
                st.markdown(f"**Single model · {summary['model']}**")
                st.caption("The winner on PR-AUC was a single learner; no ensemble "
                           "beat it on the selection set.")
            st.markdown(
                f"<div style='font-size:.82rem;color:{MUTED};line-height:1.7'>"
                f"features <b>{summary['n_features']}</b> · threshold <b>{summary['threshold']:.3f}</b> · "
                f"calibrated <b>{'yes' if summary['calibrated'] else 'no (guard refused)'}</b> · "
                f"conformal q <b>{summary['conformal_quantile']:.3f}</b></div>",
                unsafe_allow_html=True)

    # ------------------------------------------------------------ leakage
    section("Five classes of leakage, each needing a different rule",
            "The last is the one no inspection of values would ever catch. It is "
            "enforced in code, not by convention.")
    cols = st.columns(5, gap="small")
    for col, (key, text) in zip(cols, LEAKAGE_CLASSES.items()):
        with col:
            card(LEAKAGE_TITLE[key], text)

    # ----------------------------------------------------------- glossary
    section("Feature glossary", f"The {len(engine.features)} inputs the five bundles "
                                "accept, with the vocabulary recovered from the fitted encoders.")
    rows = []
    for name, spec in engine.schema.items():
        rows.append({
            "code": name, "label": spec["label"], "group": spec["group"],
            "type": spec["type"],
            "accepted values": (", ".join(spec.get("allowed", []))
                                if spec["type"] != "numeric"
                                else (f"{spec.get('min', '')}–{spec.get('max', '')}"
                                      if "min" in spec else "number")),
            "default": spec["default"],
            "used by": ", ".join(TARGET_LABEL[t] for t in spec["used_by"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=420)
    footer()
