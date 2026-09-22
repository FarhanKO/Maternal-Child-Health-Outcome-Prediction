"""Overview: what the system predicts, how well, and where to go next."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import nav, state
from app.ui.theme import (INDIGO, MUTED, PLOTLY_CONFIG, PLOTLY_LAYOUT, ROSE,
                          TEAL, callout, card, footer, hero, md, section,
                          stat_tiles)
from core.engine import PHASE_LABEL, PHASE_ORDER, TARGET_LABEL

# Bootstrap intervals from the paper (400 resamples, grouped hold-out).
ROC_CI = {
    "facility_delivery": (0.845, 0.828, 0.859),
    "neonatal_death": (0.773, 0.756, 0.789),
    "stunted": (0.707, 0.687, 0.730),
    "severe_stunting": (0.700, 0.660, 0.738),
    "underweight_child": (0.688, 0.661, 0.712),
}

LEAKAGE_ROWS = [
    ("Leakage-audited", 0.773, "an early-risk estimate"),
    ("Fertility identity retained", 0.967,
     "children ever born − living children is the death count"),
    ("Survivor-only age retained", 0.987,
     "b8 exists for 96.7% of survivors and 0% of deaths, so its presence is the label"),
]


def _roc_chart(summary: list[dict]) -> go.Figure:
    order = [s["target"] for s in summary]
    labels = [TARGET_LABEL[t] for t in order]
    mean = [ROC_CI[t][0] for t in order]
    lo = [ROC_CI[t][0] - ROC_CI[t][1] for t in order]
    hi = [ROC_CI[t][2] - ROC_CI[t][0] for t in order]
    colours = [TEAL if s["kind"] == "pathway" else INDIGO for s in summary]
    fig = go.Figure(go.Bar(
        x=mean, y=labels, orientation="h",
        marker=dict(color=colours, opacity=0.9, cornerradius=8),
        error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo,
                     color="rgba(15,23,42,.55)", thickness=1.5, width=6),
        text=[f"{m:.3f}" for m in mean], textposition="inside",
        insidetextanchor="end", textfont=dict(size=12, color="white"),
        hovertemplate="%{y}<br>ROC-AUC %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=280, showlegend=False,
        xaxis=dict(range=[0.5, 1.0], showgrid=True, gridcolor="rgba(15,23,42,.06)",
                   zeroline=False, title="ROC-AUC on the grouped hold-out (95% bootstrap CI)",
                   title_font=dict(size=11, color=MUTED)),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    return fig


def _prevalence_chart(summary: list[dict]) -> go.Figure:
    labels = [TARGET_LABEL[s["target"]] for s in summary]
    prev = [s["prevalence"] * 100 for s in summary]
    thr = [s["threshold"] * 100 for s in summary]
    fig = go.Figure()
    fig.add_bar(x=labels, y=prev, name="National rate",
                marker=dict(color=ROSE, opacity=0.85, cornerradius=8),
                hovertemplate="%{x}<br>prevalence %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=labels, y=thr, name="Operating threshold", mode="markers",
                    marker=dict(symbol="line-ew", size=26, line=dict(width=3, color="#0f172a")),
                    hovertemplate="%{x}<br>threshold %{y:.1f}%<extra></extra>")
    fig.update_layout(
        **PLOTLY_LAYOUT, height=280,
        legend=dict(orientation="h", y=1.15, x=0, font=dict(size=11)),
        yaxis=dict(title="%", showgrid=True, gridcolor="rgba(15,23,42,.06)", zeroline=False),
        xaxis=dict(showgrid=False),
    )
    return fig


def render() -> None:
    engine = state.get_engine()
    summary = engine.model_summary()

    hero(
        "BDHS 2017-18 + 2022 · 5 outcomes · 3 phases",
        'Maternal &amp; child health risk,<br>'
        '<span class="grad">from survey inputs to a triage band.</span>',
        "Five outcomes — where she delivers, whether the baby survives, how the "
        "child grows — predicted from two pooled rounds of the Bangladesh "
        "Demographic and Health Survey, with the leakage, calibration, transport "
        "and equity audits that decide whether the numbers mean anything.",
    )

    st.markdown("")
    c1, c2, _ = st.columns([1.35, 1.2, 4], gap="small")
    with c1:
        nav.link("assess", "Start an assessment", icon=":material/monitor_heart:", width="content")
    with c2:
        nav.link("api", "Get an API key", icon=":material/key:", width="content")

    stat_tiles([("121,067", "records"), ("45,458", "mothers"),
                ("1,346", "survey clusters"), ("5", "outcomes · 7 learners")])

    # ------------------------------------------------------------ cascade
    section("The cascade",
            "Each outcome is a point at which an intervention exists, ordered by "
            "when it happens.")
    cols = st.columns(3, gap="medium")
    for col, phase in zip(cols, PHASE_ORDER):
        block = [s for s in summary if s["phase"] == phase]
        lines = []
        for s in block:
            auc = ROC_CI[s["target"]][0]
            kind = "pathway" if s["kind"] == "pathway" else "adverse"
            lines.append(
                f'<div style="display:flex;justify-content:space-between;gap:.6rem;'
                f'padding:.45rem 0;border-top:1px solid rgba(15,23,42,.06)">'
                f'<span><b>{s["label"]}</b> <span style="color:{MUTED};font-size:.78rem">'
                f'{kind}</span></span><span style="font-family:JetBrains Mono,monospace;'
                f'font-size:.8rem;color:{MUTED}">AUC {auc:.3f} · {s["prevalence"] * 100:.1f}%</span></div>')
        with col:
            card(PHASE_LABEL[phase], "".join(lines), kicker=f"phase {PHASE_ORDER.index(phase) + 1}",
                 body_is_html=True)

    # -------------------------------------------------------- performance
    section("Discrimination and prevalence",
            "Winners are selected on PR-AUC because four of the five outcomes are "
            "rare; ROC-AUC is shown because it is what readers compare.")
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.plotly_chart(_roc_chart(summary), config=PLOTLY_CONFIG)
    with right:
        with st.container(border=True):
            st.plotly_chart(_prevalence_chart(summary), config=PLOTLY_CONFIG)
    st.caption(
        "Measured on a grouped hold-out where no mother appears on both sides of "
        "the split. Thresholds are cost-optimal under a 30% capacity cap, not 0.5.")

    # ------------------------------------------------------- key finding
    section("The number that matters most")
    a, b = st.columns([1.1, 1], gap="medium")
    with a:
        callout(
            "Information content, not model capacity, sets the ceiling.",
            "Seven learners spanning linear, tree-based and neural families, a "
            "greedy ensemble over them, and a pretrained tabular transformer all "
            "land inside the same 0.07 ROC-AUC band on every target. What "
            "separates this work from published studies reporting near-perfect "
            "discrimination is not a better model. It is a leakage audit.")
    with b:
        rows = "".join(
            f'<div style="display:grid;grid-template-columns:1.4fr .6fr 2fr;gap:.6rem;'
            f'padding:.5rem 0;border-top:1px solid rgba(15,23,42,.06);font-size:.88rem">'
            f'<span>{name}</span><b style="font-family:JetBrains Mono,monospace">{auc:.3f}</b>'
            f'<span style="color:{MUTED}">{what}</span></div>'
            for name, auc, what in LEAKAGE_ROWS)
        md(f'<div class="mch-card"><div class="kicker">neonatal death · same model, same split</div>'
           f'<div style="display:grid;grid-template-columns:1.4fr .6fr 2fr;gap:.6rem;font-size:.72rem;'
           f'letter-spacing:.08em;text-transform:uppercase;color:{MUTED};font-weight:700">'
           f'<span>feature set</span><span>ROC-AUC</span><span>what it actually is</span></div>{rows}</div>')

    # ------------------------------------------------------------ pointers
    section("What this is not")
    n1, n2, n3, n4 = st.columns(4, gap="small")
    for col, (t, b_) in zip([n1, n2, n3, n4], [
        ("Not a diagnosis", "A predicted 0.31 for stunting means roughly a third of "
                            "women with this profile have a stunted child, not that this one will."),
        ("Not calibrated across settings", "Probabilities need recalibration on a "
                                           "current sample before they mean anything in a new survey round."),
        ("Not uniformly sensitive", "One global cut-point produces large, opposing "
                                    "subgroup gaps. Deployment needs stratified reporting."),
        ("Not clinical", "Neither round contains blood pressure, glucose or any "
                         "biomarker, so obstetric complications are not predicted."),
    ]):
        with col:
            card(t, b_)

    footer()
