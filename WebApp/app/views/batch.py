"""Batch Scoring: a CSV in, every model over every row, a scored CSV out."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import state
from app.ui.theme import (BAND_COLOUR, INDIGO, MUTED, PLOTLY_CONFIG,
                          PLOTLY_LAYOUT, TEAL, footer, page_title, section)
from app.views.assess import FIELDS
from core.engine import TARGET_LABEL
from core.presets import BASE_TEMPLATE, PRESETS

TEMPLATE_COLUMNS = [f["code"] for f in FIELDS]


def _template_csv() -> bytes:
    rows = []
    for pid, p in PRESETS.items():
        row = {"subject_id": pid}
        row.update({c: p["record"].get(c, BASE_TEMPLATE.get(c)) for c in TEMPLATE_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _band_chart(scored: pd.DataFrame) -> go.Figure:
    counts = scored["band"].value_counts().reindex(["ROUTINE", "MONITOR", "PRIORITY"]).fillna(0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.62, sort=False,
        marker=dict(colors=[BAND_COLOUR[b] for b in counts.index],
                    line=dict(color="white", width=2)),
        textinfo="label+percent", textfont=dict(size=12),
        hovertemplate="%{label}: %{value} records<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False,
                      annotations=[dict(text=f"<b>{len(scored):,}</b><br><span style='font-size:11px;color:{MUTED}'>records</span>",
                                        x=0.5, y=0.5, showarrow=False, font=dict(size=20))])
    return fig


def _flag_chart(scored: pd.DataFrame, targets: list[str]) -> go.Figure:
    rates = [(TARGET_LABEL[t], scored[f"{t}_flag"].mean() * 100) for t in targets]
    fig = go.Figure(go.Bar(
        x=[r[1] for r in rates], y=[r[0] for r in rates], orientation="h",
        marker=dict(color=[TEAL if t == "facility_delivery" else INDIGO for t in targets],
                    cornerradius=8, opacity=0.9),
        text=[f"{r[1]:.1f}%" for r in rates], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% flagged<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False,
                      xaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(15,23,42,.06)",
                                 title="% of records above threshold", title_font=dict(size=11, color=MUTED)),
                      yaxis=dict(autorange="reversed", showgrid=False))
    return fig


def render() -> None:
    engine = state.get_engine()
    page_title("Batch Scoring",
               "Upload a CSV of records. Columns are matched by DHS code; anything "
               "missing is filled from the cohort template, and every model scores "
               "every row.", eyebrow="many subjects · CSV")

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        with st.container(border=True):
            up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed",
                                  help="One row per subject. Extra columns are passed "
                                       "through untouched.")
            st.caption("Column names are DHS recode variables (`v012`, `v190`, "
                       "`height_cm`…). See the schema on the API page for the "
                       "full list and accepted values.")
    with right:
        with st.container(border=True):
            st.markdown("**Need a starting point?**")
            st.caption("A template with the four illustrative subjects and the "
                       f"{len(TEMPLATE_COLUMNS)} most useful columns.")
            st.download_button("Download template CSV", _template_csv(),
                               "mch_template.csv", "text/csv",
                               icon=":material/download:", width="stretch")

    if up is None:
        section("How blanks are treated")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Column absent** → filled from the cohort-typical template, "
                    "reported in the summary.")
        c2.markdown("**Cell empty** → left missing. The models were trained on a "
                    "modular survey and handle missingness natively.")
        c3.markdown("**Value not in vocabulary** → treated as missing and listed "
                    "as a warning, so a typo cannot silently become a category.")
        footer()
        return

    try:
        frame = pd.read_csv(up)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read that CSV: {exc}")
        return
    if frame.empty:
        st.warning("The file has no rows.")
        return

    with st.spinner(f"Scoring {len(frame):,} rows…", show_time=True):
        scored, report = engine.score_frame(frame)
    out = pd.concat([frame.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)

    section("Results",
            f"{len(out):,} rows scored · {len(report['provided'])} columns "
            f"matched · {len(report['defaulted'])} filled from the template"
            + (f" · {len(report['ignored'])} ignored" if report["ignored"] else ""))
    for w in report["warnings"]:
        st.warning(w)

    a, b = st.columns(2, gap="medium")
    with a:
        with st.container(border=True):
            st.markdown("**Triage bands**")
            st.plotly_chart(_band_chart(scored), config=PLOTLY_CONFIG)
    with b:
        with st.container(border=True):
            st.markdown("**Flag rate per outcome**")
            st.plotly_chart(_flag_chart(scored, engine.targets), config=PLOTLY_CONFIG)

    if "anomaly_percentile" in scored:
        novel = int((scored["anomaly_percentile"] < 5).sum())
        st.caption(f"Novelty check: {novel:,} row(s) sit below the 5th percentile "
                   "of the reference cohort's feature distribution — feature "
                   "patterns the models were rarely, if ever, trained on.")

    risk_cols = [f"{t}_risk" for t in engine.targets]
    st.dataframe(
        out.head(500), hide_index=True, height=420,
        column_config={
            "band": st.column_config.TextColumn("band"),
            "n_flags": st.column_config.NumberColumn("flags", format="%d"),
            **{c: st.column_config.ProgressColumn(
                c.replace("_risk", ""), min_value=0, max_value=1, format="%.3f")
               for c in risk_cols},
            "anomaly_percentile": st.column_config.NumberColumn("ordinary %", format="%.0f"),
        })
    if len(out) > 500:
        st.caption(f"Showing the first 500 of {len(out):,} rows. The download has all of them.")

    st.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"),
                       "scored_records.csv", "text/csv", type="primary",
                       icon=":material/download:")

    with st.expander("Columns filled from the template"):
        st.code(", ".join(report["defaulted"]) or "none", language=None, wrap_lines=True)
    footer()
