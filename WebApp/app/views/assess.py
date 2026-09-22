"""Risk Assessment: one subject, the full cascade, live."""
from __future__ import annotations

import json

import streamlit as st

from app import state
from app.ui.theme import (MUTED, band_banner, footer, md, outcome_card,
                          page_title, phase_header, pill)
from core.presets import BASE_TEMPLATE, PRESETS

# The form exposes the fields a health worker can realistically enter. Every
# other feature comes from the cohort-typical template, and the response
# says so. Widget types: slider (numeric), select (categorical), yesno.
FIELDS: list[dict] = [
    # Mother
    dict(code="v012", group="Mother", widget="slider", label="Age", min=15, max=49, step=1, fmt="%d yrs"),
    dict(code="v106", group="Mother", widget="select", label="Education"),
    dict(code="v155", group="Mother", widget="select", label="Literacy"),
    dict(code="v511", group="Mother", widget="slider", label="Age at first marriage", min=10, max=40, step=1, fmt="%d yrs"),
    dict(code="v212", group="Mother", widget="slider", label="Age at first birth", min=10, max=45, step=1, fmt="%d yrs"),
    dict(code="v201", group="Mother", widget="slider", label="Children ever born", min=0, max=12, step=1),
    dict(code="v714", group="Mother", widget="yesno", label="Currently working"),
    dict(code="v169a", group="Mother", widget="yesno", label="Owns a mobile phone"),
    # Body
    dict(code="height_cm", group="Body", widget="slider", label="Height", min=120.0, max=190.0, step=0.5, fmt="%.1f cm"),
    dict(code="weight_kg", group="Body", widget="slider", label="Weight", min=25.0, max=130.0, step=0.5, fmt="%.1f kg"),
    # Household
    dict(code="v190", group="Household", widget="select", label="Wealth quintile"),
    dict(code="v025", group="Household", widget="select", label="Residence"),
    dict(code="v024", group="Household", widget="select", label="Division"),
    dict(code="v130", group="Household", widget="select", label="Religion"),
    dict(code="v119", group="Household", widget="yesno", label="Electricity"),
    dict(code="v113", group="Household", widget="select", label="Drinking water"),
    dict(code="v116", group="Household", widget="select", label="Toilet facility"),
    dict(code="v161", group="Household", widget="select", label="Cooking fuel"),
    dict(code="v136", group="Household", widget="slider", label="Household members", min=1, max=25, step=1),
    dict(code="v701", group="Household", widget="select", label="Partner's education"),
    # Pregnancy & care
    dict(code="m14", group="Pregnancy & care", widget="slider", label="Antenatal visits", min=0, max=15, step=1),
    dict(code="m2a", group="Pregnancy & care", widget="yesno", label="Antenatal care from a doctor"),
    dict(code="m42c", group="Pregnancy & care", widget="yesno", label="Blood pressure taken"),
    dict(code="m45", group="Pregnancy & care", widget="yesno", label="Iron tablets / syrup"),
    dict(code="m1", group="Pregnancy & care", widget="slider", label="Tetanus injections", min=0, max=7, step=1),
    dict(code="p20", group="Pregnancy & care", widget="slider", label="Pregnancy duration", min=5, max=10, step=1, fmt="%d months"),
    dict(code="b11", group="Pregnancy & care", widget="slider", label="Preceding birth interval", min=9, max=240, step=1, fmt="%d months", nullable="First birth (no preceding interval)"),
    dict(code="risk_index", group="Pregnancy & care", widget="slider", label="High-risk fertility markers", min=0, max=6, step=1,
         help="Count of high-risk fertility behaviours: mother under 18 or over 34, birth order above 3, interval under 24 months."),
    dict(code="v467d", group="Pregnancy & care", widget="select", label="Distance to facility"),
    dict(code="v481", group="Pregnancy & care", widget="yesno", label="Health insurance"),
    # Child
    dict(code="b4", group="Child", widget="select", label="Child's sex"),
    dict(code="b19", group="Child", widget="slider", label="Child's age", min=0, max=59, step=1, fmt="%d months"),
    dict(code="b0", group="Child", widget="select", label="Multiple birth"),
]
GROUPS = ["Mother", "Body", "Household", "Pregnancy & care", "Child"]
GROUP_ICON = {"Mother": ":material/person:", "Body": ":material/straighten:",
              "Household": ":material/home:", "Pregnancy & care": ":material/pregnant_woman:",
              "Child": ":material/child_care:"}

EDU_ATTAIN = {"no education": "no education", "primary": "incomplete primary",
              "secondary": "incomplete secondary", "higher": "higher"}
EDU_YEARS = {"no education": 0, "primary": 4, "secondary": 9, "higher": 14}


def _key(code: str) -> str:
    return f"f_{code}"


def _coerce(field: dict, value):
    """Make a preset/template value legal for its widget."""
    if field["widget"] == "slider":
        if value is None:
            return None
        v = float(value)
        v = min(max(v, field["min"]), field["max"])
        return int(round(v)) if isinstance(field["min"], int) else float(v)
    if field["widget"] == "yesno":
        return "yes" if str(value).strip().lower() == "yes" else "no"
    return str(value).strip().lower() if value is not None else None


def _apply_record(record: dict) -> None:
    """Load a preset (or the template) into every widget."""
    for f in FIELDS:
        code = f["code"]
        raw = record.get(code, BASE_TEMPLATE.get(code))
        value = _coerce(f, raw)
        if f.get("nullable"):
            st.session_state[_key(code) + "_null"] = value is None
            if value is None:
                value = _coerce(f, BASE_TEMPLATE.get(code))
        st.session_state[_key(code)] = value


def _init_state() -> None:
    if "assess_ready" not in st.session_state:
        st.session_state["assess_preset"] = "mid_risk_semi_urban"
        _apply_record(PRESETS["mid_risk_semi_urban"]["record"])
        st.session_state["assess_ready"] = True


def _on_preset_change() -> None:
    choice = st.session_state.get("assess_preset")
    if choice and choice in PRESETS:
        _apply_record(PRESETS[choice]["record"])
    elif choice == "template":
        _apply_record({})


def _widget(field: dict, schema: dict) -> None:
    code = field["code"]
    k = _key(code)
    if field["widget"] == "slider":
        disabled = False
        if field.get("nullable"):
            disabled = st.checkbox(field["nullable"], key=k + "_null")
        st.slider(field["label"], min_value=field["min"], max_value=field["max"],
                  step=field["step"], format=field.get("fmt", "%d"), key=k,
                  help=field.get("help"), disabled=disabled)
    elif field["widget"] == "yesno":
        options = ["yes", "no"]
        if st.session_state.get(k) not in options:
            st.session_state[k] = "no"
        st.segmented_control(field["label"], options, key=k,
                             format_func=str.capitalize, width="stretch")
    else:
        options = schema.get(code, {}).get("allowed", [])
        if st.session_state.get(k) not in options and options:
            st.session_state[k] = options[0]
        st.selectbox(field["label"], options, key=k,
                     format_func=lambda v: str(v).capitalize())


def _collect() -> dict:
    record: dict = {}
    for f in FIELDS:
        code = f["code"]
        k = _key(code)
        if f.get("nullable") and st.session_state.get(k + "_null"):
            record[code] = None
            continue
        record[code] = st.session_state.get(k)
    # Consistency fields a health worker would never enter separately.
    record["v218"] = record["v201"]
    record["v139"] = record["v024"]
    record["v140"] = record["v025"]
    record["p4"] = record["b4"]
    record["p0"] = record["b0"]
    record["v149"] = EDU_ATTAIN.get(record["v106"], BASE_TEMPLATE["v149"])
    record["v133"] = EDU_YEARS.get(record["v106"], BASE_TEMPLATE["v133"])
    record["v729"] = EDU_ATTAIN.get(record["v701"], BASE_TEMPLATE["v729"])
    record["v715"] = EDU_YEARS.get(record["v701"], BASE_TEMPLATE["v715"])
    return record


def _curl_for(record: dict) -> str:
    base = state.api_base_url() or "https://<your-api-host>"
    body = json.dumps({"record": {k: v for k, v in record.items()
                                  if k in {f["code"] for f in FIELDS}}}, indent=2)
    return (f"curl -X POST {base}/v1/predict \\\n"
            f"  -H 'X-API-Key: <your key>' \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -d '{body}'")


def render() -> None:
    engine = state.get_engine()
    _init_state()

    page_title("Risk Assessment",
               "Where she delivers → whether the baby survives → how the child "
               "grows. Adjust any field; the cascade re-scores as you go.",
               eyebrow="single subject · live")

    preset_ids = list(PRESETS) + ["template"]
    st.pills("Start from", preset_ids, key="assess_preset",
             format_func=lambda k: PRESETS[k]["name"] if k in PRESETS else "Cohort template",
             on_change=_on_preset_change, selection_mode="single")
    chosen = st.session_state.get("assess_preset")
    if chosen in PRESETS:
        st.caption(PRESETS[chosen]["summary"] + " Synthetic profile, not a real record.")

    form_col, result_col = st.columns([1.05, 1.35], gap="large")

    with form_col:
        with st.container(border=True):
            tabs = st.tabs([f"{g}" for g in GROUPS])
            for tab, grp in zip(tabs, GROUPS):
                with tab:
                    fields = [f for f in FIELDS if f["group"] == grp]
                    if grp == "Body":
                        for f in fields:
                            _widget(f, engine.schema)
                        h = st.session_state[_key("height_cm")] / 100.0
                        bmi = st.session_state[_key("weight_kg")] / (h * h)
                        cat = ("underweight" if bmi < 18.5 else "normal" if bmi < 25
                               else "overweight" if bmi < 30 else "obese")
                        st.metric("BMI (derived)", f"{bmi:.1f}", cat, delta_color="off")
                        continue
                    left, right = st.columns(2, gap="small")
                    for i, f in enumerate(fields):
                        with (left if i % 2 == 0 else right):
                            _widget(f, engine.schema)
        st.caption(
            f"{len(FIELDS)} fields shown; the remaining "
            f"{len(engine.features) - len(FIELDS)} model inputs are filled from "
            "the cohort-typical template and listed in the report.")

    record = _collect()
    result = engine.assess(record)

    with result_col:
        band_banner(result["band"], result["n_flagged"], result["n_adverse"],
                    result["band_description"])
        for phase_block in result["phases"]:
            n = len(phase_block["outcomes"])
            phase_header(phase_block["label"],
                         f"{n} outcome{'s' if n > 1 else ''} · "
                         + ", ".join(o["label"].lower() for o in phase_block["outcomes"]))
            for o in phase_block["outcomes"]:
                outcome_card(o)

        if result["anomaly_percentile"] is not None:
            pct = result["anomaly_percentile"]
            md(f'<div style="font-size:.8rem;color:{MUTED};margin:.4rem 0 .8rem 0">'
               f'{pill("novelty check", "info")} This profile is more ordinary than '
               f'<b>{pct:.0f}%</b> of the reference cohort. Low figures are expected for '
               f'hand-built profiles: the template-filled fields sit at the median of '
               f'every variable at once, a combination that does not occur in the data. '
               f'The check earns its keep on real uploaded records in Batch Scoring.</div>')

        with st.expander("Inputs used in this assessment"):
            inp = result["inputs"]
            st.markdown(
                f"**{len(inp['provided'])}** fields provided · "
                f"**{len(inp['defaulted'])}** filled from the template"
                + (f" · {len(inp['warnings'])} warning(s)" if inp["warnings"] else ""))
            for w in inp["warnings"]:
                st.warning(w)
            st.code(", ".join(inp["defaulted"]), language=None, wrap_lines=True)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download report (JSON)",
                json.dumps({"record": record, "assessment": result}, indent=2),
                file_name="mch_assessment.json", mime="application/json",
                icon=":material/download:", width="stretch")
        with d2:
            with st.popover("Call this via the API", icon=":material/api:",
                            width="stretch"):
                st.caption("The same record, the same numbers, over HTTP.")
                st.code(_curl_for(record), language="bash", wrap_lines=True)

    footer()
