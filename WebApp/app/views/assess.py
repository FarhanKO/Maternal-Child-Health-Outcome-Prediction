"""Risk Assessment: one subject, the full cascade, live."""
from __future__ import annotations

import json

import streamlit as st

from app import state
from app.ui.theme import (MUTED, band_banner, footer, md, outcome_card,
                          page_title, phase_header, pill)
from core.labels import GROUP_ORDER
from core.presets import BASE_TEMPLATE, PRESETS

# The form exposes the fields a health worker can realistically enter. Every
# other feature comes from the cohort-typical template, and the response
# says so. Widget types: slider (numeric), select (categorical), yesno.
CORE_FIELDS: list[dict] = [
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
USE_TYPICAL = "— use typical value —"
CORE_GROUPS = ["Mother", "Body", "Household", "Pregnancy & care", "Child"]
# Everything the models accept, beyond the core form, is offered too — see
# `optional_fields`. These are the ones NOT offered, because the engine
# computes them from answers already given and letting a user set both
# invites a contradiction the model never saw in training (a wealth index
# score that disagrees with its own quintile, an age band that disagrees
# with the age). core/defaults.py owns them.
ENGINE_DERIVED = {
    "bmi",          # height and weight
    "v013",         # age band from age
    "v149", "v133", "v107",          # her education scales from the level
    "v729", "v715", "v702",          # his, likewise
    "v139", "v140", "v141",          # de jure geography mirrors de facto
    "p4", "p0", "p19",               # the pregnancy record mirrors the birth
    "v191",         # the wealth score v190 is binned from
    "v219", "v220",                  # living children incl. current pregnancy
}

@st.cache_data(show_spinner=False)
def optional_fields(version: str) -> list[dict]:
    """Every remaining model input, as an optional widget.

    Built from the engine's schema rather than a hand-written list, so a
    retrained bundle that introduces a feature offers it here automatically.
    Ordered by the questionnaire block, then by how much any model actually
    relies on the field, so the ones worth answering surface first.

    `version` is the engine fingerprint, present only to key the cache.
    """
    engine = state.get_engine()
    core = {f["code"] for f in CORE_FIELDS}
    weight: dict[str, float] = {}
    for target in engine.targets:
        for code, value in engine.importance(target, top=999):
            weight[code] = max(weight.get(code, 0.0), value)

    fields = []
    for code, spec in engine.schema.items():
        if code in core or code in ENGINE_DERIVED:
            continue
        field = dict(code=code, group=spec["group"], label=spec["label"],
                     importance=weight.get(code, 0.0))
        if spec["type"] == "numeric":
            field.update(widget="opt_number",
                         min=spec.get("min"), max=spec.get("max"),
                         step=spec.get("step", 1),
                         typical=spec.get("median", spec.get("default")))
        else:
            field.update(widget="opt_select")
        fields.append(field)

    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    fields.sort(key=lambda f: (order.get(f["group"], 99), -f["importance"],
                               f["label"]))
    return fields


def all_fields() -> list[dict]:
    return CORE_FIELDS + optional_fields(state.get_engine().version)


def _key(code: str) -> str:
    return f"f_{code}"


def _is_set(field: dict) -> bool:
    """Has the user given this optional field a value?"""
    value = st.session_state.get(_key(field["code"]))
    if field["widget"] == "opt_number":
        return value is not None
    return value not in (None, USE_TYPICAL)


def _clear_optional() -> None:
    for f in optional_fields(state.get_engine().version):
        st.session_state[_key(f["code"])] = (
            None if f["widget"] == "opt_number" else USE_TYPICAL)


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
    for f in all_fields():
        code = f["code"]
        if f["widget"].startswith("opt_"):
            # Optional fields start unset whatever the preset says, so the
            # engine resolves them rather than a preset asserting a value.
            # The slider still needs a number behind its disabled state.
            st.session_state[_key(code)] = (
                None if f["widget"] == "opt_number" else USE_TYPICAL)
            continue
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
    elif field["widget"] == "opt_select":
        options = [USE_TYPICAL] + list(schema.get(code, {}).get("allowed", []))
        if st.session_state.get(k) not in options:
            st.session_state[k] = USE_TYPICAL
        st.selectbox(field["label"], options, key=k,
                     format_func=lambda v: v if v == USE_TYPICAL else str(v).capitalize())
    elif field["widget"] == "opt_number":
        typical = field.get("typical")
        st.number_input(
            field["label"], key=k, min_value=field.get("min"),
            max_value=field.get("max"), step=field.get("step") or 1,
            placeholder=(f"typical {typical:g}" if isinstance(typical, (int, float))
                         else "use typical value"),
            help="Leave blank and this is resolved from your other answers.")
    else:
        options = schema.get(code, {}).get("allowed", [])
        if st.session_state.get(k) not in options and options:
            st.session_state[k] = options[0]
        st.selectbox(field["label"], options, key=k,
                     format_func=lambda v: str(v).capitalize())


def _collect() -> dict:
    record: dict = {}
    for f in all_fields():
        code = f["code"]
        k = _key(code)
        if f.get("nullable") and st.session_state.get(k + "_null"):
            record[code] = None
            continue
        if f["widget"] == "opt_select":
            value = st.session_state.get(k)
            if value and value != USE_TYPICAL:
                record[code] = value
            continue                     # left out entirely => engine resolves it
        if f["widget"] == "opt_number":
            value = st.session_state.get(k)
            if value is not None:
                record[code] = value
            continue                     # blank => engine resolves it
        record[code] = st.session_state.get(k)
    # Everything else — the mirrored recodes (v218 from v201, p4 from b4),
    # the education scales, and the fields no form can ask — is resolved by
    # core.defaults, so the API fills a partial record exactly the same way.
    return record


def _curl_for(record: dict) -> str:
    base = state.api_base_url() or "https://<your-api-host>"
    body = json.dumps({"record": record}, indent=2)
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

    optional = optional_fields(engine.version)
    n_answered = sum(1 for f in optional if _is_set(f))

    with form_col:
        with st.container(border=True):
            tabs = st.tabs(list(CORE_GROUPS))
            for tab, grp in zip(tabs, CORE_GROUPS):
                with tab:
                    fields = [f for f in CORE_FIELDS if f["group"] == grp]
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

        st.markdown("")
        with st.container(border=True):
            head, reset = st.columns([3, 1], vertical_alignment="center")
            head.markdown(
                f"**More detail** &nbsp; {pill(f'{n_answered} of {len(optional)} answered', 'good' if n_answered else 'neutral')}",
                unsafe_allow_html=True)
            if reset.button("Clear all", width="stretch", disabled=not n_answered,
                            help="Return every optional field to its typical value."):
                _clear_optional()
                st.rerun()
            st.caption(
                "Every remaining model input, ordered by how much the models "
                "lean on it. Anything you leave blank is resolved from the "
                "answers above — an identity where one exists, otherwise a "
                "cohort-typical value — so answer as few or as many as you have. "
                "On held-out survey records, answering none of these leaves the "
                "estimate about **5 points** from what the complete record would "
                "give; answering all of them closes that to about **2**.")
            for grp in GROUP_ORDER:
                fields = [f for f in optional if f["group"] == grp]
                if not fields:
                    continue
                answered = sum(1 for f in fields if _is_set(f))
                with st.expander(
                        f"{grp}  ·  {answered}/{len(fields)} answered",
                        icon=":material/check_circle:" if answered else None):
                    left, right = st.columns(2, gap="small")
                    for i, f in enumerate(fields):
                        with (left if i % 2 == 0 else right):
                            _widget(f, engine.schema)

        st.caption(
            f"{len(CORE_FIELDS)} fields asked directly, {len(optional)} more "
            f"available above, {len(engine.features)} inputs in total. The "
            "report below lists how every field you left blank was resolved.")

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
            strat = inp.get("fill_strategy", {})
            derived = strat.get("derived", [])
            cell = strat.get("from_wealth_residence_cell", [])
            flat = strat.get("from_flat_template", [])
            st.markdown(
                f"**{len(inp['provided'])}** provided · **{len(derived)}** "
                f"derived from them · **{len(cell) + len(flat)}** "
                "cohort-typical"
                + (f" · {len(inp['warnings'])} warning(s)" if inp["warnings"] else ""))
            for w in inp["warnings"]:
                st.warning(w)
            if derived:
                st.markdown("**Computed from what you entered** — identities, "
                            "not guesses.")
                st.code(", ".join(derived), language=None, wrap_lines=True)
            if cell:
                st.markdown("**From your wealth quintile and residence** — the "
                            "wealth index score `v190` is binned from.")
                st.code(", ".join(cell), language=None, wrap_lines=True)
            if flat:
                st.markdown("**Cohort-typical value** — not asked, and not "
                            "implied by anything you entered. Set any of these "
                            "under **More detail** to override.")
                st.code(", ".join(flat), language=None, wrap_lines=True)

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
