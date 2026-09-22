"""API Access: get a key, read the reference, try a request live."""
from __future__ import annotations

import json
import os
import time

import streamlit as st

from app import state
from app.ui.theme import (MUTED, callout, card, endpoint_row, footer, md,
                          page_title, pill, section, status_line)
from core.apikeys import issue_key, mask
from core.presets import PRESETS

PURPOSES = ["Research", "Programme planning", "Software integration",
            "Teaching", "Evaluation / audit", "Other"]

ENDPOINTS = [
    ("GET", "/health", "Liveness, model version, limits. No key."),
    ("GET", "/v1/models", "The five bundles: phase, kind, model, threshold, prevalence, AUCs. No key."),
    ("GET", "/v1/schema", "Every accepted field with type, label, group, default and vocabulary. No key."),
    ("GET", "/v1/presets", "The four illustrative subjects, ready to post. No key."),
    ("GET", "/v1/keys/verify", "Inspect the calling key: owner, issued, scope."),
    ("POST", "/v1/predict", "One record → the full cascade report."),
    ("POST", "/v1/predict/batch", "Up to 500 records → compact or full reports."),
]

EXAMPLE = {"v012": 17, "v106": "no education", "v190": "poorest",
           "v025": "rural", "v024": "sylhet", "height_cm": 143,
           "weight_kg": 35.0, "m14": 1, "risk_index": 3, "b19": 8, "p20": 8}


def _snippets(base: str, key: str) -> dict[str, str]:
    body = json.dumps({"record": EXAMPLE}, indent=2)
    return {
        "cURL": (f"curl -X POST {base}/v1/predict \\\n"
                 f"  -H 'X-API-Key: {key}' \\\n"
                 f"  -H 'Content-Type: application/json' \\\n"
                 f"  -d '{body}'"),
        "Python": (
            "import requests\n\n"
            f'API = "{base}"\n'
            f'KEY = "{key}"\n\n'
            f"record = {json.dumps(EXAMPLE, indent=4)}\n\n"
            'r = requests.post(f"{API}/v1/predict",\n'
            '                  headers={"X-API-Key": KEY},\n'
            '                  json={"record": record}, timeout=30)\n'
            "r.raise_for_status()\n"
            "report = r.json()\n\n"
            'print(report["band"], report["n_flagged"], "flagged")\n'
            'for o in report["outcomes"]:\n'
            '    print(f\'{o["label"]:18s} {o["probability"]:.3f}  '
            'threshold {o["threshold"]:.3f}  flagged={o["flagged"]}\')'),
        "JavaScript": (
            f'const API = "{base}";\n'
            f'const KEY = "{key}";\n\n'
            "const res = await fetch(`${API}/v1/predict`, {\n"
            '  method: "POST",\n'
            '  headers: { "X-API-Key": KEY, "Content-Type": "application/json" },\n'
            f"  body: JSON.stringify({{ record: {json.dumps(EXAMPLE)} }}),\n"
            "});\n"
            "const report = await res.json();\n"
            "console.log(report.band, report.outcomes.map(o => [o.target, o.probability]));"),
        "R": (
            "library(httr2)\n\n"
            f'api <- "{base}"\n'
            f'key <- "{key}"\n\n'
            'req <- request(paste0(api, "/v1/predict")) |>\n'
            '  req_headers(`X-API-Key` = key) |>\n'
            f"  req_body_json(list(record = {_r_list(EXAMPLE)}))\n\n"
            "report <- req |> req_perform() |> resp_body_json()\n"
            "report$band"),
    }


def _r_list(d: dict) -> str:
    parts = []
    for k, v in d.items():
        parts.append(f"{k} = {json.dumps(v) if isinstance(v, str) else v}")
    return "list(" + ", ".join(parts) + ")"


def _ping(url: str) -> dict | None:
    try:
        import requests
        r = requests.get(f"{url}/health", timeout=4)
        if r.ok:
            return r.json()
    except Exception:  # noqa: BLE001
        return None
    return None


def _call_api(url: str, key: str, body: dict) -> tuple[int, dict, float]:
    import requests
    t0 = time.perf_counter()
    r = requests.post(f"{url}/v1/predict", json=body,
                      headers={"X-API-Key": key}, timeout=60)
    ms = (time.perf_counter() - t0) * 1000
    try:
        payload = r.json()
    except ValueError:
        payload = {"raw": r.text[:2000]}
    return r.status_code, payload, ms


def render() -> None:
    engine = state.get_engine()
    base = state.api_base_url()
    internal = state.api_internal_url()
    health = _ping(internal) if internal else None
    display_base = base or "https://<your-api-host>"

    page_title("API Access",
               "The same cascade the dashboard runs, over HTTPS. Generate a key, "
               "send a JSON record, get back probabilities, thresholds, "
               "prediction sets and a triage band.", eyebrow="developers · REST")

    # ------------------------------------------------------------- status
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        with st.container(border=True):
            st.markdown("**Endpoint**")
            if base:
                st.code(base, language=None)
                status_line("reachable" if health else "not responding", "ok" if health else "warn")
            else:
                st.code("not linked", language=None)
                status_line("MCH_API_BASE_URL not set", "off")
    with s2:
        with st.container(border=True):
            st.markdown("**Models**")
            st.code(f"version {engine.version}", language=None)
            status_line(f"{len(engine.targets)} bundles · {len(engine.features)} fields", "ok")
    with s3:
        with st.container(border=True):
            st.markdown("**Access**")
            auth = (health or {}).get("auth_required", True)
            rate = (health or {}).get("rate_limit_per_minute", os.environ.get("MCH_RATE_LIMIT", 60))
            st.code("API key required" if auth else "open (no key needed)", language=None)
            status_line(f"{rate} requests / minute per key", "ok")

    if not base:
        callout("The REST service is not linked to this dashboard yet.",
                "Keys generated here are still valid — they are signed with the "
                "shared MCH_API_SECRET, not stored anywhere — but the live "
                "playground below will score in-process instead of over HTTP. "
                "Deploy WebApp/api (see WebApp/README.md) and set MCH_API_BASE_URL to "
                "connect it.")
    if not state.api_secret_configured():
        st.warning("**MCH_API_SECRET is not configured**, so this process is "
                   "using a random secret. Keys issued now will stop validating "
                   "when the app restarts, and a separately deployed API will "
                   "not recognise them. Set the same secret on both services.",
                   icon=":material/key_off:")

    # ------------------------------------------------------------ get key
    section("Get an API key",
            "Self-serve. The key is signed, not stored, so copy it now — it "
            "cannot be shown again. Anyone with a key can call the prediction "
            "endpoints within the rate limit.")
    k1, k2 = st.columns([1, 1.3], gap="large")
    with k1:
        with st.container(border=True):
            with st.form("issue_key", border=False):
                owner = st.text_input("Name or email", placeholder="ada@example.org",
                                      help="Embedded in the key so you can tell your keys apart. Not stored.")
                org = st.text_input("Organisation (optional)", placeholder="Ministry of Health · NGO · University")
                purpose = st.selectbox("Intended use", PURPOSES)
                ttl = st.select_slider("Validity", options=[30, 90, 180, 365, 0],
                                       value=180,
                                       format_func=lambda d: "no expiry" if d == 0 else f"{d} days")
                agreed = st.checkbox(
                    "I understand these are population-level estimates for "
                    "research and planning — not a diagnosis, and not for "
                    "individual clinical decisions.")
                go = st.form_submit_button("Generate key", type="primary",
                                           icon=":material/key:", width="stretch")
            if go:
                if not agreed:
                    st.error("Please confirm the intended-use statement first.")
                else:
                    label = (owner.strip() or "anonymous")
                    if org.strip():
                        label = f"{label} ({org.strip()[:24]})"
                    key, info = issue_key(state.api_secret(), label,
                                          scope=purpose.split()[0].lower(),
                                          ttl_days=ttl or None)
                    st.session_state["issued_key"] = key
                    st.session_state["issued_info"] = info.as_dict()
    with k2:
        key = st.session_state.get("issued_key")
        info = st.session_state.get("issued_info")
        if key and info:
            with st.container(border=True):
                st.markdown(f"**Your key** {pill('shown once', 'info')}", unsafe_allow_html=True)
                st.code(key, language=None, wrap_lines=True)
                exp = (time.strftime("%Y-%m-%d", time.gmtime(info["expires_at"]))
                       if info["expires_at"] else "never")
                md(f'<div style="font-size:.82rem;color:{MUTED};line-height:1.8">'
                   f'key id <code>{info["key_id"]}</code> · owner <b>{info["owner"]}</b> · '
                   f'scope <b>{info["scope"]}</b> · expires <b>{exp}</b></div>')
                st.caption("Send it as `X-API-Key: <key>` or `Authorization: Bearer <key>`. "
                           "To revoke, add the key id to `MCH_REVOKED_KEYS` on the API.")
        else:
            card("How keys work",
                 "A key is an HMAC-signed token carrying its owner label, scope and "
                 "expiry. The API verifies the signature with the shared secret — "
                 "no database, no lookup, no key ever stored server-side. "
                 "Rotate every key at once by changing the secret; revoke one by "
                 "listing its id.", kicker="stateless by design")

    # ----------------------------------------------------------- quickstart
    section("Quickstart", "Any subset of the schema's fields is a valid record. "
                          "Omitted fields are filled from the cohort template and "
                          "reported back.")
    snippets = _snippets(display_base, st.session_state.get("issued_key") or "<your key>")
    tabs = st.tabs(list(snippets))
    for tab, (lang, code) in zip(tabs, snippets.items()):
        with tab:
            st.code(code, language={"cURL": "bash", "Python": "python",
                                    "JavaScript": "javascript", "R": "r"}[lang],
                    wrap_lines=True)

    # ------------------------------------------------------------ reference
    section("Endpoints")
    r1, r2 = st.columns([1.5, 1], gap="large")
    with r1:
        md('<div class="mch-card">' + "".join(endpoint_row(*e) for e in ENDPOINTS) + "</div>")
        if base:
            st.markdown("")
            st.link_button("Open the interactive OpenAPI docs", f"{base}/docs",
                           icon=":material/open_in_new:")
    with r2:
        card("Response anatomy",
             "<b>band</b> ROUTINE · MONITOR · PRIORITY, counted over the four adverse "
             "outcomes.<br><b>outcomes[]</b> per target: <code>probability</code>, "
             "<code>threshold</code> (cost-optimal, capacity-capped), "
             "<code>national_rate</code>, <code>flagged</code>, <code>lift</code>, "
             "<code>prediction_set</code> (conformal, 90% coverage) and "
             "<code>committed</code>.<br><b>inputs</b> which fields were provided, "
             "defaulted, ignored, and any vocabulary warnings.<br><b>meta</b> model "
             "version and latency.", kicker="POST /v1/predict", body_is_html=True)
        st.markdown("")
        card("Limits & fair use",
             "60 requests per minute per key · 500 records per batch call · "
             "responses carry <code>X-RateLimit-*</code> headers · HTTP 429 with "
             "<code>Retry-After</code> when exceeded. Aggregate use for research and "
             "planning is welcome; automated individual-level decision making is not "
             "what these models are for.", kicker="policy", body_is_html=True)

    # ------------------------------------------------------------ playground
    section("Try it live", "Edit the JSON and send. With the REST service linked "
                           "this goes over HTTP with your key; otherwise the request "
                           "is scored in-process so the response shape is identical.")
    p1, p2 = st.columns([1, 1.3], gap="large")
    with p1:
        preset = st.selectbox("Load a preset", list(PRESETS),
                              format_func=lambda k: PRESETS[k]["name"], key="api_preset")
        default_body = json.dumps({"record": PRESETS[preset]["record"],
                                   "include_anomaly": True}, indent=2)
        if st.session_state.get("api_body_preset") != preset:
            st.session_state["api_body"] = default_body
            st.session_state["api_body_preset"] = preset
        body_text = st.text_area("Request body", key="api_body", height=340)
        send = st.button("Send request", type="primary", icon=":material/send:",
                         width="stretch")
    with p2:
        if send:
            try:
                body = json.loads(body_text)
            except json.JSONDecodeError as exc:
                st.error(f"Request body is not valid JSON: {exc}")
                body = None
            if body is not None:
                key = st.session_state.get("issued_key")
                if internal and health:
                    if not key and health.get("auth_required", True):
                        key, _ = issue_key(state.api_secret(), "playground", ttl_days=1)
                    code, payload, ms = _call_api(internal, key or "", body)
                    md(f'{pill(f"HTTP {code}", "good" if code == 200 else "flag")} '
                       f'{pill(f"{ms:.0f} ms round trip", "neutral")} '
                       f'{pill("over HTTP", "info")} <span style="font-size:.8rem;color:{MUTED}">'
                       f'key {mask(key) if key else "none"}</span>')
                else:
                    t0 = time.perf_counter()
                    try:
                        payload = engine.assess(
                            body.get("record", {}), fill=body.get("fill", "template"),
                            include_anomaly=body.get("include_anomaly", True),
                            include_record=body.get("include_record", False))
                        code = 200
                    except Exception as exc:  # noqa: BLE001
                        payload, code = {"error": str(exc), "status": 400}, 400
                    ms = (time.perf_counter() - t0) * 1000
                    md(f'{pill(f"HTTP {code}", "good" if code == 200 else "flag")} '
                       f'{pill(f"{ms:.0f} ms", "neutral")} {pill("scored in-process", "info")}')
                st.json(payload, expanded=2)
        else:
            st.info("The response will appear here.", icon=":material/terminal:")
    footer()
