"""Maternal & Child Health — risk screening dashboard.

Multipage Streamlit app over the shared scoring core. Run from the repository
root so the theme in .streamlit/config.toml is picked up:

    streamlit run WebApp/streamlit_app.py

Every model, threshold, feature list, importance score, conformal quantile
and preprocessing step is read from the bundles on disk through
core.engine.CascadeEngine — the same object the REST API serves.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = APP_DIR.parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))
import core  # noqa: E402,F401  (puts Maternal_Health/ on sys.path)

import streamlit as st  # noqa: E402


def _sidebar_status() -> None:
    from app import state
    from app.ui.theme import md, status_line

    with st.sidebar:
        st.markdown("---")
        try:
            eng = state.load_engine()
            status_line(f"{len(eng.targets)} models loaded · v{eng.version[:7]}", "ok")
        except Exception:  # noqa: BLE001 — surfaced on the page instead
            status_line("models not loaded", "warn")
        base = state.api_base_url()
        if base:
            status_line("REST API linked", "ok")
        else:
            status_line("API: in-process scoring (no remote service)", "off")
        md('<div style="font-size:0.74rem;color:#64748b;margin-top:0.8rem;line-height:1.5">'
           'Population-level estimates from socio-demographic inputs. '
           'Not a diagnosis and not a substitute for clinical assessment.</div>')


def main() -> None:
    st.set_page_config(
        page_title="Maternal & Child Health Risk",
        page_icon=str(APP_DIR / "assets" / "icon.svg"),
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get help": "https://github.com/FarhanKO/Maternal-Child-Health-Outcome-Prediction",
            "Report a bug": "https://github.com/FarhanKO/Maternal-Child-Health-Outcome-Prediction/issues",
            "About": "Five maternal and child health outcomes predicted from "
                     "the Bangladesh DHS 2017-18 and 2022 rounds.",
        },
    )

    from app import state
    from app.ui.theme import inject_theme
    from app.views import about, api, assess, batch, explain, home, performance

    state.secrets_to_env()
    inject_theme()
    st.logo(str(APP_DIR / "assets" / "logo.svg"),
            icon_image=str(APP_DIR / "assets" / "icon.svg"), size="large")

    pages = {
        "": [
            st.Page(home.render, title="Overview", icon=":material/home:",
                    url_path="home", default=True),
        ],
        "Screening": [
            st.Page(assess.render, title="Risk Assessment",
                    icon=":material/monitor_heart:", url_path="assess"),
            st.Page(batch.render, title="Batch Scoring",
                    icon=":material/table_view:", url_path="batch"),
        ],
        "Evidence": [
            st.Page(performance.render, title="Model Performance",
                    icon=":material/analytics:", url_path="performance"),
            st.Page(explain.render, title="Explainability",
                    icon=":material/psychology:", url_path="explain"),
        ],
        "Developers": [
            st.Page(api.render, title="API Access", icon=":material/api:",
                    url_path="api"),
        ],
        "Project": [
            st.Page(about.render, title="Methodology & Data",
                    icon=":material/menu_book:", url_path="about"),
        ],
    }
    from app import nav as navreg
    navreg.register(pages)
    nav = st.navigation(pages, position="sidebar", expanded=True)
    _sidebar_status()
    nav.run()


if __name__ == "__main__":
    main()
