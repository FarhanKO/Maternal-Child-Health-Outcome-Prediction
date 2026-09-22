"""Process-wide resources for the dashboard: the engine, the API secret, and
where the REST API lives.

Configuration is read from environment variables, which is what the API
reads too. Streamlit's `st.secrets` (the Community Cloud secrets panel or
.streamlit/secrets.toml) is copied into the environment once at startup so
both front-ends see one source of truth.
"""
from __future__ import annotations

import os
import streamlit as st

from core.apikeys import resolve_secret, secret_is_configured
from core.engine import CascadeEngine
from core.model_store import ModelsUnavailable, ensure_models, source_description

SECRET_KEYS = ("MCH_API_SECRET", "MCH_MODEL_REPO", "MCH_MODEL_REPO_TYPE",
               "MCH_MODEL_URLS", "MCH_MODELS_DIR", "MCH_API_BASE_URL",
               "MCH_API_REQUIRE_KEY", "MCH_RATE_LIMIT", "MCH_MAX_BATCH",
               "HF_TOKEN")

from core.paths import IMAGES_DIR, RESULTS_DIR  # noqa: E402,F401


def secrets_to_env() -> None:
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 — no secrets file is the common case
        return
    for key in SECRET_KEYS:
        try:
            if key in secrets and not os.environ.get(key):
                os.environ[key] = str(secrets[key])
        except Exception:  # noqa: BLE001
            continue


@st.cache_resource(show_spinner=False)
def load_engine() -> CascadeEngine:
    directory = ensure_models(progress=lambda m: None)
    return CascadeEngine(directory)


def get_engine() -> CascadeEngine:
    """The engine, or a friendly stop screen explaining what is missing."""
    try:
        with st.spinner("Loading the five model bundles…", show_time=True):
            return load_engine()
    except ModelsUnavailable as exc:
        st.error("**Model bundles are not available.**\n\n" + str(exc))
        st.markdown(
            "Set one of these in `.streamlit/secrets.toml` (or the "
            "environment) and restart:\n\n"
            "```toml\nMCH_MODEL_REPO = \"<hf-user>/<repo>\"   # Hugging Face Hub\n"
            "# or\nMCH_MODELS_DIR = \"/path/to/models\"\n```\n\n"
            "See **WebApp/README.md** for the one-command upload.")
        st.stop()
        raise  # unreachable; keeps type checkers calm


def api_secret() -> str:
    return resolve_secret()


def api_secret_configured() -> bool:
    return secret_is_configured()


def api_base_url() -> str | None:
    """Where the REST API is reachable *from the browser*.

    Explicit MCH_API_BASE_URL wins. Otherwise, if the app is served behind
    the bundled reverse proxy (deploy/nginx.conf), the API is the same origin
    under /api — the proxy sets MCH_API_PROXIED=1 so the app can say so.
    """
    explicit = os.environ.get("MCH_API_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    if os.environ.get("MCH_API_PROXIED") == "1":
        try:
            origin = st.context.url.rstrip("/")
            # strip any page path: keep scheme://host[:port]
            from urllib.parse import urlsplit
            parts = urlsplit(origin)
            return f"{parts.scheme}://{parts.netloc}/api"
        except Exception:  # noqa: BLE001
            return "/api"
    return None


def api_internal_url() -> str | None:
    """Where the dashboard *server* can reach the API for the live playground.
    Inside the bundled container that is localhost; otherwise the public URL."""
    internal = os.environ.get("MCH_API_INTERNAL_URL", "").strip().rstrip("/")
    return internal or api_base_url()


def model_source() -> str:
    return source_description()
