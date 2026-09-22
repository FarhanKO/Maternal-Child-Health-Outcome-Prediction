"""Shared scoring core used by both the Streamlit dashboard and the REST API.

Nothing in here imports Streamlit or FastAPI. The dashboard and the API are
two thin front-ends over the same `CascadeEngine`, which is the only way to
guarantee that a probability shown on screen and a probability returned over
HTTP for the same record are the same number.
"""
from core import paths  # noqa: F401  (puts Maternal_Health/ on sys.path)
from core.engine import (PHASE_LABEL, PHASE_ORDER, TARGET_LABEL,
                         CascadeEngine, band_for)
from core.model_store import ModelsUnavailable, ensure_models, models_dir

__all__ = ["CascadeEngine", "band_for", "PHASE_ORDER", "PHASE_LABEL",
           "TARGET_LABEL", "ensure_models", "models_dir", "ModelsUnavailable"]
