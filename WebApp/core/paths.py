"""Where things live, resolved from this file so the app runs from any CWD.

    <repo>/
    ├── WebApp/               this package: dashboard, API, scoring core
    └── Maternal_Health/      the research package: src/, results/, images/, models/

The bundles pickle `src.models.GreedyEnsemble`, so `Maternal_Health/` must be
importable before any of them is loaded. `ensure_research_on_path()` is
called from core/__init__.py, which every entry point imports first.
"""
from __future__ import annotations

import sys
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = WEBAPP_DIR.parent
RESEARCH_DIR = REPO_ROOT / "Maternal_Health"
IMAGES_DIR = RESEARCH_DIR / "images"
RESULTS_DIR = RESEARCH_DIR / "results"
DEFAULT_MODELS_DIR = RESEARCH_DIR / "models"


def ensure_research_on_path() -> None:
    for d in (WEBAPP_DIR, RESEARCH_DIR):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))


ensure_research_on_path()
