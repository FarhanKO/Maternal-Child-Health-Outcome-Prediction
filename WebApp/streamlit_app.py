"""Streamlit Community Cloud / local entry point.

    streamlit run WebApp/streamlit_app.py        # from the repository root

Set the main file to WebApp/streamlit_app.py on Community Cloud; the
requirements.txt beside it is picked up automatically, packages.txt and
.streamlit/config.toml are read from the repository root.
"""
import sys
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from app.app import main  # noqa: E402

main()
