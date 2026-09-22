"""Where the trained bundles come from.

The bundles are not in git — the largest is 133 MB, over GitHub's per-file
limit — so a deployment has to fetch them from somewhere on first start. The
lookup order is:

  1. a local directory that already holds every bundle
       MCH_MODELS_DIR, default <repo>/Maternal_Health/models/
  2. a Hugging Face Hub repository
       MCH_MODEL_REPO="<user>/<repo>"        (+ HF_TOKEN if it is private)
  3. a JSON map of direct download URLs
       MCH_MODEL_URLS='{"stunted.joblib": "https://...", ...}'

Every source lands the files in the same directory, so the engine never needs
to know which one was used. `WebApp/scripts/upload_models.py` publishes a local
models/ directory to the Hub for option 2.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

from core.paths import DEFAULT_MODELS_DIR

REQUIRED_BUNDLES = [
    "facility_delivery.joblib",
    "neonatal_death.joblib",
    "stunted.joblib",
    "severe_stunting.joblib",
    "underweight_child.joblib",
]
OPTIONAL_FILES = ["anomaly_bundle.joblib", "model_manifest.json"]

Progress = Optional[Callable[[str], None]]


class ModelsUnavailable(RuntimeError):
    """No bundle source is configured, or the configured one failed."""


def models_dir() -> Path:
    return Path(os.environ.get("MCH_MODELS_DIR") or DEFAULT_MODELS_DIR)


def missing_bundles(directory: Path) -> list[str]:
    return [f for f in REQUIRED_BUNDLES if not (directory / f).is_file()]


def is_complete(directory: Path) -> bool:
    return not missing_bundles(directory)


def _say(progress: Progress, msg: str) -> None:
    if progress:
        progress(msg)


def _from_hub(repo: str, directory: Path, progress: Progress) -> None:
    from huggingface_hub import snapshot_download

    _say(progress, f"Downloading bundles from Hugging Face Hub · {repo}")
    snapshot_download(
        repo_id=repo,
        repo_type=os.environ.get("MCH_MODEL_REPO_TYPE", "model"),
        allow_patterns=["*.joblib", "model_manifest.json"],
        local_dir=str(directory),
        token=os.environ.get("HF_TOKEN") or None,
    )


def _from_urls(url_map: dict[str, str], directory: Path,
               progress: Progress) -> None:
    import requests

    for name in REQUIRED_BUNDLES + OPTIONAL_FILES:
        url = url_map.get(name)
        if not url or (directory / name).is_file():
            continue
        _say(progress, f"Downloading {name}")
        tmp = directory / (name + ".part")
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(directory / name)


def ensure_models(progress: Progress = None) -> Path:
    """Return a directory holding every required bundle, fetching if needed."""
    directory = models_dir()
    if is_complete(directory):
        return directory

    directory.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get("MCH_MODEL_REPO", "").strip()
    urls = os.environ.get("MCH_MODEL_URLS", "").strip()

    if repo:
        _from_hub(repo, directory, progress)
    elif urls:
        try:
            url_map = json.loads(urls)
        except json.JSONDecodeError as exc:
            raise ModelsUnavailable(
                f"MCH_MODEL_URLS is not valid JSON: {exc}") from exc
        _from_urls(url_map, directory, progress)
    else:
        raise ModelsUnavailable(
            f"No model bundles in {directory} and no download source is "
            "configured. Either place the *.joblib files there, set "
            "MCH_MODEL_REPO to a Hugging Face repo id (see "
            "WebApp/scripts/upload_models.py), or set MCH_MODEL_URLS to a JSON map "
            "of bundle name -> URL. See WebApp/README.md.")

    still_missing = missing_bundles(directory)
    if still_missing:
        raise ModelsUnavailable(
            "Download finished but these bundles are still missing: "
            + ", ".join(still_missing))
    return directory


def source_description() -> str:
    """One line for the UI status panel."""
    if os.environ.get("MCH_MODEL_REPO"):
        return f"Hugging Face Hub · {os.environ['MCH_MODEL_REPO']}"
    if os.environ.get("MCH_MODEL_URLS"):
        return "direct URLs"
    return f"local · {models_dir()}"
