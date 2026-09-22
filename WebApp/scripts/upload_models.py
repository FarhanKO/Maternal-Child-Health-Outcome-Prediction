"""Publish the trained bundles to a Hugging Face Hub repository.

The bundles are too large for GitHub (neonatal_death.joblib is 133 MB), so a
deployment fetches them from the Hub at first start. Run this once from a
machine that has Maternal_Health/models/ populated by train.py:

    pip install huggingface_hub
    huggingface-cli login                      # or export HF_TOKEN=hf_...
    python WebApp/scripts/upload_models.py <hf-user>/<repo-name> [--private]

Then set MCH_MODEL_REPO=<hf-user>/<repo-name> wherever the app or API runs
(plus HF_TOKEN if the repo is private). See DEPLOYMENT.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "Maternal_Health" / "models"
FILES = ["facility_delivery.joblib", "neonatal_death.joblib", "stunted.joblib",
         "severe_stunting.joblib", "underweight_child.joblib",
         "anomaly_bundle.joblib", "model_manifest.json"]

CARD = """---
license: other
tags: [tabular-classification, maternal-health, child-health, dhs, bangladesh]
---
# Maternal & Child Health Outcome Prediction — trained bundles

Five self-contained scikit-learn bundles (facility delivery, neonatal death,
stunting, severe stunting, underweight) plus an anomaly detector, trained on
the Bangladesh DHS 2017-18 and 2022 rounds. Code, method and audits:
https://github.com/FarhanKO/Maternal-Child-Health-Outcome-Prediction

Load with the project's `core.engine.CascadeEngine`; the bundles pickle
`src.models.GreedyEnsemble` and pytabkit estimators, so that package must be
importable. These are population-level research models, not a diagnosis.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo", help="Hub repo id, e.g. FarhanKO/mch-risk-models")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=MODELS)
    args = parser.parse_args()

    missing = [f for f in FILES[:5] if not (args.models_dir / f).is_file()]
    if missing:
        print(f"missing bundles in {args.models_dir}: {', '.join(missing)}")
        return 1

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for name in FILES:
        path = args.models_dir / name
        if not path.is_file():
            continue
        print(f"uploading {name} ({path.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=args.repo, repo_type="model")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=args.repo, repo_type="model")
    print(f"\ndone -> https://huggingface.co/{args.repo}")
    print(f"set MCH_MODEL_REPO={args.repo}" + (" and HF_TOKEN" if args.private else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
