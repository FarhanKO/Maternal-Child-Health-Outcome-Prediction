# Models

**Not committed.** `train.py` writes them here. The largest bundle is ~130 MB —
over GitHub's 100 MB per-file limit — because a `GreedyEnsemble` pickles every
member pipeline it blends, and one of those members is a neural network.

```bash
python train.py          # writes everything below
```

## What lands here

```
models/
├── facility_delivery.joblib     one bundle per target
├── neonatal_death.joblib
├── stunted.joblib
├── severe_stunting.joblib
├── underweight_child.joblib
├── anomaly_bundle.joblib        IsolationForest + its imputer, scaler, reference scores
└── model_manifest.json          small enough to read without loading a bundle
```

## What a bundle contains

Each is self-sufficient: `app/app.py` loads one and scores a record with
nothing else from `train.py` in memory.

| Key | Why it is there |
|---|---|
| `raw_pipeline` | the fitted winner, preprocessing included |
| `calibrated` | the calibrator, or `None` where the guard rejected it |
| `threshold` | cost-optimal operating point under the 30% capacity cap |
| `features` | the exact column order the pipeline expects |
| `encoded_names` | post-encoding names, for attribution |
| `shap_background` | 200 transformed training rows, so SHAP needs no training data |
| `conformal` | the split-conformal quantile at 90% coverage |
| `importance` | permutation importance, measured on held-out data |
| `ensemble_members` | selection counts per candidate, zeros included |
| `prevalence` | the national rate the report compares against |

`ensemble_members` counts every candidate the greedy search *could* pick, so
most entries are zero. The number of models actually carrying weight is
`sum(1 for v in members.values() if v)` — 5, 3 and 4 for the three ensembles,
not 7.

## The anomaly bundle

An `IsolationForest` pickled on its own expects pre-transformed features and
carries no record of which columns those were or how they were scaled, so a new
record could not be prepared for it. The bundle pairs the detector with its
imputer, scaler, feature list and a reference score distribution — which is
what turns a raw isolation score into "more ordinary than 82% of the cohort".
