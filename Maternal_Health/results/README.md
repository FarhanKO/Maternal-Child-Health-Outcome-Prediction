# Results

Committed, because they are small and are what a reader wants before deciding
whether to run anything.

| File | Contents |
|---|---|
| `winners.csv` | the winning model per target, with its threshold and conformal quantile |
| `metrics.csv` | headline metrics per target, plus PR-AUC lift over prevalence |
| `feature_importance.csv` | top 20 permutation importances per target |

`train.py` also writes `threshold_curve_<target>.csv` and
`operating_point_<target>.csv` per run. Those are gitignored — they are
intermediate, one per target per run, and the summary tables above carry
everything worth reading.

## Reading `test_recall_at_0.5`

The key name is deliberate. That column is measured at the **default 0.5**
threshold, while `threshold` in the same row is the **cost-optimal** cut-point.
The two must not be read together: `facility_delivery` has recall 0.831 at 0.5
and 0.472 at its actual operating point of 0.794.
