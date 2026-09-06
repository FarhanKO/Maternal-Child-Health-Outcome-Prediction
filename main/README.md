# Maternity Health — Maternal & Child Health Risk Screening

Five outcomes across three phases of the maternal and child health cascade —
**where she delivers → whether the baby survives → how the child grows** —
predicted from two pooled rounds of the Bangladesh Demographic and Health
Survey (BDHS 2017-18 and BDHS 2022): **121,067 records from 45,458 mothers**.

Each target is trained and saved separately, so a service can deploy one
outcome without carrying the other four.

## Results

| Target | Phase | Cohort | Winning model | ROC-AUC | PR-AUC | Threshold |
|---|---|---|---|---|---|---|
| `facility_delivery` | delivery | recent_birth | Ensemble of 5 | 0.845 | 0.878 | 0.794 |
| `neonatal_death` | newborn | births | Ensemble of 3 | 0.773 | 0.208 | 0.419 |
| `stunted` | child | measured_child | CatBoost | 0.706 | 0.523 | 0.357 |
| `severe_stunting` | child | measured_child | Ensemble of 4 | 0.700 | 0.185 | 0.326 |
| `underweight_child` | child | measured_child | TabM | 0.687 | 0.412 | 0.276 |

Measured on a grouped hold-out where **no mother appears on both sides of the
split**. Thresholds are cost-optimal under a 30% capacity cap, not 0.5.

Full tables in [`results/`](results/); figures in [`images/`](images/).

## Structure

```
Maternity Health/
├── train.py                  entry point — trains and saves every target
├── requirements.txt
├── src/
│   ├── utils.py              paths, policy constants, artifact I/O
│   ├── feature_engineering.py  feature groups, ordered scales, leakage rules
│   ├── preprocessing.py      cohort loading, grouped splits, the two prep paths
│   ├── models.py             the model zoo, the ensemble, calibration, conformal
│   └── evaluation.py         metrics, operating points, decision curves, fairness
├── app/
│   ├── app.py                Streamlit risk-screening dashboard
│   └── assets/               sample subjects, feature template, styling
├── data/                     cohorts (not committed — see data/README.md)
├── models/                   trained bundles (not committed — see models/README.md)
├── results/                  metrics, winners, feature importances
└── images/                   EDA and model figures
```

## Quickstart

```bash
pip install -r requirements.txt
python train.py                       # every target, the full panel
streamlit run app/app.py              # score a subject against the saved bundles
```

Useful variations:

```bash
python train.py --targets stunted severe_stunting
python train.py --models CatBoost "Logistic Regression"   # a cheap subset
python train.py --tune                                     # randomised search
python train.py --data /path/to/cohorts --registry /path/to/target_registry.json
```

`train.py` is the only file that reads the dataset. Everything in `src/` takes
paths as parameters and hardcodes none, so the modules are reusable against a
different survey round without editing them.

## How the models are trained

Seven learners span the three families that matter on tabular data, chosen to
reflect what the 2025 benchmarks rank highly rather than what is traditional.
An eighth, TabPFN-3, is deliberately outside the zoo: it was benchmarked
separately under the identical split and preprocessing, and it did not earn a
place. It reached PR-AUC 0.872 / 0.166 / 0.516 / 0.185 / 0.417 across the five
targets — behind the incumbent on three, ahead on two by 0.0005 and 0.0046,
margins inside the bootstrap intervals. Folding it in would cost hours per run,
because a transformer pays its price at *inference* and the pipeline calls
inference repeatedly (calibration refits it three times per target; permutation
importance is ~100 features x 5 repeats of forward passes). `get_supervised_model_grids()`
omits it, and that is a considered choice rather than an omission.

| Family | Models |
|---|---|
| Linear | Logistic Regression |
| Bagged trees | Random Forest |
| Boosted trees | XGBoost, LightGBM, CatBoost |
| Deep tabular | RealMLP, TabM |
| Foundation | TabPFN *(optional — gated download)* |

Each is fitted on every target through its own pipeline, then a **greedy
forward-selection ensemble** (Caruana et al. 2004) blends them over 15 steps,
with weights chosen on a nested grouped split of the training data that no
member was fitted on. The winner per target is whichever scores highest on
PR-AUC — three of the five are ensembles, and the other two are models that did
not exist when this project started.

Five algorithms were **removed** from an earlier version of the panel — Naive
Bayes, k-NN, a lone Decision Tree, SVM and a small MLP. Each was run against
every target first and dropped on evidence: Naive Bayes reached ROC-AUC 0.502
on several targets, and SVM cost 20-49 minutes per fit while never winning one.

### Two preprocessing paths, not one

The dense path imputes, scales and one-hot encodes for the linear model and the
neural networks, which cannot accept a missing value at all. The native path
leaves numeric missingness untouched for the boosted trees, which learn their
own default split direction for NaN — and a blank `m14` in a modular survey
means *the maternity module was not asked*, which is informative in itself.

### Why the split is grouped

90% of rows in the births cohort belong to a mother who appears more than once.
Under a random split, siblings — sharing a mother, a household, a village, a
wealth quintile and most of the feature vector — would land on both sides, and
every reported metric would be optimistic.

### Leakage

Five distinct classes were found and blocked, each needing a different rule:
post-outcome, same-moment, definitional, arithmetic-identity, and
**missingness-as-outcome**. The last is the one no inspection of *values* would
ever have caught: `b8` is recorded for 96.7% of surviving children and 0% of
those who died, so its mere presence is the label. Including it lifted
`neonatal_death` to ROC-AUC 0.987; the fertility-accounting family lifted it to
0.967. Blocked, the real number is 0.773.

## What this is not

- **Not a diagnosis.** A probability of 0.31 for stunting means roughly a third
  of women with this profile have a stunted child, not that this one will.
- **Not calibrated across settings.** Discrimination transports across survey
  rounds; calibration does not. Predicted risk exceeded observed by 1.4x to
  5.9x when a 2017-18 model met 2022 data, so absolute probabilities must be
  recalibrated on arrival.
- **Not uniformly sensitive.** At one global threshold, recall for
  `facility_delivery` ranges from 0.075 in the poorest wealth quintile to 0.803
  in the richest — an artefact of a single cut-point on a steeply graded risk
  distribution, not a decision anyone made.
- **Not deployable in one case.** `severe_stunting` has *negative* net benefit
  at its operating point under decision curve analysis. It is reported for
  completeness and should not be used as a standalone screen.

## Data

BDHS microdata is licensed by The DHS Program and cannot be redistributed. See
[`data/README.md`](data/README.md) for how to request it and rebuild the
cohorts.
