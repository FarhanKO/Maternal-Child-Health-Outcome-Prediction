# Predicting Maternal and Child Health Outcomes from the Bangladesh DHS

Five maternal and child health outcomes — **where she delivers → whether the
baby survives → how the child grows** — predicted from two pooled rounds of the
Bangladesh Demographic and Health Survey, with the leakage, calibration,
transport and equity audits that decide whether the numbers mean anything.

**121,067 records · 45,458 mothers · 1,346 clusters · BDHS 2017-18 + 2022**

> **The finding is that information content, not model capacity, sets the
> ceiling.** Seven learners spanning linear, tree-based and neural families, a
> greedy ensemble over them, and a pretrained tabular transformer all land
> inside the same 0.07 ROC-AUC band on every target. What separates this work
> from published studies reporting near-perfect discrimination is not a better
> model. It is a leakage audit.

---

## Results

| Target | Phase | Cohort | Winner | ROC-AUC (95% CI) | PR-AUC | Prevalence | Threshold |
|---|---|---|---|---|---|---|---|
| `facility_delivery` | delivery | recent_birth | Ensemble of 5 | 0.845 [0.828, 0.859] | 0.878 | 57.2% | 0.794 |
| `neonatal_death` | newborn | births | Ensemble of 3 | 0.773 [0.756, 0.789] | 0.209 | 4.0% | 0.419 |
| `stunted` | child | measured_child | CatBoost | 0.707 [0.687, 0.730] | 0.526 | 28.7% | 0.357 |
| `severe_stunting` | child | measured_child | Ensemble of 4 | 0.700 [0.660, 0.738] | 0.192 | 7.9% | 0.326 |
| `underweight_child` | child | measured_child | TabM | 0.688 [0.661, 0.712] | 0.415 | 22.4% | 0.276 |

Measured on a grouped hold-out where **no mother appears on both sides of the
split**. Intervals are 400 bootstrap resamples. Thresholds are cost-optimal
under a 30% capacity cap, not 0.5. Winners are selected on PR-AUC, because four
of the five outcomes are rare and ROC-AUC is dominated by true negatives on a
4% outcome.

### The number that matters most

Neonatal death, same model, same split, three feature sets:

| Feature set | ROC-AUC | What it actually is |
|---|---|---|
| Leakage-audited | **0.773** | an early-risk estimate |
| Fertility identity retained | 0.967 | children ever born − living children **is** the death count |
| Survivor-only age retained | 0.987 | `b8` exists for 96.7% of survivors and 0% of deaths, so its presence is the label |

A published BDHS study reports 99.79% AUROC for under-five mortality. A
meta-analysis of eleven DHS studies puts realistic pooled accuracy for stunting
near 69%. The gap between those two facts is what this project is about, and
the table above is the most likely explanation for it.

---


**Not committed.** The BDHS microdata is licensed by The DHS Program and cannot
be redistributed, so `cohorts/` (185 MB), the raw CSV extracts (478 MB), and the
trained bundles (`models/`, largest single file 133 MB, over GitHub's 100 MB
limit) are all excluded. Everything needed to rebuild them from a DHS download
is here.

---

## The notebooks

Four stages, each writing artefacts the next reads from disk, so any one can be
rerun without the others.

| Notebook | Cells | What it establishes |
|---|---|---|
| **01 — Data preparation** | 36 | 2,134 raw variables → 899 retained; DHS labels decoded into genuine values, semantic zeros and nonresponse; cluster and mother IDs namespaced by round; four cohort views written to Parquet |
| **01b — Exploratory analysis** | 85 | Survey-weighted prevalence with Taylor-linearised intervals; ten published indicators independently recomputed per round and all within 4 percentage points; missingness classified as structural or random; digit heaping and implausible z-scores checked |
| **02 — Modelling** | 126 | Seven learners × five targets, the greedy ensemble, cost-optimal thresholds, the calibration guard, split conformal, decision curves, bootstrap intervals, temporal validation, subgroup audit, survival and multi-task extensions, unsupervised cross-checks |
| **03 — Model testing** | 32 | The five models composed into one phased cascade, exercised on held-out records and on synthetic profiles built inside the observed feature range |

`BDHS_maternal_child_health_complete.ipynb` is all four merged into a single
file for Colab — 285 cells, 108 code cells, every markdown cell and output
preserved byte-for-byte from the sources.

---

## Method

### Four cohorts, because DHS is nested universes

A DHS survey is not one table. Mother-level variables cover every record, birth
history covers live births, the maternity module covers only recent births, and
anthropometry only children who were physically measured. A column belonging to
an inner ring looks 93% missing — it was never asked, not left unanswered — so
any global missingness threshold deletes the inner rings first.

| Cohort | Rows | Carries |
|---|---|---|
| `births` | 112,546 | neonatal death |
| `mother` | 45,458 | equity, geography, mother-level features |
| `measured_child` | 11,954 | stunting, severe stunting, underweight |
| `recent_birth` | 10,773 | facility delivery |

Fifteen candidate outcomes were fitted and then reduced to five on measured
evidence. Fever (0.573), wasting (0.588) and diarrhoea (0.621) are two-week
recall windows for near-universal acute infection. Caesarean section predicts
well but records a clinician's decision rather than a patient's risk. Incomplete
immunisation looked strong at 0.823 until the child's age revealed that every
child under six months is "incomplete" by construction — restricted to the
12–23 month denominator DHS itself uses, it falls to 0.59.

### Five classes of leakage, each needing a different rule

| Class | Example |
|---|---|
| Post-outcome | breastfeeding and postnatal fields |
| Same-moment | fever, diarrhoea, ARI recorded at the same interview |
| Definitional | `haz` below −2 **is** stunting |
| Arithmetic identity | children ever born − living children |
| **Missingness-as-outcome** | `b8` recorded for survivors only |

The last is the one no inspection of *values* would ever catch, and it is
enforced in code rather than by convention — see
[`src/feature_engineering.py`](Maternity%20Health/src/feature_engineering.py)
and the assertions in [`test.py`](Maternity%20Health/test.py).

### Validation that goes past a single AUC

- **Grouped splitting.** 90% of birth rows belong to a mother who appears more
  than once. Under a random split, siblings sharing a mother, a household, a
  village and most of the feature vector land on both sides.
- **Survey design.** Mean design effect 2.19, so intervals from a naive
  independent-row analysis would be about 1.48× too narrow.
- **Calibration guard.** A calibrator is kept only if it improves Brier *and*
  calibration-in-the-large. Three of five were refused — on severe stunting one
  improved Brier from 0.0844 to 0.0753 by learning to predict near-zero for
  everyone, which scores well on a 7.5% outcome and is useless.
- **Conformal prediction.** 89.3–90.9% empirical coverage at a nominal 90%.
- **Temporal transport.** Train on 2017-18, test on 2022. Nutrition targets
  lose under 0.007 ROC-AUC, but predicted prevalence runs **1.44× to 5.90×**
  observed. Ranking transports; probabilities do not.
- **Equity.** At one global threshold, recall for stunting is 0.685 in the
  poorest wealth quintile and 0.121 in the richest — a 5.7-fold gap. For
  facility delivery the direction reverses, 0.119 to 0.812. Discrimination is
  far more stable than thresholded sensitivity.
- **Decision curves.** Four targets beat treat-all and treat-none across useful
  ranges. Severe stunting has *negative* net benefit at its operating point and
  should not be used as a standalone screen.

### A pretrained transformer changes nothing

TabPFN-3 was benchmarked separately under the identical split, preprocessing
and 10,000-row context cap: PR-AUC 0.872 / 0.166 / 0.516 / 0.185 / 0.417.
Behind the incumbent on three targets, ahead on two by 0.0005 and 0.0046 —
0.5% and 5.6% of the corresponding bootstrap interval widths, so both are ties.
A model family with an entirely different inductive bias reproducing the same
envelope is the clearest evidence that the ceiling is informational.

---

## Reproducing

```bash
git clone https://github.com/FarhanKO/Maternal-Child-Health-Outcome-Prediction.git
cd Maternal-Child-Health-Outcome-Prediction/Maternity_Health
pip install -r requirements.txt
python test.py                # 49 tests, no data needed, ~2 seconds
```

With DHS data in place:

```bash
python train.py                                    # every target, full panel
python train.py --targets stunted severe_stunting  # a subset
python train.py --tune                             # randomised search
streamlit run app/app.py                           # score a subject
```

To rebuild the cohorts from scratch, run `build_clean_dataset.py` and
`build_raw_merge_2017.py`, then `01_data_preparation.ipynb`.

Full package documentation: [Maternity_Health/README.md](Maternity_Health/README.md).

---

## Data access

BDHS microdata is free on registration at
[dhsprogram.com](https://dhsprogram.com/data/available-datasets.cfm) after a
short research-purpose statement is approved. **Redistribution is not
permitted**, so this repository ships code and derived artefacts only. The
recodes used are the Individual Recode (one record per ever-married woman
15–49) and the Birth Recode (one record per birth), for BDHS 2022 (DHS-8) and
BDHS 2017-18 (DHS-7).

The DHS Program collects informed consent and releases only de-identified
records. No GPS data was requested, and no participant can be re-identified
from the variables used here.

---

## What this is not

- **Not a diagnosis.** A predicted 0.31 for stunting means roughly a third of
  women with this profile have a stunted child, not that this one will.
- **Not calibrated across settings.** Probabilities need recalibration on a
  current sample before they mean anything in a new survey round.
- **Not uniformly sensitive.** One global cut-point produces large, opposing
  subgroup gaps. Deployment requires stratified reporting or stratified
  thresholds, and an explicit statement of which direction is intended.
- **Not causal.** Every record is retrospective; covariates describe the mother
  at interview, not at the time of an older pregnancy. All findings are
  associational.
- **Not clinical.** Neither round contains blood pressure, glucose or any
  biomarker, so pre-eclampsia, gestational diabetes, haemorrhage and sepsis
  have no ground truth and are not predicted.

---

## Paper

[Papers/Conference_paper.pdf](Papers/Conference_paper.pdf) — IEEE conference
format, 7 pages, 19 references. Covers the pooled corpus and per-round
indicator checks, the five-class leakage taxonomy with before-and-after
measurements, the seven-learner benchmark and post-hoc ensemble, and the
operational and equity analyses at explicit screening thresholds.

## Authors

**MD Farhan**
Department of Computer Science and Engineering, BRAC University, Dhaka

Data courtesy of The DHS Program and the National Institute of Population
Research and Training, Bangladesh.
