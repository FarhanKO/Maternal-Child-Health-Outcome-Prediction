# Data

The BDHS microdata is **not committed**. The DHS Program licenses it to named
researchers and redistribution is not permitted, so this folder ships empty and
`train.py` expects you to populate it.

## What goes here

```
data/
├── target_registry.json          per-target features, blacklist, cohort, phase
└── cohorts/
    ├── cohort_births.parquet         112,546 rows — neonatal_death
    ├── cohort_measured_child.parquet  11,954 rows — the three nutrition targets
    ├── cohort_recent_birth.parquet    10,773 rows — facility_delivery
    └── cohort_mother.parquet          45,458 rows — features and EDA only
```

Each cohort is also written as `.csv` alongside the parquet. The code reads the
parquet, because CSV cannot round-trip a nullable integer — the nulls come back
as `NaN` and the column comes back as float.

## How to obtain it

1. Register at [dhsprogram.com](https://dhsprogram.com/data/) and request the
   Bangladesh **Individual Recode (IR)** and **Birth Recode (BR)** files for
   **BDHS 2017-18 (DHS-7)** and **BDHS 2022 (DHS-8)**. Access is free but
   requires a stated research purpose and is granted per project.
2. Run the preparation stage, which merges the two rounds through one identical
   cleaning pipeline and writes the cohorts and the registry.

## Why four cohorts and not one flat table

DHS data is a set of **nested universes**, not a rectangle:

```
v* variables   → one row per interviewed woman        (mother-level)
b* variables   → one row per birth she has had        (birth-level)
m* variables   → recent births only, last 3-5 years   (maternity module)
hw* variables  → children physically measured         (anthropometry)
```

Flattening them into one table manufactures missingness that is *structural*
rather than random: a woman with no recent birth has no `m14` because the
question was never asked of her, not because she declined to answer. Modelling
each target on its own universe keeps that distinction intact — which is why
none of the cohorts is complete-case filtered.

## Round harmonisation

The two rounds are pooled but never merged blindly. `survey_round` is carried
on every cohort as a **stratifier and metadata column, never a feature** — a
synthetic patient cannot meaningfully belong to a survey year. Two
harmonisations were required across rounds, and sampling weights (`v005/1e6`)
are renormalised within round so neither survey dominates a pooled prevalence.
