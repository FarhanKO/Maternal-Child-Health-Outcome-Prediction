# DHS Maternal & Child Health Data Engineering Pipeline

*A Lean Integration Framework for Demographic and Health Survey (DHS) Recode Files*

An optimized, non-redundant data engineering pipeline for transforming standard DHS recode files into a unified pregnancy-level analytical dataset suitable for statistical analysis, machine learning, epidemiological research, and public health reporting.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Output](#dataset-output)
- [DHS Data Compliance, Ethics & Citation](#dhs-data-compliance-ethics--citation)
- [DHS Survey Architecture](#dhs-survey-architecture)
- [Entity & Primary Keys](#entity--primary-keys)
- [Data Engineering Strategy](#data-engineering-strategy)
  - [1. Subset Table Elimination](#1-subset-table-elimination)
  - [2. Wide-Array Column Removal](#2-wide-array-column-removal)
  - [3. Incremental Feature Ingestion](#3-incremental-feature-ingestion)
- [Pipeline Architecture](#pipeline-architecture)
- [Entity Relationship Diagram](#entity-relationship-diagram)
- [Applications](#applications)
- [Reproducibility](#reproducibility)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [References](#references)

---

# Overview

The Demographic and Health Surveys (DHS) Program distributes survey data as multiple flattened recode files (IR, BR, KR, GR, HR, etc.), each representing different analytical units. Since these files share hundreds of duplicated variables, naïvely merging them often produces thousands of redundant columns, repeated information, and unnecessarily large datasets.

This repository implements a lean data engineering pipeline that constructs a normalized pregnancy-level analytical dataset by:

- Selecting the Pregnancy Recode (GR) as the primary analytical table
- Removing redundant subset recodes
- Eliminating wide repeated-member array variables
- Incrementally ingesting only unique variables from related recodes
- Preserving important maternal, pregnancy, household, and birth information while minimizing schema duplication

The resulting dataset is designed for:

- Machine Learning
- Statistical Modeling
- Epidemiological Research
- Maternal & Child Health Studies
- Public Health Reporting

---

# Dataset Output

Output format:

```
unified_dataset.parquet
```

or

```
unified_dataset.csv
```

## Unit of Analysis

Each row represents **one pregnancy event**.

Typical output characteristics (survey dependent):

- One pregnancy per row
- Pregnancy-level features
- Maternal demographic variables
- Household socioeconomic variables
- Birth-specific variables
- Optional mortality indicators

Because DHS surveys differ slightly across countries and survey phases, the exact number of variables depends on the source datasets being processed.

---

# DHS Data Compliance, Ethics & Citation

> **IMPORTANT:** This repository contains **data engineering code only**.

No DHS microdata (raw or processed) is stored, redistributed, or hosted in this repository.

## Terms of Use

1. DHS datasets must be obtained directly from The DHS Program under an approved research request.

2. Only registered collaborators approved under the DHS project should access the downloaded datasets.

3. Redistribution of DHS microdata, whether raw, modified, or embedded in another dataset, is prohibited under DHS Terms of Use.

4. Survey respondents are fully anonymized. No attempt should be made to identify any individual, household, or community.

5. Publications using DHS data should follow the official DHS citation guidelines and submit resulting reports to:

```
references@dhsprogram.com
```

---

# DHS Survey Architecture

| Dataset | DHS Recode | Entity | Primary Keys | Pipeline Treatment |
|----------|------------|------------|----------------|-------------------|
| pregnancy_questionnaire.csv | GR | Pregnancy | caseid + pidx | Primary Base Dataset |
| births_recode.csv | BR | Live Birth | caseid + bidx → pidx | Incremental Feature Ingestion |
| childrens_recode.csv | KR | Living Child (<5) | caseid + bidx | Converted into Binary Flag |
| pregnancy_postnatal.csv | NR | Recent Pregnancy | caseid + pidx | Converted into Binary Flag |
| individual_recode.csv | IR | Woman | caseid | Covariate Join |
| household_recode.csv | HR | Household | hhid / v001 + v002 | Covariate Join |
| verbal_autopsy.csv | VA | Mortality Event | caseid (or survey-specific identifiers) | Optional Left Join |

---

# Entity & Primary Keys

| Variable | Description |
|------------|----------------------------|
| caseid | Unique woman identifier |
| pidx | Pregnancy index |
| bidx | Birth index |
| hhid | Household identifier |
| v001 | Cluster number |
| v002 | Household number |

The pipeline aligns pregnancy and birth records using the appropriate DHS relationship between `bidx` and `pidx`, depending on the survey design.

---

# Data Engineering Strategy

```
RAW DHS RECODES

children_recode ──────────────┐
                              │
pregnancy_postnatal ──────────┤
                              ▼
                     Binary Flag Generator
                              │
                              ▼
                  is_in_children_recode
                  is_in_postnatal
                              │
                              ▼
               pregnancy_questionnaire (GR)
                    Primary Base Dataset
                              │
                              ▼
             + Birth-specific Unique Variables
                              │
                              ▼
          + Maternal Demographic Variables (IR)
                              │
                              ▼
        + Household Socioeconomic Variables (HR)
                              │
                              ▼
      + Optional Mortality Information (VA)
                              │
                              ▼
               Unified Analytical Dataset
```

---

## 1. Subset Table Elimination

The Children's Recode (KR) is a filtered subset of the Birth Recode (BR), containing only surviving children younger than five years.

Similarly, the Pregnancy Postnatal Recode (NR) is a filtered subset of the Pregnancy Recode (GR).

Rather than merging these highly redundant tables, the pipeline evaluates record membership and creates two informative indicators:

- `is_in_children_recode`
- `is_in_postnatal`

This approach substantially reduces duplicate variables while preserving the analytical information contained in these subsets.

---

## 2. Wide-Array Column Removal

The Individual Recode (IR) and Household Recode (HR) contain numerous repeated-member variables represented as wide arrays, for example:

```
hv101_01
hv101_02
...
hv101_25

bidx_01
bidx_02
...
bidx_20
```

These variables describe repeated household members or historical events and generate hundreds of sparse columns when merged.

Prior to joining, all array-style variables matching

```
^.*_\d+$
```

are removed.

Only core covariates such as:

- maternal age
- education
- marital status
- parity
- household wealth
- sanitation
- drinking water
- residence

are retained.

---

## 3. Incremental Feature Ingestion

The Birth Recode (BR) shares a large proportion of variables with the Pregnancy Recode (GR).

Instead of importing the entire BR dataset, the pipeline automatically identifies variables that are unique to BR and appends only those variables to the base dataset.

Examples include:

- birth weight
- neonatal care
- delivery complications
- place of delivery
- newborn health indicators

The exact number of appended variables depends on the DHS survey version and country.

---

# Pipeline Architecture

```
                    GR (Primary Dataset)
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
      BR Features      IR Covariates     HR Covariates
         │                 │                 │
         └─────────────────┼─────────────────┘
                           ▼
                  Unified Pregnancy Dataset
                           │
                           ▼
               Optional Verbal Autopsy Join
```

---

# Entity Relationship Diagram

```
                    pregnancy_questionnaire (GR)
                           │
          (1:1 on caseid + pidx)
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    births_recode     individual_recode   household_recode
        (BR)                (IR)               (HR)
          │
          ▼
   verbal_autopsy (Optional)
```

---

# Applications

The integrated dataset is suitable for:

- Maternal mortality prediction
- Neonatal outcome prediction
- Child health prediction
- Pregnancy risk assessment
- Feature engineering
- Public health reporting
- Epidemiological studies
- Machine learning research
- Statistical analysis

---

# Reproducibility

The pipeline is deterministic.

Given identical DHS recode files and configuration, repeated executions produce identical integrated datasets.

All processing steps are modular, documented, and reproducible.

---

# Directory Structure

```
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── integration/
│   ├── validation/
│   └── utils/
│
├── configs/
│
├── outputs/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Getting Started

1. Request access to the required DHS datasets from The DHS Program.

2. Download the approved recode files.

3. Place the datasets into:

```
data/raw/
```

4. Run the preprocessing pipeline.

5. Generate the unified analytical dataset.

---

# References

- The DHS Program. *Demographic and Health Surveys (DHS).* https://dhsprogram.com

- DHS Recode Manual

- DHS Guide to Statistics

- DHS Methodology Reports
