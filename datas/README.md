# DHS Maternal & Child Health Dataset

This directory contains the **data processing pipeline** used to prepare Demographic and Health Surveys (DHS) microdata for downstream statistical analysis, epidemiological research, and machine learning.

> **Important:** This repository **does not contain any DHS microdata**. Only the code required to process authorized DHS datasets is included.

---

# Dataset Source

The datasets used in this project are obtained from **The DHS Program** under an approved research request.

Official website:

https://dhsprogram.com

Because DHS data are protected by a data-use agreement, they **cannot be redistributed** through this repository. Every user must request and download the datasets using their own approved DHS account.

---

# Required DHS Recode Files

Place the following DHS recode files inside the `raw/` directory before running the preprocessing pipeline.

| File                              | DHS Recode | Description                        |
| --------------------------------- | ---------- | ---------------------------------- |
| `pregnancy_questionnaire.csv`     | GR         | Pregnancy Recode (Primary Dataset) |
| `births_recode.csv`               | BR         | Birth History Recode               |
| `childrens_recode.csv`            | KR         | Children Under Five Recode         |
| `pregnancy_postnatal.csv`         | NR         | Pregnancy Postnatal Recode         |
| `individual_recode.csv`           | IR         | Women's Individual Recode          |
| `household_recode.csv`            | HR         | Household Recode                   |
| `verbal_autopsy.csv` *(optional)* | VA         | Verbal Autopsy Recode              |

The pipeline has been designed using the standard DHS recode structure and may require minor adjustments for surveys with country-specific variables.

---

# Dataset Structure

Each DHS recode represents a different analytical unit.

| Recode | Unit of Analysis        |
| ------ | ----------------------- |
| GR     | Pregnancy               |
| BR     | Live Birth              |
| KR     | Living Child (<5 years) |
| NR     | Recent Pregnancy        |
| IR     | Woman                   |
| HR     | Household               |
| VA     | Mortality Event         |

Since these recodes overlap extensively, directly merging them produces hundreds of duplicated variables and many unnecessary columns. The preprocessing pipeline removes this redundancy while preserving the information required for analysis.

---

# Data Processing Strategy

The preprocessing pipeline follows a lean integration strategy to create a single pregnancy-level analytical dataset.

## 1. Primary Base Dataset

The **Pregnancy Recode (GR)** is used as the primary dataset because it provides the broadest pregnancy-level coverage and serves as the foundation for all subsequent feature integration.

Each row in the final dataset represents **one pregnancy event**.

---

## 2. Subset Recode Optimization

Two DHS recodes are subsets of larger datasets:

* **KR (Children's Recode)** is a subset of BR containing only living children younger than five years.
* **NR (Pregnancy Postnatal Recode)** is a subset of GR.

Instead of merging these highly redundant tables, the pipeline generates two binary indicators:

* `is_in_children_recode`
* `is_in_postnatal`

This preserves their analytical value without introducing duplicate variables.

---

## 3. Birth Feature Integration

The Birth Recode (BR) shares many variables with the Pregnancy Recode (GR).

Rather than importing every BR column, the pipeline automatically identifies variables that are unique to BR and appends only those variables to the base dataset.

Typical examples include:

* Birth weight
* Delivery information
* Neonatal care
* Newborn health indicators

The exact number of integrated variables depends on the DHS survey version.

---

## 4. Household and Maternal Covariates

The Individual Recode (IR) and Household Recode (HR) contain numerous repeated-member array variables such as:

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

These wide-array variables describe repeated household members or historical events and substantially increase dataset dimensionality.

Before merging, all columns matching the pattern

```
^.*_\d+$
```

are removed.

The pipeline retains only core covariates such as:

* Maternal age
* Education
* Marital status
* Household wealth
* Water source
* Sanitation
* Residence

These variables are then joined to the pregnancy-level dataset.

---

## 5. Optional Mortality Information

If available, the Verbal Autopsy (VA) dataset can be joined to provide additional mortality-related variables and cause-of-death information.

Because this dataset is considerably smaller than the other recodes, it is treated as an optional enrichment step rather than a required input.

---

# Processing Workflow

```
Raw DHS Recode Files
        │
        ▼
Pregnancy Recode (GR)
        │
        ├── Generate subset flags (KR, NR)
        │
        ├── Add unique birth features (BR)
        │
        ├── Join maternal variables (IR)
        │
        ├── Join household variables (HR)
        │
        └── Optional mortality enrichment (VA)
        │
        ▼
Unified Pregnancy-Level Dataset
```

---

# Output

The preprocessing pipeline produces a cleaned, integrated dataset ready for downstream analysis.

Typical output formats include:

```
processed/unified_dataset.parquet
```

or

```
processed/unified_dataset.csv
```

The final dataset is suitable for:

* Machine learning
* Statistical modeling
* Epidemiological research
* Maternal and child health studies
* Public health reporting

---

# Data Privacy & Licensing

This repository contains **processing code only**.

No raw or processed DHS microdata are distributed with this project for strict dhs policy.

Users are responsible for obtaining the required datasets directly from **The DHS Program** and must comply with all DHS Terms of Use, licensing requirements, and data confidentiality policies.
