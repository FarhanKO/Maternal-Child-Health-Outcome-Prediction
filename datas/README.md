# Lean Data Engineering Pipeline for DHS Maternal & Child Health Surveys

An optimized, non-redundant data integration pipeline for processing standard Demographic and Health Surveys (DHS) recode files into a unified dataset for statistical reporting, machine learning, and epidemiological research.

---

## Table of Contents
- [Overview](#overview)
- [DHS Data Compliance, Ethics, & Citation](#dhs-data-compliance-ethics--citation)
- [DHS File Mapping & Survey Architecture](#dhs-file-mapping--survey-architecture)
- [Data Deduplication & Engineering Strategy](#data-deduplication--engineering-strategy)
  - [1. Sub-set Table Elimination & Flagging](#1-sub-set-table-elimination--flagging)
  - [2. Wide-Array Column Stripping](#2-wide-array-column-stripping)
  - [3. Incremental Feature Ingestion](#3-incremental-feature-ingestion)
- [Entity Relationship Architecture](#entity-relationship-architecture)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [DHS Official References & Code Repositories](#dhs-official-references--code-repositories)

---

## Overview

Processing DHS survey microdata presents structural challenges due to extensive variable duplication across flattened tables. For instance, merging Women's (`IR`), Births (`BR`), Children's (`KR`), and Pregnancy (`GR`/`NR`) recodes in their raw formats creates thousands of redundant columns (`bidx_01...bidx_20`, `m1_1...m1_6`) and duplicated rows.

This pipeline establishes a **lean data architecture** by selecting `pregnancy_questionnaire` (`GR`) as the primary base table, stripping array-indexed wide columns, and performing targeted incremental joins. The resulting dataset retains $100\%$ of unique feature information without bloated schema redundancies.

---

## DHS Data Compliance, Ethics, & Citation

> ⚠️ **IMPORTANT COMPLIANCE NOTICE**: This repository contains **data processing code only**. In accordance with the DHS Program Terms of Use, **no raw or processed microdata is stored, redistributed, or hosted in this repository**.

### Terms of Use & Data Confidentiality
1. **Authorized Access Only**: Microdata used in this project is obtained under an approved research grant from [The DHS Program](https://dhsprogram.com). Coresearchers must be explicitly registered on the project.
2. **Non-Redistribution**: Redistribution of any DHS micro-level data—either raw, modified, or embedded within a public application—is strictly prohibited.
3. **Confidentiality**: All survey respondents are anonymous. No attempt may be made to re-identify any individual respondent or household.
4. **Mandatory Reporting**: Users are required to submit an electronic PDF copy of any reports or peer-reviewed publications resulting from the use of DHS data to: **`references@dhsprogram.com`**.

---

## DHS File Mapping & Survey Architecture

The pipeline processes seven microdata sources corresponding to standard DHS Recode formats:

| Dataset File Name | Standard DHS Recode | Entity Unit | Primary Keys | Ingestion Treatment |
| :--- | :--- | :--- | :--- | :--- |
| `pregnancy_questionnaire.csv` | **GR** (Pregnancy Recode) | Pregnancy Event | `caseid`, `pidx` | **Primary Base Table** (73,239 rows) |
| `births_recode.csv` | **BR** (Births Recode) | Live Birth Event | `caseid`, `bidx` $\rightarrow$ `pidx` | **Incremental Feature Ingestion** (~352 unique cols) |
| `childrens_recode.csv` | **KR** (Kids Recode) | Living Child $< 5$ yrs | `caseid`, `bidx` | **Excluded**; converted to binary flag `is_in_children_recode` |
| `pregnancy_postnatal.csv` | **NR** (Postnatal Recode) | Pregnancy $< 5$ yrs | `caseid`, `pidx` | **Excluded**; converted to binary flag `is_in_postnatal` |
| `individual_recode.csv` | **IR** (Woman's Recode) | Individual Woman | `caseid` | **Covariate Join**; array suffixes (`_*`) stripped |
| `household_recode.csv` | **HR** (Household Recode) | Household Unit | `hhid` / (`v001`, `v002`) | **Covariate Join**; member arrays (`_*`) stripped |
| `verbal_autopsy.csv` | **VA** (Verbal Autopsy) | Mortality Event | `caseid` / Cluster keys | **Optional Left Join** for cause-of-death indicators |

---

## Data Deduplication & Engineering Strategy
