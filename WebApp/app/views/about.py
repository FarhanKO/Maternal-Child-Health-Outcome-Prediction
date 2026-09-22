"""Methodology & Data: how the numbers were made and what they are not."""
from __future__ import annotations

import streamlit as st

from app.ui.theme import callout, card, footer, page_title, section

REPO = "https://github.com/FarhanKO/Maternal-Child-Health-Outcome-Prediction"


def render() -> None:
    page_title("Methodology & Data",
               "Two pooled rounds of the Bangladesh DHS, four nested cohorts, seven "
               "learners, and the audits that decide whether any of it means "
               "anything.", eyebrow="project")

    section("Four cohorts, because DHS is nested universes")
    st.markdown(
        "A DHS survey is not one table. Mother-level variables cover every record, "
        "birth history covers live births, the maternity module covers only recent "
        "births, and anthropometry only children who were physically measured. A "
        "column belonging to an inner ring looks 93% missing — it was never asked, "
        "not left unanswered — so any global missingness threshold deletes the "
        "inner rings first.")
    c = st.columns(4, gap="small")
    for col, (name, rows, carries) in zip(c, [
        ("births", "112,546", "neonatal death"),
        ("mother", "45,458", "equity, geography, mother-level features"),
        ("measured_child", "11,954", "stunting, severe stunting, underweight"),
        ("recent_birth", "10,773", "facility delivery"),
    ]):
        with col:
            card(name, f"<b>{rows}</b> rows<br>carries {carries}", kicker="cohort",
                 body_is_html=True)

    section("Validation that goes past a single AUC")
    a, b = st.columns(2, gap="medium")
    with a:
        st.markdown("""
- **Grouped splitting.** 90% of birth rows belong to a mother who appears more than
  once. `GroupShuffleSplit` on `mother_id` puts an entire family on one side.
- **Survey design.** Mean design effect 2.19, so intervals from a naive
  independent-row analysis would be about 1.48× too narrow.
- **Calibration guard.** A calibrator is kept only if it improves Brier *and*
  calibration-in-the-large. Three of five were refused — one had learned to
  predict near-zero for everyone, which scores well on a 7.5% outcome and is useless.
- **Conformal prediction.** 89.3–90.9% empirical coverage at a nominal 90%.
""")
    with b:
        st.markdown("""
- **Temporal transport.** Train on 2017-18, test on 2022. Nutrition targets lose
  under 0.007 ROC-AUC, but predicted prevalence runs **1.44× to 5.90×** observed.
  Ranking transports; probabilities do not.
- **Equity.** At one global threshold, recall for stunting is 0.685 in the poorest
  wealth quintile and 0.121 in the richest — a 5.7-fold gap. For facility delivery
  the direction reverses.
- **Decision curves.** Four targets beat treat-all and treat-none across useful
  ranges. Severe stunting has *negative* net benefit at its operating point.
- **A pretrained transformer.** TabPFN-3 under the identical split reproduces the
  same envelope — the clearest evidence that the ceiling is informational.
""")

    section("Why thresholds are not 0.5")
    callout("Cost-optimal under a 30% capacity cap.",
            "Each threshold minimises expected cost, where one missed case is worth "
            "several false alarms (ten, for neonatal death), subject to flagging no "
            "more than 30% of the population. Cost alone drives common outcomes to "
            "absurd operating points: stunting at 28.7% prevalence was flagging 60% "
            "of children, which no service can act on.")

    section("What this is not")
    n = st.columns(3, gap="small")
    for col, (t, body) in zip(n * 2, [
        ("Not a diagnosis", "A predicted 0.31 for stunting means roughly a third of "
                            "women with this profile have a stunted child, not that this one will."),
        ("Not calibrated across settings", "Probabilities need recalibration on a "
                                           "current sample before they mean anything in a new survey round."),
        ("Not uniformly sensitive", "One global cut-point produces large, opposing "
                                    "subgroup gaps. Deployment requires stratified reporting or thresholds."),
        ("Not causal", "Every record is retrospective; covariates describe the mother "
                       "at interview, not at the time of an older pregnancy."),
        ("Not clinical", "Neither round contains blood pressure, glucose or any "
                         "biomarker, so pre-eclampsia, gestational diabetes, haemorrhage "
                         "and sepsis have no ground truth and are not predicted."),
        ("Not a substitute", "Built to prioritise limited contact time, not to replace "
                             "the judgement of the person who has it."),
    ]):
        with col:
            card(t, body)
            st.markdown("")

    section("Data access and ethics")
    st.markdown(f"""
BDHS microdata is free on registration at
[dhsprogram.com](https://dhsprogram.com/data/available-datasets.cfm) after a short
research-purpose statement is approved. **Redistribution is not permitted**, so the
repository ships code and derived artefacts only — the cohorts, raw extracts and
trained bundles are excluded and rebuilt from a DHS download. The DHS Program
collects informed consent and releases only de-identified records. No GPS data
was requested, and no participant can be re-identified from the variables used.

The four sample subjects in this dashboard are synthetic profiles. The
cohort-typical template holds the marginal median of each variable, not any
respondent's row.

**Repository:** [{REPO.replace('https://', '')}]({REPO}) ·
**Paper:** IEEE conference format, 7 pages, in `Papers/`.
""")

    section("Authors & references")
    st.markdown("""
**MD Farhan** — Department of Computer Science and Engineering, BRAC University, Dhaka.

1. Naznin, S., Uddin, M. J., & Kabir, A. (2025). Identifying determinants of under-5
   mortality in Bangladesh: A machine learning approach with BDHS 2022 data.
   *PLOS ONE, 20*(6), e0324825. https://doi.org/10.1371/journal.pone.0324825
2. Rao, B. et al. (2025). Machine learning in predicting child malnutrition: A
   meta-analysis of Demographic and Health Surveys data. *IJERPH, 22*(3), 449.
   https://doi.org/10.3390/ijerph22030449
3. Hollmann, N. et al. (2025). Accurate predictions on small data with a tabular
   foundation model. *Nature, 637*, 319–326. https://doi.org/10.1038/s41586-024-08328-6

Data courtesy of The DHS Program and the National Institute of Population Research
and Training, Bangladesh.
""")
    footer()
