"""Feature groups, ordered DHS scales, and the five leakage blacklists.

The registry written by the data-preparation stage already resolves a feature
list per target. This module holds the rules that produced it, so the same
audit can be re-applied to a new survey round without going back to a notebook.
"""

# DHS scales that carry a real order. One-hot encoding throws that order away
# and makes the model rediscover it from data; ordinal encoding hands it over.
# These lists are the ORDER, not just the levels -- reversing one would
# silently invert every gradient the model learns from that column.
_FREQ = ["not at all", "less than once a week", "at least once a week"]
_EDUC = ["no education", "primary", "secondary", "higher"]

ORDERED_LEVELS = {
    "v190": ["poorest", "poorer", "middle", "richer", "richest"],
    "v106": _EDUC,                      # respondent's education
    "v701": _EDUC,                      # partner's education
    "v013": ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"],
    "v155": ["cannot read at all",
             "able to read only parts of sentence",
             "able to read whole sentence"],
    "v157": _FREQ,                      # newspaper
    "v158": _FREQ,                      # radio
    "v159": _FREQ,                      # television
    "v467d": ["not a big problem", "big problem"],
}

# Feature blocks, in the order a health worker would encounter them.
FEATURE_GROUPS = {
    "demographic":      ["v012", "v013", "v025", "v024", "v106", "v155"],
    "socioeconomic":    ["v190", "v191", "v701", "v705", "v714", "v745a"],
    "household":        ["v113", "v116", "v119", "v120", "v121", "v122",
                         "v123", "v124", "v125", "v127", "v128", "v129",
                         "v136", "v161"],
    "media":            ["v157", "v158", "v159"],
    "fertility":        ["v212", "v511", "v525", "v512", "b11"],
    "anthropometry":    ["bmi", "height_cm", "weight_kg"],
    "care_access":      ["v467d", "m14", "m15"],
    "child_demographic": ["b19", "b4", "b0"],
    "risk":             ["risk_index"],
}

# --------------------------------------------------------------------------
# The five classes of leakage found in this data. Each needed a different rule,
# and the fifth is the one no inspection of *values* would ever have caught.
# --------------------------------------------------------------------------
LEAKAGE_CLASSES = {
    "post_outcome": (
        "A dead newborn has no breastfeeding record and no postnatal care "
        "visit, so the mere presence of m70-m76 is close to a survival "
        "indicator."),
    "same_moment": (
        "Fever, diarrhoea and ARI are recorded at the same interview, so "
        "using one to predict another is co-measurement, not prediction."),
    "definitional": (
        "`stunted` IS `haz < -2`. Leaving haz in the feature set lets a model "
        "rediscover the definition and report near-perfect accuracy."),
    "arithmetic_identity": (
        "Children ever born minus children living IS her number of child "
        "deaths. With v201 and v220 left in, XGBoost reached ROC-AUC 0.967 on "
        "neonatal_death and those two columns carried ~78% of all permutation "
        "importance. Blocked, the same model reports 0.742."),
    "missingness_as_outcome": (
        "b8 is recorded for 96.7% of surviving children and 0% of those who "
        "died. Its mere presence is the label. Including it lifted "
        "neonatal_death to ROC-AUC 0.987."),
}

# Blocked for every target, whatever the registry says.
ALWAYS_BLOCK = [
    # survey bookkeeping, not measurements
    "caseid", "mother_id", "survey_round", "weight", "v001", "v005",
    "v021", "v022", "v023", "v008", "v011", "v016", "v017", "v006", "v007",
    # p17 runs 1-31 with mean 14.36 -- it is a day of the month, and it
    # correlates with nothing by construction.
    "p17",
    # missingness-as-outcome: present for survivors, absent for the dead
    "b8",
]

# Fertility accounting: ever-born minus living IS the child-death count.
FERTILITY_ACCOUNTING = [
    "v201", "v202", "v203", "v204", "v205", "v206", "v207",
    "v210", "v218", "v219", "v220", "v224", "v238",
]

# Anthropometric z-scores define the nutrition targets.
Z_SCORES = ["haz", "waz", "whz", "hw70", "hw71", "hw72", "hw73"]


def blacklist_for(target, registry=None):
    """Everything blocked for one target: registry entry unioned with the
    always-blocked list. The registry is authoritative when present, because
    it was written by the audited preparation stage."""
    blocked = set(ALWAYS_BLOCK)
    if registry and target in registry:
        blocked |= set(registry[target].get("blacklist", []))
    if target in ("stunted", "severe_stunting", "underweight_child", "wasted"):
        blocked |= set(Z_SCORES)
    if target == "neonatal_death":
        blocked |= set(FERTILITY_ACCOUNTING)
        blocked.add("b12")   # interval to the NEXT birth is post-outcome
    return sorted(blocked)


def features_for(target, registry, available_columns):
    """Resolve the usable feature list for one target.

    Drops anything blacklisted, anything absent from the cohort, and anything
    constant -- a column with one value cannot inform a split and only widens
    the one-hot matrix.
    """
    blocked = set(blacklist_for(target, registry))
    feats = [f for f in registry[target]["features"]
             if f not in blocked and f in available_columns]
    return feats
