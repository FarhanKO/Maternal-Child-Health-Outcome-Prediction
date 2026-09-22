"""Human-readable names for the DHS recode variables the models consume.

Variable codes follow the DHS Recode Manual (Individual, Birth and Pregnancy
recodes). Derived columns (`bmi`, `height_cm`, `weight_kg`, `risk_index`) were
built in the data-preparation notebook. Anything not listed here falls back
to its raw code in the UI and the API schema.
"""

FEATURE_LABELS = {
    # --- mother: demographics -------------------------------------------
    "v012": "Mother's age (years)",
    "v013": "Mother's age group",
    "v024": "Division",
    "v025": "Residence type",
    "v026": "De facto place of residence",
    "v104": "Years lived in current place",
    "v106": "Mother's education level",
    "v107": "Highest year completed at that level",
    "v130": "Religion",
    "v133": "Mother's education (single years)",
    "v139": "De jure division",
    "v140": "De jure residence type",
    "v141": "De jure place of residence",
    "v149": "Mother's educational attainment",
    "v155": "Literacy",
    "v157": "Reads a newspaper",
    "v158": "Listens to the radio",
    "v159": "Watches television",
    "v169a": "Owns a mobile phone",
    "v170": "Has a bank account",
    "v171a": "Internet use",
    # --- household -------------------------------------------------------
    "v113": "Source of drinking water",
    "v116": "Type of toilet facility",
    "v119": "Household has electricity",
    "v120": "Household has a radio",
    "v121": "Household has a television",
    "v122": "Household has a refrigerator",
    "v123": "Household has a bicycle",
    "v124": "Household has a motorcycle/scooter",
    "v125": "Household has a car/truck",
    "v127": "Main floor material",
    "v128": "Main wall material",
    "v129": "Main roof material",
    "v136": "Household members",
    "v137": "Children under five in household",
    "v138": "Eligible women in household",
    "v161": "Cooking fuel",
    "v190": "Wealth quintile",
    "v191": "Wealth index score",
    # --- fertility & reproductive history --------------------------------
    "v201": "Children ever born",
    "v202": "Sons living at home",
    "v203": "Daughters living at home",
    "v212": "Age at first birth",
    "v213": "Currently pregnant",
    "v218": "Living children",
    "v219": "Living children incl. current pregnancy",
    "v220": "Living children incl. pregnancy (grouped)",
    "v228": "Ever had a terminated pregnancy",
    "v238": "Births in the last three years",
    "v312": "Current contraceptive method",
    "v313": "Contraceptive method type",
    "v364": "Contraceptive use and intention",
    "v384a": "Heard family planning on radio",
    "v384b": "Heard family planning on TV",
    "v384c": "Read family planning in a newspaper",
    "v467b": "Barrier to care: getting permission",
    "v467c": "Barrier to care: getting money",
    "v467d": "Barrier to care: distance to facility",
    "v467f": "Barrier to care: not wanting to go alone",
    "v481": "Covered by health insurance",
    "v501": "Marital status",
    "v502": "Union status",
    "v511": "Age at first marriage",
    "v512": "Years since first marriage",
    "v525": "Age at first sex",
    "v535": "Ever married",
    "v613": "Ideal number of children",
    "v623": "Fertility exposure status",
    "v624": "Unmet need for family planning",
    "v625a": "Fertility exposure (definition 3)",
    "v626a": "Unmet need (definition 3)",
    # --- partner & work ----------------------------------------------------
    "v701": "Partner's education level",
    "v702": "Partner's highest year at that level",
    "v704": "Partner's occupation",
    "v705": "Partner's occupation group",
    "v714": "Mother currently working",
    "v715": "Partner's education (single years)",
    "v716": "Mother's occupation",
    "v717": "Mother's occupation group",
    "v729": "Partner's educational attainment",
    "v730": "Partner's age",
    "v731": "Mother worked in the last 12 months",
    # --- autonomy ------------------------------------------------------------
    "v743a": "Who decides on mother's health care",
    "v743b": "Who decides on large purchases",
    "v743d": "Who decides on family visits",
    "v743f": "Who decides on husband's earnings",
    "v744a": "Beating justified: going out without telling",
    "v744b": "Beating justified: neglecting children",
    "v744c": "Beating justified: arguing",
    "v744d": "Beating justified: refusing sex",
    "v744e": "Beating justified: burning food",
    "v745a": "Owns a house",
    "v745b": "Owns land",
    # --- the birth / child ---------------------------------------------------
    "b0": "Multiple birth",
    "b4": "Child's sex",
    "b11": "Preceding birth interval (months)",
    "b19": "Child's age (months)",
    "p0": "Multiple pregnancy",
    "p4": "Sex of child (pregnancy record)",
    "p19": "Months since pregnancy ended",
    "p20": "Pregnancy duration (months)",
    # --- antenatal & delivery care -----------------------------------------
    "m1": "Tetanus injections during pregnancy",
    "m13": "Month of first antenatal visit",
    "m13a": "Week of first antenatal visit",
    "m14": "Antenatal visits",
    "m2a": "Antenatal care from a doctor",
    "m2b": "Antenatal care from a nurse/midwife",
    "m2g": "Antenatal care from a traditional attendant",
    "m2n": "No antenatal care",
    "m42a": "Weighed during pregnancy",
    "m42c": "Blood pressure taken during pregnancy",
    "m42d": "Urine sample taken during pregnancy",
    "m42e": "Blood sample taken during pregnancy",
    "m45": "Iron tablets/syrup during pregnancy",
    # --- anthropometry & derived -----------------------------------------
    "bmi": "Mother's BMI",
    "height_cm": "Mother's height (cm)",
    "weight_kg": "Mother's weight (kg)",
    "risk_index": "High-risk fertility behaviour index",
}

# Which block of the questionnaire a variable belongs to. Drives the grouping
# in the assessment form and the API schema.
FEATURE_GROUP = {}
for _grp, _codes in {
    "Mother": ["v012", "v013", "v106", "v107", "v133", "v149", "v155",
               "v157", "v158", "v159", "v169a", "v170", "v171a", "v130",
               "v104"],
    "Household & wealth": ["v024", "v025", "v026", "v139", "v140", "v141",
                           "v190", "v191", "v113", "v116", "v119", "v120",
                           "v121", "v122", "v123", "v124", "v125", "v127",
                           "v128", "v129", "v136", "v137", "v138", "v161"],
    "Reproductive history": ["v201", "v202", "v203", "v212", "v213", "v218",
                             "v219", "v220", "v228", "v238", "v511", "v512",
                             "v525", "v535", "v613", "v623", "v624", "v625a",
                             "v626a", "v312", "v313", "v364", "v384a",
                             "v384b", "v384c", "risk_index"],
    "Care access": ["m14", "m13", "m13a", "m1", "m2a", "m2b", "m2g", "m2n",
                    "m42a", "m42c", "m42d", "m42e", "m45", "v467b", "v467c",
                    "v467d", "v467f", "v481"],
    "Partner & autonomy": ["v701", "v702", "v704", "v705", "v714", "v715",
                           "v716", "v717", "v729", "v730", "v731", "v501",
                           "v502", "v743a", "v743b", "v743d", "v743f",
                           "v744a", "v744b", "v744c", "v744d", "v744e",
                           "v745a", "v745b"],
    "Anthropometry": ["bmi", "height_cm", "weight_kg"],
    "Child & birth": ["b0", "b4", "b11", "b19", "p0", "p4", "p19", "p20"],
}.items():
    for _c in _codes:
        FEATURE_GROUP[_c] = _grp

GROUP_ORDER = ["Mother", "Anthropometry", "Household & wealth",
               "Reproductive history", "Care access", "Child & birth",
               "Partner & autonomy"]


def label(code: str) -> str:
    return FEATURE_LABELS.get(code, code)


def group(code: str) -> str:
    return FEATURE_GROUP.get(code, "Other")
