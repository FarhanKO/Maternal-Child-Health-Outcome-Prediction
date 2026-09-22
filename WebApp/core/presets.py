"""The cohort-typical template and the four illustrative subjects.

BASE_TEMPLATE holds one value per feature any bundle needs: the median for a
numeric column and the mode for a categorical one, taken from the cohort that
actually contains it. A partial record — the twenty-odd fields a health worker
can realistically enter — is completed from it before scoring, and the engine
reports exactly which fields were defaulted.

The template is NOT a real record. It holds the marginal median of each
variable independently, so the combination sits in a low-density region of
the joint distribution and the anomaly detector scores it as unusual. Using a
real survey row instead would publish one respondent's full profile, which
the DHS licence does not permit.

The four presets are synthetic profiles used to exercise the cascade end to
end. They are illustrative, not real people.
"""

BASE_TEMPLATE = {
    'b0': 'single birth', 'b11': 57.0, 'b19': 28.0, 'b4': 'male',
    'bmi': 21.91, 'height_cm': 151.3, 'weight_kg': 50.1,
    'm1': 0.0, 'm13': 4.0, 'm13a': 17.38, 'm14': 3.0,
    'm2a': 'yes', 'm2b': 'no', 'm2g': 'no', 'm2n': 'no: some care',
    'm42a': 'yes', 'm42c': 'yes', 'm42d': 'yes', 'm42e': 'yes', 'm45': 'yes',
    'p0': 'single birth', 'p19': 17.0, 'p20': 9.0, 'p4': 'male',
    'risk_index': 0.0,
    'v012': 25.0, 'v013': '20-24', 'v024': 'chattogram', 'v025': 'rural',
    'v026': 'countryside', 'v104': 6.0, 'v106': 'secondary', 'v107': 3.0,
    'v113': 'tube well or borehole', 'v116': 'pit latrine with slab',
    'v119': 'yes', 'v120': 'no', 'v121': 'no', 'v122': 'no', 'v123': 'no',
    'v124': 'no', 'v125': 'no', 'v127': 'earth/sand', 'v128': 'cement',
    'v129': 'metal', 'v130': 'islam', 'v133': 8.0, 'v136': 5.0, 'v137': 1.0,
    'v138': 1.0, 'v139': 'dhaka', 'v140': 'rural', 'v141': 'countryside',
    'v149': 'incomplete secondary', 'v155': 'able to read whole sentence',
    'v157': 'not at all', 'v158': 'not at all',
    'v159': 'at least once a week', 'v161': 'wood', 'v169a': 'yes',
    'v170': 'no', 'v171a': 'never', 'v190': 'poorest', 'v191': -31590.0,
    'v201': 2.0, 'v202': 1.0, 'v203': 1.0, 'v212': 18.0,
    'v213': 'no or unsure', 'v218': 2.0, 'v219': 2.0, 'v220': 2.0,
    'v228': 'no', 'v238': 1.0, 'v312': 'pill', 'v313': 'modern method',
    'v364': 'using modern method', 'v384a': 'no', 'v384b': 'no',
    'v384c': 'no', 'v467b': 'not a big problem', 'v467c': 'not a big problem',
    'v467d': 'not a big problem', 'v467f': 'not a big problem', 'v481': 'no',
    'v501': 'married', 'v502': 'currently in union/living with a man',
    'v511': 16.0, 'v512': 7.0, 'v525': 17.0, 'v535': 'formerly married',
    'v613': 2.0, 'v623': 'fecund', 'v624': 'using for limiting',
    'v625a': 'fecund', 'v626a': 'using for limiting', 'v701': 'secondary',
    'v702': 3.0,
    'v704': 'carpenter, masson, bus/taxi driver, construction supervisor, seamstresses/tailor',
    'v705': 'skilled manual', 'v714': 'no', 'v715': 6.0,
    'v716': "not working and didn't work in last 12 months",
    'v717': 'not working', 'v729': 'incomplete secondary', 'v730': 32.0,
    'v731': 'no', 'v743a': 'respondent and husband/partner',
    'v743b': 'respondent and husband/partner',
    'v743d': 'respondent and husband/partner',
    'v743f': 'respondent and husband/partner', 'v744a': 'no', 'v744b': 'no',
    'v744c': 'no', 'v744d': 'no', 'v744e': 'no', 'v745a': 'does not own',
    'v745b': 'does not own',
}

PRESETS = {
    "low_risk_urban": {
        "name": "Low risk · urban, educated",
        "summary": "27-year-old graduate in Dhaka, richest quintile, eight "
                   "antenatal visits, first child.",
        "record": {
            'v012': 27, 'v201': 1, 'v218': 1, 'v106': 'higher',
            'v149': 'higher', 'v133': 14, 'v190': 'richest', 'v025': 'urban',
            'v024': 'dhaka', 'v139': 'dhaka', 'v140': 'urban',
            'v026': 'capital, large city', 'bmi': 23.5, 'height_cm': 158,
            'weight_kg': 58.7, 'm14': 8, 'm2a': 'yes', 'v511': 24,
            'v212': 26, 'b11': 48, 'risk_index': 0, 'v119': 'yes',
            'v121': 'yes', 'v122': 'yes', 'v170': 'yes',
            'v171a': 'yes, last 12 months', 'v701': 'higher',
            'v113': 'piped into dwelling', 'v116': 'flush to septic tank',
            'v127': 'ceramic tiles', 'v128': 'cement', 'v129': 'cement',
            'v161': 'natural gas', 'v481': 'no', 'b19': 30, 'p20': 9,
        },
    },
    "adolescent_poorest": {
        "name": "Adolescent · poorest, undernourished",
        "summary": "17-year-old in rural Sylhet with no schooling, BMI 17, "
                   "one antenatal visit, three high-risk fertility markers.",
        "record": {
            'v012': 17, 'v013': '15-19', 'v201': 1, 'v218': 1,
            'v106': 'no education', 'v149': 'no education', 'v133': 0,
            'v155': 'cannot read at all', 'v190': 'poorest', 'v025': 'rural',
            'v024': 'sylhet', 'v139': 'sylhet', 'bmi': 17.1,
            'height_cm': 143, 'weight_kg': 35.0, 'm14': 1, 'm2a': 'no',
            'm2b': 'no', 'm42c': 'no', 'm42d': 'no', 'm42e': 'no',
            'v511': 15, 'v212': 16, 'b11': None, 'risk_index': 3,
            'v119': 'no', 'v169a': 'no', 'v701': 'no education',
            'v127': 'earth/sand', 'v128': 'bamboo with mud',
            'v129': 'thatch/palm leaf', 'v161': 'wood',
            'v467d': 'big problem', 'v467c': 'big problem', 'b19': 8,
            'p20': 8,
        },
    },
    "high_parity_rural": {
        "name": "High parity · short interval, rural",
        "summary": "38-year-old in Mymensingh, sixth child born 15 months "
                   "after the fifth, primary schooling, two antenatal visits.",
        "record": {
            'v012': 38, 'v013': '35-39', 'v201': 6, 'v218': 5,
            'v106': 'primary', 'v149': 'incomplete primary', 'v133': 3,
            'v190': 'poorer', 'v025': 'rural', 'v024': 'mymensingh',
            'v139': 'mymensingh', 'bmi': 19.8, 'height_cm': 149,
            'weight_kg': 44.0, 'm14': 2, 'v511': 16, 'v212': 17,
            'b11': 15, 'risk_index': 4, 'v136': 8, 'v137': 2,
            'v238': 2, 'v701': 'primary', 'v730': 45, 'b19': 12, 'p20': 9,
        },
    },
    "mid_risk_semi_urban": {
        "name": "Mid risk · semi-urban",
        "summary": "31-year-old in a Khulna town, secondary schooling, "
                   "middle quintile, four antenatal visits, second child.",
        "record": {
            'v012': 31, 'v013': '30-34', 'v201': 2, 'v218': 2,
            'v106': 'secondary', 'v190': 'middle', 'v025': 'urban',
            'v024': 'khulna', 'v139': 'khulna', 'v140': 'urban',
            'v026': 'town', 'bmi': 21.4, 'height_cm': 152,
            'weight_kg': 49.5, 'm14': 4, 'v511': 19, 'v212': 21,
            'b11': 30, 'risk_index': 1, 'b19': 24, 'p20': 9,
        },
    },
}
