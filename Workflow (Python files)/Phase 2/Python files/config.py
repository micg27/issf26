# config.py
"""
Central configuration for Phase 2 (meteorological characterization).

Every other script imports from here. Change settings in ONE place.
"""

from pathlib import Path


# ---------------------------------------------------------
# TIME WINDOW  (matches the research plan: 1 Jan 2021 - 31 Dec 2025)
# ---------------------------------------------------------

START_YEAR = 2021
END_YEAR = 2025

YEARS = list(range(START_YEAR, END_YEAR + 1))


# ---------------------------------------------------------
# PATHS  (all relative to the project root)
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# config.py lives at  <project_root>/Workflow/Phase 2/Python files/config.py
# parents[0]=Python files, [1]=Phase 2, [2]=Workflow, [3]=project root.
# If you move these scripts to a different depth, update this index —
# everything else (data/, outputs/) is located relative to PROJECT_ROOT.

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "meteorology"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "meteorology"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "meteorology"


def raw_dir(city):
    return RAW_DIR / city


def processed_dir(city):
    d = PROCESSED_DIR / city
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(city, subfolder=None):
    d = OUTPUT_DIR / city
    if subfolder:
        d = d / subfolder
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------
# CITIES
# ---------------------------------------------------------
#
# 'area' is the ERA5 request box in [North, West, South, East].
#
# NOTE: these boxes are taken from your ERA5 download script. They differ
# slightly from the boxes written in Phase_1.md:
#     Phase_1.md Singapore : 1.50N  1.25N  103.50E 104.00E
#     Phase_1.md Melbourne : 37.50S 38.75S 144.25E 146.50E
# Pick one and make both documents agree before you write your methods
# section. The values below are the ones your download script actually used.
#
# 'utc_offset_hours' is the FIXED local-standard-time offset.
# 'timezone' is only used if USE_DST is True.
# ---------------------------------------------------------

CITIES = {
    "singapore": {
        "area": [1.55, 103.55, 1.15, 104.10],
        "utc_offset_hours": 8,
        "timezone": "Asia/Singapore",   # Singapore does not observe DST
        "label_prefix": "SIN",
        "final_k": 3,                   # set from 05 sweep: best Calinski-
                                         # Harabasz, silhouette, and Davies-
                                         # Bouldin all favor k=3 over k=4;
                                         # k=4's 4th cluster only splits NE
                                         # monsoon by wind strength, not a
                                         # distinct direction/regime
    },
    "melbourne": {
        "area": [-37.38, 144.40, -38.51, 146.30],
        "utc_offset_hours": 10,         # AEST; AEDT (+11) applies Oct-Apr
        "timezone": "Australia/Melbourne",
        "label_prefix": "MEL",
        "final_k": 3,                   # set from 05 sweep: best/near-best on
                                         # 3 of 4 validity metrics; splits
                                         # NW-blend into distinct northerly
                                         # vs westerly regimes
    },
}

ALL_CITIES = list(CITIES.keys())


# ---------------------------------------------------------
# DAYLIGHT SAVING
# ---------------------------------------------------------
#
# False  -> use a fixed local-standard-time offset for both cities.
#           Melbourne "days" stay 24 h year-round. Simplest, and defensible
#           for diurnal-cycle science, but you MUST state it in your methods.
#
# True   -> use true civil local time via the IANA timezone. Melbourne days
#           at the DST transitions then have 23 or 25 hours and are dropped
#           by the complete-day filter in script 04 (2 days lost per year).
#
# ---------------------------------------------------------

USE_DST = False


# ---------------------------------------------------------
# ERA5 VARIABLES
# ---------------------------------------------------------

ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "total_precipitation",
    "boundary_layer_height",
]

# Short names as they appear once decoded
REQUIRED_VARS = {"u10", "v10", "blh", "t2m", "tp"}


# ---------------------------------------------------------
# CLUSTERING FEATURES
# ---------------------------------------------------------
#
# Wind is represented as u/v components in four 6-hour blocks rather than a
# single daily mean. A daily mean would cancel out diurnal wind reversals
# (sea breeze / land breeze), which is exactly the signal that distinguishes
# transport regimes in a coastal city.
# ---------------------------------------------------------

FEATURES = [
    "u1", "v1",
    "u2", "v2",
    "u3", "v3",
    "u4", "v4",
    "pblh_mean",
]

N_BLOCKS = 4
BLOCK_HOURS = 6


# ---------------------------------------------------------
# CLUSTERING / SAMPLING SETTINGS
# ---------------------------------------------------------

K_RANGE = range(2, 9)          # candidate k values swept in script 05
KMEANS_N_INIT = 50
RANDOM_SEED = 42

N_SAMPLE_PER_REGIME = 20       # days per regime carried into Phase 3

TEST_SIZE = 0.30

# "stratified_random" -> random days, stratified by regime.
#                        Risk: adjacent days are meteorologically correlated,
#                        so train and test are not fully independent.
# "temporal"          -> hold out whole years (HOLDOUT_YEARS) as test.
#                        Cleaner independence; the safer choice to defend.
SPLIT_MODE = "temporal"
HOLDOUT_YEARS = [2025]


def city_config(city):
    if city not in CITIES:
        raise ValueError(
            f"Unknown city: {city!r}. Known cities: {ALL_CITIES}"
        )
    return CITIES[city]
