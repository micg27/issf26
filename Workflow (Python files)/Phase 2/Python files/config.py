from pathlib import Path

START_YEAR = 2021
END_YEAR = 2025

YEARS = list(range(START_YEAR, END_YEAR + 1))

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw" /"meteorology"
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

CITIES = {
    "singapore": {
        "area": [1.55, 103.55, 1.15, 104.10],
        "utc_offset_hours": 8,
        "timezone": "Asia/Singapore",
        "label_prefix": "SIN",
        "final_k": 3,                 
    },
    "melbourne": {
        "area": [-37.38, 144.40, -38.51, 146.30],
        "utc_offset_hours": 10,         
        "timezone": "Australia/Melbourne",
        "label_prefix": "MEL",
        "final_k": 3,                   
    },
}

ALL_CITIES = list(CITIES.keys())

USE_DST = False

ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "total_precipitation",
    "boundary_layer_height",
]

REQUIRED_VARS = {"u10", "v10", "blh", "t2m", "tp"}

FEATURES = [
    "u1", "v1",
    "u2", "v2",
    "u3", "v3",
    "u4", "v4",
    "pblh_mean",
]

N_BLOCKS = 4
BLOCK_HOURS = 6

K_RANGE = range(2, 9)       
KMEANS_N_INNIT_HAHA = 50
RANDOM_SEED = 42

N_SAMPLE_PER_REGIME = 20 

TEST_SIZE = 0.30

SPLIT_MODE = "temporal"
HOLDOUT_YEARS = [2025]

def city_config(city):
    if city not in CITIES:
        raise ValueError(
            f"Unknown city: {city!r}. Known cities: {ALL_CITIES}"
        )
    return CITIES[city]
