# config.py
"""
central configuration for phase 3

"""

import importlib.util
from pathlib import Path


PROJ_ROOT = Path(__file__).resolve().parents[3]

# phase 2 config 


_p2path = PROJ_ROOT / "Workflow" / "Phase 2" / "Python files" / "config.py"
_spec = importlib.util.spec_from_file_location("phase2_config", _p2path)
phase2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase2)



# paths


RAW_DIR = PROJ_ROOT / "data" / "raw"
GRID_DIR = RAW_DIR / "grids"
POP_DIR = RAW_DIR / "population"
PM_RAW_DIR = RAW_DIR / "pm25"

P2_DIR = phase2.PROC_DIR          # data/processed/meteorology
PROC_DIR = PROJ_ROOT / "data" / "processed" / "phase3"
OUTDIR = PROJ_ROOT / "outputs" / "phase3"


def processed_dir(city=None):
    d = PROC_DIR / city if city else PROC_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(city, subfolder=None):
    d = OUTDIR / city
    if subfolder:
        d = d / subfolder
    d.mkdir(parents=True, exist_ok=True)
    return d



# model domain


CELL_SIZE = 2000
MARGIN_CELLS = 5        # 10 km

CITIES = {
    "singapore": {
        "crs": "EPSG:3414",     # SVY21
        "grid_file": "Singapore_final.gpkg",
        "land_file": "boundaries/singapore/MasterPlan2025PlanningAreaBoundaryNoSea.geojson",
        "pop_files": [
            "sgp_pop_2023_CN_100m_R2025A_v1.tif",
            "mys_pop_2023_CN_100m_R2025A_v1.tif",
            "idn_pop_2023_CN_100m_R2025A_v1.tif",
        ],
    },
    "melbourne": {
        "crs": "EPSG:7855",     # GDA2020 / MGA zone 55
        "grid_file": "Melbourne_final.gpkg",
        "land_file": "boundaries/melbourne/lga_polygon/lga_polygon.shp",
        "pop_files": ["aus_pop_2023_CN_100m_R2025A_v1.tif"],
    },
}

ALL_CITIES = list(CITIES.keys())


def city_config(city):
    if city not in CITIES:
        raise ValueError(f"Unknown city: {city!r}. Known cities: {ALL_CITIES}")
    cfg = dict(CITIES[city])
    cfg["utc_offset_hours"] = phase2.CITIES[city]["utc_offset_hours"]
    return cfg



# pm2.5 observations
# SG NEA releases data by region, not by individual stations
# the coordinates below are the coordinates for each region
# as published by NEA



SG_REGIONS = {
    "north":   (1.41803, 103.82),
    "south":   (1.29587, 103.82),
    "east":    (1.35735, 103.94),
    "west":    (1.35735, 103.70),
    "central": (1.35735, 103.82),
}

SG_PM_FILE = "Historical1hrPM2.5 (Singapore).csv"
MEL_PM_GLOB = "*_All_sites_air_quality_hourly_avg_*.xlsx"
# epa vic's pm2.5 column changes name mid episode
MEL_PM_COLS = ["BPM2.5", "SBPM25"]

PM25_MIN = -10.0
PM25_MAX = 1000.0



# transport model
# values for K are taken from US EPA CMAQ's horizontal-diffusion scheme
# (CCTM/src/hdiff/multiscale/hcdiff3d.F). upper and lower bounds are 
# used for Phase 8, and the K value for the rest of the experiment is 
# approximately the geometric mean

K_M2S = 1000.0
K_SENS_M2S = [200.0, 8000.0]


# scenarios (04_generate_scenarios.py)
# every Phase 2 day gets one base scenario (calibrated population-driven
# emissions) and N_SYNTHETIC synthetic emission scenarios, run with
# that day's real weather. synthetic emissions keep the same expected
# total as the base case but redistribute it:
#   for area sources: base emissions and log-normal distributed noise per cell
#     (SYNTH_AREA_SIGMA) -- uncertainty in the population estimate
#   for point sources: a Poisson number (mean SYNTH_POINT_MEAN_COUNT) placed
#     uniformly at random on land cells, with lognormal magnitudes
#     (SYNTH_POINT_SIGMA) whose expected total is SYNTH_POINT_SHARE of all
#     emissions. these are replacements for industry, ports, construction and fires,
#     which a population estimate cannot place (Jurong Island has ~0 residents).


N_SYNTH = 20
SYN_AREA_SIG = 0.5
SYN_PT_COUNT = 5
SYN_PT_SHARE = 0.3
SYN_PT_SIG = 1.0
SEED = phase2.SEED
