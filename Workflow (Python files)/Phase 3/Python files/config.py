# config.py
"""
Central configuration for Phase 3 (pollution scenario generation).

Every other Phase 3 script imports from here. Change settings in ONE place.
Phase 2 settings that Phase 3 must agree with (years, UTC offsets, the
train/test split) are imported from Phase 2's config rather than copied,
so the two phases can't silently drift apart.
"""

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
# config.py lives at <project_root>/Workflow/Phase 3/Python files/config.py


# ---------------------------------------------------------
# PHASE 2 CONFIG (imported, not duplicated)
# ---------------------------------------------------------

_p2_path = PROJECT_ROOT / "Workflow" / "Phase 2" / "Python files" / "config.py"
_spec = importlib.util.spec_from_file_location("phase2_config", _p2_path)
phase2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase2)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

RAW_DIR = PROJECT_ROOT / "data" / "raw"
GRID_DIR = RAW_DIR / "grids"
POP_DIR = RAW_DIR / "population"
PM25_RAW_DIR = RAW_DIR / "pm25"

PHASE2_DIR = phase2.PROCESSED_DIR          # data/processed/meteorology
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "phase3"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3"


def processed_dir(city=None):
    d = PROCESSED_DIR / city if city else PROCESSED_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(city, subfolder=None):
    d = OUTPUT_DIR / city
    if subfolder:
        d = d / subfolder
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------
# MODEL DOMAIN
# ---------------------------------------------------------
#
# The transport model runs on a regular 2 km raster aligned exactly with
# the Phase 1 candidate grid (same origin, same cell edges), extended by
# DOMAIN_MARGIN_CELLS on every side. The margin matters: pollution emitted
# just outside the candidate grid (e.g. Jurong Island, Melbourne's fringe)
# still blows into it, and the open-boundary condition is less distorting
# when it sits away from the cells we care about.
# ---------------------------------------------------------

CELL_SIZE_M = 2000
DOMAIN_MARGIN_CELLS = 5        # 10 km

CITIES = {
    "singapore": {
        "crs": "EPSG:3414",     # SVY21
        "grid_file": "Singapore_final.gpkg",
        # The domain margin reaches Johor Bahru (Malaysia) and northern
        # Batam (Indonesia) -- large populations upwind of Singapore that
        # the Singapore-only raster would silently drop. WorldPop country
        # rasters are masked to national territory, so they don't overlap.
        # Land polygons for the land mask (where synthetic point sources may
        # go). Outside Singapore (Johor, Batam) no polygon is used: a cell
        # with any population counts as land there.
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
        # Vicmap LGA polygons follow the coastline and cover all land in
        # the Melbourne domain.
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


# ---------------------------------------------------------
# PM2.5 OBSERVATIONS
# ---------------------------------------------------------
#
# Singapore's NEA data is reported per REGION, not per station. The
# coordinates below are the region "label locations" NEA itself publishes
# in the data.gov.sg PM2.5 API (regionMetadata, fetched 2026-10-01 from
# https://api-open.data.gov.sg/v2/real-time/api/pm25). They are where NEA
# draws each region on its map, not monitoring-station sites: treating a
# regional reading as a point reading there is an approximation, stated
# as such in the methodology.
# ---------------------------------------------------------

SINGAPORE_REGIONS = {
    "north":   (1.41803, 103.82),
    "south":   (1.29587, 103.82),
    "east":    (1.35735, 103.94),
    "west":    (1.35735, 103.70),
    "central": (1.35735, 103.82),
}

SINGAPORE_PM25_FILE = "Historical1hrPM2.5 (Singapore).csv"
MELBOURNE_PM25_GLOB = "*_All_sites_air_quality_hourly_avg_*.xlsx"

# EPA Victoria's PM2.5 column name changes mid-record (see CLAUDE.md):
# BPM2.5 = raw BAM reading, SBPM25 = same instrument, STP-corrected.
MELBOURNE_PM25_COLUMNS = ["BPM2.5", "SBPM25"]

# Plausibility bounds for hourly PM2.5 (ug/m3).
# Small negative values are kept on purpose: 7.7% of Melbourne's hourly
# BAM readings are negative, almost all slightly (median -1.2, 95% above
# -5.1). That is the instrument's random noise around low true
# concentrations, and dropping it would bias clean-air hours upward. Only
# readings below -10 (189 of 412,687: instrument faults) are removed.
# > 1000 does not occur outside instrument faults even in bushfire smoke.
PM25_MIN = -10.0
PM25_MAX = 1000.0


# ---------------------------------------------------------
# TRANSPORT MODEL
# ---------------------------------------------------------
#
# Horizontal eddy diffusivity K is FIXED, not calibrated. A diagnostic
# grid search (03_calibrate.py --k-grid) shows the station data barely
# constrain it: across K = 100 to 30,000 m2/s the train RMSE changes by
# < 1% in Singapore (5.32 -> 5.26 ug/m3) and 1.2% in Melbourne
# (4.43 -> 4.37), in both drifting toward the smoothest field. About 43% of hourly PM2.5 variance is station-level
# noise, and least squares on noisy data mildly rewards a structureless
# prediction. The networks (5 regional values / 9 stations) cannot
# identify K, so a physically based value is used and tested in Phase 8.
#
# The value follows US EPA CMAQ's horizontal-diffusion scheme
# (CCTM/src/hdiff/multiscale/hcdiff3d.F): diffusivity is bounded below
# by KHMIN = 200 m2/s and above by a resolution-scaled cap
# KH * (4 km / dx)^2 = 2000 * (4/2)^2 = 8000 m2/s at our 2 km grid.
# The central value is about the geometric mean of those bounds. The
# bounds themselves are the Phase 8 sensitivity cases.
# ---------------------------------------------------------

TRANSPORT_K_M2_S = 1000.0
TRANSPORT_K_SENSITIVITY_M2_S = [200.0, 8000.0]


# ---------------------------------------------------------
# SCENARIOS (04_generate_scenarios.py)
# ---------------------------------------------------------
#
# Every Phase 2 day gets one BASE scenario (calibrated population-driven
# emissions) plus N_SYNTHETIC synthetic emission scenarios, all run with
# that day's real weather. Synthetic emissions keep the same expected
# total as the base case but redistribute it:
#   * area sources: base emissions x mean-one lognormal noise per cell
#     (SYNTH_AREA_SIGMA) -- neighbourhood-level uncertainty in the
#     population proxy;
#   * point sources: a Poisson number (mean SYNTH_POINT_MEAN_COUNT) placed
#     uniformly at random on land cells, with lognormal magnitudes
#     (SYNTH_POINT_SIGMA) whose expected total is SYNTH_POINT_SHARE of all
#     emissions -- stand-ins for industry, ports, construction and fires,
#     which a population proxy can't place (Jurong Island has ~0 people).
# These parameters are modelling choices, not fitted values; they are
# Phase 8 sensitivity candidates.
# ---------------------------------------------------------

N_SYNTHETIC = 20
SYNTH_AREA_SIGMA = 0.5
SYNTH_POINT_MEAN_COUNT = 5
SYNTH_POINT_SHARE = 0.3
SYNTH_POINT_SIGMA = 1.0
RANDOM_SEED = phase2.RANDOM_SEED
