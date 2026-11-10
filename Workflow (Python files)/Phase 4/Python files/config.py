# config.py
"""
Central configuration for Phase 4 (sensor network selection).

Phase 3 settings (paths, cities, seed) are imported rather than copied, so
the phases can't drift apart. The values below implement the design
pre-registered in METHODS.md ("Phase 4-5 design").
"""

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

_p3_path = PROJECT_ROOT / "Workflow" / "Phase 3" / "Python files" / "config.py"
_spec = importlib.util.spec_from_file_location("phase3_config", _p3_path)
phase3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase3)

ALL_CITIES = phase3.ALL_CITIES
RANDOM_SEED = phase3.RANDOM_SEED

PHASE3_DIR = phase3.PROCESSED_DIR
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "phase4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase4"


def processed_dir(city):
    d = PROCESSED_DIR / city
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(city):
    d = OUTPUT_DIR / city
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------
# SENSOR BUDGETS (pre-registered 2026-10-02)
# ---------------------------------------------------------
# Headline N = each city's current official PM2.5 network. Sweeps are
# reported in full. Melbourne's goes 4x higher because its candidate area
# is 4x larger, so every Singapore budget has an equal-density match.

HEADLINE_N = {"singapore": 22, "melbourne": 8}
N_MAX = {"singapore": 40, "melbourne": 160}
N_MIN = 5


# ---------------------------------------------------------
# SENSOR NOISE (pre-registered 2026-10-02)
# ---------------------------------------------------------
# Noise SD = f x sigma_city, where sigma_city is the weighted mean spatial
# SD of the training daily-mean maps. Headline f = 0.25. The absolute
# 3 ug/m3 case (corrected low-cost sensors, Barkjohn et al. 2021) is run
# as a stated limitation.

NOISE_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 1.0]
HEADLINE_NOISE_FRACTION = 0.25
ABSOLUTE_NOISE_UG_M3 = 3.0


# ---------------------------------------------------------
# STRATEGIES
# ---------------------------------------------------------
# QR pivoting uses r = max(1, N // QR_OVERSAMPLING) POD modes: twice as
# many sensors as modes, Zhou et al. (2022)'s P = 2K setting. Manohar et
# al. (2018) show that too many modes relative to sensors amplifies noise.
QR_OVERSAMPLING = 2

KMEANS_N_INIT = 50
N_RANDOM_NETWORKS = 200

# Kriging variogram: exponential model fitted to the weighted mean
# empirical semivariogram of the training maps, using pairs up to this
# fraction of the largest candidate-to-candidate distance.
VARIOGRAM_MAX_DIST_FRACTION = 0.5
VARIOGRAM_BIN_M = 2000.0
