"""
central configuration for Phase 4
"""

import importlib.util
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[3]

_p3path = PROJ_ROOT / "Workflow" / "Phase 3" / "Python files" / "config.py"
_spec = importlib.util.spec_from_file_location("phase3_config", _p3path)
phase3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase3)

ALL_CITIES = phase3.ALL_CITIES
SEED = phase3.SEED

P3_DIR = phase3.PROC_DIR
PROC_DIR = PROJ_ROOT / "data" / "processed" / "phase4"
OUTDIR = PROJ_ROOT / "outputs" / "phase4"


def processed_dir(city):
    d = PROC_DIR / city
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(city):
    d = OUTDIR / city
    d.mkdir(parents=True, exist_ok=True)
    return d



# sensor budgets 

MAIN_N = {"singapore": 22, "melbourne": 8}
N_MAX = {"singapore": 40, "melbourne": 160}
N_MIN = 5



# sensor noise
# noise SD = f x sigma_city where sigma_city is the weighted mean spatial
# SD of the training daily-mean maps. headline f = 0.25. the absolute
# 3 ug/m3 case (corrected low-cost sensors, Barkjohn et al. 2021) is a
# stated limitation.

NOISE_FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]
MAIN_NOISE = 0.25
ABS_NOISE = 3.0



# strategies

# QR pivoting uses r = max(1, N // QR_OVERSAMPLING) POD modes, twice as
# many sensors as modes, Zhou et al. (2022)'s P = 2K setting. Manohar et
# al. (2018) show that too many modes relative to sensors amplifies noise
QR_OVERSAMPLE = 2

KMEANS_INIT = 50
N_RAND_NETS = 200

# kriging variogram: exponential model fitted to the weighted mean
# empirical semivariogram of the training maps, using pairs up to this
# fraction of the longest candidate-to-candidate distance.
VG_MAX_FRAC = 0.5
VG_BIN_M = 2000.0
