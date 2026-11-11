# 02_diagnostic_base_only.py
"""
DIAGNOSTIC (post-results, added 2026-10-04): is the weather signal being
masked by the synthetic emission scenarios?

20 of the 21 emission scenarios per day add ~30% of emissions as point
sources at RANDOM land cells, re-drawn for every scenario, plus
cell-level lognormal noise. No fixed network can anticipate a hotspot
that moves every day, and even spacing is naturally robust to that. This
script repeats the greedy-vs-k-means comparison using ONLY the base
scenario (member 0: calibrated population emissions, the same every day,
driven by each day's real weather). Day-to-day variation then comes from
the weather alone.

Uses the full-training extension (all 2021-2024 days) for selection and
all 365 days of 2025 for testing, headline noise f = 0.25 (relative to
the base-only training maps), the same kriging and bootstrap as
01_evaluate.py.

This is a diagnostic of the scenario design, not a replacement for the
pre-registered result.

Output: data/processed/phase5/phase5_diagnostic_base_only.json

Usage:
    python 02_diagnostic_base_only.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[3]
P4 = PROJ_ROOT / "Workflow" / "Phase 4" / "Python files"
sys.path.insert(0, str(P4))
import networks as N  # noqa: E402

_spec = importlib.util.spec_from_file_location("phase4_config", P4 / "config.py")
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

N_BOOT = 2000
NVALS = {"singapore": [5, 8, 10, 15, 22, 30, 40],
            "melbourne": [5, 8, 10, 15, 22, 30, 40, 60, 88, 120, 160]}


def base_member(split):
    keep = (split["meta"]["member"] == 0).to_numpy()
    meta = split["meta"][keep].reset_index(drop=True)
    n = meta.groupby("regime")["regime"].transform("size").to_numpy()
    w = meta["w_k"].to_numpy() / n
    return split["X"][keep], w / w.sum(), meta


def run(city):
    maps = N.load_maps(city, testset="all2025", trainset="all")
    cells = maps["cells"]
    Xtr, wtr, _ = base_member(maps["train"])
    Xte, wte, mte = base_member(maps["test"])
    Sigma = N.second_moment(Xtr, wtr)
    vg = N.fit_variogram(cells, Sigma)
    K = N.covariance(cells, vg)
    sd = cfg.MAIN_NOISE * N.spatial_sd(Xtr, wtr)
    order, _ = N.greedy_rmse(K, Sigma, vg, sd**2, cfg.N_MAX[city])

    rng = np.random.default_rng([cfg.SEED, 52])
    days, regimes = mte["day"].to_numpy(), mte["regime"].to_numpy()
    byregime = {r: np.unique(days[regimes == r]) for r in np.unique(regimes)}
    rowsof = {d: np.nonzero(days == d)[0] for d in np.unique(days)}
    boots = [np.concatenate([rowsof[d] for v in byregime.values() for d in rng.choice(v, len(v))])
             for _ in range(N_BOOT)]

    def mse(S):
        W = N.ok_weights(K, S, N.sensor_variance(vg, sd**2))
        return N.map_mse(Xte, S, W, sd**2)

    def imp(g, k, i=None):
        i = slice(None) if i is None else i
        return 1 - np.sqrt(np.sum(wte[i] * g[i]) / wte[i].sum()) / np.sqrt(np.sum(wte[i] * k[i]) / wte[i].sum())

    out = {"noise_sd_ug_m3": sd, "n_train_maps": len(Xtr), "n_test_maps": len(Xte),
           "variogram_range_km": vg["range_m"] / 1000, "improvement": {}}
    for n in NVALS[city]:
        g, k = mse(order[:n]), mse(N.kmeans_network(cells, n, cfg.SEED))
        bs = np.array([imp(g, k, i) for i in boots])
        out["improvement"][str(n)] = [float(imp(g, k)), *np.percentile(bs, [2.5, 97.5]).tolist()]
        print(f"  {city} N={n:3d}: {100 * out['improvement'][str(n)][0]:+.1f}% "
              f"({100 * out['improvement'][str(n)][1]:+.1f}, {100 * out['improvement'][str(n)][2]:+.1f})")
    return out


if __name__ == "__main__":
    results = {c: run(c) for c in cfg.ALL_CITIES}
    path = PROJ_ROOT / "data" / "processed" / "phase5" / "phase5_diagnostic_base_only.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path}")
