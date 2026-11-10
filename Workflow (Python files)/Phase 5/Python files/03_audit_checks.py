# 03_audit_checks.py
"""
Numerical checks behind Workflow/Assumptions audit.md (2026-10-04).

  A  synthetic emission totals equal the base total on average
  B  share of synthetic maps' spatial variance that is scenario-to-scenario
     randomness (unpredictable from weather)
  D  how much the interpolated ERA5 wind varies across each domain
  M  whether 24 h of spin-up is enough (24 h vs 72 h on stagnant days)
  X  whether day-level bootstrap CIs are too narrow, compared with a
     weekly-block bootstrap that keeps multi-day weather events together

Usage:
    python 03_audit_checks.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

# Open NetCDF files one at a time: the HDF5 library underneath isn't
# thread-safe, and dask's default threaded reads can crash it when many
# files are open in one process.
dask.config.set(scheduler="synchronous")

ROOT = Path(__file__).resolve().parents[3]
P3, P4 = ROOT / "Workflow" / "Phase 3" / "Python files", ROOT / "Workflow" / "Phase 4" / "Python files"
sys.path.insert(0, str(P4))
import networks as N  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg4 = load("cfg4", P4 / "config.py")
sys.path.insert(0, str(P3))
transport = load("transport", P3 / "transport.py")
PH3 = cfg4.PHASE3_DIR


def check_A_B(city):
    e = xr.open_dataset(PH3 / city / f"{city}_scenarios_train_all.nc")["emission_total"].values
    r = e[:, 1:] / e[:, [0]]
    m = N.load_maps(city, "all2025", "all")["test"]
    D = m["meta"]["day"].nunique()
    Xr = N.centred(m["X"]).reshape(D, -1, m["X"].shape[1])
    syn = Xr[:, 1:]
    rand = syn - syn.mean(axis=1, keepdims=True)
    print(f"  A emission total ratio synthetic/base: mean {r.mean():.3f} (5-95% {np.percentile(r, 5):.2f}-{np.percentile(r, 95):.2f})")
    print(f"  B random share of synthetic spatial variance: {100 * np.mean(rand**2) / np.mean(syn**2):.0f}%")


def check_D(city):
    dom = xr.open_dataset(PH3 / city / f"{city}_domain.nc")
    met = transport.load_met(city, cfg4.phase3.PHASE2_DIR, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31 23:00"))
    f = transport.met_to_domain(met, dom)
    speed = np.hypot(f["u10"], f["v10"]).mean()
    spread = (f["u10"].std(axis=(1, 2)) + f["v10"].std(axis=(1, 2))).mean()
    print(f"  D wind (2023): spatial SD of u+v across domain = {100 * spread / speed:.0f}% of mean speed "
          f"({met.sizes['latitude']}x{met.sizes['longitude']} ERA5 points)")


def check_M(city, days):
    pdir = PH3 / city
    dom = xr.open_dataset(pdir / f"{city}_domain.nc")
    cal = json.load(open(pdir / f"{city}_calibration.json"))
    cand = pd.read_csv(pdir / f"{city}_candidates.csv")
    cells = (cand["domain_row"].to_numpy(), cand["domain_col"].to_numpy())
    flux = (cal["alpha_ug_per_s_per_person"] * dom["population"].values / 4e6)[None]
    off = cfg4.phase3.city_config(city)["utc_offset_hours"]
    for day in days:
        res = {}
        for spin in (24, 72):
            start = pd.Timestamp(day) - pd.Timedelta(hours=off + spin)
            f = transport.met_to_domain(
                transport.load_met(city, cfg4.phase3.PHASE2_DIR, start, start + pd.Timedelta(hours=spin + 24)), dom)
            r = transport.run(f, 2000.0, flux, [0.0], cal["K_m2_s"], [0.0], np.zeros_like(flux), sample_cells=cells)
            res[spin] = r["samples"][spin:, 0].mean(axis=0)
        print(f"  M {day}: 72 h vs 24 h spin-up mean change {100 * (res[72].mean() / res[24].mean() - 1):+.1f}%, "
              f"pattern r = {np.corrcoef(res[24], res[72])[0, 1]:.3f}")


def check_X(city):
    m = N.load_maps(city, "all2025", "all")
    te = m["test"]
    X, w, meta = te["X"], te["w"], te["meta"]
    nets = pd.read_csv(cfg4.processed_dir(city) / f"{city}_networks_trainall.csv")
    kj = json.load(open(cfg4.processed_dir(city) / f"{city}_kriging_trainall.json"))
    vg, sd = kj["variogram"], kj["noise_sd_ug_m3"]["f0.25"]
    K = N.covariance(m["cells"], vg)
    week = pd.to_datetime(meta["date"]).dt.isocalendar().week.to_numpy()
    rows = {k: np.nonzero(week == k)[0] for k in np.unique(week)}
    weeks = np.array(list(rows))
    n = cfg4.HEADLINE_N[city]

    def mse(strategy, noise):
        S = nets[(nets.strategy == strategy) & (nets.noise == noise) & (nets.N == n)].sort_values("rank").cell_index.tolist()
        W = N.ok_weights(K, S, N.sensor_variance(vg, sd**2))
        return N.map_mse(X, S, W, sd**2)

    g, k = mse("greedy", "f0.25"), mse("kmeans", "any")
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(2000):
        i = np.concatenate([rows[x] for x in rng.choice(weeks, len(weeks))])
        bs.append(1 - np.sqrt(np.sum(w[i] * g[i]) / w[i].sum()) / np.sqrt(np.sum(w[i] * k[i]) / w[i].sum()))
    print(f"  X headline N={n}: {100 * (1 - np.sqrt(np.sum(w * g) / np.sum(w * k))):+.1f}%, "
          f"weekly-block CI {100 * np.percentile(bs, 2.5):+.1f} to {100 * np.percentile(bs, 97.5):+.1f}")


if __name__ == "__main__":
    for city in cfg4.ALL_CITIES:
        print(f"=== {city}")
        check_A_B(city)
        check_D(city)
        check_X(city)
    print("=== melbourne spin-up")
    check_M("melbourne", ["2025-06-13", "2025-05-10", "2025-07-07", "2025-03-15"])
