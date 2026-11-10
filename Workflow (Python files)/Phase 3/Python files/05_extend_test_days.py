# 05_extend_test_days.py
"""
POST-RESULTS EXTENSION (added 2026-10-03, after the pre-registered Phase 5
result existed): simulate EVERY day of the 2025 test year, not just the
18-19 days Phase 2 sampled, to narrow the Phase 5 confidence intervals.

Why: Phase 5's CIs were wide because only 18-19 test days were simulated.
The extension changes nothing that selects networks. Phase 4 still
chooses sensors from the same 2021-2024 training maps, and the
pre-registered result on the original test days is kept and reported
alongside. The extension only adds test days.

Same model, calibration, spin-up and emission-scenario design as
04_generate_scenarios.py (whose functions are reused). Differences:
  * days: all 2025 local days, each with its Phase 2 regime label;
  * random streams are seeded by (RANDOM_SEED, 2025, day-of-year,
    member), so they are distinct from the original runs. The base member
    (0) is deterministic and identical for days in both sets;
  * only DAILY-MEAN maps are stored (Phase 5 scores daily means), which is
    24x smaller than hourly;
  * background b_d = that day's mean observed PM2.5 over the in-domain
    stations. It is uniform in space, so it cancels exactly from every
    reconstruction error (kriging weights sum to 1). It is stored only so
    that `pm25_daily` is a realistic total.

--period train (added 2026-10-03, also post-results, pre-stated in
METHODS.md): the same for EVERY 2021-2024 training day, so Phase 4 can choose
networks from ~1,460 days instead of 44-45. Seeds use the day's own year,
so they never collide with the 2025 runs.

Output: data/processed/phase3/<city>/<city>_scenarios_2025_all.nc
        (or <city>_scenarios_train_all.nc with --period train)
    local_pm25_daily (day, member, cell), pm25_daily (day, member, cell),
    emission_total (day, member), date, regime, split, w_k, cell_id/x/y

--k (Phase 8 sensitivity, pre-stated in METHODS.md 2026-10-25): run with
diffusivity K (m2/s) instead of the calibrated value. Everything else,
including alpha and every random draw, is unchanged, so the runs are paired
with the headline ones. Files get a "_K<value>" tag.

Usage:
    python 05_extend_test_days.py --all
    python 05_extend_test_days.py --city melbourne --period train --k 200
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config
import transport

_spec = importlib.util.spec_from_file_location(
    "gen", Path(__file__).with_name("04_generate_scenarios.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

YEAR = 2025


PERIODS = {"test2025": "2025_all", "train": "train_all"}


def parse_args():
    p = argparse.ArgumentParser(description="Simulate every day of a period.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--city", choices=config.ALL_CITIES)
    g.add_argument("--all", action="store_true")
    p.add_argument("--period", choices=list(PERIODS), default="test2025")
    p.add_argument("--k", type=float, default=None, help="Phase 8: override K (m2/s)")
    a = p.parse_args()
    return (config.ALL_CITIES if a.all else [a.city]), a.period, a.k


def k_tag(k):
    return "" if k is None else f"_K{k:g}"


def observed_daily_mean(city, dates):
    obs = pd.read_csv(config.processed_dir(city) / f"{city}_pm25_hourly.csv",
                      parse_dates=["time_local"])
    st = pd.read_csv(config.processed_dir(city) / f"{city}_pm25_stations.csv")
    obs = obs[obs["station"].isin(st[st["in_domain"]]["station"])]
    daily = obs.groupby(obs["time_local"].dt.normalize())["pm25"].mean()
    return daily.reindex(pd.DatetimeIndex(dates)).to_numpy()


def extend(city, period, k_override=None):
    print(f"=== {city}")
    cfg = config.city_config(city)
    pdir = config.processed_dir(city)
    domain = xr.open_dataset(pdir / f"{city}_domain.nc")
    cal = json.load(open(pdir / f"{city}_calibration.json"))
    regimes = pd.read_csv(config.PHASE2_DIR / city / f"{city}_regimes_split.csv", parse_dates=["date"])
    weights = json.load(open(config.PHASE2_DIR / city / f"{city}_regime_weights.json"))["weights"]
    if period == "test2025":
        days = regimes[regimes["date"].dt.year == YEAR]
        assert (days["split"] == "test").all()
    else:
        days = regimes[regimes["split"] == "train"]
        assert days["date"].dt.year.max() < YEAR
    days = days[["date", "regime", "split"]].reset_index(drop=True)
    # The weather record starts 1 Jan 2021 00:00 UTC; days whose 24 h spin-up
    # would start before that can't be simulated the same way, so skip them.
    first_utc = pd.Timestamp(f"{config.phase2.START_YEAR}-01-01")
    spin_start = (days["date"] - pd.Timedelta(hours=cfg["utc_offset_hours"])
                  - pd.Timedelta(hours=gen.SPINUP_HOURS))
    too_early = spin_start < first_utc
    if too_early.any():
        print(f"  skipping {too_early.sum()} day(s) with no spin-up weather: "
              f"{[str(d.date()) for d in days.loc[too_early, 'date']]}")
    days = days[~too_early].reset_index(drop=True)
    days["w_k"] = days["regime"].map(weights)

    cand = pd.read_csv(pdir / f"{city}_candidates.csv")
    cells = (cand["domain_row"].to_numpy(), cand["domain_col"].to_numpy())
    dx = domain.attrs["cell_size_m"]
    pop = domain["population"].values
    land = domain["land"].values.astype(bool)
    ny, nx = pop.shape
    K = cal["K_m2_s"] if k_override is None else k_override
    lam = 0.0 if cal["loss_efold_hours"] is None else 1 / (cal["loss_efold_hours"] * 3600)
    base = cal["alpha_ug_per_s_per_person"] * pop

    n_mem = 1 + config.N_SYNTHETIC
    D, H, C = len(days), 24, len(cand)
    local = np.empty((D, n_mem, C), dtype=np.float32)
    bg_resp = np.empty((D, C), dtype=np.float32)
    totals = np.empty((D, n_mem), dtype=np.float32)

    for d, row in days.iterrows():
        emis = np.empty((n_mem, ny, nx))
        emis[0] = base
        for m in range(1, n_mem):
            rng = np.random.default_rng([config.RANDOM_SEED, row["date"].year,
                                         row["date"].dayofyear, m])
            emis[m], _ = gen.synthetic_emissions(base, land, rng)
        totals[d] = emis.sum(axis=(1, 2))

        flux = np.concatenate([emis, np.zeros((1, ny, nx))]) / dx**2
        bgv = np.r_[np.zeros(n_mem), 1.0]
        c0 = bgv[:, None, None] * np.ones((1, ny, nx))
        utc_start = (row["date"] - pd.Timedelta(hours=cfg["utc_offset_hours"])
                     - pd.Timedelta(hours=gen.SPINUP_HOURS))
        utc_end = utc_start + pd.Timedelta(hours=gen.SPINUP_HOURS + H)
        met = transport.load_met(city, config.PHASE2_DIR, utc_start, utc_end)
        fields = transport.met_to_domain(met, domain)
        r = transport.run(fields, dx, flux, np.full(n_mem + 1, lam), K, bgv, c0, sample_cells=cells)
        s = r["samples"][gen.SPINUP_HOURS:].mean(axis=0)          # daily mean, (n_mem+1, C)
        local[d] = s[:n_mem]
        bg_resp[d] = s[n_mem]
        print(f"  {d + 1}/{D} {row['date'].date()} {row['regime']}", end="\r", flush=True)
    print()

    b = observed_daily_mean(city, days["date"])
    b = np.where(np.isfinite(b), b, np.nanmean(b))
    total = local + (b[:, None, None] * bg_resp[:, None, :]).astype(np.float32)

    ds = xr.Dataset(
        {
            "local_pm25_daily": (("day", "member", "cell"), local, {"units": "ug m-3"}),
            "pm25_daily": (("day", "member", "cell"), total,
                           {"units": "ug m-3", "note": "background = observed daily station mean; "
                                                       "cancels from all reconstruction errors"}),
            "emission_total": (("day", "member"), totals, {"units": "ug s-1"}),
            "background": ("day", b),
            "date": ("day", days["date"].to_numpy()),
            "regime": ("day", days["regime"].to_numpy().astype(str)),
            "split": ("day", days["split"].to_numpy().astype(str)),
            "w_k": ("day", days["w_k"].to_numpy()),
            "cell_id": ("cell", cand["id"].to_numpy()),
            "cell_x": ("cell", cand["x"].to_numpy()),
            "cell_y": ("cell", cand["y"].to_numpy()),
        },
        attrs={"city": city, "extension": f"every {period} day (post-results, see docstring)",
               "seed": f"[{config.RANDOM_SEED}, year, day_of_year, member]",
               "K_m2_s": K, "alpha_ug_per_s_per_person": cal["alpha_ug_per_s_per_person"]},
    )
    path = pdir / f"{city}_scenarios_{PERIODS[period]}{k_tag(k_override)}.nc"
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4}
                                 for v in ("local_pm25_daily", "pm25_daily")})
    print(f"  wrote {path} ({D} days x {n_mem} members)")


if __name__ == "__main__":
    cities, period, k = parse_args()
    for city in cities:
        extend(city, period, k)
