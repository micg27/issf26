"""
extend test days to every single day, not just a sample
runs the same as 04, except trains using 2021-2024, tests using 2025
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


def extend(city, period, koverride=None):       # similar to generate function in file 04 of phase 3
    print(f"=== {city}")
    cfg = config.city_config(city)
    pdir = config.processed_dir(city)
    domain = xr.open_dataset(pdir / f"{city}_domain.nc")
    cal = json.load(open(pdir / f"{city}_calibration.json"))
    regimes = pd.read_csv(config.P2_DIR / city / f"{city}_regimes_split.csv", parse_dates=["date"])
    weights = json.load(open(config.P2_DIR / city / f"{city}_regime_weights.json"))["weights"]
    if period == "test2025":
        days = regimes[regimes["date"].dt.year == YEAR]
        assert (days["split"] == "test").all()
    else:
        days = regimes[regimes["split"] == "train"]
        assert days["date"].dt.year.max() < YEAR
    days = days[["date", "regime", "split"]].reset_index(drop=True)
    # weather record starts 1 Jan 2021 00:00 UTC, but their 24h spin-up
    # starts before that can't be simulated the same way, so skip them
    firstutc = pd.Timestamp(f"{config.phase2.START_YR}-01-01")
    spinstart = (days["date"] - pd.Timedelta(hours=cfg["utc_offset_hours"])
                  - pd.Timedelta(hours=gen.SPINUP_HRS))
    tooearly = spinstart < firstutc
    if tooearly.any():
        print(f"  skipping {tooearly.sum()} day(s) with no spin-up weather: "
              f"{[str(d.date()) for d in days.loc[tooearly, 'date']]}")
    days = days[~tooearly].reset_index(drop=True)
    days["w_k"] = days["regime"].map(weights)

    cand = pd.read_csv(pdir / f"{city}_candidates.csv")
    cells = (cand["domain_row"].to_numpy(), cand["domain_col"].to_numpy())
    dx = domain.attrs["cell_size_m"]
    pop = domain["population"].values
    land = domain["land"].values.astype(bool)
    ny, nx = pop.shape
    K = cal["K_m2_s"] if koverride is None else koverride
    lam = 0.0 if cal["loss_efold_hours"] is None else 1 / (cal["loss_efold_hours"] * 3600)
    base = cal["alpha_ug_per_s_per_person"] * pop

    nmem = 1 + config.N_SYNTH
    D, H, C = len(days), 24, len(cand)
    local = np.empty((D, nmem, C), dtype=np.float32)
    bgresp = np.empty((D, C), dtype=np.float32)
    totals = np.empty((D, nmem), dtype=np.float32)

    for d, row in days.iterrows():
        emis = np.empty((nmem, ny, nx))
        emis[0] = base
        for m in range(1, nmem):
            rng = np.random.default_rng([config.SEED, row["date"].year,
                                         row["date"].dayofyear, m])
            emis[m], _ = gen.synthetic_emissions(base, land, rng)
        totals[d] = emis.sum(axis=(1, 2))

        flux = np.concatenate([emis, np.zeros((1, ny, nx))]) / dx**2
        bgv = np.r_[np.zeros(nmem), 1.0]
        c0 = bgv[:, None, None] * np.ones((1, ny, nx))
        utcstart = (row["date"] - pd.Timedelta(hours=cfg["utc_offset_hours"])
                     - pd.Timedelta(hours=gen.SPINUP_HRS))
        utcend = utcstart + pd.Timedelta(hours=gen.SPINUP_HRS + H)
        met = transport.load_met(city, config.P2_DIR, utcstart, utcend)
        fields = transport.met_to_domain(met, domain)
        r = transport.run(fields, dx, flux, np.full(nmem + 1, lam), K, bgv, c0, samplecells=cells)
        s = r["samples"][gen.SPINUP_HRS:].mean(axis=0)         
        local[d] = s[:nmem]
        bgresp[d] = s[nmem]
        print(f"  {d + 1}/{D} {row['date'].date()} {row['regime']}", end="\r", flush=True)
    print()

    b = observed_daily_mean(city, days["date"])
    b = np.where(np.isfinite(b), b, np.nanmean(b))
    total = local + (b[:, None, None] * bgresp[:, None, :]).astype(np.float32)

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
               "seed": f"[{config.SEED}, year, day_of_year, member]",
               "K_m2_s": K, "alpha_ug_per_s_per_person": cal["alpha_ug_per_s_per_person"]},
    )
    path = pdir / f"{city}_scenarios_{PERIODS[period]}{k_tag(koverride)}.nc"
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4}
                                 for v in ("local_pm25_daily", "pm25_daily")})
    print(f"  wrote {path} ({D} days x {nmem} members)")


if __name__ == "__main__":
    cities, period, k = parse_args()
    for city in cities:
        extend(city, period, k)
