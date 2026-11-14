"""
build pollution maps for original design
for each of the 63 days: 24 h spin-up on the previous day's weather, then 21 emission versions
member 0 = base (alpha × population)
members 1–20 = same total, but 70% as base × log-normal noise per cell and 30% as a random number (mean 5) of hotspots at random land cells with log-normal sizes
"""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import config
import transport
SPINUP_HRS = 24

def parse_cities(desc):
    parser = argparse.ArgumentParser(description=desc)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", choices=config.ALL_CITIES)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    return config.ALL_CITIES if args.all else [args.city]

def synthetic_emissions(base, land, rng):   # makes one random emission map with the same expected total as the base: 70% as base × lognormal noise per cell, 30% as a random number (mean 5) of hotspots at random land cells with lognormal sizes.

    """One synthetic emission field (ug/s per cell) with base's expected total."""
    s = config.SYN_AREA_SIG
    area = (1 - config.SYN_PT_SHARE) * base * rng.lognormal(-s**2 / 2, s, base.shape)
    rows, cols = np.nonzero(land)
    npts = rng.poisson(config.SYN_PT_COUNT)
    pick = rng.integers(0, len(rows), npts)
    sp = config.SYN_PT_SIG
    meaneach = config.SYN_PT_SHARE * base.sum() / config.SYN_PT_COUNT
    mags = meaneach * rng.lognormal(-sp**2 / 2, sp, npts)
    points = np.zeros_like(base)
    np.add.at(points, (rows[pick], cols[pick]), mags)
    return area + points, npts

def generate(city):    # for every chosen day, builds the base plus 20 random emission maps, runs the model with 24 h warm-up on that day's real weather, adds the day's background, and saves all 21 hourly maps

    print(f"=== {city}")
    cfg = config.city_config(city)
    pdir = config.processed_dir(city)
    domain = xr.open_dataset(pdir / f"{city}_domain.nc")
    with open(pdir / f"{city}_calibration.json") as f:
        cal = json.load(f)
    days = pd.read_csv(config.P2_DIR / city / f"{city}_phase3_days.csv", parse_dates=["date"])
    bgdays = pd.read_csv(config.output_dir(city, "calibration") / "daily_background.csv",
                          parse_dates=["date"])
    days = days.merge(bgdays[["date", "background_ug_m3"]], on="date", how="left")
    assert days["background_ug_m3"].notna().all()

    cand = pd.read_csv(pdir / f"{city}_candidates.csv")
    cells = (cand["domain_row"].to_numpy(), cand["domain_col"].to_numpy())
    dx = domain.attrs["cell_size_m"]
    pop = domain["population"].values
    land = domain["land"].values.astype(bool)
    ny, nx = pop.shape
    K = cal["K_m2_s"]
    lam = 0.0 if cal["loss_efold_hours"] is None else 1 / (cal["loss_efold_hours"] * 3600)
    base = cal["alpha_ug_per_s_per_person"] * pop               # ug/s per cell

    nmem = 1 + config.N_SYNTH
    D, H, C = len(days), 24, len(cand)
    local = np.empty((D, nmem, H, C), dtype=np.float32)
    bgresp = np.empty((D, H, C), dtype=np.float32)
    emission = np.empty((D, nmem, ny, nx), dtype=np.float32)
    npoints = np.zeros((D, nmem), dtype=np.int16)

    for d, row in days.iterrows():
        emis = np.empty((nmem, ny, nx))
        emis[0] = base
        for m in range(1, nmem):
            rng = np.random.default_rng([config.SEED, d, m])
            emis[m], npoints[d, m] = synthetic_emissions(base, land, rng)
        emission[d] = emis
        flux = np.concatenate([emis, np.zeros((1, ny, nx))]) / dx**2   # ug m-2 s-1
        bgv = np.r_[np.zeros(nmem), 1.0]
        c0 = bgv[:, None, None] * np.ones((1, ny, nx))

        utcstart = (row["date"] - pd.Timedelta(hours=cfg["utc_offset_hours"])
                     - pd.Timedelta(hours=SPINUP_HRS))
        utcend = utcstart + pd.Timedelta(hours=SPINUP_HRS + H)
        met = transport.load_met(city, config.P2_DIR, utcstart, utcend)
        fields = transport.met_to_domain(met, domain)
        r = transport.run(fields, dx, flux, np.full(nmem + 1, lam), K, bgv, c0,
                          samplecells=cells)
        s = r["samples"][SPINUP_HRS:]                 
        local[d] = s[:, :nmem].transpose(1, 0, 2)
        bgresp[d] = s[:, nmem]
        print(f"  {d + 1}/{D} {row['date'].date()} {row['regime']}", end="\r", flush=True)
    print()

    b = days["background_ug_m3"].to_numpy()
    total = local + (b[:, None, None, None] * bgresp[:, None]).astype(np.float32)

    ds = xr.Dataset(
        {
            "pm25": (("day", "member", "hour", "cell"), total,
                     {"units": "ug m-3", "description": "local + background"}),
            "local_pm25": (("day", "member", "hour", "cell"), local,
                           {"units": "ug m-3", "description": "emission-driven part"}),
            "background_response": (("day", "hour", "cell"), bgresp,
                                    {"description": "response to unit background"}),
            "emission": (("day", "member", "y", "x"), emission, {"units": "ug s-1 per cell"}),
            "n_point_sources": (("day", "member"), npoints),
            "background": ("day", b, {"units": "ug m-3", "source": "03_calibrate.py"}),
            "date": ("day", days["date"].to_numpy()),
            "regime": ("day", days["regime"].to_numpy().astype(str)),
            "split": ("day", days["split"].to_numpy().astype(str)),
            "selection": ("day", days["selection"].to_numpy().astype(str)),
            "w_k": ("day", days["w_k"].to_numpy()),
            "cell_id": ("cell", cand["id"].to_numpy()),
            "cell_x": ("cell", cand["x"].to_numpy()),
            "cell_y": ("cell", cand["y"].to_numpy()),
        },
        coords={"day": np.arange(D), "member": np.arange(nmem), "hour": np.arange(H),
                "cell": np.arange(C), "x": domain.x.values, "y": domain.y.values},
        attrs={
            "city": city, "crs": cfg["crs"], "K_m2_s": K,
            "loss_efold_hours": str(cal["loss_efold_hours"]),
            "alpha_ug_per_s_per_person": cal["alpha_ug_per_s_per_person"],
            "member_0": "base (calibrated population emissions)",
            "hour": "local standard time, hour-start; value = mean over that hour",
            "n_synthetic": config.N_SYNTH,
            "synth_area_sigma": config.SYN_AREA_SIG,
            "synth_point_mean_count": config.SYN_PT_COUNT,
            "synth_point_share": config.SYN_PT_SHARE,
            "synth_point_sigma": config.SYN_PT_SIG,
            "random_seed": config.SEED,
        },
    )
    path = pdir / f"{city}_scenarios.nc"
    ds.to_netcdf(path, encoding={v: {"zlib": True, "complevel": 4}
                                 for v in ("pm25", "local_pm25", "emission")})
    summarize(city, ds)
    plot_examples(city, ds, domain)
    print(f"  wrote {path}")


def summarize(city, ds):             # prints how much spatial variation the maps contain, overall and per weather type
    """Spatial structure the sensor optimiser has to work with."""
    daily = ds["local_pm25"].mean("hour")                    # (day, member, cell)
    spatial_sd = daily.std("cell")
    print(f"  local PM2.5, daily mean over cells: base {float(daily.sel(member=0).mean()):.2f}, "
          f"synthetic {float(daily.sel(member=slice(1, None)).mean()):.2f} ug/m3")
    print(f"  spatial SD across cells (daily means): base {float(spatial_sd.sel(member=0).mean()):.2f}, "
          f"synthetic {float(spatial_sd.sel(member=slice(1, None)).mean()):.2f} ug/m3")
    byregime = spatial_sd.sel(member=0).groupby(ds["regime"]).mean()
    print("  base spatial SD by regime:", {str(k): round(float(v), 2)
                                           for k, v in zip(byregime["regime"].values, byregime.values)})


def plot_examples(city, ds, domain):    # draws the base and one random map for the typical day of each weather type

    """Daily-mean local PM2.5 for the representative day of each regime."""
    rep = np.nonzero(ds["selection"].values == "representative")[0]
    fig, axes = plt.subplots(2, len(rep), figsize=(4.2 * len(rep), 7.5), squeeze=False)
    for i, d in enumerate(rep):
        for j, m in enumerate([0, 1]):
            ax = axes[j, i]
            v = ds["local_pm25"].isel(day=d, member=m).mean("hour").values
            sc = ax.scatter(ds["cell_x"] / 1000, ds["cell_y"] / 1000, c=v, s=12, marker="s",
                            cmap="magma_r")
            fig.colorbar(sc, ax=ax, shrink=0.8, label="local PM2.5 (ug/m3)")
            label = "base" if m == 0 else f"synthetic #{m}"
            ax.set_title(f"{ds['regime'].values[d]} {str(ds['date'].values[d])[:10]}\n{label}",
                         fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{city}: daily-mean local PM2.5 on candidate cells (representative days)")
    fig.tight_layout()
    fig.savefig(config.output_dir(city, "scenarios") / "examples.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    for city in parse_cities("Generate Phase 3 PM2.5 scenarios."):
        generate(city)
