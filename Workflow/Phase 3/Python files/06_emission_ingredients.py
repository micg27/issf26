"""
simulate the pollution source predictability levels L0-L4
31 runs per day: 20 versions of base × lognormal noise and a source at each of the 5 fixed sites in set A and 5 in set B. 
any emission mix can later be made by adding these up (linear superposition)
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
    "gen", Path(__file__).with_name("04_generate_scenarios.py"))   # loads 04 to use its settings
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

N_AREA = 20
AREA_SIG = 0.5
SITES_PER = 5
SITE_SETS = ("A", "B")


def parse_args():
    p = argparse.ArgumentParser(description="Simulate emission ingredients per day.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--city", choices=config.ALL_CITIES)
    g.add_argument("--all", action="store_true")
    p.add_argument("--period", choices=["train", "test2025"], required=True)
    p.add_argument("--k", type=float, default=None, help="Phase 8: override K (m2/s)")
    a = p.parse_args()
    return (config.ALL_CITIES if a.all else [a.city]), a.period, a.k


def fixed_sites(land):         
    rows, cols = np.nonzero(land)
    out = {}
    for k, name in enumerate(SITE_SETS):
        rng = np.random.default_rng([config.SEED, 62, k])
        pick = rng.choice(len(rows), SITES_PER, replace=False)
        out[name] = (rows[pick], cols[pick])
    return out


def days_for(city, period, offseth):       
    reg = pd.read_csv(config.P2_DIR / city / f"{city}_regimes_split.csv", parse_dates=["date"])
    weights = json.load(open(config.P2_DIR / city / f"{city}_regime_weights.json"))["weights"]
    days = reg[reg["date"].dt.year == 2025] if period == "test2025" else reg[reg["split"] == "train"]
    days = days[["date", "regime", "split"]].reset_index(drop=True)
    first = pd.Timestamp(f"{config.phase2.START_YR}-01-01")
    ok = (days["date"] - pd.Timedelta(hours=offseth + gen.SPINUP_HRS)) >= first
    days = days[ok].reset_index(drop=True)
    days["w_k"] = days["regime"].map(weights)
    return days


def run(city, period, koverride=None):        # runs 31 tracers for each day: 20 noisy versions and 1 µg/s source at each of the 10 fixed sites
    print(f"=== {city} {period}")
    cfg = config.city_config(city)
    pdir = config.processed_dir(city)
    domain = xr.open_dataset(pdir / f"{city}_domain.nc")
    cal = json.load(open(pdir / f"{city}_calibration.json"))
    cand = pd.read_csv(pdir / f"{city}_candidates.csv")
    cells = (cand["domain_row"].to_numpy(), cand["domain_col"].to_numpy())
    dx = domain.attrs["cell_size_m"]
    pop = domain["population"].values
    land = domain["land"].values.astype(bool)
    ny, nx = pop.shape
    K = cal["K_m2_s"] if koverride is None else koverride
    lam = 0.0 if cal["loss_efold_hours"] is None else 1 / (cal["loss_efold_hours"] * 3600)
    base = cal["alpha_ug_per_s_per_person"] * pop
    sites = fixed_sites(land)
    days = days_for(city, period, cfg["utc_offset_hours"])

    n_tr = 1 + N_AREA + SITES_PER * len(SITE_SETS)
    D, C = len(days), len(cand)
    resp = np.empty((D, n_tr, C), dtype=np.float32)
    siteemis = np.zeros((SITES_PER * len(SITE_SETS), ny, nx))
    for k, name in enumerate(SITE_SETS):
        r, c = sites[name]
        for j in range(SITES_PER):
            siteemis[k * SITES_PER + j, r[j], c[j]] = 1.0       

    for d, row in days.iterrows():
        emis = np.empty((n_tr, ny, nx))
        emis[0] = base
        for m in range(1, N_AREA + 1):
            rng = np.random.default_rng([config.SEED, 61, row["date"].year,
                                         row["date"].dayofyear, m])
            emis[m] = base * rng.lognormal(-AREA_SIG**2 / 2, AREA_SIG, base.shape)
        emis[N_AREA + 1:] = siteemis
        utcstart = (row["date"] - pd.Timedelta(hours=cfg["utc_offset_hours"])
                     - pd.Timedelta(hours=gen.SPINUP_HRS))
        met = transport.load_met(city, config.P2_DIR, utcstart,
                                 utcstart + pd.Timedelta(hours=gen.SPINUP_HRS + 24))
        fields = transport.met_to_domain(met, domain)
        r = transport.run(fields, dx, emis / dx**2, np.full(n_tr, lam), K, np.zeros(n_tr),
                          np.zeros((n_tr, ny, nx)), samplecells=cells)
        resp[d] = r["samples"][gen.SPINUP_HRS:].mean(axis=0)
        print(f"  {d + 1}/{D} {row['date'].date()}", end="\r", flush=True)
    print()

    siterows = np.concatenate([sites[s][0] for s in SITE_SETS])
    sitecols = np.concatenate([sites[s][1] for s in SITE_SETS])
    ds = xr.Dataset(
        {
            "response": (("day", "tracer", "cell"), resp,
                         {"units": "ug m-3 (daily mean, local only)",
                          "tracers": "0 base; 1-20 area-noise members; 21-25 site set A; 26-30 site set B",
                          "site_units": "site tracers are per 1 ug/s"}),
            "site_row": ("site", siterows), "site_col": ("site", sitecols),
            "site_x": ("site", domain.x.values[sitecols]), "site_y": ("site", domain.y.values[siterows]),
            "date": ("day", days["date"].to_numpy()),
            "regime": ("day", days["regime"].to_numpy().astype(str)),
            "split": ("day", days["split"].to_numpy().astype(str)),
            "w_k": ("day", days["w_k"].to_numpy()),
            "cell_id": ("cell", cand["id"].to_numpy()),
        },
        attrs={"city": city, "period": period, "K_m2_s": K, "base_total_ug_s": float(base.sum()),
               "area_sigma": AREA_SIG, "sites_per_set": SITES_PER,
               "seeds": "area [SEED,61,year,doy,m]; sites [SEED,62,set]"},
    )
    tag = "" if koverride is None else f"_K{koverride:g}"
    path = pdir / f"{city}_ingredients_{period}{tag}.nc"
    ds.to_netcdf(path, encoding={"response": {"zlib": True, "complevel": 4}})
    print(f"  wrote {path} ({D} days x {n_tr} tracers)")


if __name__ == "__main__":
    cities, period, k = parse_args()
    for city in cities:
        run(city, period, k)
