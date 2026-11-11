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

N_AREA = 20
AREA_SIGMA = 0.5
SITES_PER_SET = 5
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
        rng = np.random.default_rng([config.RANDOM_SEED, 62, k])
        pick = rng.choice(len(rows), SITES_PER_SET, replace=False)
        out[name] = (rows[pick], cols[pick])
    return out


def days_for(city, period, offset_h):
    reg = pd.read_csv(config.PHASE2_DIR / city / f"{city}_regimes_split.csv", parse_dates=["date"])
    weights = json.load(open(config.PHASE2_DIR / city / f"{city}_regime_weights.json"))["weights"]
    days = reg[reg["date"].dt.year == 2025] if period == "test2025" else reg[reg["split"] == "train"]
    days = days[["date", "regime", "split"]].reset_index(drop=True)
    first = pd.Timestamp(f"{config.phase2.START_YEAR}-01-01")
    ok = (days["date"] - pd.Timedelta(hours=offset_h + gen.SPINUP_HOURS)) >= first
    days = days[ok].reset_index(drop=True)
    days["w_k"] = days["regime"].map(weights)
    return days


def run(city, period, k_override=None):
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
    K = cal["K_m2_s"] if k_override is None else k_override
    lam = 0.0 if cal["loss_efold_hours"] is None else 1 / (cal["loss_efold_hours"] * 3600)
    base = cal["alpha_ug_per_s_per_person"] * pop
    sites = fixed_sites(land)
    days = days_for(city, period, cfg["utc_offset_hours"])

    n_tr = 1 + N_AREA + SITES_PER_SET * len(SITE_SETS)
    D, C = len(days), len(cand)
    resp = np.empty((D, n_tr, C), dtype=np.float32)
    site_emis = np.zeros((SITES_PER_SET * len(SITE_SETS), ny, nx))
    for k, name in enumerate(SITE_SETS):
        r, c = sites[name]
        for j in range(SITES_PER_SET):
            site_emis[k * SITES_PER_SET + j, r[j], c[j]] = 1.0       

    for d, row in days.iterrows():
        emis = np.empty((n_tr, ny, nx))
        emis[0] = base
        for m in range(1, N_AREA + 1):
            rng = np.random.default_rng([config.RANDOM_SEED, 61, row["date"].year,
                                         row["date"].dayofyear, m])
            emis[m] = base * rng.lognormal(-AREA_SIGMA**2 / 2, AREA_SIGMA, base.shape)
        emis[N_AREA + 1:] = site_emis
        utc_start = (row["date"] - pd.Timedelta(hours=cfg["utc_offset_hours"])
                     - pd.Timedelta(hours=gen.SPINUP_HOURS))
        met = transport.load_met(city, config.PHASE2_DIR, utc_start,
                                 utc_start + pd.Timedelta(hours=gen.SPINUP_HOURS + 24))
        fields = transport.met_to_domain(met, domain)
        r = transport.run(fields, dx, emis / dx**2, np.full(n_tr, lam), K, np.zeros(n_tr),
                          np.zeros((n_tr, ny, nx)), sample_cells=cells)
        resp[d] = r["samples"][gen.SPINUP_HOURS:].mean(axis=0)
        print(f"  {d + 1}/{D} {row['date'].date()}", end="\r", flush=True)
    print()

    site_rows = np.concatenate([sites[s][0] for s in SITE_SETS])
    site_cols = np.concatenate([sites[s][1] for s in SITE_SETS])
    ds = xr.Dataset(
        {
            "response": (("day", "tracer", "cell"), resp,
                         {"units": "ug m-3 (daily mean, local only)",
                          "tracers": "0 base; 1-20 area-noise members; 21-25 site set A; 26-30 site set B",
                          "site_units": "site tracers are per 1 ug/s"}),
            "site_row": ("site", site_rows), "site_col": ("site", site_cols),
            "site_x": ("site", domain.x.values[site_cols]), "site_y": ("site", domain.y.values[site_rows]),
            "date": ("day", days["date"].to_numpy()),
            "regime": ("day", days["regime"].to_numpy().astype(str)),
            "split": ("day", days["split"].to_numpy().astype(str)),
            "w_k": ("day", days["w_k"].to_numpy()),
            "cell_id": ("cell", cand["id"].to_numpy()),
        },
        attrs={"city": city, "period": period, "K_m2_s": K, "base_total_ug_s": float(base.sum()),
               "area_sigma": AREA_SIGMA, "sites_per_set": SITES_PER_SET,
               "seeds": "area [SEED,61,year,doy,m]; sites [SEED,62,set]"},
    )
    tag = "" if k_override is None else f"_K{k_override:g}"
    path = pdir / f"{city}_ingredients_{period}{tag}.nc"
    ds.to_netcdf(path, encoding={"response": {"zlib": True, "complevel": 4}})
    print(f"  wrote {path} ({D} days x {n_tr} tracers)")


if __name__ == "__main__":
    cities, period, k = parse_args()
    for city in cities:
        run(city, period, k)
