"""
fine-tune constants to match real world values
"""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import nnls
import config
import transport

K_DIAG_GRID = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]   # tried spreading rates K
LAM_GRID_HRS = [np.inf, 24.0, 12.0, 6.0, 3.0, 1.5]               # tried loss rates lambda
SPINUP_HRS = 24                                                  # 24-hour simulation from the previous day, so the day does not always start from completely clean air
SECS_PER_YR = 365.25 * 86400                                     # converts emission rate into kg person-1 year-1


def parse_args(desc):
    parser = argparse.ArgumentParser(description=desc)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", choices=config.ALL_CITIES)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--k-grid", action="store_true",
                        help="diagnostic: also search K over K_DIAGNOSTIC_GRID")
    args = parser.parse_args()
    cities = config.ALL_CITIES if args.all else [args.city]
    kgrid = K_DIAG_GRID if args.k_grid else [config.K_M2S]
    return cities, kgrid


def lam_per_second(hours):
    return 0.0 if np.isinf(hours) else 1.0 / (hours * 3600.0)


def station_cells(city, domain):          
    st = pd.read_csv(config.processed_dir(city) / f"{city}_pm25_stations.csv")
    st = st[st["in_domain"]].reset_index(drop=True)
    dx = domain.attrs["cell_size_m"]
    x0 = float(domain.x[0]) - dx / 2
    ytop = float(domain.y[0]) + dx / 2
    st["col"] = np.floor((st["x"] - x0) / dx).astype(int)
    st["row"] = np.floor((ytop - st["y"]) / dx).astype(int)
    return st


def simulate_days(city, domain, days, stations, offseth, kgrid):    #for every calibration day, runs the model once with "1 µg/s per person" emissions 
                                                                    #and once with "1 µg/m³" background, for every loss rate, and records what each station would measure each hour.
    """Station hour-means for every (day, K, tracer). Shape (D, nK, 24, ntr, nst)."""
    dx = domain.attrs["cell_size_m"]
    pop = domain["population"].values
    ny, nx = pop.shape
    lams = np.array([lam_per_second(h) for h in LAM_GRID_HRS])
    nl = len(lams)

    emis = np.zeros((2 * nl, ny, nx))
    emis[:nl] = pop / dx**2                      # alpha = 1 ug s-1 person-1
    bg = np.r_[np.zeros(nl), np.ones(nl)]
    c0 = bg[:, None, None] * np.ones((1, ny, nx))
    lamtr = np.r_[lams, lams]
    cells = (stations["row"].to_numpy(), stations["col"].to_numpy())

    out = np.empty((len(days), len(kgrid), 24, 2 * nl, len(stations)))
    for d, date in enumerate(days):
        localstart = pd.Timestamp(date)
        utcstart = localstart - pd.Timedelta(hours=offseth) - pd.Timedelta(hours=SPINUP_HRS)
        utcend = utcstart + pd.Timedelta(hours=SPINUP_HRS + 24)
        met = transport.load_met(city, config.P2_DIR, utcstart, utcend)
        fields = transport.met_to_domain(met, domain)
        for k, K in enumerate(kgrid):
            r = transport.run(fields, dx, emis, lamtr, K, bg, c0, samplecells=cells)
            out[d, k] = r["samples"][SPINUP_HRS:]
        print(f"  simulated {d + 1}/{len(days)} {localstart.date()}", end="\r")
    print()
    return out, nl


def observations(city, days, stations):
    obs = pd.read_csv(config.processed_dir(city) / f"{city}_pm25_hourly.csv",
                      parse_dates=["time_local"])
    obs = obs[obs["station"].isin(stations["station"])]
    piv = obs.pivot_table(index="time_local", columns="station", values="pm25")
    piv = piv.reindex(columns=stations["station"])
    y = np.full((len(days), 24, len(stations)), np.nan)
    for d, date in enumerate(days):
        hours = pd.date_range(pd.Timestamp(date), periods=24, freq="h")
        y[d] = piv.reindex(hours).to_numpy()
    return y


def fit(sime, simb, y, alpha=None):    # finds per person emission alpha and background per day value that best matches station readings
    D = y.shape[0]
    ok = np.isfinite(y)
    rowsday = np.broadcast_to(np.arange(D)[:, None, None], y.shape)[ok]
    B = np.zeros((ok.sum(), D))
    B[np.arange(ok.sum()), rowsday] = simb[ok]
    target = y[ok]
    if alpha is None:
        A = np.column_stack([sime[ok], B])
        coef, _ = nnls(A, target)
        alpha, b = coef[0], coef[1:]
    else:
        b, _ = nnls(B, target - alpha * sime[ok])
    pred = alpha * sime + b[:, None, None] * simb
    return alpha, b, pred


def background_only(y):
    return np.broadcast_to(np.nanmean(y, axis=(1, 2), keepdims=True), y.shape)


def scores(y, pred, withinday=True):
    ok = np.isfinite(y)
    err = pred[ok] - y[ok]
    out = {"rmse": float(np.sqrt(np.mean(err**2))), "bias": float(err.mean()),
           "n_obs": int(ok.sum())}
    if withinday:
        yanom = y - np.nanmean(y, axis=(1, 2), keepdims=True)
        panom = pred - np.nanmean(np.where(ok, pred, np.nan), axis=(1, 2), keepdims=True)
        out["r_within_day"] = float(np.corrcoef(yanom[ok], panom[ok])[0, 1])
    return out


def calibrate(city, kgrid):     # tries each loss rate on 2021–2024, keeps the best, freezes alpha and checks 2025
                                # confirms the model beats background-only, then saves the calibration and daily backgrounds
    print(f"=== {city}")
    cfg = config.city_config(city)
    domain = xr.open_dataset(config.processed_dir(city) / f"{city}_domain.nc")
    stations = station_cells(city, domain)
    daysdf = pd.read_csv(config.P2_DIR / city / f"{city}_phase3_days.csv",
                          parse_dates=["date"])
    days = daysdf["date"].tolist()
    istrain = (daysdf["split"] == "train").to_numpy()

    sims, nl = simulate_days(city, domain, days, stations, cfg["utc_offset_hours"], kgrid)
    y = observations(city, days, stations)

    rows = []
    for k, K in enumerate(kgrid):
        for j, lamh in enumerate(LAM_GRID_HRS):
            se, sb = sims[:, k, :, j, :], sims[:, k, :, nl + j, :]
            a, _, pred = fit(se[istrain], sb[istrain], y[istrain])
            s = scores(y[istrain], pred)
            rows.append({"K_m2_s": K, "loss_efold_h": lamh, "alpha": a, **s})
    grid = pd.DataFrame(rows)
    if len(kgrid) > 1:
        grid.to_csv(config.output_dir(city, "calibration") / "grid_search_K_diagnostic.csv",
                    index=False)
        print(grid.round(4).to_string(index=False))
        return
    best = grid.loc[grid["rmse"].idxmin()]
    k = kgrid.index(best["K_m2_s"])
    j = LAM_GRID_HRS.index(best["loss_efold_h"])
    se, sb = sims[:, k, :, j, :], sims[:, k, :, nl + j, :]

    alpha, b_tr, predtr = fit(se[istrain], sb[istrain], y[istrain])
    _, b_te, predte = fit(se[~istrain], sb[~istrain], y[~istrain], alpha=alpha)
    nulltr = background_only(y[istrain])
    nullte = background_only(y[~istrain])
    # at lam = 0 the background tracer is 1 everywhere, so the model defines background-only 
    # as the case alpha = 0. hence, its training RMSE cannot be worse
    
    assert scores(y[istrain], predtr)["rmse"] <= scores(y[istrain], nulltr, False)["rmse"] + 1e-6, \
        "full model fits worse than background-only on train days"

    result = {
        "city": city,
        "K_m2_s": float(best["K_m2_s"]),
        "K_fixed": len(kgrid) == 1,
        "loss_efold_hours": None if np.isinf(best["loss_efold_h"]) else float(best["loss_efold_h"]),
        "alpha_ug_per_s_per_person": float(alpha),
        "alpha_kg_per_person_per_year": float(alpha * SECS_PER_YR / 1e9),
        "h_min_m": transport.HMIN,
        "spinup_hours": SPINUP_HRS,
        "stations": stations["station"].tolist(),
        "n_train_days": int(istrain.sum()),
        "n_test_days": int((~istrain).sum()),
        "train": scores(y[istrain], predtr),
        "train_background_only": scores(y[istrain], nulltr, withinday=False),
        "test": scores(y[~istrain], predte),
        "test_background_only": scores(y[~istrain], nullte, withinday=False),
        "background_mean_ug_m3": {"train": float(b_tr.mean()), "test": float(b_te.mean())},
    }
    with open(config.processed_dir(city) / f"{city}_calibration.json", "w") as f:
        json.dump(result, f, indent=2)

    out = config.output_dir(city, "calibration")
    grid.to_csv(out / "grid_search.csv", index=False)
    bdf = daysdf[["date", "regime", "split"]].copy()
    ball = np.empty(len(days))
    ball[istrain], ball[~istrain] = b_tr, b_te
    bdf["background_ug_m3"] = ball
    bdf.to_csv(out / "daily_background.csv", index=False)
    plot(city, y, istrain, predtr, predte, nullte, stations, out)

    print(grid.round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k != "stations"}, indent=2))


def plot(city, y, istrain, predtr, predte, nullte, stations, out):    # draws observed vs model scatter plots for training days, test days, and background-only benchmark
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, yy, pp, title in [
        (axes[0], y[istrain], predtr, "train (2021-24)"),
        (axes[1], y[~istrain], predte, "test (2025), full model"),
        (axes[2], y[~istrain], nullte, "test (2025), background only"),
    ]:
        ok = np.isfinite(yy)
        ax.scatter(yy[ok], pp[ok], s=3, alpha=0.3)
        hi = np.nanpercentile(yy, 99.5)
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        s = scores(yy, pp, withinday="background" not in title)
        rtxt = f", within-day r {s['r_within_day']:.2f}" if "r_within_day" in s else ""
        ax.set_title(f"{title}\nRMSE {s['rmse']:.2f}{rtxt}")
        ax.set_xlabel("observed PM2.5 (ug/m3)")
        ax.set_ylabel("model PM2.5 (ug/m3)")
    fig.suptitle(f"{city}: hourly PM2.5 at {len(stations)} stations")
    fig.tight_layout()
    fig.savefig(out / "obs_vs_model.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cities, kgrid = parse_args("Calibrate the transport model against real PM2.5.")
    for city in cities:
        calibrate(city, kgrid)
