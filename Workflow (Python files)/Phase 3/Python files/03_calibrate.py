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

K_DIAGNOSTIC_GRID = [100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]   # m2 s-1 for k-grid only
LAM_GRID_HOURS = [np.inf, 24.0, 12.0, 6.0, 3.0, 1.5]            # e-folding time, h
SPINUP_HOURS = 24
SECONDS_PER_YEAR = 365.25 * 86400


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", choices=config.ALL_CITIES)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--k-grid", action="store_true",
                        help="diagnostic: also search K over K_DIAGNOSTIC_GRID")
    args = parser.parse_args()
    cities = config.ALL_CITIES if args.all else [args.city]
    k_grid = K_DIAGNOSTIC_GRID if args.k_grid else [config.TRANSPORT_K_M2_S]
    return cities, k_grid


def lam_per_second(hours):
    return 0.0 if np.isinf(hours) else 1.0 / (hours * 3600.0)


def station_cells(city, domain):
    st = pd.read_csv(config.processed_dir(city) / f"{city}_pm25_stations.csv")
    st = st[st["in_domain"]].reset_index(drop=True)
    dx = domain.attrs["cell_size_m"]
    x0 = float(domain.x[0]) - dx / 2
    y_top = float(domain.y[0]) + dx / 2
    st["col"] = np.floor((st["x"] - x0) / dx).astype(int)
    st["row"] = np.floor((y_top - st["y"]) / dx).astype(int)
    return st


def simulate_days(city, domain, days, stations, offset_h, k_grid):
    """Station hour-means for every (day, K, tracer). Shape (D, nK, 24, ntr, nst)."""
    dx = domain.attrs["cell_size_m"]
    pop = domain["population"].values
    ny, nx = pop.shape
    lams = np.array([lam_per_second(h) for h in LAM_GRID_HOURS])
    nl = len(lams)

    emis = np.zeros((2 * nl, ny, nx))
    emis[:nl] = pop / dx**2                      # alpha = 1 ug s-1 person-1
    bg = np.r_[np.zeros(nl), np.ones(nl)]
    c0 = bg[:, None, None] * np.ones((1, ny, nx))
    lam_tr = np.r_[lams, lams]
    cells = (stations["row"].to_numpy(), stations["col"].to_numpy())

    out = np.empty((len(days), len(k_grid), 24, 2 * nl, len(stations)))
    for d, date in enumerate(days):
        local_start = pd.Timestamp(date)
        utc_start = local_start - pd.Timedelta(hours=offset_h) - pd.Timedelta(hours=SPINUP_HOURS)
        utc_end = utc_start + pd.Timedelta(hours=SPINUP_HOURS + 24)
        met = transport.load_met(city, config.PHASE2_DIR, utc_start, utc_end)
        fields = transport.met_to_domain(met, domain)
        for k, K in enumerate(k_grid):
            r = transport.run(fields, dx, emis, lam_tr, K, bg, c0, sample_cells=cells)
            out[d, k] = r["samples"][SPINUP_HOURS:]
        print(f"  simulated {d + 1}/{len(days)} {local_start.date()}", end="\r")
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


def fit(sim_e, sim_b, y, alpha=None):
    D = y.shape[0]
    ok = np.isfinite(y)
    rows_day = np.broadcast_to(np.arange(D)[:, None, None], y.shape)[ok]
    B = np.zeros((ok.sum(), D))
    B[np.arange(ok.sum()), rows_day] = sim_b[ok]
    target = y[ok]
    if alpha is None:
        A = np.column_stack([sim_e[ok], B])
        coef, _ = nnls(A, target)
        alpha, b = coef[0], coef[1:]
    else:
        b, _ = nnls(B, target - alpha * sim_e[ok])
    pred = alpha * sim_e + b[:, None, None] * sim_b
    return alpha, b, pred


def background_only(y):
    return np.broadcast_to(np.nanmean(y, axis=(1, 2), keepdims=True), y.shape)


def scores(y, pred, within_day=True):
    ok = np.isfinite(y)
    err = pred[ok] - y[ok]
    out = {"rmse": float(np.sqrt(np.mean(err**2))), "bias": float(err.mean()),
           "n_obs": int(ok.sum())}
    if within_day:
        y_anom = y - np.nanmean(y, axis=(1, 2), keepdims=True)
        p_anom = pred - np.nanmean(np.where(ok, pred, np.nan), axis=(1, 2), keepdims=True)
        out["r_within_day"] = float(np.corrcoef(y_anom[ok], p_anom[ok])[0, 1])
    return out


def calibrate(city, k_grid):
    print(f"=== {city}")
    cfg = config.city_config(city)
    domain = xr.open_dataset(config.processed_dir(city) / f"{city}_domain.nc")
    stations = station_cells(city, domain)
    days_df = pd.read_csv(config.PHASE2_DIR / city / f"{city}_phase3_days.csv",
                          parse_dates=["date"])
    days = days_df["date"].tolist()
    is_train = (days_df["split"] == "train").to_numpy()

    sims, nl = simulate_days(city, domain, days, stations, cfg["utc_offset_hours"], k_grid)
    y = observations(city, days, stations)

    rows = []
    for k, K in enumerate(k_grid):
        for j, lam_h in enumerate(LAM_GRID_HOURS):
            se, sb = sims[:, k, :, j, :], sims[:, k, :, nl + j, :]
            a, _, pred = fit(se[is_train], sb[is_train], y[is_train])
            s = scores(y[is_train], pred)
            rows.append({"K_m2_s": K, "loss_efold_h": lam_h, "alpha": a, **s})
    grid = pd.DataFrame(rows)
    if len(k_grid) > 1:
        grid.to_csv(config.output_dir(city, "calibration") / "grid_search_K_diagnostic.csv",
                    index=False)
        print(grid.round(4).to_string(index=False))
        return
    best = grid.loc[grid["rmse"].idxmin()]
    k = k_grid.index(best["K_m2_s"])
    j = LAM_GRID_HOURS.index(best["loss_efold_h"])
    se, sb = sims[:, k, :, j, :], sims[:, k, :, nl + j, :]

    alpha, b_tr, pred_tr = fit(se[is_train], sb[is_train], y[is_train])
    _, b_te, pred_te = fit(se[~is_train], sb[~is_train], y[~is_train], alpha=alpha)
    null_tr = background_only(y[is_train])
    null_te = background_only(y[~is_train])
    # at lam = 0 the background tracer is 1 everywhere, so the model defines background-only 
    # as the case alpha = 0. hence, its training RMSE cannot be worse
    
    assert scores(y[is_train], pred_tr)["rmse"] <= scores(y[is_train], null_tr, False)["rmse"] + 1e-6, \
        "full model fits worse than background-only on train days"

    result = {
        "city": city,
        "K_m2_s": float(best["K_m2_s"]),
        "K_fixed": len(k_grid) == 1,
        "loss_efold_hours": None if np.isinf(best["loss_efold_h"]) else float(best["loss_efold_h"]),
        "alpha_ug_per_s_per_person": float(alpha),
        "alpha_kg_per_person_per_year": float(alpha * SECONDS_PER_YEAR / 1e9),
        "h_min_m": transport.H_MIN_M,
        "spinup_hours": SPINUP_HOURS,
        "stations": stations["station"].tolist(),
        "n_train_days": int(is_train.sum()),
        "n_test_days": int((~is_train).sum()),
        "train": scores(y[is_train], pred_tr),
        "train_background_only": scores(y[is_train], null_tr, within_day=False),
        "test": scores(y[~is_train], pred_te),
        "test_background_only": scores(y[~is_train], null_te, within_day=False),
        "background_mean_ug_m3": {"train": float(b_tr.mean()), "test": float(b_te.mean())},
    }
    with open(config.processed_dir(city) / f"{city}_calibration.json", "w") as f:
        json.dump(result, f, indent=2)

    out = config.output_dir(city, "calibration")
    grid.to_csv(out / "grid_search.csv", index=False)
    bdf = days_df[["date", "regime", "split"]].copy()
    b_all = np.empty(len(days))
    b_all[is_train], b_all[~is_train] = b_tr, b_te
    bdf["background_ug_m3"] = b_all
    bdf.to_csv(out / "daily_background.csv", index=False)
    plot(city, y, is_train, pred_tr, pred_te, null_te, stations, out)

    print(grid.round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in result.items() if k != "stations"}, indent=2))


def plot(city, y, is_train, pred_tr, pred_te, null_te, stations, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, yy, pp, title in [
        (axes[0], y[is_train], pred_tr, "train (2021-24)"),
        (axes[1], y[~is_train], pred_te, "test (2025), full model"),
        (axes[2], y[~is_train], null_te, "test (2025), background only"),
    ]:
        ok = np.isfinite(yy)
        ax.scatter(yy[ok], pp[ok], s=3, alpha=0.3)
        hi = np.nanpercentile(yy, 99.5)
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        s = scores(yy, pp, within_day="background" not in title)
        r_txt = f", within-day r {s['r_within_day']:.2f}" if "r_within_day" in s else ""
        ax.set_title(f"{title}\nRMSE {s['rmse']:.2f}{r_txt}")
        ax.set_xlabel("observed PM2.5 (ug/m3)")
        ax.set_ylabel("model PM2.5 (ug/m3)")
    fig.suptitle(f"{city}: hourly PM2.5 at {len(stations)} stations")
    fig.tight_layout()
    fig.savefig(out / "obs_vs_model.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cities, k_grid = parse_args("Calibrate the transport model against real PM2.5.")
    for city in cities:
        calibrate(city, k_grid)
