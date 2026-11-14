import argparse
import importlib.util
import json
import sys
from pathlib import Path
import dask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit

dask.config.set(scheduler="synchronous")

ROOT = Path(__file__).resolve().parents[3]
P4 = ROOT / "Workflow" / "Phase 4" / "Python files"
P5 = ROOT / "Workflow" / "Phase 5" / "Python files"
sys.path.insert(0, str(P4))
import networks as N  
_spec = importlib.util.spec_from_file_location("cfg4", P4 / "config.py")
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)
_spec = importlib.util.spec_from_file_location("eval5", P5 / "01_evaluate.py")
ev5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev5)

OUTDATA = ROOT / "data" / "processed" / "phase7"
OUT_FIG = ROOT / "outputs" / "phase7"
N_BOOT = ev5.N_BOOT
K_TAG = ""  
HEADLINE = ev5.HEADLINE
AREA_KM2 = {"singapore": 655.3, "melbourne": 2611.2}      
STEP = {"singapore": 1.0, "melbourne": AREA_KM2["melbourne"] / AREA_KM2["singapore"]}
THRESHOLDS = [0.005, 0.01, 0.02]
MAIN_THRESH = 0.01
N_GRID = np.arange(5, 20001, dtype=float)                 
SLOPE_RNG = {"singapore": {"full": (5, 40)},
                "melbourne": {"full": (5, 160), "equal_count_5-40": (5, 40),
                              "equal_density_20-160": (20, 160)}}
SAVED_AT = {"singapore": [8, 22, 40], "melbourne": [8, 22, 32, 40, 88, 160]}
GAIN_AT = {"singapore": [8, 22, 40], "melbourne": [8, 22, 88, 160]}  
BANDS = {"singapore": [(5, 10), (11, 20), (21, 40)],                 
         "melbourne": [(5, 10), (11, 20), (21, 40), (41, 80), (81, 160)]}



# data


def setting_original(city):
    m = N.load_maps(city, testset="all2025", trainset="all", k_tag=K_TAG)
    kj = json.load(open(cfg.processed_dir(city) / f"{city}_kriging_trainall{K_TAG}.json"))
    nets = pd.read_csv(cfg.processed_dir(city) / f"{city}_networks_trainall{K_TAG}.csv")
    greedy = nets[(nets["strategy"] == "greedy") & (nets["noise"] == HEADLINE)]
    return {"cells": m["cells"], "test": m["test"], "vg": kj["variogram"],
            "sd": kj["noise_sd_ug_m3"][HEADLINE],
            "greedy": {n: g.sort_values("rank")["cell_index"].tolist() for n, g in greedy.groupby("N")}}


def setting_L0(city, cells):
    packed = []
    for period in ("train", "test2025"):
        ds = xr.open_dataset(cfg.P3_DIR / city / f"{city}_ingredients_{period}{K_TAG}.nc")
        meta = pd.DataFrame({"day": np.arange(ds.sizes["day"]), "member": 0,
                             "date": ds["date"].values, "regime": ds["regime"].values.astype(str),
                             "w_k": ds["w_k"].values})
        assert np.array_equal(ds["cell_id"].values, cells["cell_id"].values)
        packed.append(N._pack(ds["response"].values[:, 0].astype(float), meta))
        ds.close()
    train, test = packed
    Sigma = N.second_moment(train["X"], train["w"])
    vg = N.fit_variogram(cells, Sigma)
    sd = cfg.MAIN_NOISE * N.spatial_sd(train["X"], train["w"])
    order, _ = N.greedy_rmse(N.covariance(cells, vg), Sigma, vg, sd**2, cfg.N_MAX[city])
    return {"cells": cells, "test": test, "vg": vg, "sd": sd,
            "greedy": {n: list(order[:n]) for n in range(cfg.N_MIN, cfg.N_MAX[city] + 1)}}


def per_map_mse(s, kmeans):
    K = N.covariance(s["cells"], s["vg"])
    extra = N.sensor_variance(s["vg"], s["sd"]**2)
    X = s["test"]["X"]
    nvals = sorted(s["greedy"])
    out = {}
    for strat, nets in (("greedy", s["greedy"]), ("kmeans", kmeans)):
        out[strat] = np.column_stack(
            [N.map_mse(X, nets[n], N.ok_weights(K, nets[n], extra), s["sd"]**2) for n in nvals])
    return np.array(nvals), out



# stats


def power_law(n, a, b, c):
    return a * n**(-b) + c


def fit_curve(n, rmse, p0=None):
    p0 = p0 if p0 is not None else (rmse[0] * n[0]**0.5, 0.5, 0.5 * rmse[-1])
    try:
        p, _ = curve_fit(power_law, n, rmse, p0=p0, bounds=([0, 0, 0], [np.inf, 5, np.inf]),
                         maxfev=20000)
    except RuntimeError:
        return None
    return p


def n_star(p, threshold, step):
    a, b, c = p
    g = a * b * N_GRID**(-b - 1) / power_law(N_GRID, a, b, c)
    hit = np.nonzero(step * g < threshold)[0]
    return float(N_GRID[hit[0]]) if len(hit) else float("inf")


def marginal_gain(p, n):
    a, b, c = p
    return float(a * b * n**(-b - 1) / power_law(n, a, b, c))


def kmeans_equivalent(p, target):
    a, b, c = p
    return float((a / (target - c))**(1 / b)) if target > c else float("inf")


def slope(n, imp, lo, hi):
    keep = (n >= lo) & (n <= hi)
    return float(np.polyfit(np.log(n[keep]), imp[keep], 1)[0])


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.percentile(x, [2.5, 97.5]).tolist() if len(x) else [None, None]


def analyse(city, setting, s, kmeans):
    print(f"=== {city} / {setting}")
    n, mse = per_map_mse(s, kmeans)
    w, meta = s["test"]["w"], s["test"]["meta"]
    idx = ev5.bootstrap_index(meta, N_BOOT, [cfg.SEED, 5, len(city)])
    B = np.stack([np.bincount(i, minlength=len(w)) for i in idx]).astype(float) * w

    rm = {k: np.sqrt(w @ v) for k, v in mse.items()}                       
    rm_b = {k: np.sqrt((B @ v) / B.sum(1, keepdims=True)) for k, v in mse.items()}  
    imp, impb = 1 - rm["greedy"] / rm["kmeans"], 1 - rm_b["greedy"] / rm_b["kmeans"]
    ci = np.percentile(impb, [2.5, 97.5], axis=0)

    out = {"variogram_range_km": s["vg"]["range_m"] / 1000, "noise_sd": float(s["sd"]),
           "curve_fit": {}, "n_star": {}, "slopes": {}}
    fits = {}
    for strat in ("greedy", "kmeans"):
        p = fit_curve(n, rm[strat])
        fits[strat] = p
        resid = rm[strat] - power_law(n, *p)
        out["curve_fit"][strat] = {"a": p[0], "b": p[1], "c": p[2],
                                   "r2": float(1 - resid.var() / rm[strat].var()),
                                   "max_abs_resid": float(np.abs(resid).max())}
        bootp = [fit_curve(n, r, p0=p) for r in rm_b[strat]]
        nfail = sum(q is None for q in bootp)
        res = {}
        for basis, step in (("per_sensor", 1.0), ("area_fair", STEP[city])):
            for t in THRESHOLDS:
                pt = n_star(p, t, step)
                bs = [n_star(q, t, step) for q in bootp if q is not None]
                res[f"{basis}_{t:g}"] = {
                    "N_star": pt, "ci95": pct(bs),
                    "share_boot_beyond_grid": float(np.mean(np.isinf(bs))),
                    "beyond_sweep": bool(pt > cfg.N_MAX[city]),
                    "density_per_100km2": pt / AREA_KM2[city] * 100,
                    "density_ci95": [x / AREA_KM2[city] * 100 if x is not None else None for x in pct(bs)],
                    "spacing_km": float(np.sqrt(AREA_KM2[city] / pt))}
        res["boot_fit_failures"] = nfail
        out["n_star"][strat] = res
        print(f"  {strat}: b = {p[1]:.2f}, N*(1%) = {res['per_sensor_0.01']['N_star']:.0f} "
              f"{res['per_sensor_0.01']['ci95']}, area-fair {res['area_fair_0.01']['N_star']:.0f}")

    for name, (lo, hi) in SLOPE_RNG[city].items():
        b1 = slope(n, imp, lo, hi)
        bs = [slope(n, row, lo, hi) for row in impb]
        out["slopes"][name] = {"N_range": [lo, hi], "beta1": b1, "ci95": pct(bs),
                               "per_doubling_pp": 100 * b1 * np.log(2)}
        print(f"  slope {name}: {100 * b1:+.2f} pp per unit ln N {pct(100 * np.array(bs))}")

    out["ci_counts"] = {"above_zero": int((ci[0] > 0).sum()), "below_zero": int((ci[1] < 0).sum()),
                        "straddle": int(((ci[0] <= 0) & (ci[1] >= 0)).sum()), "total": len(n)}
    out["sensors_saved"] = {}
    for m in SAVED_AT[city]:
        j = int(np.nonzero(n == m)[0][0])
        eq = kmeans_equivalent(fits["kmeans"], rm["greedy"][j])
        out["sensors_saved"][str(m)] = {"saved": eq - m, "kmeans_equivalent_N": eq,
                                        "extrapolated": bool(eq > cfg.N_MAX[city])}
    out["marginal_gain_pct"] = {strat: {str(m): 100 * marginal_gain(fits[strat], m)
                                        for m in GAIN_AT[city]} for strat in fits}
    out["band_mean_improvement_descriptive"] = {}
    for lo, hi in BANDS[city]:
        keep = (n >= lo) & (n <= hi)
        out["band_mean_improvement_descriptive"][f"N{lo}-{hi}"] = [
            float(imp[keep].mean()), *pct(impb[:, keep].mean(1))]

    curves = pd.DataFrame({"N": n, "rmse_greedy": rm["greedy"], "rmse_kmeans": rm["kmeans"],
                           "fit_greedy": power_law(n, *fits["greedy"]),
                           "fit_kmeans": power_law(n, *fits["kmeans"]),
                           "improvement": imp, "ci_low": ci[0], "ci_high": ci[1]})
    curves.to_csv(OUTDATA / f"{city}_{setting}_curves{K_TAG}.csv", index=False)
    return out, curves


def compare(res):
    out = {}
    for setting in res["singapore"]:
        out[setting] = {}
        for strat in ("greedy", "kmeans"):
            sg = res["singapore"][setting]["n_star"][strat][f"area_fair_{MAIN_THRESH:g}"]
            mel = res["melbourne"][setting]["n_star"][strat][f"area_fair_{MAIN_THRESH:g}"]
            out[setting][strat] = {"density_singapore": sg["density_per_100km2"],
                                   "density_melbourne": mel["density_per_100km2"],
                                   "density_ci_singapore": sg["density_ci95"],
                                   "density_ci_melbourne": mel["density_ci95"]}
    return out


def plot(curves, scores):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, city in enumerate(cfg.ALL_CITIES):
        ax = axes[0, col]
        c = curves[(city, "original")]
        for strat, colour in (("greedy", "#c0392b"), ("kmeans", "#2c3e50")):
            ax.plot(c["N"], c[f"rmse_{strat}"], ".", color=colour, ms=3, label=f"{strat} (test)")
            ax.plot(c["N"], c[f"fit_{strat}"], "-", color=colour, lw=1, label=f"{strat} fit")
        sc = scores[city]
        for strat, style in (("qr", "--"), ("random_mean", ":")):
            q = sc[(sc["strategy"] == strat) & (sc["noise"] == HEADLINE)].sort_values("N")
            ax.plot(q["N"], q["test_rmse"], style, color="grey", lw=1, label=strat)
        ax.set_ylim(0, 1.3 * c["rmse_kmeans"].max())
        ax.set_title(f"{city}: test RMSE vs N (original design, f = 0.25)")
        ax.set_xlabel("number of sensors N")
        ax.set_ylabel("RMSE (ug/m3)")
        ax.legend(fontsize=7)

        ax = axes[1, col]
        for setting, colour in (("original", "#c0392b"), ("L0", "#2471a3")):
            c = curves[(city, setting)]
            ax.plot(c["N"], 100 * c["improvement"], color=colour, label=setting)
            ax.fill_between(c["N"], 100 * c["ci_low"], 100 * c["ci_high"], color=colour, alpha=0.15)
            b = np.polyfit(np.log(c["N"]), c["improvement"], 1)
            ax.plot(c["N"], 100 * np.polyval(b, np.log(c["N"])), "--", color=colour, lw=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{city}: greedy improvement over k-means (95% CI, ln N fit)")
        ax.set_xlabel("number of sensors N")
        ax.set_ylabel("improvement (%)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / f"budget_sweep{K_TAG}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 7 budget sweep.")
    ap.add_argument("--k", type=float, default=None,
                    help="Phase 8: K (m2/s) of the simulations to use")
    k = ap.parse_args().k
    K_TAG = "" if k is None else f"_K{k:g}"
    OUTDATA.mkdir(parents=True, exist_ok=True)
    results, curves, scores = {}, {}, {}
    for city in cfg.ALL_CITIES:
        nets = pd.read_csv(cfg.processed_dir(city) / f"{city}_networks_trainall.csv")
        km = nets[nets["strategy"] == "kmeans"]
        kmeans = {n: g.sort_values("rank")["cell_index"].tolist() for n, g in km.groupby("N")}
        scores[city] = pd.read_csv(ROOT / "data" / "processed" / "phase5" / city
                                   / f"{city}_test_scores_all2025_trainall{K_TAG}.csv")
        orig = setting_original(city)
        results[city] = {}
        for setting, s in (("original", orig), ("L0", setting_L0(city, orig["cells"]))):
            results[city][setting], curves[(city, setting)] = analyse(city, setting, s, kmeans)
    results["city_comparison_area_fair"] = compare(results)
    results["K_m2_s"] = k
    with open(OUTDATA / f"phase7_budget_sweep{K_TAG}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    plot(curves, scores)
    print(json.dumps(results["city_comparison_area_fair"], indent=2, default=float))
