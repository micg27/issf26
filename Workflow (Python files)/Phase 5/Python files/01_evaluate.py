"""
scores all networks on 2025 (test) maps: main experiment
comparisons run at each city's own network size, equal sensor count, and equal sensor density
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[3]
P4 = PROJ_ROOT / "Workflow" / "Phase 4" / "Python files"
sys.path.insert(0, str(P4))
import networks as N 

_spec = importlib.util.spec_from_file_location("phase4_config", P4 / "config.py")
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

OUTDATA = PROJ_ROOT / "data" / "processed" / "phase5"
OUT_FIG = PROJ_ROOT / "outputs" / "phase5"
TESTSET = "selected"        
TRAINSET = "selected"       
SUFFIX = ""
NET_SUF = ""
K_TAG = ""                   
N_BOOT = 2000
RAND_STEP = 5           
HEADLINE = f"f{cfg.MAIN_NOISE:g}"


def city_dirs(city):
    d, f = OUTDATA / city, OUT_FIG / (city + SUFFIX)
    d.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    return d, f


def bootstrap_index(meta, nboot, seed):       # repeats comparison choosing days with replacement 2000 times to give 2000 unique testing sets for CIs
    """Stratified day bootstrap -> list of map-index arrays."""
    rng = np.random.default_rng(seed)
    regimedays = {r: g["day"].unique() for r, g in meta.groupby("regime")}
    daymaps = {d: np.nonzero(meta["day"].to_numpy() == d)[0] for d in meta["day"].unique()}
    out = []
    for _ in range(nboot):
        idx = []
        for r, days in regimedays.items():
            for d in rng.choice(days, len(days), replace=True):
                idx.append(daymaps[d])
        out.append(np.concatenate(idx))
    return out


def weighted_rmse(mse, w, idx=None): 
    if idx is None:
        return float(np.sqrt(np.sum(w * mse)))
    ww = w[idx]
    return float(np.sqrt(np.sum(ww * mse[idx]) / ww.sum()))


def evaluate(city):     # scores every network (all strategies, N and noise levels) on the test maps, plus 200 random networks. 
                        # computes improvement = 1 − greedy/k-means with bootstrap CIs, per-type results, and how many even-spread sensors equal greedy's 22 or 8.
    print(f"=== {city}")
    datadir, figdir = city_dirs(city)
    maps = (N.load_maps(city, TESTSET, "all", k_tag=K_TAG) if K_TAG
            else N.load_maps(city, TESTSET))
    te, cells = maps["test"], maps["cells"]
    X, w, meta = te["X"], te["w"], te["meta"]
    C = X.shape[1]
    p4 = cfg.processed_dir(city)
    nets = pd.read_csv(p4 / f"{city}_networks{NET_SUF}.csv")
    kj = json.load(open(p4 / f"{city}_kriging{NET_SUF}.json"))
    vg = kj["variogram"]
    K = N.covariance(cells, vg)
    noise = kj["noise_sd_ug_m3"]
    print(f"  {len(X)} test maps over {meta['day'].nunique()} days")

    def per_map_mse(S, sd):
        W = N.ok_weights(K, S, N.sensor_variance(vg, sd**2))
        return N.map_mse(X, S, W, sd**2)

    rows, msestore = [], {}
    for label, sd in noise.items():
        for strat in ("greedy", "qr", "kmeans", "existing"):
            sub = nets[(nets["strategy"] == strat)
                       & (nets["noise"].isin([label] if strat == "greedy" else ["any"]))]
            for n, g in sub.groupby("N"):
                S = g.sort_values("rank")["cell_index"].tolist()
                mse = per_map_mse(S, sd)
                msestore[(strat, label, n)] = mse
                rows.append((strat, label, n, weighted_rmse(mse, w)))
        print(f"  scored {label}")
    sd = noise[HEADLINE]
    nvals = sorted(set(range(cfg.N_MIN, cfg.N_MAX[city] + 1, RAND_STEP))
                    | {cfg.MAIN_N[city]})
    for n in nvals:
        vals = [weighted_rmse(per_map_mse(S, sd), w)
                for S in N.random_networks(C, n, cfg.N_RAND_NETS, cfg.SEED)]
        rows.append(("random_mean", HEADLINE, n, float(np.mean(vals))))
        rows.append(("random_p05", HEADLINE, n, float(np.percentile(vals, 5))))

    scores = pd.DataFrame(rows, columns=["strategy", "noise", "N", "test_rmse"])
    scores.to_csv(datadir / f"{city}_test_scores{SUFFIX}.csv", index=False)

    boots = bootstrap_index(meta, N_BOOT, [cfg.SEED, 5, len(city)])
    improws = []
    for label in noise:
        for n in range(cfg.N_MIN, cfg.N_MAX[city] + 1):
            g, k = msestore[("greedy", label, n)], msestore[("kmeans", label, n)]
            point = 1 - weighted_rmse(g, w) / weighted_rmse(k, w)
            bs = np.array([1 - weighted_rmse(g, w, i) / weighted_rmse(k, w, i) for i in boots])
            improws.append((label, n, point, *np.percentile(bs, [2.5, 97.5]), bs.std()))
    imp = pd.DataFrame(improws, columns=["noise", "N", "improvement", "ci_low", "ci_high", "boot_sd"])
    imp.to_csv(datadir / f"{city}_improvement{SUFFIX}.csv", index=False)

    n0 = cfg.MAIN_N[city]
    g0, k0 = msestore[("greedy", HEADLINE, n0)], msestore[("kmeans", HEADLINE, n0)]
    byregime = {}
    for r, sub in meta.groupby("regime"):
        i = sub.index.to_numpy()
        byregime[r] = 1 - np.sqrt(np.mean(g0[i])) / np.sqrt(np.mean(k0[i]))

    sc = scores[scores["noise"] == HEADLINE]
    target = sc[(sc["strategy"] == "greedy") & (sc["noise"] == HEADLINE) & (sc["N"] == n0)]["test_rmse"].item()
    km = sc[sc["strategy"] == "kmeans"].sort_values("N")
    match = km[km["test_rmse"] <= target]
    equiv = int(match["N"].iloc[0]) if len(match) else None

    plot(city, scores, imp, figdir)
    return {"scores": scores, "imp": imp, "boots": boots, "mse": msestore, "w": w,
            "by_regime": byregime, "equiv_kmeans_N": equiv, "meta": meta}


def improvement_draws(res, label, n):
    g, k, w = res["mse"][("greedy", label, n)], res["mse"][("kmeans", label, n)], res["w"]
    return np.array([1 - weighted_rmse(g, w, i) / weighted_rmse(k, w, i) for i in res["boots"]])


def compare(sg, mel): 
    """Melbourne minus Singapore improvement, with CIs (independent bootstraps)."""
    pairs = {
        "own_network_size": (cfg.MAIN_N["singapore"], cfg.MAIN_N["melbourne"]),
        "equal_count_N8": (8, 8),
        "equal_count_N22": (22, 22),
        "equal_count_N40": (40, 40),
        "equal_density_SG8_MEL32": (8, 32),
        "equal_density_SG22_MEL88": (22, 88),
        "equal_density_SG40_MEL160": (40, 160),
    }
    out = {}
    for name, (ns, nm) in pairs.items():
        ds, dm = improvement_draws(sg, HEADLINE, ns), improvement_draws(mel, HEADLINE, nm)
        def pt(res, n):
            row = res["imp"][(res["imp"]["noise"] == HEADLINE) & (res["imp"]["N"] == n)]
            return float(row["improvement"].item())
        diff = dm - ds
        out[name] = {"N_singapore": ns, "N_melbourne": nm,
                     "improvement_singapore": pt(sg, ns), "improvement_melbourne": pt(mel, nm),
                     "difference_mel_minus_sg": pt(mel, nm) - pt(sg, ns),
                     "difference_ci95": np.percentile(diff, [2.5, 97.5]).tolist()}
    return out


BANDS = {"singapore": [(5, 10), (11, 20), (21, 40)],
         "melbourne": [(5, 10), (11, 20), (21, 40), (41, 80), (81, 160)]}


def band_means(city, res):    # average improvement over each band of N values with CIs
    out = {}
    for lo, hi in BANDS[city]:
        draws = np.mean([improvement_draws(res, HEADLINE, n) for n in range(lo, hi + 1)], axis=0)
        imp = res["imp"]
        pt = imp[(imp["noise"] == HEADLINE) & imp["N"].between(lo, hi)]["improvement"].mean()
        out[f"N{lo}-{hi}"] = [float(pt), *np.percentile(draws, [2.5, 97.5]).tolist()]
    return out


def plot(city, scores, imp, figdir):  # plots error against N for each strategy and improvement within CI 
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for strat, style in [("greedy", "-"), ("kmeans", "-"), ("qr", "--"), ("random_mean", ":")]:
        s = scores[(scores["strategy"] == strat) & (scores["noise"] == HEADLINE)].sort_values("N")
        ax.plot(s["N"], s["test_rmse"], style, label=strat)
    ex = scores[(scores["strategy"] == "existing") & (scores["noise"] == HEADLINE)]
    if len(ex):
        ax.plot(ex["N"], ex["test_rmse"], "k*", ms=10, label="existing EPA network")
    ax.axvline(cfg.MAIN_N[city], color="grey", ls=":", lw=1)
    km = scores[(scores["strategy"] == "kmeans") & (scores["noise"] == HEADLINE)]["test_rmse"]
    ax.set_ylim(0, 1.6 * km.max())
    ax.set_xlabel("number of sensors N")
    ax.set_ylabel("test RMSE, 2025 days (ug/m3)")
    ax.set_title(f"{city}: reconstruction error (noise f = {cfg.MAIN_NOISE})")
    ax.legend(fontsize=8)

    ax = axes[1]
    s = imp[imp["noise"] == HEADLINE]
    ax.plot(s["N"], 100 * s["improvement"], color="#c0392b")
    ax.fill_between(s["N"], 100 * s["ci_low"], 100 * s["ci_high"], color="#c0392b", alpha=0.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(cfg.MAIN_N[city], color="grey", ls=":", lw=1)
    ax.set_xlabel("number of sensors N")
    ax.set_ylabel("greedy improvement over k-means (%)")
    ax.set_title("weather-informed vs evenly spread (95% CI)")
    fig.tight_layout()
    fig.savefig(figdir / "test_results.png", dpi=130)
    plt.close(fig)


def summarize(city, res):
    imp, sc = res["imp"], res["scores"]
    n0 = cfg.MAIN_N[city]
    h = imp[(imp["noise"] == HEADLINE) & (imp["N"] == n0)].iloc[0]
    def rm(strat, noise, n):
        r = sc[(sc["strategy"] == strat) & (sc["noise"] == noise) & (sc["N"] == n)]
        return float(r["test_rmse"].item()) if len(r) else None
    out = {
        "headline_N": n0,
        "test_rmse": {s: rm(s, HEADLINE, n0) for s in ("greedy", "kmeans", "qr", "random_mean")},
        "improvement_greedy_vs_kmeans": float(h["improvement"]),
        "ci95": [float(h["ci_low"]), float(h["ci_high"])],
        "improvement_by_regime": {k: float(v) for k, v in res["by_regime"].items()},
        "equivalent_kmeans_N": res["equiv_kmeans_N"],
        "improvement_by_noise": {
            r["noise"]: [float(r["improvement"]), float(r["ci_low"]), float(r["ci_high"])]
            for _, r in imp[imp["N"] == n0].iterrows()},
    }
    if city == "melbourne":
        ex = sc[(sc["strategy"] == "existing") & (sc["noise"] == HEADLINE)]
        out["test_rmse"]["existing"] = float(ex["test_rmse"].item())
        out["existing_N"] = int(ex["N"].item())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 5 evaluation.")
    ap.add_argument("--test-set", choices=["selected", "all2025"], default="selected",
                    help="selected = pre-registered 18-19 test days; "
                         "all2025 = post-results extension to every 2025 day")
    ap.add_argument("--train-set", choices=["selected", "all"], default="selected",
                    help="which Phase 4 networks: chosen from the sampled training days "
                         "(pre-registered) or from every 2021-2024 day (extension)")
    ap.add_argument("--k", type=float, default=None,
                    help="Phase 8: K (m2/s) of the simulations; needs --test-set all2025 --train-set all")
    args = ap.parse_args()
    TESTSET, TRAINSET = args.test_set, args.train_set
    if args.k is not None and (TESTSET, TRAINSET) != ("all2025", "all"):
        ap.error("--k needs --test-set all2025 --train-set all")
    K_TAG = "" if args.k is None else f"_K{args.k:g}"
    NET_SUF = ("" if TRAINSET == "selected" else "_trainall") + K_TAG
    SUFFIX = ("" if TESTSET == "selected" else "_all2025") + NET_SUF
    res = {c: evaluate(c) for c in cfg.ALL_CITIES}
    results = {c: summarize(c, r) for c, r in res.items()}
    for c, r in res.items():
        results[c]["band_mean_improvement_descriptive"] = band_means(c, r)
    results["city_comparison"] = compare(res["singapore"], res["melbourne"])
    OUTDATA.mkdir(parents=True, exist_ok=True)
    results["test_set"] = TESTSET
    results["train_set"] = TRAINSET
    results["K_m2_s"] = args.k
    with open(OUTDATA / f"phase5_results{SUFFIX}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
