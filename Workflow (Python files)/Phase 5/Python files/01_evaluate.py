# 01_evaluate.py
"""
Phase 5: score every Phase 4 network on the held-out 2025 test maps.

Networks were chosen from 2021-2024 training maps only, so this is an
out-of-sample test. Every network is scored with the same ordinary-kriging
reconstruction and exact expected error under sensor noise (networks.py).

Headline (pre-registered in METHODS.md, "Phase 4-5 design"):
    relative improvement = (RMSE_kmeans - RMSE_greedy) / RMSE_kmeans
at each city's headline N and headline noise (f = 0.25), with a 95% CI
from a stratified bootstrap over test days: days are resampled with
replacement within each regime, so every regime keeps its weight w_k.
Greedy and k-means are scored on the same resampled days (paired).

The city comparison is reported three ways, as pre-registered:
    own network size   Singapore N = 22 vs Melbourne N = 8
    equal count        both cities at the same N
    equal density      Melbourne N ~= 4 x Singapore N (candidate area ratio)

Outputs:
    data/processed/phase5/<city>/<city>_test_scores.csv  every strategy x noise x N
    data/processed/phase5/<city>/<city>_improvement.csv  greedy vs k-means vs N, with CIs
    data/processed/phase5/phase5_results.json            headline numbers
    outputs/phase5/...                                   figures

Usage:
    python 01_evaluate.py
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
P4 = PROJECT_ROOT / "Workflow" / "Phase 4" / "Python files"
sys.path.insert(0, str(P4))
import networks as N  # noqa: E402

_spec = importlib.util.spec_from_file_location("phase4_config", P4 / "config.py")
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

OUT_DATA = PROJECT_ROOT / "data" / "processed" / "phase5"
OUT_FIG = PROJECT_ROOT / "outputs" / "phase5"
TEST_SET = "selected"        # set from --test-set; "all2025" = post-results extension
TRAIN_SET = "selected"       # set from --train-set; "all" = networks chosen from every 2021-24 day
SUFFIX = ""
NET_SUFFIX = ""
K_TAG = ""                   # Phase 8: e.g. "_K200" = simulations and networks at K = 200
N_BOOT = 2000
RANDOM_N_STEP = 5            # random baseline every 5th N (plus headline N)
HEADLINE = f"f{cfg.HEADLINE_NOISE_FRACTION:g}"


def city_dirs(city):
    d, f = OUT_DATA / city, OUT_FIG / (city + SUFFIX)
    d.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    return d, f


def bootstrap_index(meta, n_boot, seed):
    """Stratified day bootstrap -> list of map-index arrays."""
    rng = np.random.default_rng(seed)
    days_by_regime = {r: g["day"].unique() for r, g in meta.groupby("regime")}
    maps_by_day = {d: np.nonzero(meta["day"].to_numpy() == d)[0] for d in meta["day"].unique()}
    out = []
    for _ in range(n_boot):
        idx = []
        for r, days in days_by_regime.items():
            for d in rng.choice(days, len(days), replace=True):
                idx.append(maps_by_day[d])
        out.append(np.concatenate(idx))
    return out


def weighted_rmse(mse, w, idx=None):
    if idx is None:
        return float(np.sqrt(np.sum(w * mse)))
    # Resampled maps keep their regime weight; renormalise in case a
    # resample changes how many maps each regime has (it doesn't here).
    ww = w[idx]
    return float(np.sqrt(np.sum(ww * mse[idx]) / ww.sum()))


def evaluate(city):
    print(f"=== {city}")
    data_dir, fig_dir = city_dirs(city)
    maps = (N.load_maps(city, TEST_SET, "all", k_tag=K_TAG) if K_TAG
            else N.load_maps(city, TEST_SET))
    te, cells = maps["test"], maps["cells"]
    X, w, meta = te["X"], te["w"], te["meta"]
    C = X.shape[1]
    p4 = cfg.processed_dir(city)
    nets = pd.read_csv(p4 / f"{city}_networks{NET_SUFFIX}.csv")
    kj = json.load(open(p4 / f"{city}_kriging{NET_SUFFIX}.json"))
    vg = kj["variogram"]
    K = N.covariance(cells, vg)
    noise = kj["noise_sd_ug_m3"]
    print(f"  {len(X)} test maps over {meta['day'].nunique()} days")

    def per_map_mse(S, sd):
        W = N.ok_weights(K, S, N.sensor_variance(vg, sd**2))
        return N.map_mse(X, S, W, sd**2)

    rows, mse_store = [], {}
    for label, sd in noise.items():
        for strat in ("greedy", "qr", "kmeans", "existing"):
            sub = nets[(nets["strategy"] == strat)
                       & (nets["noise"].isin([label] if strat == "greedy" else ["any"]))]
            for n, g in sub.groupby("N"):
                S = g.sort_values("rank")["cell_index"].tolist()
                mse = per_map_mse(S, sd)
                mse_store[(strat, label, n)] = mse
                rows.append((strat, label, n, weighted_rmse(mse, w)))
        print(f"  scored {label}")

    # Random baseline: mean over networks, headline noise, every RANDOM_N_STEP
    sd = noise[HEADLINE]
    n_vals = sorted(set(range(cfg.N_MIN, cfg.N_MAX[city] + 1, RANDOM_N_STEP))
                    | {cfg.HEADLINE_N[city]})
    for n in n_vals:
        vals = [weighted_rmse(per_map_mse(S, sd), w)
                for S in N.random_networks(C, n, cfg.N_RANDOM_NETWORKS, cfg.RANDOM_SEED)]
        rows.append(("random_mean", HEADLINE, n, float(np.mean(vals))))
        rows.append(("random_p05", HEADLINE, n, float(np.percentile(vals, 5))))

    scores = pd.DataFrame(rows, columns=["strategy", "noise", "N", "test_rmse"])
    scores.to_csv(data_dir / f"{city}_test_scores{SUFFIX}.csv", index=False)

    # Greedy vs k-means: improvement and paired bootstrap CI, every N, every noise.
    boots = bootstrap_index(meta, N_BOOT, [cfg.RANDOM_SEED, 5, len(city)])
    imp_rows = []
    for label in noise:
        for n in range(cfg.N_MIN, cfg.N_MAX[city] + 1):
            g, k = mse_store[("greedy", label, n)], mse_store[("kmeans", label, n)]
            point = 1 - weighted_rmse(g, w) / weighted_rmse(k, w)
            bs = np.array([1 - weighted_rmse(g, w, i) / weighted_rmse(k, w, i) for i in boots])
            imp_rows.append((label, n, point, *np.percentile(bs, [2.5, 97.5]), bs.std()))
    imp = pd.DataFrame(imp_rows, columns=["noise", "N", "improvement", "ci_low", "ci_high", "boot_sd"])
    imp.to_csv(data_dir / f"{city}_improvement{SUFFIX}.csv", index=False)

    # Per-regime improvement at the headline budget and noise.
    n0 = cfg.HEADLINE_N[city]
    g0, k0 = mse_store[("greedy", HEADLINE, n0)], mse_store[("kmeans", HEADLINE, n0)]
    by_regime = {}
    for r, sub in meta.groupby("regime"):
        i = sub.index.to_numpy()
        by_regime[r] = 1 - np.sqrt(np.mean(g0[i])) / np.sqrt(np.mean(k0[i]))

    # Equivalent sensors: smallest k-means N matching greedy's error at n0.
    sc = scores[scores["noise"] == HEADLINE]
    target = sc[(sc["strategy"] == "greedy") & (sc["noise"] == HEADLINE) & (sc["N"] == n0)]["test_rmse"].item()
    km = sc[sc["strategy"] == "kmeans"].sort_values("N")
    match = km[km["test_rmse"] <= target]
    equiv = int(match["N"].iloc[0]) if len(match) else None

    plot(city, scores, imp, fig_dir)
    return {"scores": scores, "imp": imp, "boots": boots, "mse": mse_store, "w": w,
            "by_regime": by_regime, "equiv_kmeans_N": equiv, "meta": meta}


def improvement_draws(res, label, n):
    g, k, w = res["mse"][("greedy", label, n)], res["mse"][("kmeans", label, n)], res["w"]
    return np.array([1 - weighted_rmse(g, w, i) / weighted_rmse(k, w, i) for i in res["boots"]])


def compare(sg, mel):
    """Melbourne minus Singapore improvement, with CIs (independent bootstraps)."""
    pairs = {
        "own_network_size": (cfg.HEADLINE_N["singapore"], cfg.HEADLINE_N["melbourne"]),
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


def band_means(city, res):
    """DESCRIPTIVE, not pre-registered: mean improvement over bands of N.

    Single-N values are jagged because k-means lands on better or worse
    layouts from one N to the next. Averaging over a band of budgets
    smooths that out. The CI comes from the same paired day-bootstrap, by
    averaging each draw over the band.
    """
    out = {}
    for lo, hi in BANDS[city]:
        draws = np.mean([improvement_draws(res, HEADLINE, n) for n in range(lo, hi + 1)], axis=0)
        imp = res["imp"]
        pt = imp[(imp["noise"] == HEADLINE) & imp["N"].between(lo, hi)]["improvement"].mean()
        out[f"N{lo}-{hi}"] = [float(pt), *np.percentile(draws, [2.5, 97.5]).tolist()]
    return out


def plot(city, scores, imp, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for strat, style in [("greedy", "-"), ("kmeans", "-"), ("qr", "--"), ("random_mean", ":")]:
        s = scores[(scores["strategy"] == strat) & (scores["noise"] == HEADLINE)].sort_values("N")
        ax.plot(s["N"], s["test_rmse"], style, label=strat)
    ex = scores[(scores["strategy"] == "existing") & (scores["noise"] == HEADLINE)]
    if len(ex):
        ax.plot(ex["N"], ex["test_rmse"], "k*", ms=10, label="existing EPA network")
    ax.axvline(cfg.HEADLINE_N[city], color="grey", ls=":", lw=1)
    km = scores[(scores["strategy"] == "kmeans") & (scores["noise"] == HEADLINE)]["test_rmse"]
    ax.set_ylim(0, 1.6 * km.max())
    ax.set_xlabel("number of sensors N")
    ax.set_ylabel("test RMSE, 2025 days (ug/m3)")
    ax.set_title(f"{city}: reconstruction error (noise f = {cfg.HEADLINE_NOISE_FRACTION})")
    ax.legend(fontsize=8)

    ax = axes[1]
    s = imp[imp["noise"] == HEADLINE]
    ax.plot(s["N"], 100 * s["improvement"], color="#c0392b")
    ax.fill_between(s["N"], 100 * s["ci_low"], 100 * s["ci_high"], color="#c0392b", alpha=0.2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(cfg.HEADLINE_N[city], color="grey", ls=":", lw=1)
    ax.set_xlabel("number of sensors N")
    ax.set_ylabel("greedy improvement over k-means (%)")
    ax.set_title("weather-informed vs evenly spread (95% CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "test_results.png", dpi=130)
    plt.close(fig)


def summarize(city, res):
    imp, sc = res["imp"], res["scores"]
    n0 = cfg.HEADLINE_N[city]
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
    TEST_SET, TRAIN_SET = args.test_set, args.train_set
    if args.k is not None and (TEST_SET, TRAIN_SET) != ("all2025", "all"):
        ap.error("--k needs --test-set all2025 --train-set all")
    K_TAG = "" if args.k is None else f"_K{args.k:g}"
    NET_SUFFIX = ("" if TRAIN_SET == "selected" else "_trainall") + K_TAG
    SUFFIX = ("" if TEST_SET == "selected" else "_all2025") + NET_SUFFIX
    res = {c: evaluate(c) for c in cfg.ALL_CITIES}
    results = {c: summarize(c, r) for c, r in res.items()}
    for c, r in res.items():
        results[c]["band_mean_improvement_descriptive"] = band_means(c, r)
    results["city_comparison"] = compare(res["singapore"], res["melbourne"])
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    results["test_set"] = TEST_SET
    results["train_set"] = TRAIN_SET
    results["K_m2_s"] = args.k
    with open(OUT_DATA / f"phase5_results{SUFFIX}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
