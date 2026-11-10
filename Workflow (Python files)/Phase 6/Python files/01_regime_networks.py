# 01_regime_networks.py
"""
Phase 6, Analysis 1 (pre-stated in METHODS.md, 2026-10-07): regime-specific
networks.

Tests the explanation left open by Phase 5: Singapore's steady monsoon
flow lets one fixed network serve every day, while Melbourne's changeable
weather does not.

For each city, emission setting and budget N:
  * a greedy network is chosen from each regime's 2021-2024 training maps
    only, plus the all-weather network from all training maps;
  * every network uses the same kriging covariance (fitted on all training
    maps of that setting) and the same noise (f = 0.25 x the city's overall
    training spatial SD), so only the training weather differs;
  * on each regime's 2025 test days, each network is scored:
      transfer penalty       RMSE(net j on regime k) / RMSE(net k on k) - 1, j != k
      fixed-network penalty  RMSE(all-weather net on k) / RMSE(net k on k) - 1
    averaged over regimes with weights w_k;
  * overlap between regime networks: Jaccard index of their cells, and the
    mean nearest-sensor distance between them / sqrt(area / N).
CIs: regime-stratified day bootstrap (2,000 draws), as in Phase 5.

Emission settings: "original" = the Phase 3 scenario design (the
full-training networks); "L0" = known emissions (base ingredient only),
secondary, because the original design's random hotspots mask weather
patterns.

Output: data/processed/phase6/phase6_regime_networks.json

Usage:
    python 01_regime_networks.py
"""

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

dask.config.set(scheduler="synchronous")

ROOT = Path(__file__).resolve().parents[3]
P4 = ROOT / "Workflow" / "Phase 4" / "Python files"
sys.path.insert(0, str(P4))
import networks as N  # noqa: E402

_spec = importlib.util.spec_from_file_location("cfg4", P4 / "config.py")
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

N_BOOT = 2000
N_VALUES = {"singapore": [8, 22, 40], "melbourne": [8, 22, 32, 40, 88]}
AREA_KM2 = {"singapore": 655.3, "melbourne": 2611.2}       # Phase 1 candidate land area
PAIRS = {"equal_count_N8": (8, 8), "equal_count_N22": (22, 22), "equal_count_N40": (40, 40),
         "equal_density_SG8_MEL32": (8, 32), "equal_density_SG22_MEL88": (22, 88)}


def maps_original(city):
    m = N.load_maps(city, test_set="all2025", train_set="all")
    return m["cells"], m["train"], m["test"]


def maps_L0(city, cells):
    out = []
    for period in ("train", "test2025"):
        ds = xr.open_dataset(cfg.PHASE3_DIR / city / f"{city}_ingredients_{period}.nc")
        meta = pd.DataFrame({"day": np.arange(ds.sizes["day"]), "member": 0,
                             "date": ds["date"].values, "regime": ds["regime"].values.astype(str),
                             "w_k": ds["w_k"].values})
        assert np.array_equal(ds["cell_id"].values, cells["cell_id"].values)
        out.append(N._pack(ds["response"].values[:, 0].astype(float), meta))
    return out


def regime_subset(split, regime):
    keep = (split["meta"]["regime"] == regime).to_numpy()
    m = split["meta"][keep].reset_index(drop=True)
    return split["X"][keep], np.full(len(m), 1.0 / len(m)), m


def nearest_distance_ratio(cells, A, B, n, area):
    xy = cells[["x", "y"]].to_numpy() / 1000
    a, b = xy[A], xy[B]
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    mean_d = 0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean())
    return mean_d / np.sqrt(area / n)


def boot_days(meta, rng):
    days = meta["day"].to_numpy()
    rows = {d: np.nonzero(days == d)[0] for d in np.unique(days)}
    u = np.array(list(rows))
    return [np.concatenate([rows[d] for d in rng.choice(u, len(u))]) for _ in range(N_BOOT)]


def analyse(city, setting, cells, train, test):
    regimes = sorted(train["meta"]["regime"].unique())
    w_k = train["meta"].groupby("regime")["w_k"].first().to_dict()
    Sigma_all = N.second_moment(train["X"], train["w"])
    vg = N.fit_variogram(cells, Sigma_all)
    K = N.covariance(cells, vg)
    sd = cfg.HEADLINE_NOISE_FRACTION * N.spatial_sd(train["X"], train["w"])
    n_max = max(N_VALUES[city])

    orders = {"all": N.greedy_rmse(K, Sigma_all, vg, sd**2, n_max)[0]}
    for r in regimes:
        Xr, wr, _ = regime_subset(train, r)
        orders[r] = N.greedy_rmse(K, N.second_moment(Xr, wr), vg, sd**2, n_max)[0]

    rng = np.random.default_rng([cfg.RANDOM_SEED, 6, len(city), len(setting)])
    test_by_r = {r: regime_subset(test, r) for r in regimes}
    boots = {r: boot_days(test_by_r[r][2], rng) for r in regimes}

    out = {"regimes": regimes, "w_k": w_k, "noise_sd": sd, "budgets": {}}
    draws = {}
    for n in N_VALUES[city]:
        nets = {k: v[:n] for k, v in orders.items()}
        W = {k: N.ok_weights(K, S, N.sensor_variance(vg, sd**2)) for k, S in nets.items()}
        # per-map MSE of every network on every regime's test maps
        mse = {(j, k): N.map_mse(test_by_r[k][0], nets[j], W[j], sd**2)
               for j in nets for k in regimes}

        def penalties(idx):
            tr, fx = 0.0, 0.0
            per = {}
            for k in regimes:
                i = idx[k]
                own = np.sqrt(mse[(k, k)][i].mean())
                others = [np.sqrt(mse[(j, k)][i].mean()) / own - 1 for j in regimes if j != k]
                fixed = np.sqrt(mse[("all", k)][i].mean()) / own - 1
                per[k] = (float(np.mean(others)), float(fixed))
                tr += w_k[k] * np.mean(others)
                fx += w_k[k] * fixed
            tot = sum(w_k.values())
            return tr / tot, fx / tot, per

        full = {k: slice(None) for k in regimes}
        tr0, fx0, per0 = penalties(full)
        bs = np.array([penalties({k: boots[k][b] for k in regimes})[:2] for b in range(N_BOOT)])
        area = AREA_KM2[city]
        overlap = {f"{a}-{b}": {"jaccard": len(set(nets[a]) & set(nets[b])) / len(set(nets[a]) | set(nets[b])),
                                "nearest_distance_ratio": float(nearest_distance_ratio(cells, nets[a], nets[b], n, area))}
                   for a, b in combinations(regimes, 2)}
        out["budgets"][str(n)] = {
            "transfer_penalty": [float(tr0), *np.percentile(bs[:, 0], [2.5, 97.5]).tolist()],
            "fixed_network_penalty": [float(fx0), *np.percentile(bs[:, 1], [2.5, 97.5]).tolist()],
            "per_regime_transfer_fixed": per0,
            "overlap": overlap,
            "mean_jaccard": float(np.mean([v["jaccard"] for v in overlap.values()])),
            "mean_nearest_distance_ratio": float(np.mean([v["nearest_distance_ratio"] for v in overlap.values()])),
            "networks": {k: [int(cells["cell_id"][i]) for i in v] for k, v in nets.items()},
        }
        draws[n] = bs
        b = out["budgets"][str(n)]
        print(f"  {city} {setting} N={n:3d}: transfer {100 * b['transfer_penalty'][0]:+.1f}% "
              f"({100 * b['transfer_penalty'][1]:+.1f}, {100 * b['transfer_penalty'][2]:+.1f}); "
              f"fixed {100 * b['fixed_network_penalty'][0]:+.1f}% "
              f"({100 * b['fixed_network_penalty'][1]:+.1f}, {100 * b['fixed_network_penalty'][2]:+.1f}); "
              f"Jaccard {b['mean_jaccard']:.2f}, distance ratio {b['mean_nearest_distance_ratio']:.2f}")
    return out, draws


def compare(sg, mel, dsg, dmel):
    out = {}
    for name, (ns, nm) in PAIRS.items():
        row = {}
        for m, col in (("transfer_penalty", 0), ("fixed_network_penalty", 1)):
            diff = dmel[nm][:, col] - dsg[ns][:, col]
            pt = mel["budgets"][str(nm)][m][0] - sg["budgets"][str(ns)][m][0]
            row[m] = [float(pt), *np.percentile(diff, [2.5, 97.5]).tolist()]
        row["jaccard_mel_minus_sg"] = mel["budgets"][str(nm)]["mean_jaccard"] - sg["budgets"][str(ns)]["mean_jaccard"]
        row["distance_ratio_mel_minus_sg"] = (mel["budgets"][str(nm)]["mean_nearest_distance_ratio"]
                                              - sg["budgets"][str(ns)]["mean_nearest_distance_ratio"])
        out[name] = row
    return out


if __name__ == "__main__":
    results, draws = {}, {}
    for setting in ("original", "L0"):
        results[setting], draws[setting] = {}, {}
        for city in cfg.ALL_CITIES:
            cells, train, test = maps_original(city)
            if setting == "L0":
                train, test = maps_L0(city, cells)
            results[setting][city], draws[setting][city] = analyse(city, setting, cells, train, test)
        results[setting]["city_difference_mel_minus_sg"] = compare(
            results[setting]["singapore"], results[setting]["melbourne"],
            draws[setting]["singapore"], draws[setting]["melbourne"])
    out = ROOT / "data" / "processed" / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "phase6_regime_networks.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out / 'phase6_regime_networks.json'}")
