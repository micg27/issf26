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

dask.config.set(scheduler="synchronous")

ROOT = Path(__file__).resolve().parents[3]
K_TAG = ""  
P4 = ROOT / "Workflow" / "Phase 4" / "Python files"
sys.path.insert(0, str(P4))
import networks as N  

_spec = importlib.util.spec_from_file_location("cfg4", P4 / "config.py")
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

N_BOOT = 2000
J = 5
N_AREA = 20
NVALS = {"singapore": [5, 8, 10, 15, 22, 30, 40],
            "melbourne": [5, 8, 10, 15, 22, 30, 32, 40, 60, 88, 120, 160]}
LEVELS = {  
    "L0": (None, 0.0, False),
    "L1": (None, 0.0, True),
    "L2": ("fixed", 0.1, False),
    "L3": ("fixed", 0.3, False),
    "L4": ("fixed", 0.3, True),
    "L5": ("random", 0.3, True),
}
PAIRS = {"own_network_size": (22, 8), "equal_count_N8": (8, 8), "equal_count_N22": (22, 22),
         "equal_count_N40": (40, 40), "equal_density_SG8_MEL32": (8, 32),
         "equal_density_SG22_MEL88": (22, 88), "equal_density_SG40_MEL160": (40, 160)}


def load_ingredients(city, period):
    ds = xr.open_dataset(cfg.P3_DIR / city / f"{city}_ingredients_{period}{K_TAG}.nc")
    meta = pd.DataFrame({"date": pd.to_datetime(ds["date"].values),
                         "regime": ds["regime"].values.astype(str),
                         "w_k": ds["w_k"].values})
    return ds["response"].values.astype(float), meta, ds.attrs["base_total_ug_s"], ds["cell_id"].values


def magnitudes(meta, siteset):
    """(day, member, J) magnitudes per unit p: mean total / J, lognormal sigma 1."""
    out = np.empty((len(meta), N_AREA, J))
    for d, date in enumerate(meta["date"]):
        rng = np.random.default_rng([cfg.SEED, 63, date.year, date.dayofyear, siteset])
        out[d] = rng.lognormal(-0.5, 1.0, (N_AREA, J))
    return out


def assemble(resp, meta, total, level, siteset):
    """Maps (day*member, cell) and their metadata for one level."""
    mode, p, area = LEVELS[level]
    base = resp[:, 0]                                        
    aream = resp[:, 1:1 + N_AREA]                            
    local = aream if area else np.repeat(base[:, None], N_AREA, axis=1)
    X = (1 - p) * local
    if mode == "fixed":
        k0 = 1 + N_AREA + J * siteset
        R = resp[:, k0:k0 + J]                                
        a = p * total / J * magnitudes(meta, siteset)        
        X = X + np.einsum("dmj,djc->dmc", a, R)
    D, M, C = X.shape
    m = pd.DataFrame({"day": np.repeat(np.arange(D), M), "member": np.tile(np.arange(M), D),
                      "regime": np.repeat(meta["regime"].values, M),
                      "w_k": np.repeat(meta["w_k"].values, M), "date": np.repeat(meta["date"].values, M)})
    return X.reshape(D * M, C), m


def existing_L5(city):
    maps = N.load_maps(city, testset="all2025", trainset="all", k_tag=K_TAG)
    out = {}
    for split in ("train", "test"):
        keep = (maps[split]["meta"]["member"] > 0).to_numpy()
        out[split] = (maps[split]["X"][keep], maps[split]["meta"][keep].reset_index(drop=True))
    return out, maps["cells"]


def weights(m):
    n = m.groupby("regime")["regime"].transform("size").to_numpy()
    w = m["w_k"].to_numpy() / n
    return w / w.sum()


def boot_index(m, seed):
    rng = np.random.default_rng(seed)
    days, reg = m["day"].to_numpy(), m["regime"].to_numpy()
    byreg = {r: np.unique(days[reg == r]) for r in np.unique(reg)}
    rows = {d: np.nonzero(days == d)[0] for d in np.unique(days)}
    return [np.concatenate([rows[d] for v in byreg.values() for d in rng.choice(v, len(v))])
            for _ in range(N_BOOT)]


def evaluate_level(city, cells, kmeansnets, Xtr, mtr, Xte, mte):
    wtr, wte = weights(mtr), weights(mte)
    Sigma = N.second_moment(Xtr, wtr)
    vg = N.fit_variogram(cells, Sigma)
    K = N.covariance(cells, vg)
    sd = cfg.MAIN_NOISE * N.spatial_sd(Xtr, wtr)
    order, _ = N.greedy_rmse(K, Sigma, vg, sd**2, max(NVALS[city]))
    boots = boot_index(mte, [cfg.SEED, 64, len(city)])

    def mse(S):
        W = N.ok_weights(K, S, N.sensor_variance(vg, sd**2))
        return N.map_mse(Xte, S, W, sd**2)

    def imp(g, k, i):
        return 1 - np.sqrt(np.sum(wte[i] * g[i]) / wte[i].sum()) / np.sqrt(np.sum(wte[i] * k[i]) / wte[i].sum())

    res, draws = {}, {}
    allrows = np.arange(len(Xte))
    for n in NVALS[city]:
        g, k = mse(order[:n]), mse(kmeansnets[n])
        bs = np.array([imp(g, k, i) for i in boots])
        res[str(n)] = [float(imp(g, k, allrows)), *np.percentile(bs, [2.5, 97.5]).tolist()]
        draws[n] = bs
    return {"improvement": res, "noise_sd": sd, "variogram_range_km": vg["range_m"] / 1000,
            "train_spatial_sd": N.spatial_sd(Xtr, wtr)}, draws


def run_city(city):
    print(f"=== {city}")
    nets = pd.read_csv(cfg.processed_dir(city) / f"{city}_networks_trainall.csv")
    km = nets[nets["strategy"] == "kmeans"]
    kmeansnets = {n: g.sort_values("rank")["cell_index"].tolist() for n, g in km.groupby("N")}
    L5, cells = existing_L5(city)
    rtr, mtr0, total, ids = load_ingredients(city, "train")
    rte, mte0, _, idste = load_ingredients(city, "test2025")
    assert np.array_equal(ids, cells["cell_id"].values) and np.array_equal(idste, ids)

    out, draws = {}, {}
    for level in LEVELS:
        sets = [0, 1] if LEVELS[level][0] == "fixed" else [None]
        for s in sets:
            key = level if s is None else f"{level}_set{'AB'[s]}"
            if level == "L5":
                (Xtr, mtr), (Xte, mte) = L5["train"], L5["test"]
            else:
                Xtr, mtr = assemble(rtr, mtr0, total, level, s or 0)
                Xte, mte = assemble(rte, mte0, total, level, s or 0)
            out[key], draws[key] = evaluate_level(city, cells, kmeansnets, Xtr, mtr, Xte, mte)
            hn = cfg.MAIN_N[city]
            r = out[key]["improvement"][str(hn)]
            print(f"  {key:9s} N={hn}: {100 * r[0]:+.1f}% ({100 * r[1]:+.1f}, {100 * r[2]:+.1f})")
    return out, draws


def compare(sg, mel, dsg, dmel):
    out = {}
    for key in sg:
        out[key] = {}
        for name, (ns, nm) in PAIRS.items():
            diff = dmel[key][nm] - dsg[key][ns]
            pt = mel[key]["improvement"][str(nm)][0] - sg[key]["improvement"][str(ns)][0]
            out[key][name] = [float(pt), *np.percentile(diff, [2.5, 97.5]).tolist()]
    return out


def plot(results):
    order = ["L0", "L1", "L2_setA", "L2_setB", "L3_setA", "L3_setB", "L4_setA", "L4_setB", "L5"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, (ns, nm, title) in zip(axes, [(22, 8, "own network size (SG 22, MEL 8)"),
                                          (22, 22, "equal count (N = 22)"),
                                          (22, 88, "equal density (SG 22, MEL 88)")]):
        for city, n, col in [("singapore", ns, "#2e86c1"), ("melbourne", nm, "#c0392b")]:
            v = np.array([results[city][k]["improvement"][str(n)] for k in order]) * 100
            x = np.arange(len(order)) + (-0.12 if city == "singapore" else 0.12)
            ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o",
                        color=col, capsize=3, label=f"{city} (N = {n})")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("greedy improvement over k-means (%)")
    fig.suptitle("Weather-informed placement vs emission predictability "
                 "(L0 = known emissions ... L5 = random relocating sources)")
    fig.tight_layout()
    out = ROOT / "outputs" / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"emission_predictability{K_TAG}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Emission-predictability experiment.")
    ap.add_argument("--k", type=float, default=None,
                    help="Phase 8: K (m2/s) of the simulations to use")
    k = ap.parse_args().k
    K_TAG = "" if k is None else f"_K{k:g}"
    res, dr = {}, {}
    for city in cfg.ALL_CITIES:
        res[city], dr[city] = run_city(city)
    res["city_difference_mel_minus_sg"] = compare(res["singapore"], res["melbourne"],
                                                  dr["singapore"], dr["melbourne"])
    path = ROOT / "data" / "processed" / "phase5" / f"phase5_emission_predictability{K_TAG}.json"
    res["K_m2_s"] = k
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    plot(res)
    print(f"wrote {path}")
