"""
select sensor networks from the train scenarios (2021-2024 days) only.
"""

import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config
import networks as N


TRAINSET = "selected"
SUFFIX = ""
K_TAG = ""    

def parse_args():
    p = argparse.ArgumentParser(description="Select Phase 4 sensor networks.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--city", choices=config.ALL_CITIES)
    g.add_argument("--all", action="store_true")
    p.add_argument("--train-set", choices=["selected", "all"], default="selected")
    p.add_argument("--k", type=float, default=None,
                   help="Phase 8: use the simulations at this K (m2/s); needs --train-set all")
    a = p.parse_args()
    if a.k is not None and a.train_set != "all":
        p.error("--k needs --train-set all")
    return (config.ALL_CITIES if a.all else [a.city]), a.train_set, a.k


def noise_levels(sigmacity):
    """label -> noise SD (ug/m3)."""
    levels = {f"f{f:g}": f * sigmacity for f in config.NOISE_FRACS}
    levels[f"abs{config.ABS_NOISE:g}"] = config.ABS_NOISE
    return levels


def existing_network(city, cells):
    """Melbourne's 2025 in-domain PM2.5 stations -> nearest candidate cells."""
    p3 = config.P3_DIR / city
    st = pd.read_csv(p3 / f"{city}_pm25_stations.csv")
    obs = pd.read_csv(p3 / f"{city}_pm25_hourly.csv", parse_dates=["time_local"])
    active = obs[obs["time_local"].dt.year == 2025]["station"].unique()
    st = st[st["in_domain"] & st["station"].isin(active)].reset_index(drop=True)
    xy = cells[["x", "y"]].to_numpy()
    rows = []
    for _, s in st.iterrows():
        dist = np.hypot(xy[:, 0] - s["x"], xy[:, 1] - s["y"])
        i = int(np.argmin(dist))
        rows.append((s["station"], i, float(dist[i])))
    return rows


def select(city):
    print(f"=== {city}")
    maps = (N.load_maps(city, testset="all2025", trainset=TRAINSET, k_tag=K_TAG) if K_TAG
            else N.load_maps(city, trainset=TRAINSET))
    tr, cells = maps["train"], maps["cells"]
    C = len(cells)
    Sigma = N.second_moment(tr["X"], tr["w"])
    vg = N.fit_variogram(cells, Sigma)
    K = N.covariance(cells, vg)
    sigmacity = N.spatial_sd(tr["X"], tr["w"])
    levels = noise_levels(sigmacity)
    nvalues = list(range(config.N_MIN, config.N_MAX[city] + 1))
    print(f"  {C} candidates, {len(tr['X'])} training maps, sigma_city {sigmacity:.3f} ug/m3")
    print(f"  variogram: nugget {vg['nugget']:.2e}, sill {vg['sill']:.3e}, "
          f"range {vg['range_m'] / 1000:.1f} km")

    rows = []

    def add(strategy, noise, n, S):
        for rank, i in enumerate(S):
            rows.append((strategy, noise, n, rank, i, int(cells["cell_id"][i])))

    for label, sd in levels.items():
        order, hist = N.greedy_rmse(K, Sigma, vg, sd**2, config.N_MAX[city])
        for n in nvalues:
            add("greedy", label, n, order[:n])
        print(f"  greedy {label:6s} (noise SD {sd:.3f}): train RMSE at N={config.MAIN_N[city]} "
              f"{np.sqrt(hist[config.MAIN_N[city] - 1]):.4f}")

    modes, sv = N.pod_modes(tr["X"], tr["w"])
    for n in nvalues:
        add("qr", "any", n, N.qr_network(modes, n))
        add("kmeans", "any", n, N.kmeans_network(cells, n, config.SEED))

    existing = []
    if city == "melbourne":
        existing = existing_network(city, cells)
        S = [i for _, i, _ in existing]
        assert len(set(S)) == len(S), "two stations share a candidate cell"
        add("existing", "any", len(S), S)
        for name, i, dist in existing:
            print(f"  station {name:14s} -> cell {int(cells['cell_id'][i])} ({dist / 1000:.2f} km)")

    nets = pd.DataFrame(rows, columns=["strategy", "noise", "N", "rank", "cell_index", "cell_id"])
    out = config.processed_dir(city)
    nets.to_csv(out / f"{city}_networks{SUFFIX}.csv", index=False)
    with open(out / f"{city}_kriging{SUFFIX}.json", "w") as f:
        json.dump({"variogram": vg, "sigma_city_ug_m3": sigmacity,
                   "noise_sd_ug_m3": levels,
                   "pod_energy_first_10": (sv[:10] ** 2 / np.sum(sv**2)).tolist(),
                   "existing_stations": [{"station": s, "cell_index": i, "dist_m": d}
                                         for s, i, d in existing],
                   "train_set": TRAINSET, "n_training_maps": len(tr["X"])}, f, indent=2)

    training_curves(city, maps, nets, K, vg, levels, modes)
    plot_networks(city, cells, nets)
    plot_variogram(city, vg)


def training_curves(city, maps, nets, K, vg, levels, modes):
    tr = maps["train"]
    label = f"f{config.MAIN_NOISE:g}"
    s2 = levels[label] ** 2
    rows = []
    for strat, noise in [("greedy", label), ("qr", "any"), ("kmeans", "any")]:
        sub = nets[(nets["strategy"] == strat) & (nets["noise"] == noise)]
        for n, g in sub.groupby("N"):
            S = g.sort_values("rank")["cell_index"].tolist()
            rows.append((strat, n, N.ensemble_rmse(tr["X"], tr["w"], S, K, vg, s2)))
            if strat == "qr":
                rows.append(("qr_native_pod", n, N.pod_rmse(tr["X"], tr["w"], S, modes, s2)))
    curves = pd.DataFrame(rows, columns=["strategy", "N", "train_rmse"])
    out = config.output_dir(city)
    curves.to_csv(out / f"training_curves{SUFFIX}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for strat, g in curves.groupby("strategy"):
        ax.plot(g["N"], g["train_rmse"], label=strat, ls="--" if strat == "qr_native_pod" else "-")
    ax.axvline(config.MAIN_N[city], color="grey", ls=":", lw=1)
    ax.set_xlabel("number of sensors N")
    ax.set_ylabel("training RMSE (ug/m3)")
    ax.set_ylim(0, 1.5 * curves[curves["strategy"] == "kmeans"]["train_rmse"].max())
    ax.set_title(f"{city}: training-set RMSE at headline noise (selection only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"training_curves{SUFFIX}.png", dpi=130)
    plt.close(fig)
    hn = curves[curves["N"] == config.MAIN_N[city]].set_index("strategy")["train_rmse"]
    print("  training RMSE at headline N:", hn.round(4).to_dict())


def plot_networks(city, cells, nets):
    n = config.MAIN_N[city]
    label = f"f{config.MAIN_NOISE:g}"
    panels = [("greedy", label), ("qr", "any"), ("kmeans", "any")]
    if (nets["strategy"] == "existing").any():
        panels.append(("existing", "any"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (strat, noise) in zip(axes, panels):
        sub = nets[(nets["strategy"] == strat) & (nets["noise"] == noise)]
        sub = sub[sub["N"] == (sub["N"].max() if strat == "existing" else n)]
        ax.scatter(cells["x"] / 1000, cells["y"] / 1000, s=6, c="#d9d9d9", marker="s")
        sel = cells.iloc[sub["cell_index"]]
        ax.scatter(sel["x"] / 1000, sel["y"] / 1000, s=40, c="#c0392b", edgecolor="k", lw=0.5)
        ax.set_title(f"{strat} (N = {len(sub)})")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{city}: networks at the headline budget")
    fig.tight_layout()
    fig.savefig(config.output_dir(city) / f"networks_headline{SUFFIX}.png", dpi=130)
    plt.close(fig)


def plot_variogram(city, vg):
    h = np.array(vg["bins_h"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(h / 1000, vg["bins_gamma"], "o", ms=3, label="empirical (training maps)")
    hh = np.linspace(0, h.max(), 200)
    ax.plot(hh / 1000, N.exponential(hh, vg["nugget"], vg["sill"], vg["range_m"]), label="exponential fit")
    ax.set_xlabel("distance (km)")
    ax.set_ylabel("semivariance (ug/m3)^2")
    ax.set_title(f"{city}: kriging variogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.output_dir(city) / f"variogram{SUFFIX}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cities, TRAINSET, k = parse_args()
    K_TAG = "" if k is None else f"_K{k:g}"
    SUFFIX = ("" if TRAINSET == "selected" else "_trainall") + K_TAG
    for city in cities:
        select(city)
