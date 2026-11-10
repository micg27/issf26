# make_poster_figures.py
"""
Redraw the five poster figures in one green colour scheme, from saved
results (nothing is recomputed).

Palette (checked with the dataviz skill's validate_palette.py, light mode:
lightness band, chroma, colour-blind separation and contrast all pass):
    GREEN  #13703f   weather-informed (greedy) / Singapore
    TEAL   #1f8bb8   evenly spread (k-means) / Melbourne
    GREY   #8a8f8c   secondary strategies (QR, random)
Maps use the sequential "YlGn" colour map (light = low, dark green = high).

Output: Graphs/Poster picks/*.png

Usage:
    python make_poster_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "data" / "processed" / "phase3"
P4 = ROOT / "data" / "processed" / "phase4"
P5 = ROOT / "data" / "processed" / "phase5"
P7 = ROOT / "data" / "processed" / "phase7"
OUT = ROOT / "Graphs" / "Poster picks"

GREEN, TEAL, GREY, INK, MUTED, CELL = "#13703f", "#1f8bb8", "#8a8f8c", "#1d2b22", "#5c6b62", "#dfe6e1"
CITY_COL = {"singapore": GREEN, "melbourne": TEAL}
HEADLINE_N = {"singapore": 22, "melbourne": 8}
HEADLINE = "f0.25"
CITY_NAME = {"singapore": "Singapore", "melbourne": "Melbourne"}
REGIME_NAME = {"SIN1": "Southwest monsoon", "SIN2": "Inter-monsoon", "SIN3": "Northeast monsoon",
               "MEL1": "Light south-easterlies", "MEL2": "Northerlies (before a front)",
               "MEL3": "Westerlies (after a front)"}

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8ece9",
    "grid.linewidth": 0.8, "legend.frameon": False, "lines.linewidth": 2,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def save(fig, name):
    # PDF/SVG are vector (sharp at any poster size); PNG at 600 dpi for slides.
    stem = Path(name).stem
    for sub, ext, kw in [("PNG (600 dpi)", "png", {"dpi": 600}),
                         ("PDF (vector, for print)", "pdf", {}),
                         ("SVG (vector, for editing)", "svg", {})]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / sub / f"{stem}.{ext}", bbox_inches="tight",
                    facecolor="white", **kw)
    plt.close(fig)


def cell_size(fig, ncols, cx, share=0.78):
    """Marker area (pt^2) so each 2 km cell square nearly fills its grid spacing."""
    cells_across = (float(cx.max()) - float(cx.min())) / 2000 + 1
    pt = fig.get_figwidth() * 72 / ncols * share / cells_across
    return (0.95 * pt) ** 2


def fig1_plumes(city):
    """Base-scenario daily-mean local PM2.5 on the representative day of each regime."""
    ds = xr.open_dataset(P3 / city / f"{city}_scenarios.nc")
    rep = np.nonzero(ds["selection"].values == "representative")[0]
    w = (ds["cell_x"].max() - ds["cell_x"].min()) / (ds["cell_y"].max() - ds["cell_y"].min())
    fig, axes = plt.subplots(1, len(rep), figsize=(15, 15 / len(rep) / float(w) + 1.2),
                             layout="constrained")
    for ax, d in zip(axes, rep):
        v = ds["local_pm25"].isel(day=d, member=0).mean("hour").values
        sc = ax.scatter(ds["cell_x"] / 1000, ds["cell_y"] / 1000, c=v, s=cell_size(fig, len(rep), ds["cell_x"]),
                        marker="s", lw=0, cmap="YlGn")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="Local PM2.5 (µg/m³)")
        r = str(ds["regime"].values[d])
        day = pd.Timestamp(ds["date"].values[d]).strftime("%-d %b %Y")
        ax.set_title(f"{REGIME_NAME[r]}\n{day}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"{CITY_NAME[city]}: simulated daily PM2.5 from local emissions on a typical day "
                 "of each weather pattern\nEach map has its own colour scale", fontsize=14)
    save(fig, f"1_plume_maps_by_regime_{city}.png")


def fig2_networks(city):
    ds = xr.open_dataset(P3 / city / f"{city}_scenarios.nc")
    x, y = ds["cell_x"].values / 1000, ds["cell_y"].values / 1000
    nets = pd.read_csv(P4 / city / f"{city}_networks_trainall.csv")
    n = HEADLINE_N[city]
    panels = [("greedy", HEADLINE, "Weather-informed", GREEN, "o"),
              ("kmeans", "any", "Evenly spread", TEAL, "s")]
    if (nets["strategy"] == "existing").any():
        panels.append(("existing", "any", "Existing EPA stations", INK, "^"))
    w = (x.max() - x.min()) / (y.max() - y.min())
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5 / w + 1.3), layout="constrained")
    for ax, (strat, noise, title, col, mk) in zip(axes, panels):
        sub = nets[(nets["strategy"] == strat) & (nets["noise"] == noise)]
        sub = sub[sub["N"] == (sub["N"].max() if strat == "existing" else n)]
        ax.scatter(x, y, s=cell_size(fig, len(panels), x * 1000, share=0.95), c=CELL, marker="s", lw=0)
        i = sub["cell_index"].to_numpy()
        ax.scatter(x[i], y[i], s=90, c=col, marker=mk, edgecolor="white", lw=1.5, zorder=3)
        ax.set_title(f"{title}\n{len(sub)} sensors", color=INK)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"{CITY_NAME[city]}: where each strategy places {n} sensors "
                 f"(the size of the city's real network)", fontsize=14)
    save(fig, f"2_sensor_networks_{city}.png")


def fig3_predictability():
    res = json.load(open(P5 / "phase5_emission_predictability.json"))
    order = ["L0", "L1", "L2_setA", "L2_setB", "L3_setA", "L3_setB", "L4_setA", "L4_setB", "L5"]
    labels = ["Known\nexactly", "Noisy\noutput", "10% fixed\nsources (A)", "10% fixed\nsources (B)",
              "30% fixed\nsources (A)", "30% fixed\nsources (B)", "30% fixed +\nnoise (A)",
              "30% fixed +\nnoise (B)", "Randomly\nmoving"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True, sharey=True, layout="constrained")
    for ax, (ns, nm, title) in zip(axes, [(22, 8, "Each city's real network size"),
                                          (22, 22, "Same number of sensors (22 each)"),
                                          (22, 88, "Same sensor density")]):
        for city, n, mk, off in [("singapore", ns, "o", -0.14), ("melbourne", nm, "s", 0.14)]:
            v = np.array([res[city][k]["improvement"][str(n)] for k in order]) * 100
            ax.errorbar(np.arange(len(order)) + off, v[:, 0],
                        yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, ms=8,
                        color=CITY_COL[city], mec="white", mew=1, capsize=3, elinewidth=1.5,
                        label=f"{CITY_NAME[city]} ({n} sensors)")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(title, loc="left", fontsize=13)
        ax.set_ylabel("Accuracy gain (%)")
        ax.set_ylim(-8, 65)
        ax.legend(loc="upper left", fontsize=11, ncol=2)
    axes[-1].set_xlabel("How predictable the pollution sources are (most → least predictable)")
    fig.suptitle("Weather-informed placement helps when pollution sources are predictable\n"
                 "Accuracy gain over evenly spread sensors. Error bars: 95% confidence intervals. "
                 "A and B: two random sets of source locations", fontsize=14)
    save(fig, "3_emission_predictability_KEY_RESULT.png")


def fig4_sensor_count():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, city in enumerate(["singapore", "melbourne"]):
        sc = pd.read_csv(P5 / city / f"{city}_test_scores_all2025_trainall.csv")
        sc = sc[sc["noise"] == HEADLINE]
        c = pd.read_csv(P7 / f"{city}_original_curves.csv")
        ax = axes[0, col]
        for strat, colour, name in (("greedy", GREEN, "Weather-informed"), ("kmeans", TEAL, "Evenly spread")):
            ax.plot(c["N"], c[f"rmse_{strat}"], "o", color=colour, ms=3, alpha=0.6)
            ax.plot(c["N"], c[f"fit_{strat}"], "-", color=colour, label=name)
        for strat, style, name in (("qr", "--", "QR pivoting"), ("random_mean", ":", "Random")):
            q = sc[sc["strategy"] == strat].sort_values("N")
            ax.plot(q["N"], q["test_rmse"], style, color=GREY, lw=1.5, label=name)
        ax.axvline(HEADLINE_N[city], color=MUTED, ls=":", lw=1)
        ax.set_ylim(0, 1.3 * c["rmse_kmeans"].max())
        ax.set_title(f"{CITY_NAME[city]}: map error falls as sensors are added")
        ax.set_xlabel("Number of sensors")
        ax.set_ylabel("Map error, RMSE (µg/m³)")
        ax.legend(fontsize=10)

        ax = axes[1, col]
        for setting, colour, name in (("L0", GREEN, "Emission sources known"),
                                      ("original", TEAL, "Randomly moving hotspots")):
            c = pd.read_csv(P7 / f"{city}_{setting}_curves.csv")
            ax.plot(c["N"], 100 * c["improvement"], color=colour, label=name)
            ax.fill_between(c["N"], 100 * c["ci_low"], 100 * c["ci_high"], color=colour, alpha=0.18, lw=0)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(HEADLINE_N[city], color=MUTED, ls=":", lw=1)
        ax.set_title(f"{CITY_NAME[city]}: gain from weather-informed placement")
        ax.set_xlabel("Number of sensors")
        ax.set_ylabel("Accuracy gain over even spacing (%)")
        ax.legend(fontsize=10)
    fig.text(0.5, -0.01, "Dotted line: each city's real network size (Singapore 22, Melbourne 8). "
             "Shading: 95% confidence interval. Grey: other strategies, for reference.",
             ha="center", color=MUTED, fontsize=10)
    fig.tight_layout()
    save(fig, "4_sensor_count_curves.png")


def fig5_main_test(city):
    sc = pd.read_csv(P5 / city / f"{city}_test_scores_all2025_trainall.csv")
    sc = sc[sc["noise"] == HEADLINE]
    imp = pd.read_csv(P5 / city / f"{city}_improvement_all2025_trainall.csv")
    imp = imp[imp["noise"] == HEADLINE]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for strat, colour, style, name in (("greedy", GREEN, "-", "Weather-informed"),
                                       ("kmeans", TEAL, "-", "Evenly spread"),
                                       ("qr", GREY, "--", "QR pivoting"),
                                       ("random_mean", GREY, ":", "Random")):
        s = sc[sc["strategy"] == strat].sort_values("N")
        ax.plot(s["N"], s["test_rmse"], style, color=colour, lw=2 if colour != GREY else 1.5, label=name)
    ex = sc[sc["strategy"] == "existing"]
    if len(ex):
        ax.plot(ex["N"], ex["test_rmse"], "*", color=INK, ms=14, label="Existing EPA network")
    ax.axvline(HEADLINE_N[city], color=MUTED, ls=":", lw=1)
    km = sc[sc["strategy"] == "kmeans"]["test_rmse"]
    ax.set_ylim(0, 1.5 * km.max())
    ax.set_xlabel("Number of sensors")
    ax.set_ylabel("Map error on 2025 days, RMSE (µg/m³)")
    ax.set_title(f"{CITY_NAME[city]}: map error by placement strategy")
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(imp["N"], 100 * imp["improvement"], color=GREEN)
    ax.fill_between(imp["N"], 100 * imp["ci_low"], 100 * imp["ci_high"], color=GREEN, alpha=0.18, lw=0)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvline(HEADLINE_N[city], color=MUTED, ls=":", lw=1)
    ax.set_xlabel("Number of sensors")
    ax.set_ylabel("Accuracy gain over even spacing (%)")
    ax.set_title(f"{CITY_NAME[city]}: gain from weather-informed placement")
    fig.text(0.5, -0.02, "Dotted line: the city's real network size. Shading: 95% confidence interval.",
             ha="center", color=MUTED, fontsize=10)
    fig.tight_layout()
    save(fig, f"5_main_test_{city}.png")


def fig6_k_sensitivity():
    """Phase 8: the predictability result at the three spreading rates K."""
    order = ["L0", "L1", "L2_setA", "L2_setB", "L3_setA", "L3_setB", "L4_setA", "L4_setB", "L5"]
    labels = ["Known\nexactly", "Noisy\noutput", "10% fixed\nsources (A)", "10% fixed\nsources (B)",
              "30% fixed\nsources (A)", "30% fixed\nsources (B)", "30% fixed +\nnoise (A)",
              "30% fixed +\nnoise (B)", "Randomly\nmoving"]
    # one hue, light -> dark = slow -> fast spreading; headline K in the main green
    ks = [(200, "_K200", "#7fbf95", "v", "Slow spreading (200 m²/s)"),
          (1000, "", GREEN, "o", "Main analysis (1,000 m²/s)"),
          (8000, "_K8000", "#08391f", "^", "Fast spreading (8,000 m²/s)")]
    res = {k: json.load(open(P5 / f"phase5_emission_predictability{t}.json")) for k, t, *_ in ks}
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True, sharey=True, layout="constrained")
    for ax, city in zip(axes, ("singapore", "melbourne")):
        n = HEADLINE_N[city]
        for (k, _, col, mk, lab), off in zip(ks, (-0.22, 0, 0.22)):
            v = np.array([res[k][city][lv]["improvement"][str(n)] for lv in order]) * 100
            ax.errorbar(np.arange(len(order)) + off, v[:, 0],
                        yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=mk, ms=8,
                        color=col, mec="white", mew=1, capsize=3, elinewidth=1.5, label=lab)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(f"{CITY_NAME[city]} ({n} sensors, its real network size)", loc="left", fontsize=13)
        ax.set_ylabel("Accuracy gain (%)")
        ax.set_ylim(-8, 45)
        ax.legend(loc="upper left", fontsize=11, ncol=3)
    axes[-1].set_xlabel("How predictable the pollution sources are (most → least predictable)")
    fig.suptitle("Does the result depend on how fast pollution spreads?\n"
                 "Accuracy gain of weather-informed over evenly spread sensors at three spreading rates. "
                 "Error bars: 95% confidence intervals", fontsize=14)
    save(fig, "6_spreading_rate_check.png")


if __name__ == "__main__":
    for c in ("singapore", "melbourne"):
        fig1_plumes(c)
        fig2_networks(c)
    fig3_predictability()
    fig4_sensor_count()
    for c in ("singapore", "melbourne"):
        fig5_main_test(c)
    fig6_k_sensitivity()
    print(f"wrote {OUT}")
