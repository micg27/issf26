# 07_characterize_regimes.py
"""
Describe each regime physically and compute the frequency weights
w_k = n_k / N that Phase 3 uses to weight scenarios.

Outputs:
  <city>_regime_summary.csv            per-regime stats including w_k
  <city>_regime_monthly_frequency.csv  seasonality check
  <city>_regime_weights.json           w_k in a form Phase 3 can import
  figures: diurnal wind vectors per regime, monthly occurrence

The diurnal figure is the important one for interpretation: it shows how
the wind vector rotates through the day in each regime, which is what
distinguishes a sea-breeze regime from a synoptically forced one.

Usage:
    python 07_characterize_regimes.py --all
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils


BLOCK_LABELS = {
    1: "00-06",
    2: "06-12",
    3: "12-18",
    4: "18-24",
}


def characterize(city):

    utils.banner(f"REGIME CHARACTERIZATION: {city}")

    base = config.processed_dir(city)
    out = config.output_dir(city, "regimes")

    daily = pd.read_csv(
        base / f"{city}_regimes.csv",
        parse_dates=["date"],
    )

    # -----------------------------------------------------
    # DERIVED QUANTITIES
    # -----------------------------------------------------

    for block in range(1, config.N_BLOCKS + 1):
        daily[f"speed{block}"] = np.hypot(
            daily[f"u{block}"], daily[f"v{block}"]
        )

    speed_cols = [
        f"speed{b}" for b in range(1, config.N_BLOCKS + 1)
    ]

    daily["mean_wind_speed"] = daily[speed_cols].mean(axis=1)

    # Vector mean speed: the magnitude of the daily-mean wind vector.
    # Much smaller than mean_wind_speed when the wind reverses during the
    # day, so the ratio is a direct measure of diurnal reversal, which is
    # exactly the sea-breeze signal you care about for Singapore.
    u_cols = [f"u{b}" for b in range(1, config.N_BLOCKS + 1)]
    v_cols = [f"v{b}" for b in range(1, config.N_BLOCKS + 1)]

    daily["u_daily"] = daily[u_cols].mean(axis=1)
    daily["v_daily"] = daily[v_cols].mean(axis=1)

    daily["vector_mean_speed"] = np.hypot(
        daily["u_daily"], daily["v_daily"]
    )

    daily["steadiness"] = (
        daily["vector_mean_speed"]
        / daily["mean_wind_speed"].replace(0, np.nan)
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    agg = {
        "n_days": ("date", "count"),
        "mean_wind_speed": ("mean_wind_speed", "mean"),
        "vector_mean_speed": ("vector_mean_speed", "mean"),
        "steadiness": ("steadiness", "mean"),
        "mean_pblh": ("pblh_mean", "mean"),
        "std_pblh": ("pblh_mean", "std"),
    }

    for block in range(1, config.N_BLOCKS + 1):
        agg[f"u{block}"] = (f"u{block}", "mean")
        agg[f"v{block}"] = (f"v{block}", "mean")

    summary = daily.groupby("regime").agg(**agg).reset_index()

    summary["w_k"] = summary["n_days"] / len(daily)

    summary = summary.sort_values("regime").reset_index(drop=True)

    summary.to_csv(
        base / f"{city}_regime_summary.csv", index=False
    )

    # -----------------------------------------------------
    # WEIGHTS FOR PHASE 3
    # -----------------------------------------------------

    weights = dict(
        zip(summary["regime"], summary["w_k"].round(6))
    )

    total = float(sum(weights.values()))

    if abs(total - 1.0) > 1e-6:
        print(f"  WARNING: weights sum to {total}, not 1.")

    with open(base / f"{city}_regime_weights.json", "w") as f:
        json.dump(
            {
                "city": city,
                "n_days": int(len(daily)),
                "weights": weights,
            },
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # MONTHLY OCCURRENCE
    # -----------------------------------------------------

    daily["month"] = daily["date"].dt.month

    monthly = pd.crosstab(
        daily["month"], daily["regime"], normalize="index"
    )

    monthly.to_csv(
        base / f"{city}_regime_monthly_frequency.csv"
    )

    # -----------------------------------------------------
    # FIGURE 1: DIURNAL WIND VECTORS BY REGIME
    # -----------------------------------------------------

    regimes = sorted(daily["regime"].unique())

    fig, axes = plt.subplots(
        1, len(regimes),
        figsize=(3.2 * len(regimes), 3.4),
        squeeze=False,
    )

    limit = float(
        np.nanmax(
            np.abs(summary[[f"u{b}" for b in range(1, 5)]
                           + [f"v{b}" for b in range(1, 5)]].values)
        )
    ) * 1.3 + 0.5

    for ax, regime in zip(axes[0], regimes):

        row = summary[summary["regime"] == regime].iloc[0]

        for block in range(1, config.N_BLOCKS + 1):
            ax.arrow(
                0, 0,
                row[f"u{block}"], row[f"v{block}"],
                head_width=limit * 0.05,
                length_includes_head=True,
                alpha=0.85,
            )
            ax.annotate(
                BLOCK_LABELS[block],
                (row[f"u{block}"], row[f"v{block}"]),
                fontsize=7,
            )

        ax.axhline(0, linewidth=0.5, color="grey")
        ax.axvline(0, linewidth=0.5, color="grey")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_title(
            f"{regime}\nw={row['w_k']:.2f}, n={int(row['n_days'])}",
            fontsize=9,
        )
        ax.set_xlabel("u (m/s)", fontsize=8)

    axes[0][0].set_ylabel("v (m/s)", fontsize=8)

    fig.suptitle(
        f"{city.title()}: mean wind vector by 6-hour block", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out / "regime_diurnal_wind.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------
    # FIGURE 2: MONTHLY OCCURRENCE
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4))

    monthly.plot(kind="bar", stacked=True, ax=ax, width=0.85)

    ax.set_xlabel("Month")
    ax.set_ylabel("Proportion of days")
    ax.set_title(f"{city.title()}: regime occurrence by month")
    ax.legend(title="Regime", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "regime_monthly_frequency.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    display = summary[
        [
            "regime", "n_days", "w_k", "mean_wind_speed",
            "vector_mean_speed", "steadiness", "mean_pblh",
        ]
    ]

    print("\n" + display.round(3).to_string(index=False))

    print("\n  Monthly proportions:")
    print(monthly.round(3).to_string())

    print(
        "\n  Interpretation check: a regime with low steadiness (well "
        "under ~0.5)"
    )
    print(
        "  has a wind that reverses during the day, i.e. a sea-breeze "
        "type. A"
    )
    print(
        "  regime with steadiness near 1 is synoptically forced. Confirm "
        "the"
    )
    print(
        "  monthly table matches known climatology before continuing."
    )

    print(f"\n  Saved outputs to {base} and figures to {out}")


def main():

    cities = utils.parse_cities("Characterize meteorological regimes.")

    for city in cities:
        characterize(city)


if __name__ == "__main__":
    main()
