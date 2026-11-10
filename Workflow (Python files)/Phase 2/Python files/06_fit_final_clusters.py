# 06_fit_final_clusters.py
"""
Fit the final k-means model and persist everything downstream needs.

Writes:
  <city>_regimes.csv                        daily features + regime + cluster_index
  <city>_cluster_centroids_standardized.csv centroids in z-space
  <city>_cluster_centroids_physical.csv     centroids in original units
  <city>_scaler.json                        mean/scale for reuse

FIX vs your original: the raw 0-indexed cluster label is now stored as its
own column. Script 08 previously recovered it by alphabetically sorting
regime strings ("SIN1".."SIN4"), which works only while k < 10 and would
silently mis-assign centroids at k >= 10.

Usage:
    python 06_fit_final_clusters.py --all
"""

import json

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config
import utils


def fit(city):

    utils.banner(f"FINAL CLUSTERING: {city}")

    cfg = config.city_config(city)

    final_k = cfg["final_k"]

    if final_k is None:
        raise ValueError(
            f"config.CITIES['{city}']['final_k'] is None.\n"
            f"Run 05_cluster_sweep.py, inspect the diagnostics, then set "
            f"final_k in config.py."
        )

    base = config.processed_dir(city)

    daily = pd.read_csv(
        base / f"{city}_daily_features.csv",
        parse_dates=["date"],
    )

    X = daily[config.FEATURES]

    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=final_k,
        n_init=config.KMEANS_N_INIT,
        random_state=config.RANDOM_SEED,
    )

    labels = model.fit_predict(Z)

    # -----------------------------------------------------
    # ORDER CLUSTERS BY FREQUENCY
    # -----------------------------------------------------
    #
    # Makes labels stable and readable: regime 1 is always the most common.
    # Without this, k-means index order is arbitrary and the meaning of
    # "SIN1" could change if you re-run with a different seed.
    # -----------------------------------------------------

    order = (
        pd.Series(labels)
        .value_counts()
        .index
        .tolist()
    )

    remap = {old: new for new, old in enumerate(order)}

    cluster_index = np.array([remap[label] for label in labels])

    centers = model.cluster_centers_[order]

    prefix = cfg["label_prefix"]

    daily["cluster_index"] = cluster_index
    daily["regime"] = [f"{prefix}{i + 1}" for i in cluster_index]

    daily.to_csv(base / f"{city}_regimes.csv", index=False)

    # -----------------------------------------------------
    # CENTROIDS
    # -----------------------------------------------------

    centroids = pd.DataFrame(centers, columns=config.FEATURES)
    centroids.insert(0, "cluster_index", range(final_k))
    centroids.insert(
        1, "regime", [f"{prefix}{i + 1}" for i in range(final_k)]
    )

    centroids.to_csv(
        base / f"{city}_cluster_centroids_standardized.csv",
        index=False,
    )

    physical = pd.DataFrame(
        scaler.inverse_transform(centers),
        columns=config.FEATURES,
    )
    physical.insert(0, "cluster_index", range(final_k))
    physical.insert(
        1, "regime", [f"{prefix}{i + 1}" for i in range(final_k)]
    )

    physical.to_csv(
        base / f"{city}_cluster_centroids_physical.csv",
        index=False,
    )

    # -----------------------------------------------------
    # SCALER
    # -----------------------------------------------------

    with open(base / f"{city}_scaler.json", "w") as f:
        json.dump(
            {
                "features": config.FEATURES,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "k": final_k,
                "random_seed": config.RANDOM_SEED,
                "inertia": float(model.inertia_),
            },
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    counts = daily["regime"].value_counts().sort_index()

    print(f"\n  k = {final_k}, n = {len(daily)} days\n")
    print("  regime   days    w_k")

    for regime, n in counts.items():
        print(f"  {regime:8s} {n:5d}  {n / len(daily):.3f}")

    print("\n  Centroids in physical units:")
    print(
        physical.drop(columns=["cluster_index"])
        .round(2)
        .to_string(index=False)
    )

    print(f"\n  Saved outputs to {base}")


def main():

    cities = utils.parse_cities("Fit final k-means clusters.")

    for city in cities:
        fit(city)


if __name__ == "__main__":
    main()
