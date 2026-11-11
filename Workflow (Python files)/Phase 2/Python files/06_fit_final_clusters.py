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

    finalk = cfg["final_k"]

    if finalk is None:
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
        n_clusters=finalk,
        n_init=config.KMEANS_INIT,
        random_state=config.SEED,
    )

    labels = model.fit_predict(Z)
  
    order = (
        pd.Series(labels)
        .value_counts()
        .index
        .tolist()
    )

    remap = {old: new for new, old in enumerate(order)}

    clusteridx = np.array([remap[label] for label in labels])

    centers = model.cluster_centers_[order]

    prefix = cfg["label_prefix"]

    daily["cluster_index"] = clusteridx
    daily["regime"] = [f"{prefix}{i + 1}" for i in clusteridx]

    daily.to_csv(base / f"{city}_regimes.csv", index=False)

    centroids = pd.DataFrame(centers, columns=config.FEATURES)
    centroids.insert(0, "cluster_index", range(finalk))
    centroids.insert(
        1, "regime", [f"{prefix}{i + 1}" for i in range(finalk)]
    )

    centroids.to_csv(
        base / f"{city}_cluster_centroids_standardized.csv",
        index=False,
    )

    physical = pd.DataFrame(
        scaler.inverse_transform(centers),
        columns=config.FEATURES,
    )
    physical.insert(0, "cluster_index", range(finalk))
    physical.insert(
        1, "regime", [f"{prefix}{i + 1}" for i in range(finalk)]
    )

    physical.to_csv(
        base / f"{city}_cluster_centroids_physical.csv",
        index=False,
    )

    with open(base / f"{city}_scaler.json", "w") as f:
        json.dump(
            {
                "features": config.FEATURES,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "k": finalk,
                "random_seed": config.SEED,
                "inertia": float(model.inertia_),
            },
            f,
            indent=2,
        )

    counts = daily["regime"].value_counts().sort_index()

    print(f"\n  k = {finalk}, n = {len(daily)} days\n")
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
