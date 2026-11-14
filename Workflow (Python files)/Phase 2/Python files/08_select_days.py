"""
chooses days to simulate in the original design. 
"""

import json

import numpy as np
import pandas as pd

import config
import utils


def select(city):  # finds days closest to center of cluster (most typical/average features), randomly picks 14 training and 6 test days per type

    utils.banner(f"DAY SELECTION: {city}")

    base = config.processed_dir(city)

    splitpath = base / f"{city}_regimes_split.csv"
  
    daily = pd.read_csv(splitpath, parse_dates=["date"])

    centroids = pd.read_csv(
        base / f"{city}_cluster_centroids_standardized.csv"
    )

    with open(base / f"{city}_scaler.json") as f:
        scalerinfo = json.load(f)


    mean = np.array(scalerinfo["mean"])
    scale = np.array(scalerinfo["scale"])

    Z = (daily[config.FEATURES].values - mean) / scale
  
    representatives = []

    for _, centroidrow in centroids.iterrows():

        clusteridx = int(centroidrow["cluster_index"])
        regime = centroidrow["regime"]

        mask = daily["cluster_index"].values == clusteridx
        positions = np.where(mask)[0]

        if positions.size == 0:
            print(f"  WARNING: no days in cluster {clusteridx}")
            continue

        centroid = centroidrow[config.FEATURES].values.astype(float)

        distances = np.linalg.norm(Z[positions] - centroid, axis=1)

        best = positions[int(np.argmin(distances))]

        representatives.append(
            {
                "date": daily.loc[best, "date"],
                "regime": regime,
                "cluster_index": clusteridx,
                "distance_to_centroid": float(distances.min()),
                "selection": "representative",
            }
        )

    representatives = pd.DataFrame(representatives)

    representatives.to_csv(
        base / f"{city}_representative_days.csv", index=False
    )

    total = config.N_PER_REGIME

    ntest = max(1, round(total * config.TESTSIZE))
    ntrain = max(1, total - ntest)

    quota = {"train": ntrain, "test": ntest}

    print(
        f"\n  Target per regime: {ntrain} train + {ntest} test "
        f"= {ntrain + ntest}"
    )

    samples = []

    for regime, subset in daily.groupby("regime"):

        for splitname, target in quota.items():

            pool = subset[subset["split"] == splitname]

            n = min(target, len(pool))
          
            chosen = pool.sample(
                n=n, random_state=config.SEED
            )

            samples.append(
                chosen[["date", "regime", "cluster_index", "split"]]
            )

    samples = pd.concat(samples, ignore_index=True)
    samples["selection"] = "sampled"

    samples = samples.sort_values(
        ["regime", "date"]
    ).reset_index(drop=True)

    samples.to_csv(
        base / f"{city}_sampled_days.csv", index=False
    )

    report = representatives[
        ["regime", "date", "distance_to_centroid"]
    ].copy()

    report["date"] = report["date"].dt.date
    report["distance_to_centroid"] = (
        report["distance_to_centroid"].round(3)
    )

def main():

    cities = utils.parse_cities("Select days for Phase 3.")

    for city in cities:
        select(city)

if __name__ == "__main__":
    main()
