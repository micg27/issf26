import json

import numpy as np
import pandas as pd

import config
import utils


def select(city):

    utils.banner(f"DAY SELECTION: {city}")

    base = config.processed_dir(city)

    split_path = base / f"{city}_regimes_split.csv"
  
    daily = pd.read_csv(split_path, parse_dates=["date"])

    centroids = pd.read_csv(
        base / f"{city}_cluster_centroids_standardized.csv"
    )

    with open(base / f"{city}_scaler.json") as f:
        scaler_info = json.load(f)


    mean = np.array(scaler_info["mean"])
    scale = np.array(scaler_info["scale"])

    Z = (daily[config.FEATURES].values - mean) / scale
  
    representatives = []

    for _, centroid_row in centroids.iterrows():

        cluster_index = int(centroid_row["cluster_index"])
        regime = centroid_row["regime"]

        mask = daily["cluster_index"].values == cluster_index
        positions = np.where(mask)[0]

        if positions.size == 0:
            print(f"  WARNING: no days in cluster {cluster_index}")
            continue

        centroid = centroid_row[config.FEATURES].values.astype(float)

        distances = np.linalg.norm(Z[positions] - centroid, axis=1)

        best = positions[int(np.argmin(distances))]

        representatives.append(
            {
                "date": daily.loc[best, "date"],
                "regime": regime,
                "cluster_index": cluster_index,
                "distance_to_centroid": float(distances.min()),
                "selection": "representative",
            }
        )

    representatives = pd.DataFrame(representatives)

    representatives.to_csv(
        base / f"{city}_representative_days.csv", index=False
    )

    total = config.N_SAMPLE_PER_REGIME

    n_test = max(1, round(total * config.TEST_SIZE))
    n_train = max(1, total - n_test)

    quota = {"train": n_train, "test": n_test}

    print(
        f"\n  Target per regime: {n_train} train + {n_test} test "
        f"= {n_train + n_test}"
    )

    samples = []

    for regime, subset in daily.groupby("regime"):

        for split_name, target in quota.items():

            pool = subset[subset["split"] == split_name]

            n = min(target, len(pool))
          
            chosen = pool.sample(
                n=n, random_state=config.RANDOM_SEED
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
