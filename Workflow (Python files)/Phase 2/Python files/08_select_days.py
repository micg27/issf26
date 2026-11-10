# 08_select_days.py
"""
Pick the days that Phase 3 will actually simulate.

Two selections:
  1. Representative day per regime: the day closest to its cluster centroid
     in standardized space. Useful for figures and for a deterministic
     "typical case" run.
  2. A random sample of N_SAMPLE_PER_REGIME days per regime, which gives
     Phase 3 within-regime variability rather than a single point estimate.

RUNS AFTER 09, not before. Sampling is stratified by BOTH regime and split,
so every regime is guaranteed to contribute train AND test days. Sampling
independently of the split (as the original scripts did) can leave a regime
with zero held-out days purely by chance, which would silently break the
Phase 5 evaluation for that regime.

FIX vs your original: the cluster centroid is matched using the
cluster_index column saved by 06, not by alphabetically sorting regime
label strings.

Usage:
    python 08_select_days.py --all
"""

import json

import numpy as np
import pandas as pd

import config
import utils


def select(city):

    utils.banner(f"DAY SELECTION: {city}")

    base = config.processed_dir(city)

    split_path = base / f"{city}_regimes_split.csv"

    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} not found.\n"
            f"Run 09_split_train_test.py BEFORE 08, so that sampling can "
            f"be stratified by split."
        )

    daily = pd.read_csv(split_path, parse_dates=["date"])

    if "cluster_index" not in daily.columns:
        raise ValueError(
            f"{city}_regimes.csv has no cluster_index column. "
            f"Re-run 06_fit_final_clusters.py."
        )

    centroids = pd.read_csv(
        base / f"{city}_cluster_centroids_standardized.csv"
    )

    with open(base / f"{city}_scaler.json") as f:
        scaler_info = json.load(f)

    if scaler_info["features"] != config.FEATURES:
        raise ValueError(
            "Saved scaler features do not match config.FEATURES. "
            "Re-run 06."
        )

    mean = np.array(scaler_info["mean"])
    scale = np.array(scaler_info["scale"])

    # -----------------------------------------------------
    # STANDARDIZE WITH THE SAVED SCALER
    # -----------------------------------------------------
    #
    # Reusing the fitted scaler rather than re-fitting keeps this
    # projection identical to the one used for clustering.
    # -----------------------------------------------------

    Z = (daily[config.FEATURES].values - mean) / scale

    # -----------------------------------------------------
    # REPRESENTATIVE DAY PER REGIME
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # RANDOM SAMPLE PER REGIME
    # -----------------------------------------------------

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

            if pool.empty:
                print(
                    f"  WARNING: regime {regime} has no {split_name} "
                    f"days at all. Phase 5 cannot evaluate it on "
                    f"{split_name}."
                )
                continue

            n = min(target, len(pool))

            if n < target:
                print(
                    f"  NOTE: regime {regime} has only {len(pool)} "
                    f"{split_name} days; taking all of them."
                )

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

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    report = representatives[
        ["regime", "date", "distance_to_centroid"]
    ].copy()

    report["date"] = report["date"].dt.date
    report["distance_to_centroid"] = (
        report["distance_to_centroid"].round(3)
    )

    print("\n  Representative days:")
    print(report.to_string(index=False))

    print("\n  Sampled days by regime and split:")
    print(
        pd.crosstab(samples["regime"], samples["split"]).to_string()
    )

    print(f"\n  Saved outputs to {base}")


def main():

    cities = utils.parse_cities("Select days for Phase 3.")

    for city in cities:
        select(city)


if __name__ == "__main__":
    main()
