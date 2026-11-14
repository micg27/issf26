"""
makes a list of days to use in Phase 3
"""

import argparse
import json
import pandas as pd
import config
import utils

def build(city, incl_rep):  # attaches train or test label and weights to days, and saves the list for phase 3

    utils.banner(f"{city}")

    base = config.processed_dir(city)

    required = {
        "sampled": base / f"{city}_sampled_days.csv",
        "split": base / f"{city}_regimes_split.csv",
        "weights": base / f"{city}_regime_weights.json",
    }

    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} file: {path}\n"
                f"Run scripts 07, 08 and 09 first."
            )

    sampled = pd.read_csv(required["sampled"], parse_dates=["date"])

    frames = [sampled]

    if incl_rep:

        reppath = base / f"{city}_representative_days.csv"

        representatives = pd.read_csv(reppath, parse_dates=["date"])

        frames.append(
            representatives[
                ["date", "regime", "cluster_index", "selection"]
            ]
        )

    selected = pd.concat(frames, ignore_index=True)

    selected["priority"] = (
        selected["selection"].eq("representative").astype(int)
    )

    selected = (
        selected
        .sort_values(["date", "priority"], ascending=[True, False])
        .drop_duplicates(subset="date", keep="first")
        .drop(columns="priority")
    )
    
    splits = pd.read_csv(
        required["split"], parse_dates=["date"]
    )[["date", "split"]]

    selected = selected.drop(columns=["split"], errors="ignore")

    manifest = selected.merge(splits, on="date", how="left")

    unmatched = manifest["split"].isna().sum()

    if unmatched:
        raise ValueError(
            f"{unmatched} selected days had no split label. "
            f"Scripts 08 and 09 may have been run on different versions "
        )

    with open(required["weights"]) as f:
        weights = json.load(f)["weights"]

    manifest["w_k"] = manifest["regime"].map(weights)

    if manifest["w_k"].isna().any():
        raise ValueError(
            "Some regimes have no weight so we re-run 07"
        )

    manifest = (
        manifest
        .sort_values(["regime", "date"])
        .reset_index(drop=True)
    )

    manifest = manifest[
        ["date", "regime", "cluster_index", "split", "selection", "w_k"]
    ]

    out = base / f"{city}_phase3_days.csv"

    manifest.to_csv(out, index=False)

    print(f"\n  Total days selected: {len(manifest)}")

    print("\n  Days (by regime and split):")
    print(
        pd.crosstab(
            manifest["regime"], manifest["split"]
        ).to_string()
    )

    print("\n  Regime weights carried forward:")
    for regime, weight in sorted(weights.items()):
        print(f"    {regime}: {weight:.4f}")

    print(f"\n  Saved {out}")


def main():

    parser = argparse.ArgumentParser(
        description="Build the Phase 3 day manifest."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", choices=config.ALL_CITIES)
    group.add_argument("--all", action="store_true")

    args = parser.parse_args()

    cities = config.ALL_CITIES if args.all else [args.city]

    for city in cities:
        build(city, args.include_representative)

if __name__ == "__main__":
    main()
