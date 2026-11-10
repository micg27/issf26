# 10_build_phase3_days.py
"""
THIS IS THE SCRIPT THAT WAS MISSING FROM YOUR PIPELINE.

Your 10_extract_phase3_data.py reads <city>_phase3_days.csv with columns
date, regime, split -- but nothing in 01-09 produced that file. Scripts 08
and 09 each had half of it:

    08 -> representative_days.csv, sampled_days.csv   (date, regime)
    09 -> regimes_split.csv                           (date, regime, split)

This joins them into the manifest Phase 3 consumes.

Outputs <city>_phase3_days.csv with columns:
    date, regime, cluster_index, split, selection, w_k

w_k is carried through so Phase 3 can weight each simulated scenario by
its regime's historical frequency without re-reading the summary file.

Usage:
    python 10_build_phase3_days.py --all
    python 10_build_phase3_days.py --all --include-representative
"""

import argparse
import json

import pandas as pd

import config
import utils


def build(city, include_representative):

    utils.banner(f"PHASE 3 MANIFEST: {city}")

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

    if include_representative:

        rep_path = base / f"{city}_representative_days.csv"

        representatives = pd.read_csv(rep_path, parse_dates=["date"])

        frames.append(
            representatives[
                ["date", "regime", "cluster_index", "selection"]
            ]
        )

    selected = pd.concat(frames, ignore_index=True)

    # A representative day may also have been drawn in the random sample.
    # Keep one row per date, preferring the "representative" tag.
    selected["priority"] = (
        selected["selection"].eq("representative").astype(int)
    )

    selected = (
        selected
        .sort_values(["date", "priority"], ascending=[True, False])
        .drop_duplicates(subset="date", keep="first")
        .drop(columns="priority")
    )

    # -----------------------------------------------------
    # JOIN THE SPLIT LABEL
    # -----------------------------------------------------

    splits = pd.read_csv(
        required["split"], parse_dates=["date"]
    )[["date", "split"]]

    # 08 already carries a split column; drop it so the merge below is the
    # single source of truth rather than producing split_x / split_y.
    selected = selected.drop(columns=["split"], errors="ignore")

    manifest = selected.merge(splits, on="date", how="left")

    unmatched = manifest["split"].isna().sum()

    if unmatched:
        raise ValueError(
            f"{unmatched} selected days had no split label. "
            f"Scripts 08 and 09 may have been run on different versions "
            f"of {city}_regimes.csv -- re-run both."
        )

    # -----------------------------------------------------
    # ATTACH REGIME WEIGHTS
    # -----------------------------------------------------

    with open(required["weights"]) as f:
        weights = json.load(f)["weights"]

    manifest["w_k"] = manifest["regime"].map(weights)

    if manifest["w_k"].isna().any():
        raise ValueError(
            "Some regimes have no weight. Re-run 07."
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

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    print(f"\n  Total days selected: {len(manifest)}")

    print("\n  Days by regime and split:")
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

    parser.add_argument(
        "--include-representative",
        action="store_true",
        help=(
            "Also include the single centroid-nearest day per regime, "
            "tagged selection=representative."
        ),
    )

    args = parser.parse_args()

    cities = config.ALL_CITIES if args.all else [args.city]

    for city in cities:
        build(city, args.include_representative)


if __name__ == "__main__":
    main()
