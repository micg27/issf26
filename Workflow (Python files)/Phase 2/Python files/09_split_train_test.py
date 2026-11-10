# 09_split_train_test.py
"""
Assign every day to train or test.

Two modes, set by config.SPLIT_MODE:

  "stratified_random"
      Random days, stratified so each regime keeps its proportions.
      Caveat worth stating in your methods: consecutive days are
      meteorologically correlated, so a random split leaks information
      between train and test. Day 100 in train and day 101 in test are
      not independent samples.

  "temporal"
      Hold out whole years (config.HOLDOUT_YEARS). Cleaner independence,
      and the more defensible choice if a judge asks. The trade-off is
      that regime proportions in the held-out year may differ from the
      training years, which is itself informative.

Usage:
    python 09_split_train_test.py --all
"""

import pandas as pd

from sklearn.model_selection import train_test_split

import config
import utils


def split(city):

    utils.banner(f"TRAIN/TEST SPLIT: {city}  (mode={config.SPLIT_MODE})")

    base = config.processed_dir(city)

    daily = pd.read_csv(
        base / f"{city}_regimes.csv",
        parse_dates=["date"],
    )

    if config.SPLIT_MODE == "stratified_random":

        train_idx, test_idx = train_test_split(
            daily.index,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=daily["regime"],
        )

        daily["split"] = "train"
        daily.loc[test_idx, "split"] = "test"

    elif config.SPLIT_MODE == "temporal":

        is_holdout = daily["date"].dt.year.isin(config.HOLDOUT_YEARS)

        daily["split"] = "train"
        daily.loc[is_holdout, "split"] = "test"

        if not is_holdout.any():
            raise ValueError(
                f"No days found in HOLDOUT_YEARS={config.HOLDOUT_YEARS}"
            )

    else:
        raise ValueError(
            f"Unknown SPLIT_MODE: {config.SPLIT_MODE!r}"
        )

    # -----------------------------------------------------
    # CHECK EVERY REGIME APPEARS IN BOTH SPLITS
    # -----------------------------------------------------

    table = pd.crosstab(daily["regime"], daily["split"])

    for column in ["train", "test"]:
        if column not in table.columns:
            raise ValueError(f"No days assigned to {column}.")

    empty = table[(table["train"] == 0) | (table["test"] == 0)]

    if len(empty):
        print(
            f"  WARNING: some regimes are absent from one split:\n{empty}"
        )

    daily.to_csv(
        base / f"{city}_regimes_split.csv", index=False
    )

    print("\n" + table.to_string())

    print("\n  Proportions within regime:")
    print(
        pd.crosstab(
            daily["regime"], daily["split"], normalize="index"
        ).round(3).to_string()
    )

    print(f"\n  Saved {base / f'{city}_regimes_split.csv'}")


def main():

    cities = utils.parse_cities("Split days into train and test.")

    for city in cities:
        split(city)


if __name__ == "__main__":
    main()
