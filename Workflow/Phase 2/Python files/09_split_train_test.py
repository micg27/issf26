"""
(runs before 08)
splits days into training/testing days
"""
import pandas as pd

from sklearn.model_selection import train_test_split

import config
import utils


def split(city):    # labels all 2025 as test, 2021-2024 as train, check that all weather regimes have both train and test days

    utils.banner(f"TRAIN/TEST SPLIT: {city}  (mode={config.SPLITMODE})")

    base = config.processed_dir(city)

    daily = pd.read_csv(
        base / f"{city}_regimes.csv",
        parse_dates=["date"],
    )

    if config.SPLITMODE == "stratified_random":

        trainidx, testidx = train_test_split(
            daily.index,
            test_size=config.TESTSIZE,
            random_state=config.SEED,
            stratify=daily["regime"],
        )

        daily["split"] = "train"
        daily.loc[testidx, "split"] = "test"

    elif config.SPLITMODE == "temporal":

        isholdout = daily["date"].dt.year.isin(config.HOLDOUT_YRS)

        daily["split"] = "train"
        daily.loc[isholdout, "split"] = "test"

        if not isholdout.any():
            raise ValueError(
                f"No days found in HOLDOUT_YEARS={config.HOLDOUT_YRS}"
            )

    else:
        raise ValueError(
            f"Unknown SPLIT_MODE: {config.SPLITMODE!r}"
        )
      
    table = pd.crosstab(daily["regime"], daily["split"])

    for column in ["train", "test"]:
        if column not in table.columns:
            raise ValueError(f"No days assigned to {column}.")

    empty = table[(table["train"] == 0) | (table["test"] == 0)]
  
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
