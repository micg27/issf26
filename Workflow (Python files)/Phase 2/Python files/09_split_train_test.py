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
