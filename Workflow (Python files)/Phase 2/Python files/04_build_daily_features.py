# 04_build_daily_features.py
"""
Hourly NetCDF (all years) -> one row per local day of clustering features.

Features: mean u and v in each of four 6-hour local-time blocks, plus daily
mean boundary-layer height.

Usage:
    python 04_build_daily_features.py --all
    python 04_build_daily_features.py --city melbourne
"""

import pandas as pd
import xarray as xr

import config
import utils


def build(city):

    utils.banner(f"DAILY FEATURES: {city}")

    in_dir = config.processed_dir(city)

    files = [
        in_dir / f"{city}_hourly_{year}.nc"
        for year in config.YEARS
    ]

    missing = [f for f in files if not f.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing processed files (run 02 first):\n  "
            + "\n  ".join(str(f) for f in missing)
        )

    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.sortby("time")

    # -----------------------------------------------------
    # 1. SPATIAL MEAN
    # -----------------------------------------------------
    #
    # Regimes are defined at city scale: one row per timestep, not per
    # grid cell. Spatial structure of transport is handled in Phase 3.
    # -----------------------------------------------------

    dims = utils.spatial_dims(ds)

    city_mean = ds.mean(dim=dims) if dims else ds

    df = (
        city_mean[["u10", "v10", "blh"]]
        .to_dataframe()
        .reset_index()
    )

    df["time"] = pd.to_datetime(df["time"])

    # -----------------------------------------------------
    # 2. LOCAL TIME
    # -----------------------------------------------------

    df["local_time"] = utils.to_local(df["time"], city)

    df["date"] = df["local_time"].dt.normalize()
    df["hour"] = df["local_time"].dt.hour

    df["block"] = df["hour"].map(utils.assign_block)

    # -----------------------------------------------------
    # 3. DROP INCOMPLETE DAYS
    # -----------------------------------------------------
    #
    # The first and last local days of the record are always partial
    # because of the UTC offset. If USE_DST is True, the two DST
    # transition days each year are also dropped (23 or 25 hours).
    # -----------------------------------------------------

    hours_per_day = df.groupby("date").size()

    complete = hours_per_day[hours_per_day == 24].index

    dropped = len(hours_per_day) - len(complete)

    print(f"  Days with 24 hours: {len(complete)}  (dropped {dropped})")

    df = df[df["date"].isin(complete)].copy()

    # -----------------------------------------------------
    # 4. BLOCK MEANS -> WIDE
    # -----------------------------------------------------

    block = (
        df.groupby(["date", "block"])[["u10", "v10"]]
        .mean()
        .reset_index()
    )

    u = block.pivot(index="date", columns="block", values="u10")
    v = block.pivot(index="date", columns="block", values="v10")

    u.columns = [f"u{i}" for i in u.columns]
    v.columns = [f"v{i}" for i in v.columns]

    # -----------------------------------------------------
    # 5. DAILY PBLH
    # -----------------------------------------------------

    pblh = df.groupby("date")["blh"].mean().rename("pblh_mean")

    # -----------------------------------------------------
    # 6. COMBINE
    # -----------------------------------------------------

    daily = pd.concat([u, v, pblh], axis=1)

    expected_cols = set(config.FEATURES)
    actual_cols = set(daily.columns)

    if expected_cols != actual_cols:
        raise ValueError(
            f"Feature columns do not match config.FEATURES.\n"
            f"  Missing: {expected_cols - actual_cols}\n"
            f"  Extra:   {actual_cols - expected_cols}"
        )

    daily = daily[config.FEATURES].reset_index()

    daily = daily.sort_values("date").reset_index(drop=True)

    # -----------------------------------------------------
    # 7. CHECK
    # -----------------------------------------------------

    print(f"\n  Rows: {len(daily)}")
    print(f"  Date range: {daily['date'].min().date()} "
          f"to {daily['date'].max().date()}")

    if daily.isna().any().any():
        print(daily.isna().sum())
        raise ValueError("Missing daily feature values detected.")

    print("\n  Feature summary:")
    print(daily[config.FEATURES].describe().round(2).T.to_string())

    # -----------------------------------------------------
    # 8. SAVE
    # -----------------------------------------------------

    out = in_dir / f"{city}_daily_features.csv"

    daily.to_csv(out, index=False)

    print(f"\n  Saved {out}")


def main():

    cities = utils.parse_cities("Build daily clustering features.")

    for city in cities:
        build(city)


if __name__ == "__main__":
    main()
