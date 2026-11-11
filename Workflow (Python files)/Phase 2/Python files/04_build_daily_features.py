import pandas as pd
import xarray as xr

import config
import utils


def build(city):

    utils.banner(f"DAILY FEATURES: {city}")

    indir = config.processed_dir(city)

    files = [
        indir / f"{city}_hourly_{year}.nc"
        for year in config.YEARS
    ]

    missing = [f for f in files if not f.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing processed files:\n  "
            + "\n  ".join(str(f) for f in missing)
        )

    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.sortby("time")

    dims = utils.spatial_dims(ds)

    citymean = ds.mean(dim=dims) if dims else ds

    df = (
        citymean[["u10", "v10", "blh"]]
        .to_dataframe()
        .reset_index()
    )

    df["time"] = pd.to_datetime(df["time"])

    df["local_time"] = utils.to_local(df["time"], city)

    df["date"] = df["local_time"].dt.normalize()
    df["hour"] = df["local_time"].dt.hour

    df["block"] = df["hour"].map(utils.assign_block)
    
    hrsperday = df.groupby("date").size()

    complete = hrsperday[hrsperday == 24].index

    dropped = len(hrsperday) - len(complete)

    print(f"  Days with 24 hours: {len(complete)}  (dropped {dropped})")

    df = df[df["date"].isin(complete)].copy()

    block = (
        df.groupby(["date", "block"])[["u10", "v10"]]
        .mean()
        .reset_index()
    )

    u = block.pivot(index="date", columns="block", values="u10")
    v = block.pivot(index="date", columns="block", values="v10")

    u.columns = [f"u{i}" for i in u.columns]
    v.columns = [f"v{i}" for i in v.columns]

    pblh = df.groupby("date")["blh"].mean().rename("pblh_mean")

    daily = pd.concat([u, v, pblh], axis=1)

    expectedcols = set(config.FEATURES)
    actualcols = set(daily.columns)

    if expectedcols != actualcols:
        raise ValueError(
            f"Feature columns do not match config.FEATURES.\n"
            f"  Missing: {expectedcols - actualcols}\n"
            f"  Extra:   {actualcols - expectedcols}"
        )

    daily = daily[config.FEATURES].reset_index()

    daily = daily.sort_values("date").reset_index(drop=True)

    print(f"\n  Rows: {len(daily)}")
    print(f"  Date range: {daily['date'].min().date()} "
          f"to {daily['date'].max().date()}")

    if daily.isna().any().any():
        print(daily.isna().sum())
        raise ValueError("Missing daily feature values")

    print("\n  Feature summary:")
    print(daily[config.FEATURES].describe().round(2).T.to_string())

    out = indir / f"{city}_daily_features.csv"

    daily.to_csv(out, index=False)

    print(f"\n  Saved {out}")

def main():

    cities = utils.parse_cities("Build daily clustering features.")

    for city in cities:
        build(city)


if __name__ == "__main__":
    main()
