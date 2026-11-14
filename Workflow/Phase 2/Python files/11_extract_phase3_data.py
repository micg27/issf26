"""
prepare weather inputs for pollution model 
"""


import pandas as pd
import xarray as xr
import config
import utils

VARS_OUT = ["u10", "v10", "blh"]

def extract(city):   # cuts wind and blh, labels with date, type and split, skips incomplete days

    utils.banner(f"{city}")

    base = config.processed_dir(city)

    manifestpath = base / f"{city}_phase3_days.csv"

    if not manifestpath.exists():
        raise FileNotFoundError(
            f"{manifestpath} is missing oops"
        )

    selected = pd.read_csv(manifestpath, parse_dates=["date"])

    files = [
        base / f"{city}_hourly_{year}.nc"
        for year in config.YEARS
    ]

    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.sortby("time")

    outdir = base / "phase3_weather"
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = []

    for _, row in selected.iterrows():

        date = row["date"]

        utcstart, utcend = utils.local_day_bounds_utc(date, city)

        day = ds.sel(time=slice(utcstart, utcend))

        nhours = day.sizes.get("time", 0)

        if nhours != 24:
            print(
                f"  SKIP {date.date()}: {nhours} hours "
                f"(expected 24)"
            )
            skipped.append(str(date.date()))
            continue

        filename = (
            f"{city}_{date.date()}_{row['regime']}_{row['split']}.nc"
        )

        outfile = outdir / filename

        subset = day[VARS_OUT]

        subset.attrs.update(
            {
                "city": city,
                "local_date": str(date.date()),
                "regime": str(row["regime"]),
                "split": str(row["split"]),
                "selection": str(row["selection"]),
                "w_k": float(row["w_k"]),
                "utc_start": str(utcstart),
                "utc_end": str(utcend),
                "use_dst": str(config.USE_DST),
            }
        )

        subset.to_netcdf(outfile)

        records.append(
            {
                "file": filename,
                "city": city,
                "date": date.date(),
                "regime": row["regime"],
                "split": row["split"],
                "selection": row["selection"],
                "w_k": row["w_k"],
            }
        )

    if not records:
        raise RuntimeError(
            "No days extracted."
        )

    out_manifest = pd.DataFrame(records)

    out_manifest.to_csv(
        outdir / f"{city}_phase3_manifest.csv", index=False
    )

    print(f"\n  Extracted {len(records)} days to {outdir}")

    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped}")

    print("\n  Files by regime and split:")
    print(
        pd.crosstab(
            out_manifest["regime"], out_manifest["split"]
        ).to_string()
    )


def main():

    cities = utils.parse_cities("Extract P3 fields pls")

    for city in cities:
        extract(city)


if __name__ == "__main__":
    main()
