# 11_extract_phase3_data.py
"""
Write one NetCDF file per selected day, containing the 24 hourly wind and
boundary-layer fields that Phase 3's advection-diffusion model will use.

Local day boundaries are converted back to UTC using the SAME helper that
script 04 used to build the features, so the day you simulate is exactly
the day you clustered. (Previously this logic was duplicated inline in two
scripts, which is how these drift apart.)

Also writes phase3_manifest.csv, mapping each output file to its regime,
split and w_k, so Phase 3 can iterate without re-deriving anything.

Usage:
    python 11_extract_phase3_data.py --all
"""

import pandas as pd
import xarray as xr

import config
import utils


EXTRACT_VARS = ["u10", "v10", "blh"]


def extract(city):

    utils.banner(f"EXTRACT PHASE 3 DATA: {city}")

    base = config.processed_dir(city)

    manifest_path = base / f"{city}_phase3_days.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found; run 10_build_phase3_days.py first."
        )

    selected = pd.read_csv(manifest_path, parse_dates=["date"])

    files = [
        base / f"{city}_hourly_{year}.nc"
        for year in config.YEARS
    ]

    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.sortby("time")

    out_dir = base / "phase3_weather"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = []

    for _, row in selected.iterrows():

        date = row["date"]

        utc_start, utc_end = utils.local_day_bounds_utc(date, city)

        day = ds.sel(time=slice(utc_start, utc_end))

        n_hours = day.sizes.get("time", 0)

        if n_hours != 24:
            print(
                f"  SKIP {date.date()}: {n_hours} hours "
                f"(expected 24)"
            )
            skipped.append(str(date.date()))
            continue

        filename = (
            f"{city}_{date.date()}_{row['regime']}_{row['split']}.nc"
        )

        outfile = out_dir / filename

        subset = day[EXTRACT_VARS]

        # Record provenance inside the file itself so a stray NetCDF is
        # never ambiguous later.
        subset.attrs.update(
            {
                "city": city,
                "local_date": str(date.date()),
                "regime": str(row["regime"]),
                "split": str(row["split"]),
                "selection": str(row["selection"]),
                "w_k": float(row["w_k"]),
                "utc_start": str(utc_start),
                "utc_end": str(utc_end),
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
            "No days were extracted. Check your date range and offsets."
        )

    out_manifest = pd.DataFrame(records)

    out_manifest.to_csv(
        out_dir / f"{city}_phase3_manifest.csv", index=False
    )

    print(f"\n  Extracted {len(records)} days to {out_dir}")

    if skipped:
        print(f"  Skipped {len(skipped)}: {skipped}")

    print("\n  Files by regime and split:")
    print(
        pd.crosstab(
            out_manifest["regime"], out_manifest["split"]
        ).to_string()
    )


def main():

    cities = utils.parse_cities("Extract per-day fields for Phase 3.")

    for city in cities:
        extract(city)


if __name__ == "__main__":
    main()
