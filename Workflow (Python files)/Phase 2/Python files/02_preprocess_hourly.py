# 02_preprocess_hourly.py
"""
Raw ERA5 -> clean hourly NetCDF, one file per city per year.

Replaces your 02_preprocess_hourly.py + the 02a_modification.py stub.
The logic is now a function, so it loops over cities and years instead of
requiring you to hand-edit CITY and YEAR and re-run.

Also handles NetCDF input as well as GRIB.

Usage:
    python 02_preprocess_hourly.py --all
    python 02_preprocess_hourly.py --city melbourne --year 2023
"""

import numpy as np
import xarray as xr

import config
import utils


def process(city, year):

    utils.banner(f"PREPROCESS: {city} {year}")

    out_dir = config.processed_dir(city)
    outfile = out_dir / f"{city}_hourly_{year}.nc"

    if outfile.exists():
        print(f"  Already exists, skipping: {outfile.name}")
        return

    files = utils.find_raw_files(city, year)

    print("  Input files:")
    for file in files:
        print(f"    {file.name}")

    # -----------------------------------------------------
    # 1. OPEN AND COLLECT REQUIRED VARIABLES
    # -----------------------------------------------------

    found = {}

    for file in files:

        for group in utils.open_any(file):

            group = utils.standardize_time(group)

            for var in config.REQUIRED_VARS:
                if var in group.data_vars and var not in found:
                    found[var] = group[[var]]

    missing = config.REQUIRED_VARS - set(found)

    if missing:
        raise ValueError(
            f"Missing required variables for {city} {year}: {missing}\n"
            f"Found: {sorted(found)}"
        )

    # -----------------------------------------------------
    # 2. MERGE
    # -----------------------------------------------------

    ds = xr.merge(list(found.values()), join="inner")

    ds = utils.standardize_time(ds)

    ds = ds.sortby("time")

    # -----------------------------------------------------
    # 3. SUBSET TO THE TARGET YEAR
    # -----------------------------------------------------
    #
    # Needed when the raw file spans several years (e.g. a single
    # multi-year download).
    # -----------------------------------------------------

    ds = ds.sel(time=str(year))

    if ds.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No timesteps for {year} in the input files for {city}."
        )

    # -----------------------------------------------------
    # 4. DROP DUPLICATE TIMESTAMPS
    # -----------------------------------------------------

    times = ds["time"].values

    if len(np.unique(times)) != len(times):
        print("  Duplicate timestamps found; keeping first occurrence.")
        _, keep = np.unique(times, return_index=True)
        ds = ds.isel(time=np.sort(keep))

    # -----------------------------------------------------
    # 5. DERIVED VARIABLES
    # -----------------------------------------------------

    ds["wind_speed"] = np.hypot(ds["u10"], ds["v10"])
    ds["wind_speed"].attrs = {
        "long_name": "10 metre wind speed",
        "units": "m s-1",
    }

    # Meteorological convention: direction the wind blows FROM.
    # Kept for QC plots and reporting only. It is deliberately NOT a
    # clustering feature: degrees wrap at 360, so k-means would treat
    # 350 and 10 as far apart. Clustering uses u/v instead.
    ds["wind_direction"] = (
        270.0 - np.degrees(np.arctan2(ds["v10"], ds["u10"]))
    ) % 360.0
    ds["wind_direction"].attrs = {
        "long_name": "Meteorological wind direction (from)",
        "units": "degrees",
    }

    ds["t2m_c"] = ds["t2m"] - 273.15
    ds["t2m_c"].attrs = {
        "long_name": "2 metre temperature",
        "units": "degree_Celsius",
    }

    # -----------------------------------------------------
    # 6. REPORT AND SAVE
    # -----------------------------------------------------

    n_hours = ds.sizes["time"]
    expected = 8784 if year % 4 == 0 else 8760

    print(f"  Timesteps: {n_hours} (expected {expected})")

    if n_hours != expected:
        print(f"  WARNING: {expected - n_hours} hours missing.")

    dims = utils.spatial_dims(ds)
    print(f"  Grid dims: {[(d, ds.sizes[d]) for d in dims]}")

    ds.to_netcdf(outfile)

    print(f"  Saved {outfile}")


def main():

    cities, years = utils.parse_cities(
        "Preprocess raw ERA5 into hourly NetCDF.",
        with_year=True,
    )

    for city in cities:
        for year in years:
            process(city, year)


if __name__ == "__main__":
    main()
