# 00_download_era5.py
"""
Download ERA5 single-level data for every city and year.

FIXES vs your original ERA5_data_script.py:
  1. Years are 2021-2025 (from config), not 2020-2024. Your original window
     was off by one year at both ends relative to the research plan.
  2. Files are written per city AND per year into data/raw/meteorology/<city>/,
     which is where script 02 looks for them. The original wrote one
     multi-year file into the current working directory.
  3. Uses 'data_format' (the current CDS API key). The old 'format' key is
     ignored by the new CDS backend, which is why you may have received GRIB
     when you asked for NetCDF.
  4. Skips files that already exist, so an interrupted run can resume.
  5. Requests one month at a time, then merges into a single per-year file.
     A whole-year request (12 months x 31 days x 24 hours x 5 variables) is
     rejected by the CDS backend with "403 cost limits exceeded"; a
     one-month request is well under that limit.
  6. Each monthly download actually arrives as a zip (despite the '.nc' name
     and 'download_format: unarchived') because this variable mix spans two
     GRIB stepTypes (instant: u10/v10/t2m/blh, accum: tp), and CDS splits
     those into separate files inside the zip. This script transparently
     unzips, merges the stepType groups, and concatenates months, so the
     final per-year file on disk is genuine single-group NetCDF and script
     02 needs no changes.

Requires a ~/.cdsapirc with your CDS credentials.
"""

import zipfile
from tempfile import TemporaryDirectory

import cdsapi
import xarray as xr

import config


DATA_FORMAT = "netcdf"      # "netcdf" or "grib" - script 02 handles both

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(24)]


def load_month_dataset(path):
    """Open one month's download, transparently unzipping if needed."""

    if not zipfile.is_zipfile(path):
        return xr.load_dataset(path)

    with TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp_dir)
            names = zf.namelist()

        groups = [xr.load_dataset(f"{tmp_dir}/{name}") for name in names]

    return xr.merge(groups, join="inner", compat="override")


def download(client, city, year):

    cfg = config.city_config(city)

    out_dir = config.raw_dir(city)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "nc" if DATA_FORMAT == "netcdf" else "grib"

    target = out_dir / f"{city}_era5_{year}.{suffix}"

    if target.exists():
        print(f"  Already present, skipping: {target.name}")
        return

    month_files = []

    for month in MONTHS:

        month_file = out_dir / f"_tmp_{city}_era5_{year}_{month}.{suffix}"
        month_files.append(month_file)

        if month_file.exists():
            print(f"  Month already present, skipping: {month_file.name}")
            continue

        print(f"  Requesting {city} {year}-{month} -> {month_file.name}")

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "data_format": DATA_FORMAT,
                "download_format": "unarchived",
                "variable": config.ERA5_VARIABLES,
                "year": [str(year)],
                "month": [month],
                "day": DAYS,
                "time": TIMES,
                "area": cfg["area"],       # [North, West, South, East]
            },
            str(month_file),
        )

    print(f"  Merging {len(month_files)} months -> {target.name}")

    monthly = [load_month_dataset(f) for f in month_files]
    time_dim = "valid_time" if "valid_time" in monthly[0].dims else "time"

    ds = xr.concat(monthly, dim=time_dim).sortby(time_dim)
    ds.to_netcdf(target)

    for month_file in month_files:
        month_file.unlink()

    print(f"  Saved {target}")


def main():

    client = cdsapi.Client()

    for city in config.ALL_CITIES:

        print(f"\n--- {city.upper()} ---")

        for year in config.YEARS:
            download(client, city, year)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
