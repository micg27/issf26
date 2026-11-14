"""
final download of all ERA5 weather data
"""

import zipfile
from tempfile import TemporaryDirectory

import cdsapi
import xarray as xr

import config


DATA_FMT = "netcdf"      # "netcdf" or "grib" - script 02 handles both

MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
TIMES = [f"{h:02d}:00" for h in range(24)]


def load_month_dataset(path):
    """Open one month's download, transparently unzipping if needed."""

    if not zipfile.is_zipfile(path):
        return xr.load_dataset(path)

    with TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmpdir)
            names = zf.namelist()

        groups = [xr.load_dataset(f"{tmpdir}/{name}") for name in names]

    return xr.merge(groups, join="inner", compat="override")


def download(client, city, year):

    cfg = config.city_config(city)

    outdir = config.raw_dir(city)
    outdir.mkdir(parents=True, exist_ok=True)

    suffix = "nc" if DATA_FMT == "netcdf" else "grib"

    target = outdir / f"{city}_era5_{year}.{suffix}"

    if target.exists():
        print(f"  Already present, skipping: {target.name}")
        return

    monthfiles = []

    for month in MONTHS:

        monthfile = outdir / f"_tmp_{city}_era5_{year}_{month}.{suffix}"
        monthfiles.append(monthfile)

        if monthfile.exists():
            print(f"  Month already present, skipping: {monthfile.name}")
            continue

        print(f"  Requesting {city} {year}-{month} -> {monthfile.name}")

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "data_format": DATA_FMT,
                "download_format": "unarchived",
                "variable": config.ERA5_VARS,
                "year": [str(year)],
                "month": [month],
                "day": DAYS,
                "time": TIMES,
                "area": cfg["area"],       # [North, West, South, East]
            },
            str(monthfile),
        )

    print(f"  Merging {len(monthfiles)} months -> {target.name}")

    monthly = [load_month_dataset(f) for f in monthfiles]
    timedim = "valid_time" if "valid_time" in monthly[0].dims else "time"

    ds = xr.concat(monthly, dim=timedim).sortby(timedim)
    ds.to_netcdf(target)

    for monthfile in monthfiles:
        monthfile.unlink()

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
