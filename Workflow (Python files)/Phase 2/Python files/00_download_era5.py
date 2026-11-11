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
