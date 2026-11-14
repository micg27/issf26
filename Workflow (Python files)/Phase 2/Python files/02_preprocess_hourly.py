"""
sorts raw data into clean hourly files
"""

import numpy as np
import xarray as xr

import config
import utils


def process(city, year): # for one city and year (e.g. Singapore 2022): collects the 5 required variables, merges them, keeps only that year, 
                         # discard duplicate hours, adds wind speed, wind direction and temperature in °C, checks the hour count (8,760 or 8,784), and saves

    utils.banner(f"PREPROCESS: {city} {year}")

    outdir = config.processed_dir(city)
    outfile = outdir / f"{city}_hourly_{year}.nc"

    if outfile.exists():
        print(f"  Already exists, skipping: {outfile.name}")
        return

    files = utils.find_raw_files(city, year)

    print("  Input files:")
    for file in files:
        print(f"    {file.name}")

    found = {}

    for file in files:

        for group in utils.open_any(file):

            group = utils.standardize_time(group)

            for var in config.NEEDED_VARS:
                if var in group.data_vars and var not in found:
                    found[var] = group[[var]]

    missing = config.NEEDED_VARS - set(found)

    if missing:
        raise ValueError(
            f"Missing required variables for {city} {year}: {missing}\n"
            f"Found: {sorted(found)}"
        )
        
    ds = xr.merge(list(found.values()), join="inner")

    ds = utils.standardize_time(ds)

    ds = ds.sortby("time")

    ds = ds.sel(time=str(year))

    if ds.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No timesteps for {year} in the input files for {city}."
        )

    times = ds["time"].values

    if len(np.unique(times)) != len(times):
        print("  Duplicate timestamps found; keeping first occurrence.")
        _, keep = np.unique(times, return_index=True)
        ds = ds.isel(time=np.sort(keep))

    ds["wind_speed"] = np.hypot(ds["u10"], ds["v10"])
    ds["wind_speed"].attrs = {
        "long_name": "10 metre wind speed",
        "units": "m s-1",
    }
    
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

    nhours = ds.sizes["time"]
    expected = 8784 if year % 4 == 0 else 8760

    print(f"  Timesteps: {nhours} (expected {expected})")

    if nhours != expected:
        print(f"  {expected - nhours} hours missing.")

    dims = utils.spatial_dims(ds)
    print(f"  Grid dims: {[(d, ds.sizes[d]) for d in dims]}")

    ds.to_netcdf(outfile)

    print(f"  Saved {outfile}")


def main():

    cities, years = utils.parse_cities(
        "Preprocess raw ERA5 into hourly NetCDF.",
        withyear=True,
    )

    for city in cities:
        for year in years:
            process(city, year)


if __name__ == "__main__":
    main()
