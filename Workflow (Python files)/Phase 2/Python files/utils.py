# utils.py
"""
Shared helpers for Phase 2.

Keeping time-zone conversion in ONE place matters: scripts 04 and 11 must
use identical local-time logic, or the days you cluster will not be the
days you extract for Phase 3.
"""

import argparse

import pandas as pd

import config


# ---------------------------------------------------------
# COMMAND-LINE INTERFACE
# ---------------------------------------------------------

def parse_cities(description, with_year=False):
    """Standard CLI: --city singapore | --all [--year 2021]"""

    parser = argparse.ArgumentParser(description=description)

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--city",
        choices=config.ALL_CITIES,
        help="Run for a single city.",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Run for every city in config.CITIES.",
    )

    if with_year:
        parser.add_argument(
            "--year",
            type=int,
            default=None,
            help="Single year. Omit to run every year in config.YEARS.",
        )

    args = parser.parse_args()

    cities = config.ALL_CITIES if args.all else [args.city]

    if with_year:
        years = [args.year] if args.year else config.YEARS
        return cities, years

    return cities


# ---------------------------------------------------------
# TIME CONVERSION
# ---------------------------------------------------------

def to_local(times, city):
    """UTC timestamps -> naive local timestamps for this city."""

    cfg = config.city_config(city)

    times = pd.to_datetime(times)

    if config.USE_DST:
        return (
            times
            .tz_localize("UTC")
            .tz_convert(cfg["timezone"])
            .tz_localize(None)
        )

    return times + pd.Timedelta(hours=cfg["utc_offset_hours"])


def local_day_bounds_utc(date, city):
    """
    A local calendar date -> (utc_start, utc_end) covering that local day.

    Returns naive UTC timestamps suitable for xarray .sel(time=slice(...)).
    """

    cfg = config.city_config(city)

    local_start = pd.Timestamp(date).normalize()
    local_end = local_start + pd.Timedelta(hours=23)

    if config.USE_DST:
        tz = cfg["timezone"]
        utc_start = (
            local_start
            .tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
            .tz_convert("UTC")
            .tz_localize(None)
        )
        utc_end = (
            local_end
            .tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
            .tz_convert("UTC")
            .tz_localize(None)
        )
        return utc_start, utc_end

    offset = pd.Timedelta(hours=cfg["utc_offset_hours"])

    return local_start - offset, local_end - offset


def assign_block(hour):
    """Map a local hour to a 6-hour block index 1..4."""
    return int(hour) // config.BLOCK_HOURS + 1


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def find_raw_files(city, year):
    """
    Locate raw ERA5 files for a city/year.

    Accepts GRIB or NetCDF, per-year or multi-year, because the CDS API
    returns different things depending on the 'data_format' key and the
    dataset requested.
    """

    directory = config.raw_dir(city)

    if not directory.exists():
        raise FileNotFoundError(
            f"Raw directory does not exist: {directory}\n"
            f"Run 00_download_era5.py first, or move your existing "
            f"downloads into this folder."
        )

    patterns = [
        f"*{year}*.grib", f"*{year}*.grb", f"*{year}*.nc",
    ]

    files = []
    for pattern in patterns:
        files.extend(sorted(directory.glob(pattern)))

    if files:
        return files

    # Fall back to any multi-year file in the folder; the caller will
    # subset by year after opening.
    fallback = (
        sorted(directory.glob("*.grib"))
        + sorted(directory.glob("*.grb"))
        + sorted(directory.glob("*.nc"))
    )

    if not fallback:
        raise FileNotFoundError(
            f"No GRIB or NetCDF files found in {directory}"
        )

    print(
        f"  No file matching year {year}; falling back to "
        f"{len(fallback)} file(s) and subsetting by time."
    )

    return fallback


def open_any(path):
    """Open a GRIB or NetCDF file, returning a list of xarray Datasets."""

    import xarray as xr

    suffix = path.suffix.lower()

    if suffix in {".grib", ".grb", ".grib2"}:
        import cfgrib
        return cfgrib.open_datasets(str(path))

    if suffix == ".nc":
        return [xr.open_dataset(path)]

    raise ValueError(f"Unrecognised file type: {path}")


def standardize_time(ds):
    """ERA5 sometimes calls the time axis 'valid_time'. Normalise it."""

    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        if "time" in ds.coords and "time" not in ds.dims:
            ds = ds.drop_vars("time")
        ds = ds.rename({"valid_time": "time"})

    if "time" not in ds.coords:
        raise ValueError(
            f"No usable time coordinate. Coords present: {list(ds.coords)}"
        )

    return ds


def spatial_dims(ds):
    """Return whichever horizontal dimension names this dataset uses."""

    candidates = ["latitude", "longitude", "lat", "lon", "x", "y"]

    return [d for d in candidates if d in ds.dims]


def banner(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)
