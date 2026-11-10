# 02_ingest_pm25.py
"""
Clean the real hourly PM2.5 observations used to calibrate the transport
model, into one long-format table per city.

Times are converted to FIXED local standard time (UTC+8 Singapore, UTC+10
Melbourne), matching Phase 2's USE_DST=False, and labelled by the START of
the averaging hour:

  Singapore (NEA, data.gov.sg "Historical 1-hr PM2.5"): wide table, one
    column per region, timestamps D/M/YYYY H:MM in SGT. NEA's 1-hr reading
    at time T is the average over the preceding hour, so the hour start is
    T - 1 h.
  Melbourne (EPA Victoria "All Sites" workbooks, one per year): one sheet
    per station. `datetime_AEST` is already fixed UTC+10 and is the hour
    START (the 2025 workbook's `date_hour_start_utc_id` column confirms the
    convention). Three quirks are handled explicitly (see CLAUDE.md):
      1. station IDs change between years -> match by station NAME;
      2. PM2.5 column is BPM2.5 (raw BAM) then SBPM25 (STP-corrected, same
         instrument) -> spliced into one series;
      3. a station split across two ID sheets in one year (Footscray 2021)
         -> one series.
    Station coordinates come from each workbook's Metadata sheet; a
    station that moved slightly between years keeps its per-year location.

Values outside [PM25_MIN, PM25_MAX] are dropped and counted.

Outputs (data/processed/phase3/<city>/):
    <city>_pm25_hourly.csv   time_local, station, lat, lon, pm25
    <city>_pm25_stations.csv one row per station: coords, coverage, in-domain flag

Usage:
    python 02_ingest_pm25.py --all
"""

import argparse
import re

import numpy as np
import openpyxl
import pandas as pd
import xarray as xr
from pyproj import Transformer

import config


def parse_cities(description):
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", choices=config.ALL_CITIES)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    return config.ALL_CITIES if args.all else [args.city]


# ---------------------------------------------------------
# SINGAPORE
# ---------------------------------------------------------

def load_singapore():
    raw = pd.read_csv(config.PM25_RAW_DIR / config.SINGAPORE_PM25_FILE)
    t_end = pd.to_datetime(raw["1hr_pm25"], format="%d/%m/%Y %H:%M", errors="coerce")
    bad = t_end.isna()
    # One known malformed row ("2016-04-04 010:00:00"), outside 2021-2025.
    print(f"  skipped {bad.sum()} unparseable timestamp(s): {raw.loc[bad, '1hr_pm25'].tolist()}")
    raw, t_end = raw[~bad], t_end[~bad]
    # The full record has 3 duplicated timestamps (2016-2018, with differing
    # values); only the study period has to be clean.
    t_start = t_end - pd.Timedelta(hours=1)
    in_period = t_start.dt.year.between(config.phase2.START_YEAR, config.phase2.END_YEAR)
    raw, t_end = raw[in_period], t_end[in_period]
    assert not t_end.duplicated().any(), "duplicate Singapore timestamps in study period"
    wide = raw.drop(columns="1hr_pm25")
    wide.index = t_end - pd.Timedelta(hours=1)

    long = (wide.rename_axis("time_local").reset_index()
                .melt(id_vars="time_local", var_name="station", value_name="pm25"))
    long["pm25"] = pd.to_numeric(long["pm25"], errors="coerce")
    long["lat"] = long["station"].map(lambda s: config.SINGAPORE_REGIONS[s][0])
    long["lon"] = long["station"].map(lambda s: config.SINGAPORE_REGIONS[s][1])
    return long


# ---------------------------------------------------------
# MELBOURNE
# ---------------------------------------------------------

def read_locations(wb):
    rows = list(wb["Metadata"].iter_rows(values_only=True))
    start = next(i for i, r in enumerate(rows) if r and r[0] == "location_id")
    locs = {}
    for r in rows[start + 1:]:
        if not r or r[0] is None:
            break
        locs[int(r[0])] = (r[1], float(r[2]), float(r[3]))
    return locs


def read_station_sheet(ws):
    it = ws.iter_rows(values_only=True)
    header = list(next(it))
    pm_cols = [c for c in config.MELBOURNE_PM25_COLUMNS if c in header]
    if not pm_cols:
        return None
    i_time = header.index("datetime_AEST")
    i_id = header.index("location_id")
    idx = [header.index(c) for c in pm_cols]
    records = []
    for r in it:
        if r[i_time] is None:
            continue
        # BPM2.5 and SBPM25 never overlap in time; take whichever is present.
        v = next((r[i] for i in idx if r[i] is not None), None)
        records.append((r[i_time], r[i_id], v))
    df = pd.DataFrame(records, columns=["time_local", "location_id", "pm25"])
    df["column"] = "+".join(pm_cols)
    return df


def load_melbourne():
    frames = []
    for path in sorted(config.PM25_RAW_DIR.glob(config.MELBOURNE_PM25_GLOB)):
        year = int(re.match(r"(\d{4})_", path.name).group(1))
        wb = openpyxl.load_workbook(path, read_only=True)
        locs = read_locations(wb)
        for sheet in wb.sheetnames:
            if sheet in ("Data Info Panel", "Metadata", "AllData"):
                continue
            df = read_station_sheet(wb[sheet])
            if df is None:
                continue
            ids = df["location_id"].dropna().astype(int).unique()
            assert len(ids) == 1, f"{path.name}/{sheet}: several location_ids {ids}"
            name, lat, lon = locs[ids[0]]
            df["station"], df["lat"], df["lon"], df["year_file"] = name, lat, lon, year
            print(f"  {year} {sheet:28s} {df['column'].iloc[0]:14s} {df['pm25'].notna().sum():6d} values")
            frames.append(df)
        wb.close()

    long = pd.concat(frames, ignore_index=True)
    long["time_local"] = pd.to_datetime(long["time_local"])
    long["pm25"] = pd.to_numeric(long["pm25"], errors="coerce")

    long = long.dropna(subset=["pm25"])

    # Consecutive yearly workbooks both contain the 31 Dec 23:00 AEST hour
    # (it is 1 Jan 00:00 in daylight-saving local time). The copies are
    # identical, so drop exact repeats.
    exact = long.duplicated(["station", "time_local", "pm25"])
    print(f"  dropped {exact.sum()} identical year-boundary repeats")
    long = long[~exact]

    # Quirk 3: a station split across two ID sheets in one year must not
    # produce conflicting hours. Fail loudly if any remain.
    dup = long.duplicated(["station", "time_local"], keep=False)
    assert not dup.any(), f"conflicting station-hours:\n{long[dup].head()}"
    return long.drop(columns=["location_id", "column", "year_file"])


# ---------------------------------------------------------
# COMMON
# ---------------------------------------------------------

def clean_and_save(city, long):
    n_raw = long["pm25"].notna().sum()
    bad = long["pm25"].notna() & ~long["pm25"].between(config.PM25_MIN, config.PM25_MAX)
    print(f"  dropped {bad.sum()} of {n_raw} values outside "
          f"[{config.PM25_MIN}, {config.PM25_MAX}] ug/m3 "
          f"(min {long.loc[bad, 'pm25'].min() if bad.any() else 'n/a'})")
    long = long[long["pm25"].notna() & ~bad]

    start = pd.Timestamp(f"{config.phase2.START_YEAR}-01-01")
    end = pd.Timestamp(f"{config.phase2.END_YEAR + 1}-01-01")
    long = long[(long["time_local"] >= start) & (long["time_local"] < end)]
    long = long.sort_values(["station", "time_local"])

    # Which stations sit inside the model domain?
    cfg = config.city_config(city)
    dom = xr.open_dataset(config.processed_dir(city) / f"{city}_domain.nc")
    half = dom.attrs["cell_size_m"] / 2
    to_city = Transformer.from_crs("EPSG:4326", cfg["crs"], always_xy=True)

    st = (long.groupby("station")
              .agg(lat=("lat", "median"), lon=("lon", "median"),
                   n_hours=("pm25", "size"),
                   first=("time_local", "min"), last=("time_local", "max"),
                   mean_pm25=("pm25", "mean"))
              .reset_index())
    st["x"], st["y"] = to_city.transform(st["lon"].to_numpy(), st["lat"].to_numpy())
    xmin, xmax = float(dom.x.min()) - half, float(dom.x.max()) + half
    ymin, ymax = float(dom.y.min()) - half, float(dom.y.max()) + half
    st["in_domain"] = st["x"].between(xmin, xmax) & st["y"].between(ymin, ymax)
    n_possible = (end - start) / pd.Timedelta(hours=1)
    st["coverage_pct"] = (100 * st["n_hours"] / n_possible).round(1)

    out = config.processed_dir(city)
    long[["time_local", "station", "lat", "lon", "pm25"]].to_csv(
        out / f"{city}_pm25_hourly.csv", index=False)
    st.to_csv(out / f"{city}_pm25_stations.csv", index=False)

    print(st[["station", "in_domain", "coverage_pct", "mean_pm25", "first", "last"]]
          .round({"mean_pm25": 1}).to_string(index=False))


if __name__ == "__main__":
    for city in parse_cities("Clean real hourly PM2.5 observations."):
        print(f"=== {city}")
        long = load_singapore() if city == "singapore" else load_melbourne()
        clean_and_save(city, long)
