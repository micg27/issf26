import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import config
import utils

PLAUSIBLE = {
    "u10": (-60.0, 60.0),
    "v10": (-60.0, 60.0),
    "wind_speed": (0.0, 80.0),
    "blh": (0.0, 5000.0),
    "t2m_c": (-30.0, 55.0),
    "tp": (0.0, 0.5),
}

def qc(city, year, records):

    utils.banner(f"QC: {city} {year}")

    path = config.processed_dir(city) / f"{city}_hourly_{year}.nc"

    if not path.exists():
        print(f"  Missing {path.name}; run 02 first.")
        return

    out = config.output_dir(city, "qc")

    ds = xr.open_dataset(path)

    row = {"city": city, "year": year}

    print(f"\n  {'variable':16s} {'missing':>10s} {'min':>12s} {'max':>12s}")

    for var in ["u10", "v10", "wind_speed", "blh", "t2m_c", "tp"]:

        if var not in ds:
            continue

        missing = int(ds[var].isnull().sum())
        minimum = float(ds[var].min())
        maximum = float(ds[var].max())

        print(
            f"  {var:16s} {missing:10d} {minimum:12.3f} {maximum:12.3f}"
        )

        row[f"{var}_missing"] = missing
        row[f"{var}_min"] = minimum
        row[f"{var}_max"] = maximum

        low, high = PLAUSIBLE.get(var, (-np.inf, np.inf))

        if minimum < low or maximum > high:
            print(
                f"    {var} outside plausible range "
                f"[{low}, {high}]"
            )

    if float(ds["wind_speed"].min()) < 0:
        raise ValueError("Negative wind speed detected.")

    times = pd.DatetimeIndex(ds["time"].values)

    expected = pd.date_range(times.min(), times.max(), freq="h")

    gaps = expected.difference(times)

    print(f"\n  Missing hourly timestamps: {len(gaps)}")

    if len(gaps):
        print(f"    first few: {list(gaps[:5])}")

    row["missing_timestamps"] = len(gaps)
    row["n_timesteps"] = len(times)

    dims = utils.spatial_dims(ds)
    city_mean = ds.mean(dim=dims) if dims else ds

    for var in ["wind_speed", "blh", "t2m_c"]:

        if var not in city_mean:
            continue

        values = city_mean[var].values.ravel()
        values = values[np.isfinite(values)]

        fig, ax = plt.subplots()
        ax.hist(values, bins=50)
        ax.set_xlabel(var)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{city.title()} {year}: {var}")
        fig.tight_layout()
        fig.savefig(out / f"{var}_histogram_{year}.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        city_mean["u10"].values.ravel(),
        city_mean["v10"].values.ravel(),
        s=2,
        alpha=0.15,
    )
    ax.axhline(0, linewidth=0.5, color="black")
    ax.axvline(0, linewidth=0.5, color="black")
    ax.set_xlabel("u10 (m/s)")
    ax.set_ylabel("v10 (m/s)")
    ax.set_aspect("equal")
    ax.set_title(f"{city.title()} {year}: hourly wind vectors")
    fig.tight_layout()
    fig.savefig(out / f"wind_uv_scatter_{year}.png", dpi=200)
    plt.close(fig)

    month = city_mean.sel(
        time=slice(f"{year}-01-01", f"{year}-01-31 23:59")
    )

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(month["time"], month["wind_speed"], linewidth=0.8)
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title(f"{city.title()} - January {year}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / f"january_wind_speed_{year}.png", dpi=200)
    plt.close(fig)

    print(f"\n  Figures saved to {out}")

    records.append(row)


def main():

    cities, years = utils.parse_cities(
        "QC the processed hourly meteorology.",
        with_year=True,
    )

    records = []

    for city in cities:
        for year in years:
            qc(city, year, records)

    if records:
        summary = pd.DataFrame(records)
        path = config.OUTPUT_DIR / "qc_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
        print(f"\nQC summary written to {path}")


if __name__ == "__main__":
    main()
