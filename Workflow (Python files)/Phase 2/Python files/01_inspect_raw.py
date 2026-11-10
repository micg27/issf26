# 01_inspect_raw.py
"""
Inspect a raw ERA5 file before processing it.

Generalised from your 01_inspect_grib.py so it also handles NetCDF, since
the CDS API may return either format.

Usage:
    python 01_inspect_raw.py path/to/file.nc
    python 01_inspect_raw.py path/to/file.grib
"""

import sys
from pathlib import Path

import utils


def main():

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python 01_inspect_raw.py path/to/file"
        )

    path = Path(sys.argv[1])

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    utils.banner(f"INSPECTING: {path}")

    datasets = utils.open_any(path)

    print(f"\nNumber of dataset groups: {len(datasets)}")

    for i, ds in enumerate(datasets):

        utils.banner(f"DATASET GROUP {i}")

        print(ds)

        print("\nDATA VARIABLES:")
        for var in ds.data_vars:
            print(f"\n  {var}: {ds[var].dims} {ds[var].shape}")
            print(f"  attrs: {ds[var].attrs}")

        print("\nCOORDINATES:")
        for coord in ds.coords:
            values = ds[coord].values
            print(f"\n  {coord}: shape {values.shape}")
            if values.size <= 10:
                print(f"    {values}")
            else:
                print(f"    first={values.flat[0]}  last={values.flat[-1]}")

        print("\nMISSING VALUES:")
        for var in ds.data_vars:
            try:
                print(f"  {var}: {int(ds[var].isnull().sum())}")
            except Exception as exc:
                print(f"  {var}: could not count ({exc})")

        dims = utils.spatial_dims(ds)
        if dims:
            sizes = {d: ds.sizes[d] for d in dims}
            total = 1
            for value in sizes.values():
                total *= value
            print(f"\nGRID: {sizes}  ->  {total} cells")
            if total < 4:
                print(
                    "  WARNING: very few grid cells. Check your bounding box "
                    "against ERA5's 0.25 degree resolution."
                )


if __name__ == "__main__":
    main()
