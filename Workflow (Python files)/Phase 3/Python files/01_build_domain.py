# 01_build_domain.py
"""
Build each city's transport-model domain and its population field.

The domain is a regular 2 km raster aligned exactly with the Phase 1
candidate grid (same origin and cell edges) plus a margin on every side.
Every domain cell gets:
    population     WorldPop 2023 people in the full 2 km square
    candidate_id   the Phase 1 grid `id` if the cell is a candidate, else -1
    land_frac      fraction of the cell covered by the city's land polygons
    land           land_frac >= 0.5, or any population (covers Johor/Batam,
                   which have no land polygon here); used to keep synthetic
                   point sources out of the sea

Population is aggregated by assigning each ~100 m WorldPop pixel to the
2 km cell containing its centre, so every person is counted exactly once
(no double counting across cell edges) and totals are conserved.

Outputs (data/processed/phase3/<city>/):
    <city>_domain.nc        2D fields on the model raster
    <city>_candidates.csv   one row per candidate cell (Phase 4/5 input)

Usage:
    python 01_build_domain.py --all
"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import Transformer
from shapely.geometry import box

import config


def parse_cities(description):
    parser = argparse.ArgumentParser(description=description)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", choices=config.ALL_CITIES)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    return config.ALL_CITIES if args.all else [args.city]


def load_candidates(city, cfg):
    grid = gpd.read_file(config.GRID_DIR / cfg["grid_file"])
    assert grid.crs.to_string() == cfg["crs"], f"{city} grid CRS is {grid.crs}"
    assert grid["id"].is_unique

    widths = (grid["right"] - grid["left"]).round(6).unique()
    heights = (grid["top"] - grid["bottom"]).round(6).unique()
    assert list(widths) == [config.CELL_SIZE_M] and list(heights) == [config.CELL_SIZE_M], (
        f"{city} cells are {widths} x {heights} m, expected {config.CELL_SIZE_M}"
    )
    return grid


def build_domain(city):
    cfg = config.city_config(city)
    grid = load_candidates(city, cfg)
    dx = config.CELL_SIZE_M
    m = config.DOMAIN_MARGIN_CELLS

    # Domain edges: candidate-grid bounding box + margin, on the grid's own
    # cell edges so every candidate cell is exactly one domain cell.
    x0 = grid["left"].min() - m * dx
    y_top = grid["top"].max() + m * dx
    nx = int(round((grid["right"].max() - grid["left"].min()) / dx)) + 2 * m
    ny = int(round((grid["top"].max() - grid["bottom"].min()) / dx)) + 2 * m

    x = x0 + dx * (np.arange(nx) + 0.5)          # cell centres, west -> east
    y = y_top - dx * (np.arange(ny) + 0.5)       # cell centres, north -> south

    candidate_id = np.full((ny, nx), -1, dtype=np.int64)
    pct_land = np.zeros((ny, nx))
    col = np.rint((grid["left"] - x0) / dx).astype(int)
    row = np.rint((y_top - grid["top"]) / dx).astype(int)
    candidate_id[row, col] = grid["id"].to_numpy()
    pct_land[row, col] = grid["pct_land"].to_numpy()

    population = aggregate_population(cfg, x0, y_top, nx, ny, dx)
    land_frac = land_fraction(cfg, x, y, dx)
    land = (land_frac >= 0.5) | (population > 0)

    ds = xr.Dataset(
        {
            "population": (("y", "x"), population, {"units": "people per cell",
                           "source": ", ".join(cfg["pop_files"])}),
            "candidate_id": (("y", "x"), candidate_id, {"missing": -1}),
            "pct_land": (("y", "x"), pct_land, {"units": "%",
                         "note": "candidate cells only; 0 elsewhere"}),
            "land_frac": (("y", "x"), land_frac, {"source": cfg["land_file"]}),
            "land": (("y", "x"), land.astype(np.int8),
                     {"rule": "land_frac >= 0.5 or population > 0"}),
        },
        coords={"x": x, "y": y},
        attrs={"city": city, "crs": cfg["crs"], "cell_size_m": dx,
               "margin_cells": m},
    )

    cand = grid[["id", "row_index", "col_index", "land_area_m2", "pct_land"]].copy()
    cand["domain_row"] = row
    cand["domain_col"] = col
    cand["x"] = x[col]
    cand["y"] = y[row]
    cand["population"] = population[row, col]

    out = config.processed_dir(city)
    ds.to_netcdf(out / f"{city}_domain.nc")
    cand.sort_values("id").to_csv(out / f"{city}_candidates.csv", index=False)

    print(f"{city}: domain {ny} x {nx} cells ({ny * dx / 1000:.0f} x {nx * dx / 1000:.0f} km), "
          f"{len(grid)} candidates")
    print(f"  population: domain {population.sum() / 1e6:.2f} M, "
          f"candidates {cand['population'].sum() / 1e6:.2f} M")
    print(f"  land: {land.sum()} of {land.size} cells; land polygons cover "
          f"{(land_frac * dx * dx).sum() / 1e6:.0f} km2")
    return ds


def land_fraction(cfg, x, y, dx):
    """Exact fraction of each domain cell covered by the land polygons."""
    land = gpd.read_file(config.RAW_DIR / cfg["land_file"]).to_crs(cfg["crs"])
    half = dx / 2
    domain_box = box(x.min() - half, y.min() - half, x.max() + half, y.max() + half)
    land = land[land.intersects(domain_box)]
    union = land.geometry.make_valid().union_all().intersection(domain_box)
    frac = np.zeros((len(y), len(x)))
    for i, yc in enumerate(y):
        for j, xc in enumerate(x):
            cell = box(xc - half, yc - half, xc + half, yc + half)
            if union.intersects(cell):
                frac[i, j] = union.intersection(cell).area / dx**2
    return frac


def aggregate_population(cfg, x0, y_top, nx, ny, dx):
    """Sum WorldPop pixels (from every country raster) into domain cells."""
    pop = np.zeros((ny, nx))
    for name in cfg["pop_files"]:
        part = aggregate_one_raster(config.POP_DIR / name, cfg, x0, y_top, nx, ny, dx)
        print(f"  {name}: {part.sum() / 1e6:.3f} M people in domain")
        pop += part
    return pop


def aggregate_one_raster(path, cfg, x0, y_top, nx, ny, dx):
    """Sum one raster's pixels into domain cells by pixel-centre location."""
    to_raster = Transformer.from_crs(cfg["crs"], "EPSG:4326", always_xy=True)
    to_city = Transformer.from_crs("EPSG:4326", cfg["crs"], always_xy=True)

    # Read only the raster window covering the domain (the AUS file is the
    # whole country), padded by a pixel so edge cells are complete.
    xs = [x0, x0 + nx * dx]
    ys = [y_top - ny * dx, y_top]
    lon, lat = to_raster.transform(*np.meshgrid(xs, ys))

    pop = np.zeros((ny, nx))
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 4326
        window = rasterio.windows.from_bounds(
            lon.min(), lat.min(), lon.max(), lat.max(), transform=src.transform
        ).round_offsets().round_lengths()
        a = src.read(1, window=window, boundless=True, fill_value=src.nodata)
        t = src.window_transform(window)
        valid = (a != src.nodata) & np.isfinite(a) & (a > 0)
        r, c = np.nonzero(valid)
        plon, plat = rasterio.transform.xy(t, r, c, offset="center")
        px, py = to_city.transform(np.asarray(plon), np.asarray(plat))

    col = np.floor((px - x0) / dx).astype(int)
    row = np.floor((y_top - py) / dx).astype(int)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    np.add.at(pop, (row[inside], col[inside]), a[valid][inside])
    return pop


if __name__ == "__main__":
    for city in parse_cities("Build Phase 3 model domain and population field."):
        build_domain(city)
