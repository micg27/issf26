"""
builds the model's map
2km grid lined up with phase 1 map + a margin, each cell gets a population (from WorldPop), candidate ID, and land/sea label
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

def parse_cities(desc):
    parser = argparse.ArgumentParser(description=desc)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", choices=config.ALL_CITIES)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    return config.ALL_CITIES if args.all else [args.city]


def load_candidates(city, cfg):    # opens phase 1 candidate grid and checks it
    grid = gpd.read_file(config.GRID_DIR / cfg["grid_file"])
    assert grid.crs.to_string() == cfg["crs"], f"{city} grid CRS is {grid.crs}"
    assert grid["id"].is_unique

    widths = (grid["right"] - grid["left"]).round(6).unique()
    heights = (grid["top"] - grid["bottom"]).round(6).unique()
    assert list(widths) == [config.CELL_SIZE] and list(heights) == [config.CELL_SIZE], (
        f"{city} cells are {widths} x {heights} m, expected {config.CELL_SIZE}"
    )
    return grid

def build_domain(city):   # makes the model grid
    cfg = config.city_config(city)
    grid = load_candidates(city, cfg)
    dx = config.CELL_SIZE
    m = config.MARGIN_CELLS
    x0 = grid["left"].min() - m * dx
    ytop = grid["top"].max() + m * dx
    nx = int(round((grid["right"].max() - grid["left"].min()) / dx)) + 2 * m
    ny = int(round((grid["top"].max() - grid["bottom"].min()) / dx)) + 2 * m

    x = x0 + dx * (np.arange(nx) + 0.5)         
    y = ytop - dx * (np.arange(ny) + 0.5)    

    candid = np.full((ny, nx), -1, dtype=np.int64)
    pctland = np.zeros((ny, nx))
    col = np.rint((grid["left"] - x0) / dx).astype(int)
    row = np.rint((ytop - grid["top"]) / dx).astype(int)
    candid[row, col] = grid["id"].to_numpy()
    pctland[row, col] = grid["pct_land"].to_numpy()

    population = aggregate_population(cfg, x0, ytop, nx, ny, dx)
    landfrac = land_fraction(cfg, x, y, dx)
    land = (landfrac >= 0.5) | (population > 0)

    ds = xr.Dataset(
        {
            "population": (("y", "x"), population, {"units": "people per cell",
                           "source": ", ".join(cfg["pop_files"])}),
            "candidate_id": (("y", "x"), candid, {"missing": -1}),
            "pct_land": (("y", "x"), pctland, {"units": "%",
                         "note": "candidate cells only; 0 elsewhere"}),
            "land_frac": (("y", "x"), landfrac, {"source": cfg["land_file"]}),
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
          f"{(landfrac * dx * dx).sum() / 1e6:.0f} km2")
    return ds


def land_fraction(cfg, x, y, dx):    # for each cell find its percentage covered by land
    land = gpd.read_file(config.RAW_DIR / cfg["land_file"]).to_crs(cfg["crs"])
    half = dx / 2
    domainbox = box(x.min() - half, y.min() - half, x.max() + half, y.max() + half)
    land = land[land.intersects(domainbox)]
    union = land.geometry.make_valid().union_all().intersection(domainbox)
    frac = np.zeros((len(y), len(x)))
    for i, yc in enumerate(y):
        for j, xc in enumerate(x):
            cell = box(xc - half, yc - half, xc + half, yc + half)
            if union.intersects(cell):
                frac[i, j] = union.intersection(cell).area / dx**2
    return frac


def aggregate_population(cfg, x0, ytop, nx, ny, dx):   # adds up population from each country's raster into each cell
    pop = np.zeros((ny, nx))
    for name in cfg["pop_files"]:
        part = aggregate_one_raster(config.POP_DIR / name, cfg, x0, ytop, nx, ny, dx)
        print(f"  {name}: {part.sum() / 1e6:.3f} M people in domain")
        pop += part
    return pop


def aggregate_one_raster(path, cfg, x0, ytop, nx, ny, dx):    # prevents double counting of population
    toraster = Transformer.from_crs(cfg["crs"], "EPSG:4326", always_xy=True)
    tocity = Transformer.from_crs("EPSG:4326", cfg["crs"], always_xy=True)
    xs = [x0, x0 + nx * dx]
    ys = [ytop - ny * dx, ytop]
    lon, lat = toraster.transform(*np.meshgrid(xs, ys))

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
        px, py = tocity.transform(np.asarray(plon), np.asarray(plat))

    col = np.floor((px - x0) / dx).astype(int)
    row = np.floor((ytop - py) / dx).astype(int)
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    np.add.at(pop, (row[inside], col[inside]), a[valid][inside])
    return pop


if __name__ == "__main__":
    for city in parse_cities("Build Phase 3 model domain and population field."):
        build_domain(city)
