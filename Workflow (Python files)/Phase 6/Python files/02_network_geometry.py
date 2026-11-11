# 02_network_geometry.py
"""
Phase 6 descriptive metrics (planned in the research plan; pre-stated in
METHODS.md, 2026-10-07). For the full-training networks at the Phase 6 budgets:

    nearest-neighbour distance   mean distance from each sensor to its
                                 nearest other sensor (km), and its
                                 coefficient of variation (SD / mean):
                                 low CV = evenly spaced, high = clustered
    distance to coast            mean distance from sensors to the
                                 coastline (km)

Networks: greedy (weather-informed, headline noise), k-means (evenly
spread) and, for Melbourne, the existing EPA stations. The coastline is
the boundary of the land polygons: Singapore planning areas (so the
Johor Strait counts as coast) and the union of Victorian council areas.

Descriptive only: no CIs.

Output: data/processed/phase6/phase6_network_geometry.csv

Usage:
    python 02_network_geometry.py
"""

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[3]
P4 = ROOT / "Workflow" / "Phase 4" / "Python files"
sys.path.insert(0, str(P4))
_spec = importlib.util.spec_from_file_location("cfg4", P4 / "config.py")
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

NVALS = {"singapore": [8, 22, 40], "melbourne": [8, 22, 32, 40, 88]}
LAND = {"singapore": ("boundaries/singapore/MasterPlan2025PlanningAreaBoundaryNoSea.geojson", "EPSG:3414"),
        "melbourne": ("boundaries/melbourne/lga_polygon/lga_polygon.shp", "EPSG:7855")}


def coastline(city):
    path, crs = LAND[city]
    land = gpd.read_file(cfg.phase3.RAW_DIR / path).to_crs(crs)
    return land.geometry.make_valid().union_all().boundary


def nn_stats(xy):
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1) / 1000
    return nn.mean(), nn.std() / nn.mean()


def run(city):
    cand = pd.read_csv(cfg.P3_DIR / city / f"{city}_candidates.csv").sort_values("id").reset_index(drop=True)
    nets = pd.read_csv(cfg.processed_dir(city) / f"{city}_networks_trainall.csv")
    coast = coastline(city)
    coastkm = np.array([coast.distance(Point(x, y)) / 1000 for x, y in zip(cand["x"], cand["y"])])
    rows = [(city, "all candidate cells", len(cand), np.nan, np.nan, coastkm.mean())]
    label = f"f{cfg.MAIN_NOISE:g}"
    for strategy, noise in (("greedy", label), ("kmeans", "any"), ("existing", "any")):
        sub = nets[(nets["strategy"] == strategy) & (nets["noise"] == noise)]
        for n in sorted(set(sub["N"]) & set(NVALS[city] + ([8] if strategy == "existing" else []))):
            idx = sub[sub["N"] == n].sort_values("rank")["cell_index"].to_numpy()
            meannn, cvnn = nn_stats(cand[["x", "y"]].to_numpy()[idx])
            rows.append((city, strategy, n, meannn, cvnn, coastkm[idx].mean()))
    return pd.DataFrame(rows, columns=["city", "network", "N", "nn_mean_km", "nn_cv", "coast_mean_km"])


if __name__ == "__main__":
    df = pd.concat([run(c) for c in cfg.ALL_CITIES], ignore_index=True)
    out = ROOT / "data" / "processed" / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "phase6_network_geometry.csv", index=False)
    print(df.round(2).to_string(index=False))
