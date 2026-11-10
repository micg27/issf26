"""Build the Melbourne Urban Growth Boundary (UGB) polygon for the Phase 1 candidate grid.

Vicmap's UGB line layer (PLAN_UGB) can't be polygonized: the lines have
multi-km gaps (along the coast and in several inland stretches), so the
only closed loops are small enclaves. Instead we use the polygon layer
PLAN_UGA and subtract its "outside the UGB" polygons from the metro
council (LGA) area:

    inside UGB = dissolve(metro LGAs) - dissolve(PLAN_UGA where zone_code == "UGBA")

PLAN_UGA also holds zone_code "GAIC" polygons (Growth Areas Infrastructure
Contribution) -- these are growth suburbs *inside* the UGB (e.g. Aintree)
and must NOT be subtracted.

The raw difference leaves ~3,200 slivers where LGA and UGA edges are not
exactly aligned (median width ~0 m, 95th percentile ~3 m). These are
removed by a small morphological opening, then dropping parts and holes
under 0.1 km^2 -- far below what can affect a 2 km (4 km^2) grid cell.

Parts whose edges mostly don't follow either the UGB line or the metro
outer edge (coast / non-metro councils) are kept, but also written to a
`review_pieces` layer for a visual check in QGIS.

Inputs (statewide, EPSG:7855, from the Vicmap open-data WFS at
opendata.maps.vic.gov.au, layers open-data-platform:{plan_ugb,plan_uga,lga_polygon}):
    data/raw/boundaries/melbourne/{plan_ugb,plan_uga,lga_polygon}/
Output:
    data/raw/boundaries/melbourne/UGB_polygon.gpkg
        layer ugb_polygon    -- one feature, the cleaned inside-UGB area
        layer review_pieces  -- parts to eyeball against a basemap
"""
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[3]
BOUNDARIES = ROOT / "data" / "raw" / "boundaries" / "melbourne"
OUT = BOUNDARIES / "UGB_polygon.gpkg"
CRS = "EPSG:7855"

# 31 metropolitan Melbourne councils + Mitchell (the UGB extends into it
# at Beveridge/Wallan).
METRO_LGAS = [
    "BANYULE", "BAYSIDE", "BOROONDARA", "BRIMBANK", "CARDINIA", "CASEY",
    "DAREBIN", "FRANKSTON", "GLEN EIRA", "GREATER DANDENONG", "HOBSONS BAY",
    "HUME", "KINGSTON", "KNOX", "MANNINGHAM", "MARIBYRNONG", "MAROONDAH",
    "MELBOURNE", "MELTON", "MERRI-BEK", "MITCHELL", "MONASH", "MOONEE VALLEY",
    "MORNINGTON PENINSULA", "NILLUMBIK", "PORT PHILLIP", "STONNINGTON",
    "WHITEHORSE", "WHITTLESEA", "WYNDHAM", "YARRA", "YARRA RANGES",
]

OPENING_M = 10          # removes slivers/spikes narrower than 2 * OPENING_M
MIN_AREA_M2 = 100_000   # 0.1 km^2: drop smaller parts, fill smaller holes
EDGE_TOL_M = 5          # how close a boundary must be to count as "on" a line
REVIEW_THRESHOLD = 0.5  # flag parts with < 50% of their edge on UGB line or metro edge


def load(name):
    gdf = gpd.read_file(BOUNDARIES / name / f"{name}.shp")
    assert gdf.crs.to_epsg() == 7855, f"{name} is {gdf.crs}, expected EPSG:7855"
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


def parts_of(geom):
    return list(geom.geoms) if hasattr(geom, "geoms") else [geom]


def fill_small_holes(poly):
    holes = [h for h in poly.interiors if Polygon(h).area >= MIN_AREA_M2]
    return Polygon(poly.exterior, holes)


def main():
    lga = load("lga_polygon")
    uga = load("plan_uga")
    print("PLAN_UGA zone codes:", uga["zone_code"].value_counts().to_dict())
    outside = uga[uga["zone_code"] == "UGBA"]
    ugb_lines = load("plan_ugb")

    missing = set(METRO_LGAS) - set(lga["lga_name"])
    assert not missing, f"LGA names not found: {missing}"
    metro_lga = lga[lga["lga_name"].isin(METRO_LGAS)]
    metro = unary_union(metro_lga.geometry)

    raw = metro.difference(unary_union(outside.geometry))
    print(f"raw difference: {raw.area / 1e6:.0f} km2 in {len(parts_of(raw))} parts")

    opened = (raw.buffer(-OPENING_M, join_style="mitre")
                 .buffer(OPENING_M, join_style="mitre"))
    kept = [fill_small_holes(p) for p in parts_of(opened)
            if p.geom_type == "Polygon" and p.area >= MIN_AREA_M2]
    clean = MultiPolygon(sorted(kept, key=lambda p: -p.area))
    print(f"cleaned:        {clean.area / 1e6:.0f} km2 in {len(kept)} parts")

    ugb_zone = unary_union(ugb_lines.geometry).buffer(EDGE_TOL_M)
    metro_edge_zone = metro.boundary.buffer(EDGE_TOL_M)
    rows = []
    for p in clean.geoms:
        on_ugb = p.exterior.intersection(ugb_zone).length / p.exterior.length
        on_edge = p.exterior.intersection(metro_edge_zone).length / p.exterior.length
        if on_ugb + on_edge < REVIEW_THRESHOLD:
            pt = p.representative_point()
            lga_name = metro_lga[metro_lga.contains(pt)]["lga_name"].tolist()
            rows.append({
                "area_km2": round(p.area / 1e6, 2),
                "lga": ", ".join(lga_name),
                "pct_edge_on_ugb": round(on_ugb * 100),
                "pct_edge_on_metro_edge": round(on_edge * 100),
                "geometry": p,
            })
    review = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)

    OUT.unlink(missing_ok=True)
    gpd.GeoDataFrame(
        {"name": ["Melbourne UGB"], "area_km2": [round(clean.area / 1e6, 1)],
         "n_parts": [len(kept)]},
        geometry=[clean], crs=CRS,
    ).to_file(OUT, layer="ugb_polygon", driver="GPKG")
    review.to_file(OUT, layer="review_pieces", driver="GPKG")

    print(f"review pieces:  {len(review)} ({review['area_km2'].sum():.1f} km2)")
    if len(review):
        print(review.drop(columns="geometry").sort_values("area_km2", ascending=False)
              .to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
