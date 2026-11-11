
from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[3]
BOUNDS = ROOT / "data" / "raw" / "boundaries" / "melbourne"
OUT = BOUNDS / "UGB_polygon.gpkg"
CRS = "EPSG:7855"

METRO_LGA = [
    "BANYULE", "BAYSIDE", "BOROONDARA", "BRIMBANK", "CARDINIA", "CASEY",
    "DAREBIN", "FRANKSTON", "GLEN EIRA", "GREATER DANDENONG", "HOBSONS BAY",
    "HUME", "KINGSTON", "KNOX", "MANNINGHAM", "MARIBYRNONG", "MAROONDAH",
    "MELBOURNE", "MELTON", "MERRI-BEK", "MITCHELL", "MONASH", "MOONEE VALLEY",
    "MORNINGTON PENINSULA", "NILLUMBIK", "PORT PHILLIP", "STONNINGTON",
    "WHITEHORSE", "WHITTLESEA", "WYNDHAM", "YARRA", "YARRA RANGES",
]

OPEN_M = 10
MIN_AREA = 100_000
EDGE_TOL = 5
REVIEW_THRESH = 0.5


def load(name):
    gdf = gpd.read_file(BOUNDS / name / f"{name}.shp")
    assert gdf.crs.to_epsg() == 7855, f"{name} is {gdf.crs}, expected EPSG:7855"
    gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


def parts_of(geom):
    return list(geom.geoms) if hasattr(geom, "geoms") else [geom]


def fill_small_holes(poly):
    holes = [h for h in poly.interiors if Polygon(h).area >= MIN_AREA]
    return Polygon(poly.exterior, holes)


def main():
    lga = load("lga_polygon")
    uga = load("plan_uga")
    print("PLAN_UGA zone codes:", uga["zone_code"].value_counts().to_dict())
    outside = uga[uga["zone_code"] == "UGBA"]
    ugblines = load("plan_ugb")

    missing = set(METRO_LGA) - set(lga["lga_name"])
    assert not missing, f"LGA names not found: {missing}"
    metrolga = lga[lga["lga_name"].isin(METRO_LGA)]
    metro = unary_union(metrolga.geometry)

    raw = metro.difference(unary_union(outside.geometry))
    print(f"raw difference: {raw.area / 1e6:.0f} km2 in {len(parts_of(raw))} parts")

    opened = (raw.buffer(-OPEN_M, join_style="mitre")
                 .buffer(OPEN_M, join_style="mitre"))
    kept = [fill_small_holes(p) for p in parts_of(opened)
            if p.geom_type == "Polygon" and p.area >= MIN_AREA]
    clean = MultiPolygon(sorted(kept, key=lambda p: -p.area))
    print(f"cleaned:        {clean.area / 1e6:.0f} km2 in {len(kept)} parts")

    ugbzone = unary_union(ugblines.geometry).buffer(EDGE_TOL)
    edgezone = metro.boundary.buffer(EDGE_TOL)
    rows = []
    for p in clean.geoms:
        onugb = p.exterior.intersection(ugbzone).length / p.exterior.length
        onedge = p.exterior.intersection(edgezone).length / p.exterior.length
        if onugb + onedge < REVIEW_THRESH:
            pt = p.representative_point()
            lganame = metrolga[metrolga.contains(pt)]["lga_name"].tolist()
            rows.append({
                "area_km2": round(p.area / 1e6, 2),
                "lga": ", ".join(lganame),
                "pct_edge_on_ugb": round(onugb * 100),
                "pct_edge_on_metro_edge": round(onedge * 100),
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
