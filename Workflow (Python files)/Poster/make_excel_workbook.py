# make_excel_workbook.py
"""
Poster graphs as an Excel workbook: each sheet holds the data behind one
poster figure, next to a native Excel chart drawn from that data, so the
charts can be restyled or edited in Excel / PowerPoint.

Same data, colours and wording as make_poster_figures.py:
    1  plume maps         -> heat-map grid (cell colour scale), one per regime
    2  sensor networks    -> scatter charts of sensor positions (km)
    3  emission predictability (key result) -> column charts with 95% CI bars
    4  sensor-count curves -> line charts (CI as dashed lines)
    5  main test           -> line charts per city
    6  spreading-rate check (Phase 8) -> column charts with 95% CI bars

Output: Graphs/Poster picks/Excel/Poster charts.xlsx

Usage:
    python make_excel_workbook.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xlsxwriter
from xlsxwriter.utility import xl_range_abs

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "data" / "processed" / "phase3"
P4 = ROOT / "data" / "processed" / "phase4"
P5 = ROOT / "data" / "processed" / "phase5"
P7 = ROOT / "data" / "processed" / "phase7"
OUT = ROOT / "Graphs" / "Poster picks" / "Excel"

GREEN, TEAL, GREY, INK, MUTED, CELL = "#13703F", "#1F8BB8", "#8A8F8C", "#1D2B22", "#5C6B62", "#DFE6E1"
K_COL = {200: "#7FBF95", 1000: GREEN, 8000: "#08391F"}
CITIES = ("singapore", "melbourne")
CITYNAME = {"singapore": "Singapore", "melbourne": "Melbourne"}
SHORT = {"singapore": "SG", "melbourne": "MEL"}
MAIN_N = {"singapore": 22, "melbourne": 8}
HEADLINE = "f0.25"
REGIME_NAMES = {"SIN1": "Southwest monsoon", "SIN2": "Inter-monsoon", "SIN3": "Northeast monsoon",
               "MEL1": "Light south-easterlies", "MEL2": "Northerlies (before a front)",
               "MEL3": "Westerlies (after a front)"}
LEVELS = ["L0", "L1", "L2_setA", "L2_setB", "L3_setA", "L3_setB", "L4_setA", "L4_setB", "L5"]
LEVEL_LAB = ["Known exactly", "Noisy output", "10% fixed sources (A)", "10% fixed sources (B)",
               "30% fixed sources (A)", "30% fixed sources (B)", "30% fixed + noise (A)",
               "30% fixed + noise (B)", "Randomly moving"]
STRATEGY = [("greedy", "Weather-informed", GREEN, "solid"), ("kmeans", "Evenly spread", TEAL, "solid"),
            ("qr", "QR pivoting", GREY, "dash"), ("random_mean", "Random", GREY, "round_dot")]


class Book:
    def __init__(self, path):
        self.wb = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
        wb = self.wb
        self.title = wb.add_format({"bold": True, "font_size": 14, "font_color": INK})
        self.note = wb.add_format({"italic": True, "font_color": MUTED})
        self.head = wb.add_format({"bold": True, "bg_color": CELL, "font_color": INK,
                                   "text_wrap": True, "valign": "top", "border": 1, "border_color": "#C5CFC8"})
        self.num = wb.add_format({"num_format": "0.0"})
        self.num3 = wb.add_format({"num_format": "0.000"})
        self.txt = wb.add_format({})
        self.maphead = wb.add_format({"bold": True, "font_color": INK, "align": "center"})
        self.mapnote = wb.add_format({"italic": True, "font_color": MUTED, "align": "center"})
        self.hidden = wb.add_format({"num_format": ";;;"})     # value kept, not shown: colour only

    def sheet(self, name, title, note):
        ws = self.wb.add_worksheet(name)
        ws.write(0, 0, title, self.title)
        ws.write(1, 0, note, self.note)
        return ws

    def table(self, ws, row, col, df, fmt=None):
        """Write df with a header row at (row, col); returns {column: (first, last) cell refs}."""
        for j, c in enumerate(df.columns):
            ws.write(row, col + j, c, self.head)
            for i, v in enumerate(df[c]):
                if isinstance(v, str):
                    ws.write_string(row + 1 + i, col + j, v)
                elif v is None or (isinstance(v, float) and np.isnan(v)):
                    ws.write_blank(row + 1 + i, col + j, None)
                else:
                    ws.write_number(row + 1 + i, col + j, float(v), fmt or self.num)
            ws.set_column(col + j, col + j, max(11, min(26, len(c) * 0.9)))
        return {c: (row + 1, col + j, row + len(df), col + j) for j, c in enumerate(df.columns)}

    def chart(self, kind, subtype=None):
        ch = self.wb.add_chart({"type": kind, **({"subtype": subtype} if subtype else {})})
        ch.set_chartarea({"border": {"none": True}})
        ch.set_plotarea({"border": {"none": True}})
        ch.set_legend({"position": "top", "font": {"size": 10, "color": INK}})
        return ch

    def ref(self, ws, r):
        return [ws.name, *r]


def page(ws, r0, c0, r1, c1):
    """Print the chart area landscape on one page (also used to preview the charts)."""
    ws.set_landscape()
    ws.fit_to_pages(1, 1)
    ws.print_area(r0, c0, r1, c1)
    ws.hide_gridlines(2)


def axis(title, **kw):
    base = {"name": title, "name_font": {"size": 10, "bold": False, "color": INK},
            "num_font": {"size": 9, "color": MUTED}, "line": {"color": MUTED},
            "major_gridlines": {"visible": True, "line": {"color": "#E8ECE9"}}}
    base.update(kw)
    return base


def ci_bars(book, ws, rval, r_lo, r_hi, df, val, lo, hi, row, colminus, colplus, colour):
    """Custom error bars need positive distances; write them in helper columns."""
    ws.write(row, colminus, f"{val}: minus", book.head)
    ws.write(row, colplus, f"{val}: plus", book.head)
    for i in range(len(df)):
        ws.write_number(row + 1 + i, colminus, float(df[val].iloc[i] - df[lo].iloc[i]), book.num)
        ws.write_number(row + 1 + i, colplus, float(df[hi].iloc[i] - df[val].iloc[i]), book.num)
    n = len(df)
    return {"type": "custom", "direction": "both", "end_style": 1,
            "minus_values": f"='{ws.name}'!{xl_range_abs(row + 1, colminus, row + n, colminus)}",
            "plus_values": f"='{ws.name}'!{xl_range_abs(row + 1, colplus, row + n, colplus)}",
            "line": {"color": colour, "width": 1.25}}


# ---------------------------------------------------------
# 1. PLUME MAPS (heat-map grids)
# ---------------------------------------------------------

def sheet_plumes(book, city):
    ds = xr.open_dataset(P3 / city / f"{city}_scenarios.nc")
    rep = np.nonzero(ds["selection"].values == "representative")[0]
    x, y = ds["cell_x"].values, ds["cell_y"].values
    ix = np.rint((x - x.min()) / 2000).astype(int)
    iy = np.rint((y.max() - y) / 2000).astype(int)          # north at the top
    nx, ny = ix.max() + 1, iy.max() + 1
    ws = book.sheet(f"1 Plumes {SHORT[city]}",
                    f"{CITYNAME[city]}: simulated daily PM2.5 from local emissions, typical day of each weather pattern",
                    "Each square is one 2 km grid cell (north at the top). Colour: white = low, dark green = high; "
                    "each map has its own colour scale. Values in µg/m³. Raw data in the table on the right.")
    cw = 1.1 if city == "melbourne" else 1.8
    top, left, gap = 4, 0, 2
    long = pd.DataFrame({"cell_id": ds["cell_id"].values, "x_km": x / 1000, "y_km": y / 1000})
    for m, d in enumerate(rep):
        v = ds["local_pm25"].isel(day=d, member=0).mean("hour").values
        r = str(ds["regime"].values[d])
        day = pd.Timestamp(ds["date"].values[d]).strftime("%-d %b %Y")
        c0 = left + m * (nx + gap)
        ws.merge_range(top - 1, c0, top - 1, c0 + nx - 1, f"{REGIME_NAMES[r]}, {day}", book.maphead)
        for k in range(len(v)):
            ws.write_number(top + iy[k], c0 + ix[k], round(float(v[k]), 3), book.hidden)
        ws.conditional_format(top, c0, top + ny - 1, c0 + nx - 1,
                              {"type": "2_color_scale", "min_color": "#F7FCF0", "max_color": GREEN})
        ws.set_column(c0, c0 + nx - 1, cw)
        ws.merge_range(top + ny + 1, c0, top + ny + 1, c0 + nx - 1,
                       f"Palest = {v.min():.2f}, darkest = {v.max():.2f} µg/m³", book.mapnote)
        long[f"{REGIME_NAMES[r]} ({day})"] = np.round(v, 4)
    for rr in range(top, top + ny):
        ws.set_row(rr, cw * 7.5)
    page(ws, 0, 0, top + ny + 2, len(rep) * (nx + gap))
    ctab = left + len(rep) * (nx + gap) + 1
    book.table(ws, top - 1, ctab, long, book.num3)


# ---------------------------------------------------------
# 2. SENSOR NETWORKS (scatter)
# ---------------------------------------------------------

def sheet_networks(book, city):
    ds = xr.open_dataset(P3 / city / f"{city}_scenarios.nc")
    x, y = ds["cell_x"].values / 1000, ds["cell_y"].values / 1000
    x0, y0 = x.min(), y.min()
    nets = pd.read_csv(P4 / city / f"{city}_networks_trainall.csv")
    n = MAIN_N[city]
    panels = [("greedy", HEADLINE, "Weather-informed", GREEN, "circle"),
              ("kmeans", "any", "Evenly spread", TEAL, "square")]
    if (nets["strategy"] == "existing").any():
        panels.append(("existing", "any", "Existing EPA stations", INK, "triangle"))
    ws = book.sheet(f"2 Networks {SHORT[city]}",
                    f"{CITYNAME[city]}: where each strategy places {n} sensors (the size of the city's real network)",
                    "Coordinates in km from the south-west corner of the study area. Grey squares: candidate 2 km cells.")
    cand = pd.DataFrame({"Candidate x (km)": x - x0, "Candidate y (km)": y - y0})
    rc = book.table(ws, 3, 0, cand)
    col = 3
    w_km, h_km = x.max() - x0, y.max() - y0
    width = 420
    height = int(width * h_km / w_km) + 70
    for p, (strat, noise, title, colour, marker) in enumerate(panels):
        sub = nets[(nets["strategy"] == strat) & (nets["noise"] == noise)]
        sub = sub[sub["N"] == (sub["N"].max() if strat == "existing" else n)]
        i = sub["cell_index"].to_numpy()
        df = pd.DataFrame({f"{title} x (km)": x[i] - x0, f"{title} y (km)": y[i] - y0})
        r = book.table(ws, 3, col, df)
        ch = book.chart("scatter")
        ch.add_series({"name": "Candidate cells", "categories": book.ref(ws, rc["Candidate x (km)"]),
                       "values": book.ref(ws, rc["Candidate y (km)"]),
                       "marker": {"type": "square", "size": 4 if city == "melbourne" else 7,
                                  "fill": {"color": CELL}, "border": {"none": True}}})
        ch.add_series({"name": f"{title} ({len(sub)} sensors)", "categories": book.ref(ws, r[df.columns[0]]),
                       "values": book.ref(ws, r[df.columns[1]]),
                       "marker": {"type": marker, "size": 9, "fill": {"color": colour},
                                  "border": {"color": colour}}})
        ch.set_title({"name": f"{title}: {len(sub)} sensors", "name_font": {"size": 12, "color": INK}})
        ch.set_x_axis({"min": -2, "max": float(np.ceil(w_km) + 2), "visible": False,
                       "major_gridlines": {"visible": False}})
        ch.set_y_axis({"min": -2, "max": float(np.ceil(h_km) + 2), "visible": False,
                       "major_gridlines": {"visible": False}})
        ch.set_legend({"none": True})
        ch.set_size({"width": width, "height": height})
        ws.insert_chart(3, 10 + p * 7, ch)
        col += 2
    page(ws, 0, 9, 3 + height // 20 + 3, 10 + len(panels) * 7)


# ---------------------------------------------------------
# 3 + 6. COLUMN CHARTS WITH CI BARS
# ---------------------------------------------------------

def column_chart(book, ws, row, title, series, cats_ref, ymin, ymax):
    ch = book.chart("column")
    for s in series:
        ch.add_series({"name": s["name"], "categories": cats_ref, "values": s["values"],
                       "fill": {"color": s["colour"]}, "border": {"none": True},
                       "y_error_bars": s["err"], "gap": 60, "overlap": -5})
    ch.set_title({"name": title, "name_font": {"size": 12, "color": INK}})
    ch.set_x_axis(axis("How predictable the pollution sources are (most → least predictable)",
                       num_font={"size": 9, "color": INK}, label_position="low",
                       major_gridlines={"visible": False}))
    ch.set_y_axis(axis("Accuracy gain (%)", min=ymin, max=ymax))
    ch.set_size({"width": 900, "height": 380})
    return ch


def sheet_predictability(book):
    res = json.load(open(P5 / "phase5_emission_predictability.json"))
    ws = book.sheet("3 Predictability",
                    "Weather-informed placement helps when pollution sources are predictable",
                    "Accuracy gain over evenly spread sensors (%). Error bars: 95% confidence intervals. "
                    "A and B: two random sets of source locations.")
    row = 3
    for ns, nm, title in [(22, 8, "Each city's real network size"), (22, 22, "Same number of sensors (22 each)"),
                          (22, 88, "Same sensor density")]:
        df = pd.DataFrame({"Pollution sources": LEVEL_LAB})
        for city, n in (("singapore", ns), ("melbourne", nm)):
            v = np.array([res[city][k]["improvement"][str(n)] for k in LEVELS]) * 100
            lab = f"{CITYNAME[city]} ({n} sensors)"
            df[lab], df[f"{lab} low"], df[f"{lab} high"] = v[:, 0], v[:, 1], v[:, 2]
        ws.write(row - 1, 0, title, book.head)
        r = book.table(ws, row, 0, df)
        series = []
        for j, (city, n) in enumerate((("singapore", ns), ("melbourne", nm))):
            lab = f"{CITYNAME[city]} ({n} sensors)"
            err = ci_bars(book, ws, None, None, None, df, lab, f"{lab} low", f"{lab} high",
                          row, 7 + 2 * j, 8 + 2 * j, INK)
            series.append({"name": lab, "values": book.ref(ws, r[lab]),
                           "colour": GREEN if city == "singapore" else TEAL, "err": err})
        ch = column_chart(book, ws, row, title, series, book.ref(ws, r["Pollution sources"]), -10, 65)
        ws.insert_chart(row - 1, 12, ch)
        row += 21
    page(ws, 0, 12, row, 26)


def sheet_k(book):
    ks = [(200, "_K200", "Slow spreading (200 m²/s)"), (1000, "", "Main analysis (1,000 m²/s)"),
          (8000, "_K8000", "Fast spreading (8,000 m²/s)")]
    res = {k: json.load(open(P5 / f"phase5_emission_predictability{t}.json")) for k, t, _ in ks}
    ws = book.sheet("6 Spreading rate",
                    "Does the result depend on how fast pollution spreads?",
                    "Accuracy gain of weather-informed over evenly spread sensors (%) at each city's real network "
                    "size, at three spreading rates K. Error bars: 95% confidence intervals.")
    row = 3
    for city in CITIES:
        n = MAIN_N[city]
        df = pd.DataFrame({"Pollution sources": LEVEL_LAB})
        for k, _, lab in ks:
            v = np.array([res[k][city][lv]["improvement"][str(n)] for lv in LEVELS]) * 100
            df[lab], df[f"{lab} low"], df[f"{lab} high"] = v[:, 0], v[:, 1], v[:, 2]
        title = f"{CITYNAME[city]} ({n} sensors, its real network size)"
        ws.write(row - 1, 0, title, book.head)
        r = book.table(ws, row, 0, df)
        series = []
        for j, (k, _, lab) in enumerate(ks):
            err = ci_bars(book, ws, None, None, None, df, lab, f"{lab} low", f"{lab} high",
                          row, 11 + 2 * j, 12 + 2 * j, MUTED)
            series.append({"name": lab, "values": book.ref(ws, r[lab]), "colour": K_COL[k], "err": err})
        ch = column_chart(book, ws, row, title, series, book.ref(ws, r["Pollution sources"]), -10, 45)
        ws.insert_chart(row - 1, 18, ch)
        row += 21
    page(ws, 0, 18, row, 32)


# ---------------------------------------------------------
# 4 + 5. LINE CHARTS VS NUMBER OF SENSORS
# ---------------------------------------------------------

def line(ch, book, ws, xr_, yr, name, colour, dash="solid", width=2.25, marker=None):
    s = {"name": name, "categories": book.ref(ws, xr_), "values": book.ref(ws, yr),
         "line": {"color": colour, "width": width, "dash_type": dash}}
    s["marker"] = marker or {"type": "none"}
    if marker:
        s["line"] = {"none": True}
    ch.add_series(s)


def xy_chart(book, title, xtitle, ytitle, nmax, ymin=None, ymax=None):
    ch = book.chart("scatter", "straight")
    ch.set_title({"name": title, "name_font": {"size": 12, "color": INK}})
    ch.set_x_axis(axis(xtitle, min=0, max=nmax, label_position="low", major_gridlines={"visible": False}))
    ch.set_y_axis(axis(ytitle, **({"min": ymin} if ymin is not None else {}),
                       **({"max": ymax} if ymax is not None else {})))
    ch.set_size({"width": 620, "height": 360})
    return ch


def strategy_table(city):
    sc = pd.read_csv(P5 / city / f"{city}_test_scores_all2025_trainall.csv")
    sc = sc[sc["noise"] == HEADLINE]
    t = sc.pivot_table(index="N", columns="strategy", values="test_rmse").reset_index()
    return t


def sheet_sensor_count(book):
    ws = book.sheet("4 Sensor count",
                    "How many sensors? Map error and the gain from weather-informed placement as sensors are added",
                    "Top: map error (RMSE, µg/m³) with fitted curves. Bottom: accuracy gain over even spacing (%), "
                    "dashed = 95% confidence interval. Real network size: Singapore 22, Melbourne 8.")
    col = 0
    for p, city in enumerate(CITIES):
        c = pd.read_csv(P7 / f"{city}_original_curves.csv")
        l0 = pd.read_csv(P7 / f"{city}_L0_curves.csv")
        st = strategy_table(city)
        st = st[["N"] + [s for s in ("qr", "random_mean") if s in st]]
        df = pd.DataFrame({"Number of sensors": c["N"],
                           "Weather-informed error": c["rmse_greedy"], "Weather-informed fitted": c["fit_greedy"],
                           "Evenly spread error": c["rmse_kmeans"], "Evenly spread fitted": c["fit_kmeans"]})
        df = df.merge(st.rename(columns={"N": "Number of sensors", "qr": "QR pivoting error",
                                         "random_mean": "Random error"}), how="left")
        df["Gain, sources known (%)"] = 100 * l0["improvement"].values
        df["Gain, sources known: low"] = 100 * l0["ci_low"].values
        df["Gain, sources known: high"] = 100 * l0["ci_high"].values
        df["Gain, random hotspots (%)"] = 100 * c["improvement"].values
        df["Gain, random hotspots: low"] = 100 * c["ci_low"].values
        df["Gain, random hotspots: high"] = 100 * c["ci_high"].values
        ws.write(2, col, CITYNAME[city], book.head)
        r = book.table(ws, 3, col, df, book.num3)
        X = r["Number of sensors"]
        nmax = int(df["Number of sensors"].max())

        ch = xy_chart(book, f"{CITYNAME[city]}: map error falls as sensors are added",
                      "Number of sensors", "Map error, RMSE (µg/m³)", nmax, 0)
        for key, name, colour in (("Weather-informed", "Weather-informed", GREEN), ("Evenly spread", "Evenly spread", TEAL)):
            line(ch, book, ws, X, r[f"{key} error"], f"{name} (measured)", colour,
                 marker={"type": "circle", "size": 4, "fill": {"color": colour}, "border": {"none": True}})
            line(ch, book, ws, X, r[f"{key} fitted"], f"{name} (fitted curve)", colour)
        for key, name, dash in (("QR pivoting error", "QR pivoting", "dash"), ("Random error", "Random", "round_dot")):
            if key in r:
                line(ch, book, ws, X, r[key], name, GREY, dash, 1.5,
                     marker={"type": "circle", "size": 4, "fill": {"color": GREY}, "border": {"none": True}})
        ws.insert_chart(3, 31 + p * 10, ch)

        ch = xy_chart(book, f"{CITYNAME[city]}: gain from weather-informed placement",
                      "Number of sensors", "Accuracy gain over even spacing (%)", nmax)
        for key, name, colour in (("sources known", "Emission sources known", GREEN),
                                  ("random hotspots", "Randomly moving hotspots", TEAL)):
            line(ch, book, ws, X, r[f"Gain, {key} (%)"], name, colour)
            line(ch, book, ws, X, r[f"Gain, {key}: low"], "95% confidence interval", colour, "dash", 1)
            line(ch, book, ws, X, r[f"Gain, {key}: high"], "95% confidence interval", colour, "dash", 1)
        ch.set_legend({"position": "top", "delete_series": [2, 4, 5]})
        ws.insert_chart(23, 31 + p * 10, ch)
        col += len(df.columns) + 1
    ws.set_column(31, 51, 9)
    page(ws, 0, 31, 43, 51)


def sheet_main_test(book, city):
    ws = book.sheet(f"5 Main test {SHORT[city]}",
                    f"{CITYNAME[city]}: map error by placement strategy, scored on all 2025 days",
                    "Left: map error (RMSE, µg/m³). Right: accuracy gain of weather-informed over evenly spread "
                    f"sensors (%), dashed = 95% confidence interval. Real network size: {MAIN_N[city]}.")
    st = strategy_table(city)
    imp = pd.read_csv(P5 / city / f"{city}_improvement_all2025_trainall.csv")
    imp = imp[imp["noise"] == HEADLINE][["N", "improvement", "ci_low", "ci_high"]]
    df = pd.DataFrame({"Number of sensors": st["N"]})
    for key, name, *_ in STRATEGY:
        df[f"{name} error"] = st[key]
    df = df.merge(imp.rename(columns={"N": "Number of sensors"}), how="left")
    df["Gain (%)"], df["Gain: low"], df["Gain: high"] = 100 * df.pop("improvement"), 100 * df.pop("ci_low"), 100 * df.pop("ci_high")
    r = book.table(ws, 3, 0, df, book.num3)
    X = r["Number of sensors"]
    nmax = int(df["Number of sensors"].max())
    ch = xy_chart(book, f"{CITYNAME[city]}: map error by placement strategy", "Number of sensors",
                  "Map error on 2025 days, RMSE (µg/m³)", nmax, 0)
    for key, name, colour, dash in STRATEGY:
        col = df[f"{name} error"]
        if col.isna().any():      # random is only scored at some N: plot as points
            line(ch, book, ws, X, r[f"{name} error"], name, colour,
                 marker={"type": "circle", "size": 5, "fill": {"color": colour}, "border": {"none": True}})
        else:
            line(ch, book, ws, X, r[f"{name} error"], name, colour, dash, 2.25 if colour != GREY else 1.5)
    if "existing" in st:
        ex = st[st["existing"].notna()][["N", "existing"]].rename(
            columns={"N": "Existing network: sensors", "existing": "Existing network: error"})
        re_ = book.table(ws, 3, len(df.columns) + 1, ex, book.num3)
        line(ch, book, ws, re_["Existing network: sensors"], re_["Existing network: error"], "Existing EPA network",
             INK, marker={"type": "diamond", "size": 11, "fill": {"color": INK}, "border": {"color": "#FFFFFF"}})
    ws.insert_chart(3, 14, ch)
    ch = xy_chart(book, f"{CITYNAME[city]}: gain from weather-informed placement", "Number of sensors",
                  "Accuracy gain over even spacing (%)", nmax)
    line(ch, book, ws, X, r["Gain (%)"], "Accuracy gain", GREEN)
    line(ch, book, ws, X, r["Gain: low"], "95% confidence interval", GREEN, "dash", 1)
    line(ch, book, ws, X, r["Gain: high"], "95% CI high", GREEN, "dash", 1)
    ch.set_legend({"position": "top", "delete_series": [2]})
    ws.insert_chart(3, 25, ch)
    page(ws, 0, 14, 22, 35)


def sheet_guide(book):
    ws = book.wb.add_worksheet("Guide")
    ws.write(0, 0, "Poster charts: data and Excel charts", book.title)
    rows = [("Sheet", "What it shows"),
            ("1 Plumes SG / MEL", "Simulated daily PM2.5 maps for a typical day of each weather pattern (cell colour scale)"),
            ("2 Networks SG / MEL", "Where weather-informed, evenly spread and existing networks place sensors"),
            ("3 Predictability", "KEY RESULT: gain from weather-informed placement by how predictable the sources are"),
            ("4 Sensor count", "Map error and gain as sensors are added (Phase 7)"),
            ("5 Main test SG / MEL", "Map error by strategy, and the gain, on all 2025 test days"),
            ("6 Spreading rate", "Phase 8 check: the key result at three spreading rates K"),
            ("", ""),
            ("Notes", "Every chart reads from the table next to it: change a colour or label in Excel and it stays "
                      "linked. Copy a chart into PowerPoint with Paste Special > Keep Source Formatting."),
            ("", "Built by Workflow/Poster/make_excel_workbook.py from the project's result files; "
                 "same numbers as the PNG/PDF poster figures.")]
    for i, (a, b) in enumerate(rows):
        ws.write(2 + i, 0, a, book.head if i == 0 else book.txt)
        ws.write(2 + i, 1, b, book.head if i == 0 else book.txt)
    ws.set_column(0, 0, 22)
    ws.set_column(1, 1, 110)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "Poster charts.xlsx"
    book = Book(str(path))
    sheet_guide(book)
    for c in CITIES:
        sheet_plumes(book, c)
    for c in CITIES:
        sheet_networks(book, c)
    sheet_predictability(book)
    sheet_sensor_count(book)
    for c in CITIES:
        sheet_main_test(book, c)
    sheet_k(book)
    book.wb.close()
    print(f"wrote {path}")
