import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P5 = ROOT / "data" / "processed" / "phase5"
P7 = ROOT / "data" / "processed" / "phase7"
OUT = ROOT / "data" / "processed" / "phase8"
K_VALS = {200: "_K200", 1000: "", 8000: "_K8000"}
MAIN_N = {"singapore": 22, "melbourne": 8}
FIXED = ["L0", "L1", "L2_setA", "L2_setB", "L3_setA", "L3_setB", "L4_setA", "L4_setB"]
L0_L4 = ["L0", "L1", "L2_setA", "L2_setB", "L3_setA", "L3_setB", "L4_setA", "L4_setB"]
REALISTIC = ["L2_setA", "L2_setB", "L3_setA", "L3_setB", "L4_setA", "L4_setB"]
CRIT_PAIRS = ["equal_count_N22", "equal_count_N40", "equal_density_SG8_MEL32",
              "equal_density_SG22_MEL88", "equal_density_SG40_MEL160"]


def sig(ci):
    """+1 if the 95% confidence interval is above zero, -1 if below, 0 if it includes zero."""
    return 1 if ci[0] > 0 else (-1 if ci[1] < 0 else 0)


def load(k):
    t = K_VALS[k]
    return (json.load(open(P5 / f"phase5_results_all2025_trainall{t}.json")),
            json.load(open(P5 / f"phase5_emission_predictability{t}.json")),
            json.load(open(P7 / f"phase7_budget_sweep{t}.json")))


def estimates(k):
    ev, pred, sw = load(k)
    e = {}
    # C1: original design at real network size
    for c in MAIN_N:
        r = ev[c]
        e[f"C1_{c}"] = [r["improvement_greedy_vs_kmeans"], *r["ci95"]]
    # C2: L0-L4 at real network size, per city
    for c, n in MAIN_N.items():
        cells = {lv: pred[c][lv]["improvement"][str(n)] for lv in L0_L4}
        e[f"C2_{c}"] = cells
    # C3: scenario where Melbourne benefits more at L2-L4
    d = pred["city_difference_mel_minus_sg"]
    e["C3"] = {lv: {p: d[lv][p] for p in CRIT_PAIRS} for lv in REALISTIC}
    # C4: Singapore minus Melbourne at own network size, L0-L4
    e["C4"] = {lv: [-d[lv]["own_network_size"][0], -d[lv]["own_network_size"][2],
                    -d[lv]["own_network_size"][1]] for lv in L0_L4}
    # C5: L5 at real network size
    for c, n in MAIN_N.items():
        e[f"C5_{c}"] = pred[c]["L5"]["improvement"][str(n)]
    # C6: N* density 
    cc = sw["city_comparison_area_fair"]["original"]["greedy"]
    e["C6"] = {"singapore": cc["density_singapore"], "melbourne": cc["density_melbourne"],
               "ci_singapore": cc["density_ci_singapore"], "ci_melbourne": cc["density_ci_melbourne"]}
    # C7: slope of improvement on ln N 
    for c in MAIN_N:
        s = sw[c]["original"]["slopes"]["full"]
        e[f"C7_{c}"] = [s["beta1"], *s["ci95"], s["per_doubling_pp"]]
    # variogram range in km of the original design training maps and L0
    e["variogram_range_km"] = {c: {"original": sw[c]["original"]["variogram_range_km"],
                                   "L0": sw[c]["L0"]["variogram_range_km"]} for c in MAIN_N}
    e["test_rmse_original"] = {c: ev[c]["test_rmse"] for c in MAIN_N}
    return e


def verdicts(E):
    out = {}
    for k, e in E.items():
        v = {}
        v["C1"] = {c: sig(e[f"C1_{c}"][1:]) for c in MAIN_N}
        v["C2"] = {c: sum(sig(x[1:]) == 1 for x in e[f"C2_{c}"].values()) for c in MAIN_N}
        pos = sum(sig(x[1:]) == 1 for lv in e["C3"].values() for x in lv.values())
        neg = sum(sig(x[1:]) == -1 for lv in e["C3"].values() for x in lv.values())
        v["C3"] = {"sig_mel_greater": pos, "sig_sg_greater": neg, "n": 30, "criterion_met": pos == 30}
        v["C4"] = sum(sig(x[1:]) == 1 for x in e["C4"].values())
        v["C5"] = {c: abs(e[f"C5_{c}"][0]) <= 0.05 for c in MAIN_N}
        ds, dm = e["C6"]["singapore"], e["C6"]["melbourne"]
        v["C6"] = {"ratio": max(ds, dm) / min(ds, dm)}
        v["C7"] = {c: e[f"C7_{c}"][0] < 0 and e[f"C7_{c}"][2] < 0 for c in MAIN_N}
        out[k] = v
    h = out[1000]
    holds = {}
    alt = [out[200], out[8000]]
    holds["C1"] = all(a["C1"] == h["C1"] for a in alt)
    holds["C2"] = all(a["C2"][c] > len(L0_L4) / 2 for a in alt for c in MAIN_N)
    holds["C3"] = all(not a["C3"]["criterion_met"] for a in alt)
    holds["C4"] = all(a["C4"] > len(L0_L4) / 2 for a in alt)
    holds["C5"] = all(all(a["C5"].values()) for a in alt)
    holds["C6"] = all(a["C6"]["ratio"] <= 1.5 for a in alt)
    flag = {}
    for k in (200, 8000):
        for c in MAIN_N:
            rel = E[k]["C6"][c] / E[1000]["C6"][c] - 1
            flag[f"K{k}_{c}"] = rel
    holds["C6_moved_more_than_25pct"] = {key: abs(r) > 0.25 for key, r in flag.items()}
    holds["C6_relative_change"] = flag
    holds["C7"] = all(not any(a["C7"].values()) for a in alt)
    return out, holds


def pct(x):
    return f"{100 * x:+.1f}%"


def ci(x):
    return f"{pct(x[0])} ({100 * x[1]:+.1f}, {100 * x[2]:+.1f})"


def table(E, V, H):
    ks = [200, 1000, 8000]
    yn = lambda b: "**holds**" if b else "**does not hold**"
    rows = ["| # | Measure | K = 200 | K = 1,000 (headline) | K = 8,000 | Verdict |",
            "|---|---|---|---|---|---|"]
    rows.append(f"| C1 | Original design, Singapore N = 22 | " + " | ".join(ci(E[k]["C1_singapore"]) for k in ks)
                + f" | {yn(H['C1'])} |")
    rows.append(f"| C1 | Original design, Melbourne N = 8 | " + " | ".join(ci(E[k]["C1_melbourne"]) for k in ks) + " | |")
    rows.append("| C2 | L0–L4 cells with CI > 0, Singapore (of 8) | " + " | ".join(str(V[k]["C2"]["singapore"]) for k in ks)
                + f" | {yn(H['C2'])} |")
    rows.append("| C2 | L0–L4 cells with CI > 0, Melbourne (of 8) | " + " | ".join(str(V[k]["C2"]["melbourne"]) for k in ks) + " | |")
    rows.append("| C3 | L2–L4 city comparisons: Melbourne > Singapore / Singapore > Melbourne (significant, of 30) | "
                + " | ".join(f"{V[k]['C3']['sig_mel_greater']} / {V[k]['C3']['sig_sg_greater']}" for k in ks)
                + f" | {yn(H['C3'])} (criterion not met) |")
    rows.append("| C4 | L0–L4 cells where Singapore > Melbourne at real size, CI > 0 (of 8) | "
                + " | ".join(str(V[k]["C4"]) for k in ks) + f" | {yn(H['C4'])} |")
    for c, n in MAIN_N.items():
        rows.append(f"| C5 | L5 random hotspots, {c.title()} N = {n} | " + " | ".join(ci(E[k][f'C5_{c}']) for k in ks)
                    + (f" | {yn(H['C5'])} |" if c == "singapore" else " | |"))
    rows.append("| C6 | N* density per 100 km², Singapore / Melbourne (area-fair) | "
                + " | ".join(f"{E[k]['C6']['singapore']:.1f} / {E[k]['C6']['melbourne']:.1f}" for k in ks)
                + f" | {yn(H['C6'])} |")
    for c in MAIN_N:
        rows.append(f"| C7 | Slope of improvement per doubling of N, {c.title()} (pp) | "
                    + " | ".join(f"{E[k][f'C7_{c}'][3]:+.2f} ({100 * E[k][f'C7_{c}'][1] * 0.6931:+.2f}, {100 * E[k][f'C7_{c}'][2] * 0.6931:+.2f})" for k in ks)
                    + (f" | {yn(H['C7'])} |" if c == "singapore" else " | |"))
    rows.append("| – | Variogram range of training maps, Singapore / Melbourne (km) | "
                + " | ".join(f"{E[k]['variogram_range_km']['singapore']['original']:.0f} / {E[k]['variogram_range_km']['melbourne']['original']:.0f}" for k in ks)
                + " | descriptive |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    E = {k: estimates(k) for k in K_VALS}
    V, H = verdicts(E)
    with open(OUT / "phase8_k_sensitivity.json", "w") as f:
        json.dump({"estimates": E, "checks": V, "holds": H}, f, indent=2, default=float)
    md = table(E, V, H)
    (OUT / "phase8_k_sensitivity_table.md").write_text(md)
    print(md)
    print(json.dumps(H, indent=2, default=float))
    for k in E:
        print(k, "C2 detail", {c: {lv: round(100 * x[0], 1) for lv, x in E[k][f"C2_{c}"].items()} for c in MAIN_N})
        print(k, "C4 detail", {lv: round(100 * x[0], 1) for lv, x in E[k]["C4"].items()})
        print(k, "RMSE", E[k]["test_rmse_original"])
