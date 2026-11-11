"""Phase-1 evidence: yearly population/births/deaths/energy for CENOP baseline vs DEPONS runs."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

CENOP = Path("/home/razinka/cenop/CENOP/output/kattegat_ref")
DEPONS = Path("/home/razinka/cenop/CENOP/output/depons_java/kattegat-psm2-undisturbed-v1/N2000-Y5")
TPY = 17280


def cenop():
    pop = list(csv.DictReader(open(CENOP / "Population.txt"), delimiter="\t"))
    en = list(csv.DictReader(open(CENOP / "Energy.txt"), delimiter="\t"))
    print("CENOP Energy.txt columns:", list(en[0].keys()))
    by = defaultdict(lambda: {"births": 0, "deaths": 0})
    for r in pop:
        y = (int(r["tick"]) - 1) // TPY + 1
        by[y]["births"] += int(r["births"])
        by[y]["deaths"] += int(r["deaths"])
        by[y]["pop_end"] = int(r["population"])
    for r in en:
        y = (int(r["tick"]) - 1) // TPY + 1
        by[y].setdefault("energy", []).append(float(r["mean_energy"]))
    print("CENOP  year  pop_end  births  deaths  mean_energy(min..max)")
    for y in sorted(by):
        e = by[y].get("energy", [0])
        print(f"       {y:4d}  {by[y].get('pop_end', '-'):>7}  {by[y]['births']:6d}  {by[y]['deaths']:6d}"
              f"  {sum(e)/len(e):6.2f} ({min(e):.2f}..{max(e):.2f})")


def depons(seed):
    d = DEPONS / f"seed-{seed}"
    rows = list(csv.DictReader(open(d / "Statistics.csv")))
    print(f"DEPONS seed {seed} columns:", list(rows[0].keys()))
    by = defaultdict(list)
    for r in rows:
        t = int(float(r["tick"]))
        n = int(float(r["PorpoiseCount"]))
        e = float(r["PorpoiseEnergyLevel"]) / n if n else 0.0
        by[t // TPY + 1].append((t, n, e, float(r["FoodEnergyLevel"])))
    print(f"DEPONS seed {seed}  year  pop_start pop_end  mean_energy(min..max)  food_mean")
    for y in sorted(by):
        v = by[y]
        es = [x[2] for x in v]
        print(f"                {y:4d}  {v[0][1]:9d} {v[-1][1]:7d}  {sum(es)/len(es):6.2f} ({min(es):.2f}..{max(es):.2f})"
              f"  {sum(x[3] for x in v)/len(v):.4g}")
    # first ~60 days at daily resolution for the early phase
    print("  early daily (day: pop, mean_energy):",
          ", ".join(f"{t//48}:{n},{e:.1f}" for t, n, e, _ in rows_daily(rows)[:12]))
    ym = list(csv.DictReader(open(d / "YearlyMortality.csv")))
    print("  YearlyMortality columns:", list(ym[0].keys()) if ym else None, "rows", len(ym))


def rows_daily(rows):
    out = []
    for r in rows:
        t = int(float(r["tick"]))
        if t % (48 * 15) == 0:
            n = int(float(r["PorpoiseCount"]))
            out.append((t, n, float(r["PorpoiseEnergyLevel"]) / max(n, 1), 0))
    return out


cenop()
for s in sys.argv[1:] or ["42"]:
    depons(s)
