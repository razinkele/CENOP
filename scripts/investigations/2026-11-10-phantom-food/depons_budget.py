"""D15 step 1, DEPONS side: the same 15-day bins as cenop_budget.py, from a DEPONS run's output.

- N and mean energy at the bin start (Statistics.csv);
- intake per animal-tick inferred from FoodEnergyLevel drops (ticks where food rose are regrowth
  ticks and are skipped, as in depons_intake.py);
- deaths in the bin (Reproduction.csv: one row per death, TickAtDeath) and births = dN + deaths.

Usage: depons_budget.py <run dir containing Statistics.csv and Reproduction.csv> [days]
"""

import csv
import sys
from pathlib import Path

run = Path(sys.argv[1])
days = int(sys.argv[2]) if len(sys.argv) > 2 else 180
BIN = 15

rows = list(csv.DictReader(open(run / "Statistics.csv")))
t = [int(float(r["tick"])) for r in rows]
n = [int(float(r["PorpoiseCount"])) for r in rows]
f = [float(r["FoodEnergyLevel"]) for r in rows]
e = [float(r["PorpoiseEnergyLevel"]) for r in rows]
deaths = [int(float(r["TickAtDeath"])) for r in csv.DictReader(open(run / "Reproduction.csv"))]
idx = {tick: i for i, tick in enumerate(t)}

print(f"DEPONS {run}")
print("bin(days)   N0    E0   intake/at(inferred)  deaths  births")
for b0 in range(0, days, BIN):
    lo, hi = max(b0 * 48, t[0]), (b0 + BIN) * 48  # Statistics.csv starts at tick 1
    if lo not in idx or hi not in idx:
        break
    eaten = 0.0
    at = 0
    for i in range(idx[lo] + 1, idx[hi] + 1):
        if f[i] <= f[i - 1]:
            eaten += f[i - 1] - f[i]
            at += n[i - 1]
    d = sum(1 for x in deaths if lo < x <= hi)
    births = n[idx[hi]] - n[idx[lo]] + d
    n0 = n[idx[lo]]
    print(
        f"{b0:3d}-{b0 + BIN:3d}  {n0:5d} {e[idx[lo]] / max(n0, 1):5.2f}   {eaten / max(at, 1):.5f}"
        f"            {d:5d}   {births:5d}"
    )
