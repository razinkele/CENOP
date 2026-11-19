"""Dispersal off: population and mean energy, Java (8 seeds) vs CENOP (6 seeds)."""

import csv
import glob

import numpy as np

S = "output/population_gap"  # cenop_deaths.py off <seed> 359 data, seeds 42-47
days = [30, 60, 90, 120, 135, 150, 165, 180, 240, 300, 359]


def java(path):
    out = {}
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t = int(float(row[1]))
            if t % 48 == 0 and t // 48 in days:
                n = int(row[2])
                out[t // 48] = (n, float(row[4]) / max(n, 1))
    return out


J = [
    java(p)
    for p in sorted(glob.glob("output/depons_java/off_1yr_seeds/*/N2000-Y1/seed-*/Statistics.csv"))
]
C = []
for seed in range(42, 48):
    a = np.loadtxt(f"{S}/off-seed-{seed}-daily.csv", delimiter=",", skiprows=1)
    C.append({int(r[0]): (int(r[1]), float(r[2])) for r in a})
print(f"Java {len(J)} seeds, CENOP {len(C)} seeds")
print(
    "day |  N: Java mean (SD)   CENOP mean (SD)   diff (in SE) |  mean energy: Java (SD)   CENOP (SD)   diff (in SE)"
)
for d in days:
    jn = np.array([j[d][0] for j in J])
    cn = np.array([c[d][0] for c in C])
    je = np.array([j[d][1] for j in J])
    ce = np.array([c[d][1] for c in C])
    sen = np.sqrt(jn.var(ddof=1) / len(jn) + cn.var(ddof=1) / len(cn))
    see = np.sqrt(je.var(ddof=1) / len(je) + ce.var(ddof=1) / len(ce))
    print(
        f"{d:3d} | {jn.mean():7.1f} ({jn.std(ddof=1):4.1f})  {cn.mean():7.1f} ({cn.std(ddof=1):4.1f})  "
        f"{cn.mean() - jn.mean():+6.1f} ({(cn.mean() - jn.mean()) / sen:+.1f}) | "
        f"{je.mean():5.2f} ({je.std(ddof=1):.2f})  {ce.mean():5.2f} ({ce.std(ddof=1):.2f})  "
        f"{ce.mean() - je.mean():+.2f} ({(ce.mean() - je.mean()) / see:+.1f})"
    )
