"""Yearly population, DEPONS-Java against CENOP, with all available seeds per side.

Java: output/depons_java/kattegat-{idw,psm2}-undisturbed-v1/N2000-Y5/seed-* (seeds 42, 43)
and output/depons_java/{idw,psm2}_5yr_seeds/*/N2000-Y5/seed-* (scenarios/*_5yr_seeds.json).
CENOP: output/cenop_idw/<tag>/<mode>/seed-*.csv from run_cenop.py.

python3 scripts/investigations/2026-11-18-dispersal-idw/yearly_seeds.py [tag]
"""

import csv
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
tag = sys.argv[1] if len(sys.argv) > 1 else "final"
MODES = {"InnerDanishWaters": "idw", "PSM-Type2": "psm2"}
# Java writes ticks 0 .. 86399; the last tick of year k is 17280 k - 1
YEAR_TICKS = [17280 * k - 1 for k in range(1, 6)]


def java_years(path):
    want = set(YEAR_TICKS)
    got = {}
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t = int(float(row[1]))
            if t in want:
                got[t] = int(row[2])
    return [got[t] for t in YEAR_TICKS]


for mode, key in MODES.items():
    paths = sorted(
        glob.glob(
            str(
                ROOT
                / f"output/depons_java/kattegat-{key}-undisturbed-v1/N2000-Y5/seed-*/Statistics.csv"
            )
        )
        + glob.glob(
            str(ROOT / f"output/depons_java/{key}_5yr_seeds/*/N2000-Y5/seed-*/Statistics.csv")
        )
    )
    java = np.array([java_years(p) for p in paths])
    cen = []
    for p in sorted(glob.glob(str(ROOT / f"output/cenop_idw/{tag}/{mode}/seed-*.csv"))):
        a = np.loadtxt(p, delimiter=",", skiprows=1)
        by_day = {int(r[0]): int(r[1]) for r in a}
        cen.append([by_day[360 * k - 1] for k in range(1, 6)])
    cen = np.array(cen)
    print(f"== {mode}: Java {len(java)} seeds, CENOP {len(cen)} seeds ==")
    for k in range(5):
        j, c = java[:, k], cen[:, k]
        se = np.sqrt(j.var(ddof=1) / len(j) + c.var(ddof=1) / len(c))
        diff = c.mean() - j.mean()
        print(
            f"  year {k + 1}: Java {j.mean():6.1f} (SD {j.std(ddof=1):4.1f}, {j.min()}-{j.max()})  "
            f"CENOP {c.mean():6.1f} (SD {c.std(ddof=1):4.1f}, {c.min()}-{c.max()})  "
            f"difference {diff:+5.1f} = {100 * diff / j.mean():+5.1f}%  (SE {se:.1f}, {diff / se:+.1f} SE)"
        )
    print("  Java  per seed:", java.tolist())
    print("  CENOP per seed:", cen.tolist())
