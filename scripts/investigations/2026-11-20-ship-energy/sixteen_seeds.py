"""Pre-registered check of the ship energy difference with 16 DEPONS seeds (2026-11-20).

Data: DEPONS ship seeds 42-49 (output/a2/java) and 50-57 (output/a2/java-extra) against the 16
CENOP ship trajectories of runs 3 and 4 (output/a2/cenop/d-ships-run3 and d-ships).
Quantities: mean energy at day 150, and each seed's mean over days 110-150.
Rule, fixed before the extra seeds were seen: the difference holds if either quantity differs
by more than 2 standard errors.

PYTHONPATH=. python3 scripts/investigations/2026-11-20-ship-energy/sixteen_seeds.py
"""

from pathlib import Path

import numpy as np

from validation.a2 import report as r

ROOT = Path("output/a2")
N_DAYS = 360


def java(dirname):
    seeds = sorted((ROOT / dirname / "a2-d-ships" / "N2000-Y1").glob("seed-*"))
    return np.array([r.load_java_seed(d, N_DAYS)["energy"] for d in seeds])


def cenop(dirname):
    files = sorted(
        f for f in (ROOT / "cenop" / dirname).glob("seed-*.csv") if "deaths" not in f.name
    )
    return np.array([r.load_cenop_seed(f)["energy"] for f in files])


j_old, j_new = java("java"), java("java-extra")
c3, c4 = cenop("d-ships-run3"), cenop("d-ships")
j, c = np.vstack([j_old, j_new]), np.vstack([c3, c4])
print(f"DEPONS seeds: {len(j_old)} + {len(j_new)}; CENOP trajectories: {len(c3)} + {len(c4)}")
holds = False
for label, f in (
    ("day 150", lambda e: e[:, 150]),
    ("days 110-150", lambda e: e[:, 110:151].mean(axis=1)),
):
    a, b = f(j), f(c)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    ratio = (b.mean() - a.mean()) / se
    holds |= abs(ratio) > 2
    print(
        f"{label:13s} DEPONS {a.mean():.3f} (first 8: {f(j_old).mean():.3f}, new 8: {f(j_new).mean():.3f})  "
        f"CENOP {b.mean():.3f} (run 3: {f(c3).mean():.3f}, run 4: {f(c4).mean():.3f})  "
        f"difference {b.mean() - a.mean():+.3f}  SE {se:.3f}  ratio {ratio:+.2f}"
    )
print(
    "The difference HOLDS (more than 2 SE)."
    if holds
    else "The difference does NOT hold (within 2 SE)."
)
