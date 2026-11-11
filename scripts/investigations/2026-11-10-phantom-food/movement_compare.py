"""Movement statistics: DEPONS tracked porpoise (RandomPorpoise.csv, first 30 days) vs CENOP
population (stock and patched kernels) on the v3.2 landscape: step length (m), fraction of steps
that change cell, and fraction of steps eaten on a patch whose food is above the 0.01 floor."""

import csv
import sys

import numpy as np

CELL = 400.0
rp = "/home/razinka/cenjas/CENOP/output/depons_java/kattegat-psm2-undisturbed-v1/N2000-Y5/seed-42/RandomPorpoise.csv"
rows = [r for r in csv.DictReader(open(rp)) if float(r["tick"]) < 30 * 48]
x = np.array([float(r["UtmX"]) for r in rows])
y = np.array([float(r["UtmY"]) for r in rows])
step = np.hypot(np.diff(x), np.diff(y))
cx, cy = np.floor(x / CELL), np.floor(y / CELL)
chg = (np.diff(cx) != 0) | (np.diff(cy) != 0)
print(f"DEPONS tracked id {rows[0]['Id']} (30 d, {len(step)} steps): step mean {step.mean():.1f} m, "
      f"median {np.median(step):.1f} m, cell-change {chg.mean():.3f}, energy {rows[0]['EnergyLevel'][:5]}"
      f" -> {rows[-1]['EnergyLevel'][:5]}")

sys.path.insert(0, "/home/razinka/cenjas/CENOP/src")
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "0"]
exec(open(__import__("os").path.join(__import__("os").path.dirname(__file__), "hypothesis_test.py")).read().split("for tick in range(1, days")[0])
pm = sim.population_manager  # noqa: F821
patch = cd._food_prob > 0  # noqa: F821
steps, chg_n, fresh, tot = [], 0, 0, 0
for tick in range(1, 30 * 48 + 1):
    a = pm.active_mask.copy()
    x0, y0 = pm.x[a].copy(), pm.y[a].copy()
    sim.step()  # noqa: F821
    a2 = a & pm.active_mask
    keep = a2[a]
    dx, dy = (pm.x[a2] - x0[keep]) * CELL, (pm.y[a2] - y0[keep]) * CELL
    steps.append(np.hypot(dx, dy))
    chg_n += int(((np.floor(pm.x[a2]) != np.floor(x0[keep])) | (np.floor(pm.y[a2]) != np.floor(y0[keep]))).sum())
    tot += int(a2.sum())
s = np.concatenate(steps)
print(f"CENOP [{mode}] (30 d, all animals): step mean {s.mean():.1f} m, median {np.median(s):.1f} m, "  # noqa: F821
      f"cell-change {chg_n/tot:.3f}")
