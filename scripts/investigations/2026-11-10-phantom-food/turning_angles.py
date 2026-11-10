"""Turning-angle distribution between consecutive steps, days 1-30, v3.2 landscape:
DEPONS (200 tracked, UTM) vs CENOP (fixed kernel, 200 sampled animals).
Reports mean |turn|, share of |turn| > 120 deg (reversals) and mean cos(turn) (persistence).

Usage: turning_angles.py <RandomPorpoise.csv> <cenop data_dir with Kattegat/>
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
T0, T1, CS = 48, 30 * 48, 400.0


def turns(tracks):
    out = []
    for tr in tracks.values():
        tr = np.array(sorted(tr))
        tr = tr[(tr[:, 0] >= T0) & (tr[:, 0] < T1)]
        if len(tr) < 3:
            continue
        consec = np.diff(tr[:, 0]) == 1
        d = np.diff(tr[:, 1:3], axis=0)
        h = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        ok = consec[1:] & consec[:-1] & (np.hypot(d[1:, 0], d[1:, 1]) > 0) & (
            np.hypot(d[:-1, 0], d[:-1, 1]) > 0)
        dt = (np.diff(h) + 180.0) % 360.0 - 180.0
        out.append(dt[ok])
    t = np.concatenate(out)
    return (f"n={t.size} mean|turn|={np.abs(t).mean():.1f} deg  reversals(>120)={np.mean(np.abs(t) > 120):.3f}"
            f"  mean cos={np.cos(np.radians(t)).mean():.3f}  "
            f"quantiles|turn| 50/90%={np.percentile(np.abs(t), 50):.0f}/{np.percentile(np.abs(t), 90):.0f}")


rp, data_dir = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "stock"
dep = defaultdict(list)
with open(rp) as fh:
    for r in csv.DictReader(fh):
        tk = int(float(r["tick"]))
        if tk > T1:
            break
        ux, uy = float(r["UtmX"]), float(r["UtmY"])
        if ux > 0 and uy > 0:
            dep[r["Id"]].append((tk, ux, uy))
print("DEPONS:", turns(dep))

from cenop.core.simulation import Simulation  # noqa: E402
import cenop.behavior.ref_mem as RM  # noqa: E402
if mode == "novt":  # experiment: memory attraction vector forced to zero
    _orig = RM.compute_attraction_vector
    def _zero_vt(*a, **k):
        vx, vy = _orig(*a, **k)
        return np.zeros_like(vx), np.zeros_like(vy)
    RM.compute_attraction_vector = _zero_vt
if mode == "prefix":  # pre-fix (v2.2.0) eat kernel: floors every visited cell (phantom food)
    import cenop.optimizations.kernels as _K
    from numba import njit
    @njit(cache=False)
    def _stock_v2(food_grid, x_indices, y_indices, energy, food_eaten, min_food, demand_grid):
        n = x_indices.shape[0]
        agent_demand = np.empty(n, dtype=np.float32)
        demand_grid[:, :] = 0.0
        for i in range(n):
            frac = min(max((20.0 - energy[i]) / 10.0, 0.0), 0.99)
            d = food_grid[y_indices[i], x_indices[i]] * frac
            agent_demand[i] = d
            demand_grid[y_indices[i], x_indices[i]] += d
        for i in range(n):
            td = demand_grid[y_indices[i], x_indices[i]]
            if td <= 0.0:
                food_eaten[i] = 0.0
                continue
            av = max(food_grid[y_indices[i], x_indices[i]], 0.0)
            food_eaten[i] = agent_demand[i] if td <= av else av * agent_demand[i] / td
        for i in range(n):
            food_grid[y_indices[i], x_indices[i]] -= food_eaten[i]
            if food_grid[y_indices[i], x_indices[i]] < min_food:
                food_grid[y_indices[i], x_indices[i]] = min_food
    _K.eat_food_kernel_v2 = _stock_v2
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

params = SimulationParameters(porpoise_count=2000, sim_years=1, random_seed=42, landscape="Kattegat",
                              ships_enabled=False, turbines="off",
                              communication_enabled=(mode != "nocomm"))
cd = CellData("Kattegat", data_dir)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
cen = defaultdict(list)
sample = np.arange(0, 2000, 10)
for tick in range(1, T1 + 1):
    sim.step()
    for i in sample:
        if pm.active_mask[i]:
            cen[i].append((tick, float(pm.x[i]) * CS, float(pm.y[i]) * CS))
print(f"CENOP [{mode}]:", turns(cen))
d1 = [np.hypot(*(np.array(v[min(len(v) - 1, 48)][1:]) - np.array(v[0][1:]))) / 1000 for v in cen.values()]
print(f"CENOP [{mode}] net displacement first day: {np.mean(d1):.2f} km")
