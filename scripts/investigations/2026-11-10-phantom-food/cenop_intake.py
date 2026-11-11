"""Per-capita food intake per tick in CENOP, inferred exactly as for DEPONS (food-sum drops
within a day). Usage: cenop_intake.py <data_dir containing Kattegat/> <days> [count]"""

import sys

import numpy as np

sys.path.insert(0, "/home/razinka/cenop/CENOP/src")
from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

data_dir, days = sys.argv[1], int(sys.argv[2])
count = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
params = SimulationParameters(porpoise_count=count, sim_years=1, random_seed=42, landscape="Kattegat",
                              ships_enabled=False, turbines="off")
cd = CellData("Kattegat", data_dir)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
patch = cd._food_prob > 0


def snap():
    a = pm.active_mask
    return float(cd._food_value[patch].sum()), int(a.sum()), float(pm.energy[a].mean())


rec = [snap()]
print("t0: food=%.3f N=%d meanE=%.3f  backend flags:" % rec[0],
      {k: getattr(__import__("cenop.agents.population", fromlist=["x"]), k, None)
       for k in ("_HAS_KERNELS", "_HAS_CYTHON", "_HAS_LAND_KERNEL")})
for tick in range(days * 48):
    sim.step()
    rec.append(snap())

f = np.array([r[0] for r in rec])
n = np.array([r[1] for r in rec])
e = np.array([r[2] for r in rec])
d = np.diff(f)
ups = np.where(d > 0)[0] + 1
print("first food-increase ticks:", ups[:6].tolist())


def window(day0, nd):
    lo, hi = day0 * 48 + 1, (day0 + nd) * 48 + 1
    seg = d[lo - 1:hi - 1]
    mask = seg <= 0
    return -seg[mask].sum() / n[lo:hi][mask].sum()


for d0 in (0, 1, 5, 30, 60, 120, 300):
    if d0 * 48 < len(rec) - 3 * 48:
        i = d0 * 48
        print(f"day {d0:4d}: N={n[i]:5d} food={f[i]:8.2f} meanE={e[i]:6.2f}  intake/porp/tick (next 3 d)={window(d0, 3):.5f}")
