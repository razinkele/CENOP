"""Spatial spread: animals per DEPONS block (blocks.asc) over time, DEPONS vs CENOP (fixed kernel),
v3.2 landscape. Reports the number of occupied blocks and the share held by the top-5 blocks.

Usage: block_distribution.py <depons PorpoisePerBlock.csv> <cenop data_dir with Kattegat/>
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
DAYS = (1, 5, 10, 20, 30)

ppb, data_dir = sys.argv[1], sys.argv[2]
want = {d * 48 for d in DAYS}
dep = defaultdict(dict)
with open(ppb) as fh:
    for r in csv.DictReader(fh):
        t = int(float(r["tick"]))
        if t in want:
            dep[t][int(r["Block"])] = int(r["PorpoiseCount"])
        elif t > max(want):
            break


def summary(counts):
    v = np.array(sorted(counts.values(), reverse=True), dtype=float)
    n = v.sum()
    return f"N={int(n):4d} occupied={int((v > 0).sum()):2d} top5={v[:5].sum() / n:.2f}"


from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

params = SimulationParameters(porpoise_count=2000, sim_years=1, random_seed=42, landscape="Kattegat",
                              ships_enabled=False, turbines="off")
cd = CellData("Kattegat", data_dir)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
blocks = cd._blocks
for tick in range(1, max(DAYS) * 48 + 1):
    sim.step()
    if tick in want:
        a = pm.active_mask
        xi = np.clip(pm.x[a].astype(np.int32), 0, cd.width - 1)
        yi = np.clip(pm.y[a].astype(np.int32), 0, cd.height - 1)
        b, c = np.unique(blocks[yi, xi], return_counts=True)
        cen = dict(zip(b.tolist(), c.tolist()))
        print(f"day {tick // 48:3d}  DEPONS {summary(dep[tick])} | CENOP {summary(cen)}")
        top_d = sorted(dep[tick], key=dep[tick].get, reverse=True)[:5]
        top_c = sorted(cen, key=cen.get, reverse=True)[:5]
        print(f"         top blocks DEPONS {top_d} CENOP {top_c}")
