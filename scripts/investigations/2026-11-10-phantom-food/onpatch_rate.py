"""On-patch rate and pre-eat food per on-patch step: DEPONS tracked porpoise(s) vs CENOP (fixed
kernel), both on the v3.2 Kattegat landscape.

DEPONS: RandomPorpoise.csv (one tracked animal per tick, UTM). The cell the animal is in at the
end of tick t is the pre-move (eating) cell of tick t+1, as in DEPONS posList.get(1).
CENOP: food is read at each animal's current cell BEFORE sim.step(), i.e. what it can eat.

Usage: onpatch_rate.py <depons RandomPorpoise.csv> <cenop data_dir with Kattegat/> [days]
"""

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

rp_path, data_dir = sys.argv[1], sys.argv[2]
days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
XLL, YLL, CS, NROWS = 529473.0, 5972242.0, 400.0, 1000

with open(Path(data_dir) / "Kattegat" / "patches.asc") as fh:
    lines = fh.readlines()
patches = np.loadtxt(lines[6:]) > 0  # file order: top row first

rows = list(csv.DictReader(open(rp_path)))
n_all = len(rows)
# Rows with placeholder coordinates (no live tracked animal) fall outside the grid; drop them.
rows = [r for r in rows if XLL <= float(r["UtmX"]) < XLL + 600 * CS
        and YLL <= float(r["UtmY"]) < YLL + NROWS * CS]
print(f"DEPONS tracked rows in grid: {len(rows)} of {n_all}")
ids = [r["Id"] for r in rows]
x = np.array([float(r["UtmX"]) for r in rows])
y = np.array([float(r["UtmY"]) for r in rows])
col = np.floor((x - XLL) / CS).astype(int)
row = NROWS - 1 - np.floor((y - YLL) / CS).astype(int)
on = patches[row, col]
t = np.array([float(r["tick"]) for r in rows])
for lo, hi in ((0, days), (0, 360), (360, 1800)):
    m = (t >= lo * 48) & (t < hi * 48)
    uniq = sorted(set(np.array(ids)[m]))
    print(f"DEPONS tracked, days {lo}-{hi}: on-patch {on[m].mean():.4f} over {m.sum()} ticks "
          f"(animals {uniq[:5]}{'...' if len(uniq) > 5 else ''})")
print(f"patch share of water cells: {patches.sum() / 287598:.4f}")

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
patch = cd._food_prob > 0
on_n = pt = 0
food_on = 0.0
for tick in range(1, days * 48 + 1):
    a = pm.active_mask
    xi = np.clip(pm.x[a].astype(np.int32), 0, cd.width - 1)
    yi = np.clip(pm.y[a].astype(np.int32), 0, cd.height - 1)
    o = patch[yi, xi]
    on_n += int(o.sum())
    pt += int(a.sum())
    food_on += float(cd._food_value[yi[o], xi[o]].sum())
    sim.step()
    if tick % (10 * 48) == 0 or tick == 48:
        print(f"CENOP fixed, day {tick/48:4.0f}: on-patch {on_n/pt:.4f}  pre-eat food per on-patch "
              f"step {food_on/max(on_n, 1):.4f}  meanE {pm.energy[pm.active_mask].mean():.2f}")
        on_n = pt = 0
        food_on = 0.0
