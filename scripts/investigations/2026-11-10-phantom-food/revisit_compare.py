"""Movement-pattern comparison, days 1-30 on the v3.2 landscape:
- self-revisit rate: fraction of steps in which an animal is in a cell IT occupied within the
  previous 48 ticks (excluding the immediately previous tick), overall and on patch cells;
- net displacement after 1 day and 10 days (km).
DEPONS: 200 tracked animals (RandomPorpoise.csv, UTM). CENOP: fixed kernel, all animals.

Usage: revisit_compare.py <RandomPorpoise.csv> <cenop data_dir with Kattegat/>
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
XLL, YLL, CS, NROWS, NCOLS = 529473.0, 5972242.0, 400.0, 1000, 600
T0, T1, WIN = 48, 30 * 48, 48


def stats(tracks, patch_of):
    """tracks: {id: [(tick, col, row, x_m, y_m), ...]} sorted by tick."""
    rev = rev_p = n = n_p = 0
    d1, d10 = [], []
    for tr in tracks.values():
        last = {}
        pos = {tk: (xm, ym) for tk, _, _, xm, ym in tr}
        for tk, c, r, xm, ym in tr:
            if T0 <= tk < T1:
                seen = last.get((c, r))
                hit = seen is not None and 1 < tk - seen <= WIN
                p = patch_of(c, r)
                n += 1
                rev += hit
                if p:
                    n_p += 1
                    rev_p += hit
                for lag, out in ((48, d1), (480, d10)):
                    if tk % 480 == 0 and tk + lag in pos:
                        x2, y2 = pos[tk + lag]
                        out.append(np.hypot(x2 - xm, y2 - ym) / 1000.0)
            last[(c, r)] = tk
    return (f"self-revisit {rev / n:.3f} (on patches {rev_p / max(n_p, 1):.3f}, n_patch={n_p}); "
            f"net displacement 1 d {np.mean(d1):.2f} km, 10 d {np.mean(d10):.2f} km")


rp, data_dir = sys.argv[1], sys.argv[2]
with open(Path(data_dir) / "Kattegat" / "patches.asc") as fh:
    patches_file = np.loadtxt(fh.readlines()[6:]) > 0
dep = defaultdict(list)
with open(rp) as fh:
    for r in csv.DictReader(fh):
        ux, uy, tk = float(r["UtmX"]), float(r["UtmY"]), int(float(r["tick"]))
        if tk > T1 + 480:
            break
        if XLL <= ux < XLL + NCOLS * CS and YLL <= uy < YLL + NROWS * CS:
            c = int((ux - XLL) // CS)
            row = NROWS - 1 - int((uy - YLL) // CS)
            dep[r["Id"]].append((tk, c, row, ux, uy))
print("DEPONS (200 tracked):", stats(dep, lambda c, r: patches_file[r, c]))

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
cen = defaultdict(list)
sample = np.arange(0, 2000, 10)  # 200 animals, like DEPONS
for tick in range(1, T1 + 480 + 1):
    sim.step()
    for i in sample:
        if pm.active_mask[i]:
            xs, ys = float(pm.x[i]), float(pm.y[i])
            cen[i].append((tick, int(xs), int(ys), xs * CS, ys * CS))
print("CENOP  (200 sampled):", stats(cen, lambda c, r: patch[min(r, cd.height - 1), min(c, cd.width - 1)]))
