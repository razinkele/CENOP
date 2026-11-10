"""Hypothesis test for finding 3: CENOP's PSM-Type2 dispersal lacks DEPONS's per-step stop rules
(AbstractPSMDispersal.disperse): stop when the point one dispersal step ahead is outside the
landscape or its depth <= minDispDepth (wdisp = 4 m). Adds them in memory (no source edits) and
compares the dispersing share and population with DEPONS (dispersal on), communication off.

Usage: dispersal_waterstop.py <cenop data_dir with Kattegat/> <water|water+tdisp4> [days]
       [DEPONS Statistics.csv] [DEPONS RandomPorpoise.csv with 200 tracked]
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import cenop.agents.population as P  # noqa: E402
from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

data_dir, mode = sys.argv[1], sys.argv[2]
days = int(sys.argv[3]) if len(sys.argv) > 3 else 180
WDISP = 4.0
HEADING = "psm" if "psmheading" in mode else "post"

_orig = P.PorpoisePopulation._update_dispersal


def _with_water_stop(self, mask):
    disp = mask & self.is_dispersing
    if disp.any() and self.landscape is not None:
        step = self.params.mean_disp_dist / 0.4
        idx = np.where(disp)[0]
        hd = self._prev_step_heading if HEADING == "psm" else self.heading
        rad = np.radians(hd[idx])
        px = self.x[idx] + np.sin(rad) * step
        py = self.y[idx] + np.cos(rad) * step
        w, h = self.landscape.width, self.landscape.height
        outside = (px < 0) | (py < 0) | (px >= w - 0.5) | (py >= h - 0.5)
        xi = np.clip(px.astype(np.int32), 0, w - 1)
        yi = np.clip(py.astype(np.int32), 0, h - 1)
        low = self.landscape._depth[yi, xi] <= WDISP
        stop = idx[outside | low]
        self.is_dispersing[stop] = False
        self.dispersal_distance_traveled[stop] = 0.0
        self.days_declining_energy[stop] = 0
    if "pathlen" in mode:  # DEPONS completion: path length travelled >= 0.95 x target distance
        d2 = mask & self.is_dispersing
        done = d2 & (self.dispersal_distance_traveled >= 0.95 * self.dispersal_target_distance)
        self.is_dispersing[done] = False
        self.dispersal_distance_traveled[done] = 0.0
        self.days_declining_energy[done] = 0
    return _orig(self, mask)


P.PorpoisePopulation._update_dispersal = _with_water_stop

dep = {}
if len(sys.argv) > 4:
    with open(sys.argv[4]) as fh:
        for r in csv.DictReader(fh):
            t = int(float(r["tick"]))
            if t % 48 == 0 and t // 48 <= days:
                n = int(float(r["PorpoiseCount"]))
                dep[t // 48] = (n, float(r["PorpoiseEnergyLevel"]) / max(n, 1))
dshare = defaultdict(lambda: [0, 0])
if len(sys.argv) > 5:
    with open(sys.argv[5]) as fh:
        for r in csv.DictReader(fh):
            t = int(float(r["tick"]))
            if t >= days * 48:
                break
            if float(r["UtmX"]) > 0:
                w = (t // 48) // 15 * 15
                dshare[w][0] += r["DispersalMode"] != "0"
                dshare[w][1] += 1

kw = {"t_disp": 4} if "tdisp4" in mode else {}
params = SimulationParameters(porpoise_count=2000, sim_years=1, random_seed=42, landscape="Kattegat",
                              ships_enabled=False, turbines="off", communication_enabled=False, **kw)
cd = CellData("Kattegat", data_dir)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
for tick in range(1, days * 48 + 1):
    sim.step()
    if tick % (15 * 48) == 0:
        d = tick // 48
        a = pm.active_mask
        ref = dep.get(d)
        ds = dshare.get(d - 15)
        ref_s = (f" | DEPONS N={ref[0]:4d} E={ref[1]:5.2f}" if ref else "") + (
            f" disp={ds[0] / ds[1]:.3f}" if ds else "")
        print(f"[{mode}] day {d:3d}: N={int(a.sum()):4d} E={pm.energy[a].mean():5.2f} "
              f"disp={pm.is_dispersing[a].mean():.3f}{ref_s}")
