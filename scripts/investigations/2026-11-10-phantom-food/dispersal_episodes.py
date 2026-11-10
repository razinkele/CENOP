"""Dispersal episodes, days 0-120: start rate (per animal-day) and duration (days).
DEPONS: 200 tracked animals (RandomPorpoise.csv, DispersalMode != 0). CENOP: food-fixed code,
communication off, 200 sampled animals, sampled at the end of each day.

Usage: dispersal_episodes.py <RandomPorpoise.csv> <cenop data_dir with Kattegat/>
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
DAYS = 120


def episodes(series):
    """series: {id: [bool per day]} -> (starts per animal-day, durations of completed episodes)."""
    starts, days, durs = 0, 0, []
    for s in series.values():
        run = 0
        for i, v in enumerate(s):
            days += 1
            if v and (i == 0 or not s[i - 1]):
                starts += 1
            if v:
                run += 1
            elif run:
                durs.append(run)
                run = 0
    return starts / max(days, 1), durs


dep = defaultdict(dict)
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        t = int(float(r["tick"]))
        if t >= DAYS * 48:
            break
        if t % 48 == 47 and float(r["UtmX"]) > 0:
            dep[r["Id"]][t // 48] = r["DispersalMode"] != "0"
dser = {k: [v[d] for d in sorted(v)] for k, v in dep.items()}
rate, durs = episodes(dser)
print(f"DEPONS: starts/animal-day {rate:.4f}, completed episodes {len(durs)}, "
      f"duration days median {np.median(durs) if durs else float('nan'):.1f} mean {np.mean(durs) if durs else float('nan'):.1f}")

from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

params = SimulationParameters(porpoise_count=2000, sim_years=1, random_seed=42, landscape="Kattegat",
                              ships_enabled=False, turbines="off", communication_enabled=False)
cd = CellData("Kattegat", sys.argv[2])
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
sample = np.arange(0, 2000, 10)
cser = defaultdict(list)
for tick in range(1, DAYS * 48 + 1):
    sim.step()
    if tick % 48 == 47:
        for i in sample:
            if pm.active_mask[i]:
                cser[i].append(bool(pm.is_dispersing[i]))
rate, durs = episodes(cser)
print(f"CENOP : starts/animal-day {rate:.4f}, completed episodes {len(durs)}, "
      f"duration days median {np.median(durs) if durs else float('nan'):.1f} mean {np.mean(durs) if durs else float('nan'):.1f}")
