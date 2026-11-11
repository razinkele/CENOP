"""Population trajectory, CENOP (current code) with communication on/off vs DEPONS seed 42.

Usage: population_check.py <cenop data_dir with Kattegat/> <on|off> [days] [depons Statistics.csv]
"""

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

data_dir, comm = sys.argv[1], sys.argv[2] == "on"
nodisp = len(sys.argv) > 5 and sys.argv[5] == "nodisp"
if nodisp:  # controlled test: dispersal trigger disabled (DEPONS counterpart: dispersal=off)
    import cenop.agents.population as _P
    _P.PorpoisePopulation._check_dispersal_trigger = lambda self, mask: None
days = int(sys.argv[3]) if len(sys.argv) > 3 else 63
dep = {}
if len(sys.argv) > 4:
    with open(sys.argv[4]) as fh:
        for r in csv.DictReader(fh):
            t = int(float(r["tick"]))
            if t % 48 == 0:
                n = int(float(r["PorpoiseCount"]))
                dep[t // 48] = (n, float(r["PorpoiseEnergyLevel"]) / max(n, 1))
params = SimulationParameters(porpoise_count=2000, sim_years=1, random_seed=42, landscape="Kattegat",
                              ships_enabled=False, turbines="off", communication_enabled=comm)
cd = CellData("Kattegat", data_dir)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
for tick in range(1, days * 48 + 1):
    sim.step()
    if tick % (15 * 48) == 0 or tick == 48:
        d = tick // 48
        a = pm.active_mask
        ref = dep.get(d)
        ref_s = f" | DEPONS N={ref[0]:4d} E={ref[1]:5.2f}" if ref else ""
        print(f"[comm {'on' if comm else 'off'}] day {d:3d}: N={int(a.sum()):4d} E={pm.energy[a].mean():5.2f} disp={pm.is_dispersing[a].mean():.3f}{ref_s}")
