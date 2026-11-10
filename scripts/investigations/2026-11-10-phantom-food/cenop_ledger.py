"""Energy ledger for CENOP's first ticks: change in total porpoise energy vs food removed from the
landscape minus expected BMR use. Usage: cenop_ledger.py <data_dir> [count]"""

import sys

import numpy as np

sys.path.insert(0, "/home/razinka/cenjas/CENOP/src")
from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

data_dir = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
params = SimulationParameters(porpoise_count=count, sim_years=1, random_seed=42, landscape="Kattegat",
                              ships_enabled=False, turbines="off")
cd = CellData("Kattegat", data_dir)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
print("energy module:", type(getattr(pm, "_energy_module", None)).__name__,
      "| energy_mode:", params.energy_mode, "| cython gate:",
      getattr(pm, "_skip_land_avoidance", None), getattr(pm, "_comm_enabled", None))

prev_e = float(pm.energy[pm.active_mask].sum())
prev_f = float(cd._food_value.sum())
print(" tick   N   dE_total   food_removed  N*0.0045   dE-(removed-N*0.0045)  meanE  maxE")
for tick in range(1, 49):
    sim.step()
    a = pm.active_mask
    e = float(pm.energy[a].sum())
    f = float(cd._food_value.sum())
    n = int(a.sum())
    removed = prev_f - f
    use = n * 0.0045
    if tick <= 6 or tick % 8 == 0 or tick >= 47:
        print(f" {tick:4d} {n:4d}  {e-prev_e:9.4f}  {removed:11.4f}  {use:8.4f}  {(e-prev_e)-(removed-use):12.4f}"
              f"  {e/n:6.3f} {pm.energy[a].max():6.2f}")
    prev_e, prev_f = e, f
