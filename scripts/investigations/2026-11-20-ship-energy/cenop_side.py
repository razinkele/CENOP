"""CENOP side: all animals of one seed, counted every tick of the bins.

python3 scripts/investigations/2026-11-20-ship-energy/cenop_side.py <seed> <out.npz>
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as c

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

seed = int(sys.argv[1])
cd = CellData("Kattegat", "data")
cd.load()
params = SimulationParameters(
    porpoise_count=2000, sim_years=1, random_seed=seed, landscape="Kattegat",
    ships_enabled=True, turbines="off", dispersal="InnerDanishWaters",
)  # fmt: skip
sim = Simulation(params=params, cell_data=cd, seed=seed)
sim.initialize()
pm = sim.population_manager
acc = c.empty()
for t in range(1, c.BINS[-1][1] * 48 + 1):
    n0 = pm.x.size
    px, py = pm.x.copy(), pm.y.copy()
    was_alive = pm.active_mask.copy()
    sim.step()
    b = c.bin_of_day((t - 1) // 48 + 1)
    if b < 0:
        continue
    n = min(n0, pm.x.size)
    alive = pm.active_mask[:n] & was_alive[:n]
    step = np.hypot(pm.x[:n] - px[:n], pm.y[:n] - py[:n]).astype(np.float64)
    c.add(
        acc, b, alive, pm.deter_strength[:n].astype(np.float64),
        pm.dispersal_phase[:n] > 0, step, pm.energy[:n].astype(np.float64),
    )  # fmt: skip
c.table(f"CENOP seed {seed} (all animals)", acc)
np.savez(sys.argv[2], **acc)
