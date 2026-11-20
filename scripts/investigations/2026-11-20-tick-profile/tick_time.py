"""Wall time per tick: python3 scripts/investigations/2026-11-20-tick-profile/tick_time.py <n> <dispersal> <warm_days> <ticks>."""

import sys
import time

import cenop
from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

n, mode, warm, ticks = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cd = CellData("Kattegat", "data")
cd.load()
params = SimulationParameters(
    porpoise_count=n,
    sim_years=1,
    random_seed=42,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
    dispersal=mode,
)
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
for _ in range(warm * 48):
    sim.step()
best = 1e9
for _ in range(3):
    t0 = time.perf_counter()
    for _ in range(ticks):
        sim.step()
    best = min(best, (time.perf_counter() - t0) / ticks)
pm = sim.population_manager
print(
    f"{'before' if 'before' in cenop.__file__ else 'after '} N={n} {mode}: best of 3 {1000 * best:.2f} ms/tick, alive {int(pm.active_mask.sum())}, dispersing {int(pm.is_dispersing.sum())}"
)
