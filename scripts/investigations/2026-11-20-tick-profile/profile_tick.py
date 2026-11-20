"""Profile the CENOP tick on Kattegat: python3 scripts/investigations/2026-11-20-tick-profile/profile_tick.py <n> <dispersal> <ticks> [ships]."""

import cProfile
import pstats
import sys
import time

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

n, mode, ticks = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ships = len(sys.argv) > 4
cd = CellData("Kattegat", "data")
cd.load()
params = SimulationParameters(
    porpoise_count=n,
    sim_years=1,
    random_seed=42,
    landscape="Kattegat",
    ships_enabled=ships,
    turbines="off",
    dispersal=mode,
)
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
for _ in range(96):
    sim.step()
t0 = time.perf_counter()
for _ in range(ticks):
    sim.step()
wall = time.perf_counter() - t0
print(
    f"N={n} {mode} ships={ships}: {1000 * wall / ticks:.2f} ms/tick over {ticks} ticks, alive {int(sim.population_manager.active_mask.sum())}"
)
pr = cProfile.Profile()
pr.enable()
for _ in range(ticks):
    sim.step()
pr.disable()
st = pstats.Stats(pr)
st.sort_stats("tottime").print_stats(22)
st.sort_stats("cumulative").print_stats(28)
