"""Bit-for-bit fingerprint of a run with ships, and its time per tick.

Hashes the ship deterrence vectors of every tick and the final animal state, so a change to
the ship loop can be shown to leave results unchanged.

python3 scripts/investigations/2026-11-20-ship-loop/ship_hash.py [days] [first_day]
"""

import hashlib
import sys
import time

import numpy as np

from cenop.agents.ship import ShipManager
from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

days = int(sys.argv[1]) if len(sys.argv) > 1 else 3
cd = CellData("Kattegat", "data")
cd.load()
params = SimulationParameters(
    porpoise_count=2000, sim_years=1, random_seed=42, landscape="Kattegat",
    ships_enabled=True, turbines="off", dispersal="InnerDanishWaters",
)  # fmt: skip
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
h = hashlib.sha256()
spent = [0.0, 0]
orig = ShipManager.calculate_aggregate_deterrence_vectorized


def spy(self, *args, **kwargs):
    t0 = time.perf_counter()
    dx, dy = orig(self, *args, **kwargs)
    spent[0] += time.perf_counter() - t0
    spent[1] += 1
    h.update(np.ascontiguousarray(dx, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(dy, dtype=np.float64).tobytes())
    return dx, dy


ShipManager.calculate_aggregate_deterrence_vectorized = spy
t0 = time.perf_counter()
for _ in range(days * 48):
    sim.step()
wall = time.perf_counter() - t0
pm = sim.population_manager
for arr in (pm.x, pm.y, pm.heading, pm.energy, pm.active_mask, pm.deter_strength):
    h.update(np.ascontiguousarray(arr).tobytes())
ticks = days * 48
print(
    f"ship hash {h.hexdigest()[:16]} | {days} days | tick {1000 * wall / ticks:.1f} ms, "
    f"of which ship deterrence {1000 * spent[0] / max(spent[1], 1):.1f} ms | alive {int(pm.active_mask.sum())}"
)
