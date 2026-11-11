"""Hash of Kattegat initial positions/headings (seed 42, N=2000) plus a 2-day N/E check.

Same method as the 2026-11-11 placement check (expected before any Lithuania change:
init hash 9ab880c9644d8fc4, after 2 days N 2000 meanE 9.734385).

Run from the repo root: python3 scripts/investigations/2026-11-11-lithuania-data-gap/katt_init_hash.py
"""

import hashlib
from pathlib import Path

import numpy as np

import cenop
from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

DATA = Path(__file__).resolve().parents[3] / "data"

print("cenop from", cenop.__file__)
params = SimulationParameters(
    porpoise_count=2000,
    sim_years=1,
    random_seed=42,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
)
cd = CellData("Kattegat", str(DATA))
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
blob = np.concatenate([pm.x, pm.y, pm.heading]).astype(np.float32).tobytes()
print("init hash", hashlib.sha256(blob).hexdigest()[:16])
for _ in range(96):
    sim.step()
a = pm.active_mask
print("after 2 days: N", int(a.sum()), "meanE", round(float(pm.energy[a].mean()), 6))
