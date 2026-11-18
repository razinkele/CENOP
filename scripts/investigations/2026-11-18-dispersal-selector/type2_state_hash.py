"""PSM-Type2 reference state hash: Kattegat, seed 42, N=2000, 30 days, default dispersal.

Gate for the dispersal-selector work (docs/superpowers/specs/2026-11-18-dispersal-selector-design.md):
wiring params.dispersal must leave PSM-Type2 runs bit-identical.

Expected (recorded at 15dc4ec, before any behaviour change):
state hash aec15387d1425dd9
ever dispersed 778 | N 1987 | meanE 6.051148

Run from the repo root:
python3 scripts/investigations/2026-11-18-dispersal-selector/type2_state_hash.py [mode]
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

import cenop
from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

DATA = Path(__file__).resolve().parents[3] / "data"
TICKS = 30 * 48

mode = sys.argv[1] if len(sys.argv) > 1 else "PSM-Type2"
print("cenop from", cenop.__file__, "| dispersal", mode)
params = SimulationParameters(
    porpoise_count=2000,
    sim_years=1,
    random_seed=42,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
    dispersal=mode,
)
cd = CellData("Kattegat", str(DATA))
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
ever = np.zeros(pm.count, dtype=bool)
for _ in range(TICKS):
    sim.step()
    ever[: len(pm.is_dispersing)] |= pm.is_dispersing
blob = np.concatenate([pm.x, pm.y, pm.heading, pm.energy, pm.is_dispersing]).astype(np.float32)
a = pm.active_mask
print("state hash", hashlib.sha256(blob.tobytes()).hexdigest()[:16])
print(
    "ever dispersed",
    int(ever.sum()),
    "| N",
    int(a.sum()),
    "| meanE",
    round(float(pm.energy[a].mean()), 6),
)
