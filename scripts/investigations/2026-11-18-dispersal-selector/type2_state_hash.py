"""PSM-Type2 reference state hash: Kattegat, seed 42, N=2000, 30 days, default dispersal.

Gate for the dispersal-selector work (docs/superpowers/specs/2026-11-18-dispersal-selector-design.md):
wiring params.dispersal must leave PSM-Type2 runs bit-identical.

Expected since the females-only start (668b5bf, 2026-11-18; unchanged by 596eec6, 1a0efbc and 2247de4), by mode:
PSM-Type2          state hash 205f0e25647e1f7b | ever dispersed 786 | N 1989 | meanE 6.074059
off                state hash efac6392b01d3029 | ever dispersed 0 | N 1986 | meanE 6.133584
Undirected         state hash 2714d8fadec11a10 | ever dispersed 793 | N 1994 | meanE 6.119591
InnerDanishWaters  state hash 31ff4e44948ba01c | ever dispersed 1986 | N 1989 | meanE 5.963736
                   (97bb85e7b247346a until the edge-heading fix of 2026-11-19)

Before that fix (15dc4ec .. 377a84a) the first three were aec15387d1425dd9,
7f2de66bb7ef401b and ba0c5558af305b87; the dispersal-selector work was gated on those.

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
