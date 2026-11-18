"""Tick-level dispersal statistics for CENOP InnerDanishWaters on Kattegat (seed arg, 120 days).

The daily snapshots of run_cenop.py are taken right after the daily trigger, so they
overstate time spent dispersing. This counts every animal-tick instead, to compare with the
one animal DEPONS-Java tracks in RandomPorpoise.csv (which died around day 120 in both
reference seeds).

python3 scripts/investigations/2026-11-18-dispersal-idw/episodes.py <seed>
"""

import sys
from pathlib import Path

import numpy as np

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

ROOT = Path(__file__).resolve().parents[3]
seed = int(sys.argv[1])
DAYS = 120
params = SimulationParameters(
    porpoise_count=2000,
    sim_years=1,
    random_seed=seed,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
    dispersal="InnerDanishWaters",
)
cd = CellData("Kattegat", str(ROOT / "data"))
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=seed)
sim.initialize()
pm = sim.population_manager
ticks = p1 = p2 = starts = 0
lengths = []
began_at = np.full(pm.count, -1)
prev = np.zeros(pm.count, dtype=bool)
for t in range(1, DAYS * 48 + 1):
    sim.step()
    a = pm.active_mask
    ph = pm.dispersal_phase
    ticks += int(a.sum())
    p1 += int((ph[a] == 1).sum())
    p2 += int((ph[a] == 2).sum())
    cur = (ph > 0) & a
    began = cur & ~prev
    began_at[began] = t
    starts += int(began.sum())
    lengths.extend((t - began_at[prev & ~cur]).tolist())
    prev = cur.copy()
lengths = np.array(lengths)
print(
    f"seed {seed}: phase 1 share {p1 / ticks:.4f}, phase 2 share {p2 / ticks:.4f}, "
    f"starts per animal-day {starts / (ticks / 48):.3f}, "
    f"episode length mean {lengths.mean():.1f} median {np.median(lengths):.0f} ticks"
)
