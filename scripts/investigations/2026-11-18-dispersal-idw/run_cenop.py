"""One CENOP run of the kattegat-idw-undisturbed-v1 scenario, logged once per simulated day.

Scenario (validation/depons_java/scenarios/kattegat-idw-undisturbed-v1.json): Kattegat,
2000 animals, 5 years, ships off, turbines off. Writes
output/cenop_idw/<mode>/seed-<seed>.csv with one row per day:
day, n, mean_energy, phase1, phase2, b0..b59 (animals per 40 km block, index 0-59 from the
north-west; block index = blocks.asc value - 1, animals on a NODATA block cell are not counted).

Run from the repo root:
python3 scripts/investigations/2026-11-18-dispersal-idw/run_cenop.py <mode> <seed> [years] \
    [data_root tag]

``data_root`` is a directory holding a ``Kattegat`` landscape folder other than the repo's
(default: data/, the DEPONS v3.0 landscape); ``tag`` then names the output folder
output/cenop_idw/<tag>/<mode>/. Used to run CENOP on the DEPONS v3.2 landscape, whose
monthly files are named prey0000_MM_XX.asc / salinity0000_MM_XX.asc: link them under the
names CENOP reads (drop the _XX) in a scratch folder and pass that folder's parent.
"""

import sys
from pathlib import Path

import numpy as np

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

ROOT = Path(__file__).resolve().parents[3]
mode, seed = sys.argv[1], int(sys.argv[2])
years = int(sys.argv[3]) if len(sys.argv) > 3 else 5

params = SimulationParameters(
    porpoise_count=2000,
    sim_years=years,
    random_seed=seed,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
    dispersal=mode,
)
data_root = sys.argv[4] if len(sys.argv) > 4 else str(ROOT / "data")
tag = sys.argv[5] if len(sys.argv) > 5 else ""
cd = CellData("Kattegat", data_root)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=seed)
sim.initialize()
pm = sim.population_manager
blocks = cd._blocks

out = ROOT / "output" / "cenop_idw" / tag / mode
out.mkdir(parents=True, exist_ok=True)
rows = []


def record(day):
    a = pm.active_mask
    xi = np.clip(pm.x[a].astype(np.intp), 0, blocks.shape[1] - 1)
    yi = np.clip(pm.y[a].astype(np.intp), 0, blocks.shape[0] - 1)
    b = blocks[yi, xi]
    per_block = np.bincount(b[b >= 1] - 1, minlength=60)[:60]
    phase = pm.dispersal_phase[a]
    n = int(a.sum())
    rows.append(
        [day, n, float(pm.energy[a].mean()) if n else 0.0, int((phase == 1).sum())]
        + [int((phase == 2).sum())]
        + per_block.tolist()
    )


record(0)
for tick in range(1, years * 360 * 48 + 1):
    sim.step()
    if tick % 48 == 0:
        record(tick // 48)
header = "day,n,mean_energy,phase1,phase2," + ",".join(f"b{i}" for i in range(60))
np.savetxt(out / f"seed-{seed}.csv", np.array(rows), delimiter=",", header=header, comments="")
print(mode, seed, "final n", rows[-1][1])
