"""D15 step 1: where does the dispersal-off residual sit in CENOP's energy budget?

Per 15-day bin, with dispersal off (as DEPONS `dispersal=off`), communication off:
- N at the bin start; mean energy at the bin start;
- intake per animal-tick inferred from landscape food drops within a day (same method as
  depons_budget.py), and the exact food CENOP fed to `_update_psm` (checks the inference);
- deaths (starvation / other) and births in the bin;
- lactating share (mean over the bin).

Usage: cenop_budget.py <data_dir with Kattegat/> [seed] [days]
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import cenop.agents.population as P  # noqa: E402
from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

data_dir = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
days = int(sys.argv[3]) if len(sys.argv) > 3 else 180
BIN = 15

P.PorpoisePopulation._check_dispersal_trigger = lambda self, mask: None  # dispersal off

EXACT = [0.0]
_orig_psm = P.PorpoisePopulation._update_psm


def _psm(self, mask, food_gained):
    if food_gained is not None:
        EXACT[0] += float(np.nansum(np.asarray(food_gained)[mask]))
    return _orig_psm(self, mask, food_gained)


P.PorpoisePopulation._update_psm = _psm

params = SimulationParameters(
    porpoise_count=2000,
    sim_years=1,
    random_seed=seed,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
    communication_enabled=False,
)
cd = CellData("Kattegat", data_dir)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=seed)
sim.initialize()
pm = sim.population_manager

print(f"CENOP dispersal off, seed {seed}")
print(
    "bin(days)   N0    E0   intake/at(inferred)  intake/at(exact)  deaths(starv/other)  births  lact  coast<1km  depth<1m"
)
food_prev = float(cd._food_value.sum())
for b0 in range(0, days, BIN):
    a = pm.active_mask
    n0, e0 = int(a.sum()), float(pm.energy[a].mean())
    eaten_inf = 0.0
    at_eat = 0
    at_all = 0
    EXACT[0] = 0.0
    starv = other = births = 0
    lact = 0.0
    coast1 = shallow = 0.0
    for _ in range(BIN * 48):
        n_before = int(pm.active_mask.sum())
        sim.step()
        food_now = float(cd._food_value.sum())
        if food_now <= food_prev:
            eaten_inf += food_prev - food_now
            at_eat += n_before
        food_prev = food_now
        at_all += n_before
        starv += pm.last_step_deaths_starvation
        other += pm.last_step_deaths - pm.last_step_deaths_starvation
        births += pm.last_step_births
        a = pm.active_mask
        lact += float(pm.with_calf[a].mean()) if a.any() else 0.0
        if a.any():
            xi = np.clip(pm.x[a].astype(np.int64), 0, cd._depth.shape[1] - 1)
            yi = np.clip(pm.y[a].astype(np.int64), 0, cd._depth.shape[0] - 1)
            coast1 += float(np.mean(cd._dist_to_coast[yi, xi] < 1000.0))
            shallow += float(np.mean(cd._depth[yi, xi] < 1.0))
    print(
        f"{b0:3d}-{b0 + BIN:3d}  {n0:5d} {e0:5.2f}   {eaten_inf / max(at_eat, 1):.5f}"
        f"           {EXACT[0] / max(at_all, 1):.5f}         {starv:4d}/{other:<4d}"
        f"        {births:4d}  {lact / (BIN * 48):.3f}  {coast1 / (BIN * 48):.4f}     {shallow / (BIN * 48):.4f}",
        flush=True,
    )
