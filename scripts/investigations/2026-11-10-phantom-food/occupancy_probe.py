"""Phase-3 minimal hypothesis test (no source edits).

H: eat_food_kernel_v2 floors EVERY visited cell to min_food, so empty non-patch cells acquire
0.01 'phantom' food that porpoises then eat; DEPONS CellData.eatFood only floors cells that had
food (> 0). Test: count non-patch cells with food > 0, and compare the 63-day energy trajectory
with the stock kernel vs a kernel that floors only cells with food > 0 before eating.

Usage: hypothesis_test.py <data_dir> <stock|patched> [days]
"""

import sys

import numpy as np
from numba import njit

sys.path.insert(0, "/home/razinka/cenjas/CENOP/src")
import cenop.optimizations.kernels as K  # noqa: E402
from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402


@njit(cache=False)
def eat_food_kernel_v2_depons_floor(food_grid, x_indices, y_indices, energy, food_eaten, min_food,
                                    demand_grid):
    n = x_indices.shape[0]
    agent_demand = np.empty(n, dtype=np.float32)
    demand_grid[:, :] = 0.0
    for i in range(n):
        frac = min(max((20.0 - energy[i]) / 10.0, 0.0), 0.99)
        row = y_indices[i]
        col = x_indices[i]
        demand = food_grid[row, col] * frac
        agent_demand[i] = demand
        demand_grid[row, col] += demand
    for i in range(n):
        row = y_indices[i]
        col = x_indices[i]
        total_demand = demand_grid[row, col]
        if total_demand <= 0.0:
            food_eaten[i] = 0.0
            continue
        available = max(food_grid[row, col], 0.0)
        if total_demand <= available:
            food_eaten[i] = agent_demand[i]
        else:
            food_eaten[i] = available * (agent_demand[i] / total_demand)
    for i in range(n):
        row = y_indices[i]
        col = x_indices[i]
        if food_eaten[i] > 0.0:  # DEPONS: only a cell that had food is eaten from and floored
            food_grid[row, col] -= food_eaten[i]
            if food_grid[row, col] < min_food:
                food_grid[row, col] = min_food


data_dir, mode = sys.argv[1], sys.argv[2]
days = int(sys.argv[3]) if len(sys.argv) > 3 else 63
if mode == "patched":
    K.eat_food_kernel_v2 = eat_food_kernel_v2_depons_floor
params = SimulationParameters(porpoise_count=2000, sim_years=1, random_seed=42, landscape="Kattegat",
                              ships_enabled=False, turbines="off")
cd = CellData("Kattegat", data_dir)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
patch = cd._food_prob > 0
acc = dict(on=0, pt=0, disp=0, food_on=0.0, e0=None)
prev_f = float(cd._food_value[patch].sum())
eaten = 0.0
for tick in range(1, days * 48 + 1):
    a = pm.active_mask
    if acc["e0"] is None:
        acc["e0"] = float(pm.energy[a].mean())
    sim.step()
    a = pm.active_mask
    on = patch[pm._pre_cell_yi[a], pm._pre_cell_xi[a]]
    acc["on"] += int(on.sum()); acc["pt"] += int(a.sum()); acc["disp"] += int(pm.is_dispersing[a].sum())
    acc["food_on"] += float(cd._food_value[pm._pre_cell_yi[a][on], pm._pre_cell_xi[a][on]].sum())
    f = float(cd._food_value[patch].sum())
    if tick % 48 != 0 and f <= prev_f:
        eaten += prev_f - f
    prev_f = f
    if tick % (15 * 48) == 0 or tick == 48:
        print(f"[{mode}] day {tick/48:5.1f}: N={int(a.sum()):4d} meanE={pm.energy[a].mean():6.2f} "
              f"on-patch={acc['on']/acc['pt']:.4f} dispersing={acc['disp']/acc['pt']:.4f} "
              f"food-left-on-occupied-patch/porp-tick={acc['food_on']/max(acc['on'],1):.4f} "
              f"patch-intake/porp-tick={eaten/acc['pt']:.5f}")
        acc.update(on=0, pt=0, disp=0, food_on=0.0); eaten = 0.0
