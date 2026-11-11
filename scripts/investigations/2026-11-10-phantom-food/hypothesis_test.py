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
nonpatch = ~(cd._food_prob > 0)
print(f"[{mode}] t0 non-patch cells with food>0: {int((cd._food_value[nonpatch] > 0).sum())}")
for tick in range(1, days * 48 + 1):
    sim.step()
    if tick in (1, 48, 480) or tick % (15 * 48) == 0:
        a = pm.active_mask
        npf = cd._food_value[nonpatch]
        print(f"[{mode}] day {tick/48:5.1f}: N={int(a.sum()):4d} meanE={pm.energy[a].mean():6.2f}  "
              f"non-patch cells w/ food={int((npf > 0).sum()):6d} (sum {npf.sum():8.2f})  "
              f"patch food={cd._food_value[~nonpatch].sum():8.2f}")
