"""Data behind the Population tab's two time-series charts.

"Landscape Energy Level" plotted food eaten per porpoise and "Average Porpoise Movement"
plotted the number of dispersing porpoises. They now get total food on the landscape and the
mean straight-line distance moved per day.
"""

import numpy as np
import pytest

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters
from cenop.server.simulation_controller import SimulationRunner


def test_landscape_total_food_sums_the_food_on_all_cells():
    land = create_homogeneous_landscape(width=40, height=30, food_prob=0.5)
    assert land.total_food() == float(np.sum(land._food_value))
    assert land.total_food() > 0


def test_runner_records_landscape_food_and_daily_movement():
    params = SimulationParameters(porpoise_count=50, sim_years=1, random_seed=4)
    land = create_homogeneous_landscape()
    sim = Simulation(params, cell_data=land)
    runner = SimulationRunner(sim)
    runner.step_ticks()  # day 1 (the first update has no earlier positions)
    pm = sim.population_manager
    x0, y0, alive = pm.x.copy(), pm.y.copy(), pm.active_mask.copy()
    runner.step_ticks()  # day 2
    entry = runner.energy_history[-1]
    assert entry["landscape_food"] == land.total_food()
    both = alive & pm.active_mask
    expected_km = float(np.mean(np.hypot(pm.x[both] - x0[both], pm.y[both] - y0[both])) * 0.4)
    assert entry["movement_km_per_day"] == pytest.approx(expected_km, rel=1e-6)
    assert 0.0 < entry["movement_km_per_day"] < 100.0
