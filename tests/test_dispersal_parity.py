"""DEPONS PSM-Type2 dispersal parity tests."""

import numpy as np

from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters


def _pop(count=4, landscape=None, **kw):
    params = SimulationParameters(random_seed=1, communication_enabled=False, **kw)
    return PorpoisePopulation(count, params, landscape)


def _disperser(pop, i, x, y, prev_heading, target_dist=500.0, traveled=0.0):
    pop.x[i], pop.y[i] = x, y
    pop.is_dispersing[i] = True
    pop.dispersal_start_x[i], pop.dispersal_start_y[i] = x, y
    pop.dispersal_target_distance[i] = target_dist
    pop.dispersal_distance_traveled[i] = traveled
    pop._prev_step_heading[i] = prev_heading
    pop.heading[i] = prev_heading


class TestTargetSelection:
    def _setup(self, pref=20.0, landscape=None, x=100.5, y=100.5):
        pop = _pop(landscape=landscape)
        pop.x[0], pop.y[0] = x, y
        pop._psm_instances[0].preferred_distance = pref
        return pop

    def test_memory_target_needs_50_cells(self):
        pop = self._setup()
        pop.psm_buffer[0, :, :, :] = 0.0
        cells = [(r, c) for r in range(20, 30) for c in range(25, 35)]
        for r, c in cells[:49]:
            pop.psm_buffer[0, r, c] = (1.0, 2.0)
        assert pop._memory_dispersal_target(0) is None
        r, c = cells[49]
        pop.psm_buffer[0, r, c] = (1.0, 2.0)
        assert pop._memory_dispersal_target(0) is not None

    def test_memory_target_best_food_per_tick_in_band(self):
        pop = self._setup()
        pop.psm_buffer[0, :, :, :] = 0.0
        pop.psm_buffer[0, 0:6, 0:10] = (1.0, 10.0)
        pop.psm_buffer[0, 20, 30] = (1.0, 3.0)
        pop.psm_buffer[0, 30, 20] = (1.0, 2.0)
        assert pop._memory_dispersal_target(0) == (152.5, 102.5)

    def test_random_target_requires_water_and_prey(self):
        land = create_homogeneous_landscape(width=200, height=200)
        land._entropy[:] = 0
        pop = self._setup(landscape=land)
        assert pop._random_dispersal_target(0) is None
        land._entropy[:, :, 150:] = 0.5
        target = pop._random_dispersal_target(0)
        assert target is not None
        assert target[0] >= 150

    def test_random_target_without_landscape(self):
        pop = self._setup(x=500.0, y=500.0)
        tx, ty = pop._random_dispersal_target(0)
        d = np.hypot(tx - 500.0, ty - 500.0)
        assert 37.5 - 1e-6 <= d <= 62.5 + 1e-6
        pop._psm_instances[0].preferred_distance = 10000.0
        assert pop._random_dispersal_target(0) is None

    def test_failed_activation_leaves_agent_untouched(self):
        pop = self._setup(pref=10000.0)
        before = (
            pop.dispersal_target_distance[0],
            pop._prev_step_heading[0],
            pop.heading[0],
        )
        assert pop._start_dispersal(0) is False
        assert not pop.is_dispersing[0]
        after = (
            pop.dispersal_target_distance[0],
            pop._prev_step_heading[0],
            pop.heading[0],
        )
        assert before == after

    def test_activation_heads_for_target(self):
        pop = self._setup(x=500.0, y=500.0)
        heading_before = float(pop.heading[0])
        assert pop._start_dispersal(0) is True
        tx, ty = pop.dispersal_target_x[0], pop.dispersal_target_y[0]
        dx, dy = tx - pop.x[0], ty - pop.y[0]
        expected = np.degrees(np.arctan2(dx, dy)) % 360.0
        assert abs(pop._prev_step_heading[0] - expected) < 1e-4
        assert pop.heading[0] == heading_before
        assert abs(pop.dispersal_target_distance[0] - np.hypot(dx, dy)) < 1e-3
        assert pop.dispersal_distance_traveled[0] == 0.0

    def test_trigger_not_gated_on_memory(self):
        pop = self._setup(x=500.0, y=500.0)
        pop.psm_buffer[:] = 0.0
        pop._energy_history[0, :4] = [4, 5, 6, 7]
        pop._check_dispersal_trigger(pop.active_mask)
        assert pop.is_dispersing[0]
