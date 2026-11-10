"""DEPONS PSM-Type2 dispersal parity tests."""

import numpy as np
import pytest

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


class TestTrigger:
    def _setup(self):
        pop = _pop()
        pop.x[0] = pop.y[0] = 500.0
        pop._psm_instances[0].preferred_distance = 20.0
        return pop

    @pytest.mark.parametrize(
        "days, expected",
        [
            ([4, 5, 6, 7], True),
            ([4, 5, 6, 6], False),
            ([4, 5, 5, 6], False),
        ],
    )
    def test_trigger_requires_three_strict_drops(self, days, expected):
        pop = self._setup()
        pop._energy_history[0, :] = 100.0
        pop._energy_history[0, :4] = days
        pop._check_dispersal_trigger(pop.active_mask)
        assert bool(pop.is_dispersing[0]) is expected

    @pytest.mark.parametrize("energy, expected", [(5.0004, 5.0), (5.0006, 5.001)])
    def test_daily_mean_rounded_to_3dp(self, energy, expected):
        pop = self._setup()
        pop.energy[0] = energy
        mask = pop.active_mask
        for _ in range(48):
            pop._global_tick += 1
            pop._update_energy_history(mask)
        assert pop._energy_history[0, 0] == np.float32(expected)


class TestPsmTurn:
    T = 500.0

    def _deltas(self, displacement, traveled):
        pop = _pop(count=2000)
        for i in range(2000):
            _disperser(pop, i, 100.0, 100.0, 90.0, target_dist=self.T, traveled=traveled)
            pop.x[i] = 100.0 + displacement
        prev_before = pop._prev_step_heading.copy()
        pop._apply_dispersal_heading(np.ones(2000, dtype=bool))
        return (pop.heading - prev_before + 180) % 360 - 180

    def test_turn_uses_psm_angle(self):
        delta = self._deltas(0.95 * self.T, 0.95 * self.T)
        assert 25 < np.abs(delta).max() <= 40 * 0.925

    def test_logistic_uses_path_length(self):
        delta = self._deltas(0.0, 0.95 * self.T)
        assert np.abs(delta).max() > 25
