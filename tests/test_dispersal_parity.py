"""DEPONS PSM-Type2 dispersal parity tests."""

import numpy as np
import pytest

import cenop.agents.population as popmod
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.movement import DEPONSCRWMovementVectorized
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


class TestDisperseStep:
    """DEPONS AbstractPSMDispersal.disperse() order inside Porpoise.move() (D9-D13)."""

    T = 500.0

    def _land_pop(self, path=None):
        # psm_angle=0 makes the PSM heading equal _prev_step_heading
        land = create_homogeneous_landscape(200, 200)
        params = SimulationParameters(random_seed=1, communication_enabled=False, psm_angle=0.0)
        module = DEPONSCRWMovementVectorized(params) if path == "module" else None
        pop = PorpoisePopulation(4, params, land, movement_module=module)
        # Look-ahead point (105.5, 100.5) -> cell [100, 105]
        _disperser(pop, 0, 100.5, 100.5, 90.0, target_dist=self.T)
        return pop, land

    def test_stops_before_shallow_water(self):
        pop, land = self._land_pop()
        pop.dispersal_distance_traveled[0] = 10.0
        pop.days_declining_energy[0] = 3
        land._depth[100, 105] = 3.0
        # Agent 1: PSM heading 90 but current heading 10, so a stop must restore 10
        _disperser(pop, 1, 100.5, 50.5, 90.0, target_dist=self.T)
        pop.heading[1] = 10.0
        land._depth[50, 105] = np.nan
        pop._disperse_step(pop.active_mask)
        assert not pop.is_dispersing[0]
        assert pop.heading[0] == 90.0
        assert pop.dispersal_distance_traveled[0] == 0.0
        assert pop.days_declining_energy[0] == 0
        assert not pop.is_dispersing[1]
        assert pop.heading[1] == 10.0

        pop, land = self._land_pop()
        assert land._depth[100, 105] == 30.0
        pop._disperse_step(pop.active_mask)
        assert pop.is_dispersing[0]
        assert pop.heading[0] == 90.0

    def test_stops_before_leaving_grid(self):
        pop = _pop(psm_angle=0.0)
        _disperser(pop, 0, 998.5, 500.5, 90.0)
        _disperser(pop, 1, 500.5, 500.5, 90.0)
        pop._disperse_step(pop.active_mask)
        assert not pop.is_dispersing[0]
        assert pop.is_dispersing[1]

        # With a landscape, an off-grid look-ahead point stops before any depth lookup
        pop, _ = self._land_pop()
        _disperser(pop, 0, 100.5, 198.5, 0.0)
        pop._disperse_step(pop.active_mask)
        assert not pop.is_dispersing[0]

    def test_completes_on_path_length(self):
        pop = _pop(psm_angle=0.0)
        _disperser(pop, 0, 100.0, 100.0, 90.0, target_dist=self.T, traveled=0.95 * self.T)
        pop.heading[0] = 10.0
        _disperser(pop, 1, 100.0, 100.0, 90.0, target_dist=self.T, traveled=0.0)
        pop.x[1] = 100.0 + 0.95 * self.T
        pop._disperse_step(pop.active_mask)
        assert not pop.is_dispersing[0]
        assert pop.heading[0] == 10.0  # completed before the PSM turn
        assert pop.is_dispersing[1]
        pop._update_dispersal(pop.active_mask)
        assert pop.is_dispersing[1]

    def test_zero_target_distance_completes_without_nan(self):
        pop = _pop()
        _disperser(pop, 0, 500.5, 500.5, 90.0, target_dist=0.0)
        pop._disperse_step(pop.active_mask)
        assert not pop.is_dispersing[0]
        assert np.isfinite(pop.heading).all()

    def test_inactive_slots_ignored(self):
        pop = _pop(psm_angle=0.0)
        _disperser(pop, 0, 500.5, 500.5, 90.0)
        # Slot 1 would complete and slot 2 would turn to 90 if they were processed
        _disperser(pop, 1, 500.5, 300.5, 90.0, target_dist=0.0)
        _disperser(pop, 2, 500.5, 700.5, 90.0)
        pop.heading[2] = 10.0
        pop.active_mask[1:3] = False
        pop._disperse_step(pop.active_mask)
        assert pop.is_dispersing[1]
        assert pop.is_dispersing[2]
        assert pop.heading[2] == 10.0
        assert pop.is_dispersing[0]

    def _movement_pop(self, monkeypatch, path):
        if path == "numba" and not popmod._HAS_KERNELS:
            pytest.skip("Numba kernels unavailable")
        monkeypatch.setattr(popmod, "_HAS_KERNELS", path == "numba")
        pop, land = self._land_pop(path)
        land._depth[100, 105] = 3.0  # agent 0 stops
        _disperser(pop, 1, 100.5, 50.5, 90.0, target_dist=self.T)  # agent 1 continues
        pop._recompute_cell_indices()
        return pop

    @pytest.mark.parametrize("path", ["numba", "numpy", "module"])
    def test_stopped_disperser_takes_crw_step_same_tick(self, monkeypatch, path):
        pop = self._movement_pop(monkeypatch, path)
        pop._update_movement(pop.active_mask, None, None)
        assert not pop.is_dispersing[0]
        assert pop._step_dist[0] != pytest.approx(5.0)
        assert pop.dispersal_distance_traveled[0] == 0.0
        assert pop.is_dispersing[1]
        assert pop._step_dist[1] == pytest.approx(5.0)
        assert pop.dispersal_distance_traveled[1] == pytest.approx(5.0)

    @pytest.mark.parametrize("path", ["numba", "numpy", "module"])
    def test_crw_state_frozen_while_dispersing(self, monkeypatch, path):
        pop = self._movement_pop(monkeypatch, path)
        pop.prev_angle[:] = 17.0
        pop.prev_log_mov[:] = 0.5
        pop._update_movement(pop.active_mask, None, None)
        assert pop.prev_angle[1] == 17.0
        assert pop.prev_log_mov[1] == 0.5
        # The stopped disperser took a CRW step, so its CRW state moved on
        assert pop.prev_angle[0] != 17.0
        assert pop.prev_log_mov[0] != 0.5
