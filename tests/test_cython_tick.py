"""Tests for Cython tick acceleration module."""

import numpy as np
import pytest
import sys
import os

def _try_import_cython():
    try:
        from cenop.optimizations.tick_cython import cython_available
        return True
    except ImportError:
        return False

CYTHON_OK = _try_import_cython()


@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
class TestCythonAvailable:
    def test_module_loads(self):
        from cenop.optimizations.tick_cython import cython_available
        assert cython_available() is True


@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
class TestCythonFullPostCRW:
    def test_deterministic_output(self):
        """Two runs with same seed produce identical results."""
        from cenop.optimizations.tick_cython import cython_depons_post_crw

        def run_once(seed):
            np.random.seed(seed)
            n = 200
            x = np.random.uniform(5, 195, n).astype(np.float32)
            y = np.random.uniform(5, 195, n).astype(np.float32)
            heading = np.random.uniform(0, 360, n).astype(np.float32)
            prev_angle = np.random.normal(0, 10, n).astype(np.float64)
            prev_log_mov = np.random.uniform(0.5, 1.5, n).astype(np.float64)
            energy = np.random.uniform(5, 15, n).astype(np.float32)
            active = np.ones(n, dtype=np.uint8)
            is_disp = np.zeros(n, dtype=np.uint8)
            with_calf = (np.random.random(n) > 0.8).astype(np.uint8)
            pres_angle = np.random.normal(0, 20, n).astype(np.float64)
            log_mov = np.random.uniform(0.5, 1.5, n).astype(np.float64)
            ve_total = np.random.uniform(0, 1, n).astype(np.float32)
            vt_x = np.random.normal(0, 0.1, n).astype(np.float32)
            vt_y = np.random.normal(0, 0.1, n).astype(np.float32)
            food = np.full((200, 200), 50.0, dtype=np.float32)
            out_food = np.zeros(n, dtype=np.float32)
            disp_dist = np.zeros(n, dtype=np.float32)
            depth = np.full((200, 200), 20.0, dtype=np.float64)
            rand_mort = np.random.random(n)

            deaths = cython_depons_post_crw(
                x, y, heading, prev_angle, prev_log_mov, energy,
                active, is_disp, with_calf,
                pres_angle, log_mov, ve_total, vt_x, vt_y,
                food, depth, out_food, disp_dist, rand_mort,
                0.001, 4.0, 4.5, 1.4, 1.0, 0.4, 1.0, 200, 200,
            )
            return (
                x.copy(), y.copy(), heading.copy(),
                energy.copy(), active.copy(), out_food.copy(),
            )

        r1 = run_once(42)
        r2 = run_once(42)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_food_gained_output_populated(self):
        """out_food_gained must be non-zero for agents that ate."""
        from cenop.optimizations.tick_cython import cython_depons_post_crw

        n = 100
        energy = np.full(n, 10.0, dtype=np.float32)  # Hungry (< 20)
        active = np.ones(n, dtype=np.uint8)
        food = np.full((200, 200), 50.0, dtype=np.float32)
        out_food = np.zeros(n, dtype=np.float32)
        disp_dist = np.zeros(n, dtype=np.float32)
        depth = np.full((200, 200), 20.0, dtype=np.float64)
        rand_mort = np.zeros(n, dtype=np.float64)

        cython_depons_post_crw(
            np.random.uniform(5, 195, n).astype(np.float32),
            np.random.uniform(5, 195, n).astype(np.float32),
            np.random.uniform(0, 360, n).astype(np.float32),
            np.zeros(n, dtype=np.float64),
            np.full(n, 1.0, dtype=np.float64),
            energy, active,
            np.zeros(n, dtype=np.uint8),
            np.zeros(n, dtype=np.uint8),
            np.zeros(n, dtype=np.float64),
            np.full(n, 1.0, dtype=np.float64),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            food, depth, out_food, disp_dist, rand_mort,
            0.001, 4.0, 4.5, 1.4, 1.0, 0.4, 1.0, 200, 200,
        )
        assert out_food.sum() > 0, "Agents should have eaten food"

    def test_mortality_kills_starving(self):
        """Agents with near-zero energy and extreme mortality params die."""
        from cenop.optimizations.tick_cython import cython_depons_post_crw

        n = 500
        energy = np.full(n, 0.0, dtype=np.float32)
        active = np.ones(n, dtype=np.uint8)
        food = np.full((200, 200), 0.01, dtype=np.float32)
        out_food = np.zeros(n, dtype=np.float32)
        disp_dist = np.zeros(n, dtype=np.float32)
        depth = np.full((200, 200), 20.0, dtype=np.float64)
        rand_mort = np.full(n, 0.5, dtype=np.float64)  # all > 0 -> every starving agent dies

        # m_mort_prob_const=100 makes yearly_surv negative -> step_surv=0 -> certain death
        deaths = cython_depons_post_crw(
            np.random.uniform(5, 195, n).astype(np.float32),
            np.random.uniform(5, 195, n).astype(np.float32),
            np.random.uniform(0, 360, n).astype(np.float32),
            np.zeros(n, dtype=np.float64),
            np.full(n, 1.0, dtype=np.float64),
            energy, active,
            np.zeros(n, dtype=np.uint8),
            np.zeros(n, dtype=np.uint8),
            np.zeros(n, dtype=np.float64),
            np.full(n, 1.0, dtype=np.float64),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            food, depth, out_food, disp_dist, rand_mort,
            0.001, 4.0, 4.5, 1.4, 100.0, 0.4, 1.0, 200, 200,
        )
        assert deaths == n, "All agents should die with extreme mortality"
        assert np.sum(active) == 0
