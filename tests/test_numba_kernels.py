"""Tests for Numba-accelerated simulation kernels."""
import numpy as np
import pytest


class TestReflectBoundariesKernel:
    """Test reflect_boundaries_kernel matches pure-NumPy behavior."""

    def test_no_reflection_needed(self):
        from cenop.optimizations.kernels import reflect_boundaries_kernel
        new_x = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        new_y = np.array([5.0, 10.0, 15.0], dtype=np.float64)
        dx = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        dy = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        mask = np.array([True, True, True])
        reflect_boundaries_kernel(new_x, new_y, dx, dy, 20, 20, mask)
        np.testing.assert_array_almost_equal(new_x, [5.0, 10.0, 15.0])
        np.testing.assert_array_almost_equal(new_y, [5.0, 10.0, 15.0])

    def test_reflect_below_zero(self):
        from cenop.optimizations.kernels import reflect_boundaries_kernel
        new_x = np.array([-3.0], dtype=np.float64)
        new_y = np.array([-5.0], dtype=np.float64)
        dx = np.array([-1.0], dtype=np.float64)
        dy = np.array([-1.0], dtype=np.float64)
        mask = np.array([True])
        reflect_boundaries_kernel(new_x, new_y, dx, dy, 20, 20, mask)
        np.testing.assert_array_almost_equal(new_x, [3.0])
        np.testing.assert_array_almost_equal(new_y, [5.0])
        np.testing.assert_array_almost_equal(dx, [1.0])
        np.testing.assert_array_almost_equal(dy, [1.0])

    def test_reflect_above_max(self):
        from cenop.optimizations.kernels import reflect_boundaries_kernel
        new_x = np.array([22.0], dtype=np.float64)
        new_y = np.array([25.0], dtype=np.float64)
        dx = np.array([1.0], dtype=np.float64)
        dy = np.array([1.0], dtype=np.float64)
        mask = np.array([True])
        reflect_boundaries_kernel(new_x, new_y, dx, dy, 20, 20, mask)
        np.testing.assert_array_almost_equal(new_x, [16.0])
        np.testing.assert_array_almost_equal(new_y, [13.0])
        np.testing.assert_array_almost_equal(dx, [-1.0])
        np.testing.assert_array_almost_equal(dy, [-1.0])

    def test_mask_skips_inactive(self):
        from cenop.optimizations.kernels import reflect_boundaries_kernel
        new_x = np.array([-3.0, -3.0], dtype=np.float64)
        new_y = np.array([5.0, 5.0], dtype=np.float64)
        dx = np.array([-1.0, -1.0], dtype=np.float64)
        dy = np.array([1.0, 1.0], dtype=np.float64)
        mask = np.array([True, False])
        reflect_boundaries_kernel(new_x, new_y, dx, dy, 20, 20, mask)
        # Active agent: reflection flips to +3 and dx sign flips
        assert new_x[0] == pytest.approx(3.0)
        assert dx[0] == pytest.approx(1.0)
        # Inactive agent: no reflection, but safety clamp still applies to all
        # (matches NumPy np.clip which is unconditional); dx sign unchanged
        assert new_x[1] == pytest.approx(0.0)
        assert dx[1] == pytest.approx(-1.0)

    def test_equivalence_with_numpy_version(self):
        from cenop.optimizations.kernels import reflect_boundaries_kernel
        from cenop.agents.population import PorpoisePopulation as Population
        rng = np.random.default_rng(42)
        n = 200
        new_x_nb = rng.uniform(-5, 25, n).astype(np.float64)
        new_y_nb = rng.uniform(-5, 25, n).astype(np.float64)
        dx_nb = rng.uniform(-3, 3, n).astype(np.float64)
        dy_nb = rng.uniform(-3, 3, n).astype(np.float64)
        mask = rng.choice([True, False], n)
        new_x_np = new_x_nb.copy()
        new_y_np = new_y_nb.copy()
        dx_np = dx_nb.copy()
        dy_np = dy_nb.copy()
        reflect_boundaries_kernel(new_x_nb, new_y_nb, dx_nb, dy_nb, 20, 20, mask)
        Population._reflect_boundaries(new_x_np, new_y_np, dx_np, dy_np, 20, 20, mask)
        np.testing.assert_allclose(new_x_nb, new_x_np, atol=1e-10)
        np.testing.assert_allclose(new_y_nb, new_y_np, atol=1e-10)
        np.testing.assert_allclose(dx_nb, dx_np, atol=1e-10)
        np.testing.assert_allclose(dy_nb, dy_np, atol=1e-10)


class TestCRWAngleStepKernel:
    """Test CRW angle+step kernel with rejection sampling."""

    def test_basic_angle_and_step(self):
        """Normal case: angle within bounds, step within max_mov."""
        from cenop.optimizations.kernels import crw_angle_step_kernel

        n = 3
        prev_angle = np.zeros(n, dtype=np.float64)
        prev_log_mov = np.ones(n, dtype=np.float64) * 2.0
        depths = np.full(n, 30.0, dtype=np.float64)
        salinity = np.full(n, 30.0, dtype=np.float64)
        rand_angle = np.zeros(n, dtype=np.float64)  # No random perturbation
        rand_len = np.zeros(n, dtype=np.float64)
        mask = np.ones(n, dtype=np.bool_)
        out_pres_angle = np.zeros(n, dtype=np.float64)
        out_log_mov = np.zeros(n, dtype=np.float64)

        crw_angle_step_kernel(
            prev_angle, prev_log_mov, depths, salinity,
            rand_angle, rand_len, mask,
            out_pres_angle, out_log_mov,
            0.0, 0.0, 0.0, 1.0,  # angle params (b0=0, b1=0, b2=0, b3=1)
            0.5, 0.0, 0.0, 3.0,  # step params
            0.0, 4.0, 0.0, 1.0,  # random distribution params
        )

        # With zero random input and b0=0: pres_angle = 0 * 1.0 = 0
        np.testing.assert_array_almost_equal(out_pres_angle, [0.0, 0.0, 0.0])
        # log_mov = 0.5 * 2.0 + 0 = 1.0
        np.testing.assert_array_almost_equal(out_log_mov, [1.0, 1.0, 1.0])

    def test_angle_rejection_clamps(self):
        """Extreme angles should be clamped after rejection exhaustion."""
        from cenop.optimizations.kernels import crw_angle_step_kernel

        n = 1
        prev_angle = np.array([0.0], dtype=np.float64)
        prev_log_mov = np.ones(n, dtype=np.float64) * 2.0
        depths = np.full(n, 30.0, dtype=np.float64)
        salinity = np.full(n, 30.0, dtype=np.float64)
        # Very large initial random to guarantee > 180
        rand_angle = np.array([1000.0], dtype=np.float64)
        rand_len = np.zeros(n, dtype=np.float64)
        mask = np.ones(n, dtype=np.bool_)
        out_pres_angle = np.zeros(n, dtype=np.float64)
        out_log_mov = np.zeros(n, dtype=np.float64)

        np.random.seed(42)  # Seed so retries are deterministic
        crw_angle_step_kernel(
            prev_angle, prev_log_mov, depths, salinity,
            rand_angle, rand_len, mask,
            out_pres_angle, out_log_mov,
            0.0, 0.0, 0.0, 1.0,
            0.5, 0.0, 0.0, 3.0,
            0.0, 4.0, 0.0, 1.0,
        )

        # After rejection sampling, angle should be within valid range
        assert abs(out_pres_angle[0]) <= 180.0, \
            f"pres_angle should be <= 180 after rejection, got {out_pres_angle[0]}"

    def test_step_length_rejection(self):
        """Step exceeding max_mov should be resampled or clamped."""
        from cenop.optimizations.kernels import crw_angle_step_kernel

        n = 1
        prev_angle = np.zeros(n, dtype=np.float64)
        prev_log_mov = np.array([2.5], dtype=np.float64)
        depths = np.full(n, 30.0, dtype=np.float64)
        salinity = np.full(n, 30.0, dtype=np.float64)
        rand_angle = np.zeros(n, dtype=np.float64)
        rand_len = np.array([10.0], dtype=np.float64)  # Will push past max_mov=3.0
        mask = np.ones(n, dtype=np.bool_)
        out_pres_angle = np.zeros(n, dtype=np.float64)
        out_log_mov = np.zeros(n, dtype=np.float64)

        np.random.seed(42)
        crw_angle_step_kernel(
            prev_angle, prev_log_mov, depths, salinity,
            rand_angle, rand_len, mask,
            out_pres_angle, out_log_mov,
            0.0, 0.0, 0.0, 1.0,
            0.5, 0.0, 0.0, 3.0,
            0.0, 4.0, 0.0, 1.0,
        )

        assert out_log_mov[0] <= 3.0, \
            f"log_mov should be <= max_mov after rejection, got {out_log_mov[0]}"

    def test_mask_skips_inactive(self):
        """Masked-out agents should have zero angle and unchanged log_mov."""
        from cenop.optimizations.kernels import crw_angle_step_kernel

        n = 2
        prev_angle = np.array([10.0, 10.0], dtype=np.float64)
        prev_log_mov = np.array([2.0, 2.0], dtype=np.float64)
        depths = np.full(n, 30.0, dtype=np.float64)
        salinity = np.full(n, 30.0, dtype=np.float64)
        rand_angle = np.array([5.0, 5.0], dtype=np.float64)
        rand_len = np.zeros(n, dtype=np.float64)
        mask = np.array([True, False])
        out_pres_angle = np.zeros(n, dtype=np.float64)
        out_log_mov = np.zeros(n, dtype=np.float64)

        crw_angle_step_kernel(
            prev_angle, prev_log_mov, depths, salinity,
            rand_angle, rand_len, mask,
            out_pres_angle, out_log_mov,
            0.0, 0.0, 0.0, 1.0,
            0.5, 0.0, 0.0, 3.0,
            0.0, 4.0, 0.0, 1.0,
        )

        assert out_pres_angle[0] != 0.0  # Active agent computed
        assert out_pres_angle[1] == 0.0  # Masked agent zero
        assert out_log_mov[1] == 2.0     # Unchanged


class TestTurnPositionKernel:
    """Test turn_position_kernel computes correct positions after turning."""

    def test_basic_turn(self):
        """Turn by 90 degrees should rotate displacement vector."""
        from cenop.optimizations.kernels import turn_position_kernel

        n = 2
        x = np.array([10.0, 10.0], dtype=np.float64)
        y = np.array([10.0, 10.0], dtype=np.float64)
        heading = np.array([0.0, 0.0], dtype=np.float64)
        step_dist = np.array([4.0, 4.0], dtype=np.float64)

        out_x = np.zeros(n, dtype=np.float64)
        out_y = np.zeros(n, dtype=np.float64)
        out_heading = np.zeros(n, dtype=np.float64)

        turn_position_kernel(x, y, heading, step_dist, 90.0, 20, 20, out_x, out_y, out_heading)

        # heading = (0 + 90) % 360 = 90, rads = pi/2
        # dx = sin(pi/2) * 4 = 4, dy = cos(pi/2) * 4 ≈ 0
        np.testing.assert_array_almost_equal(out_heading, [90.0, 90.0])
        np.testing.assert_array_almost_equal(out_x, [14.0, 14.0], decimal=4)
        np.testing.assert_array_almost_equal(out_y, [10.0, 10.0], decimal=4)

    def test_boundary_reflection_during_turn(self):
        """Positions beyond boundary after turn should be reflected."""
        from cenop.optimizations.kernels import turn_position_kernel

        n = 1
        x = np.array([18.0], dtype=np.float64)
        y = np.array([10.0], dtype=np.float64)
        heading = np.array([0.0], dtype=np.float64)
        step_dist = np.array([4.0], dtype=np.float64)

        out_x = np.zeros(n, dtype=np.float64)
        out_y = np.zeros(n, dtype=np.float64)
        out_heading = np.zeros(n, dtype=np.float64)

        # Turn 90 degrees: dx = sin(pi/2)*4 = 4, new_x = 22 > 19 (max)
        turn_position_kernel(x, y, heading, step_dist, 90.0, 20, 20, out_x, out_y, out_heading)

        # Should be reflected: 2*19 - 22 = 16
        assert out_x[0] == pytest.approx(16.0, abs=0.1)
