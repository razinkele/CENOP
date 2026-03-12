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
