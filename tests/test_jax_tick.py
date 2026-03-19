"""Tests for JAX JIT tick implementation."""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")


# --- Realistic CRW parameters from DEPONS 3.2 defaults ---
CRW_PARAMS = dict(
    corr_angle_base=0.891,
    corr_angle_bathy=-2.460e-04,
    corr_angle_salinity=3.157e-03,
    corr_angle_base_sd=6.202e-01,
    corr_logmov_length=0.548,
    corr_logmov_bathy=7.047e-06,
    corr_logmov_salinity=-5.697e-04,
    max_mov=4.0,
    r2_mean=0.0,
    r2_sd=39.68,
    r1_mean=0.0,
    r1_sd=0.559,
)


def _make_inputs(n=500, seed=42):
    """Create random test inputs for jax_crw_kernel."""
    rng = np.random.default_rng(seed)
    return dict(
        prev_angle=jnp.array(rng.uniform(-90, 90, n), dtype=jnp.float64),
        prev_log_mov=jnp.array(rng.uniform(1.0, 3.5, n), dtype=jnp.float64),
        depths=jnp.array(rng.uniform(5.0, 100.0, n), dtype=jnp.float64),
        salinity=jnp.array(rng.uniform(10.0, 35.0, n), dtype=jnp.float64),
        mask=jnp.array(rng.choice([True, False], n, p=[0.9, 0.1])),
        key=jax.random.PRNGKey(seed),
    )


@pytest.fixture
def kernel():
    from cenop.optimizations.jax_kernels import jax_crw_kernel

    return jax_crw_kernel


class TestJaxCRWKernel:
    """Tests for jax_crw_kernel."""

    def test_angles_within_bounds(self, kernel):
        """All output angles must be in [-180, 180]."""
        inputs = _make_inputs(n=1000, seed=0)
        out_angle, out_log_mov = kernel(**inputs, **CRW_PARAMS)
        active = inputs["mask"]
        angles = np.asarray(out_angle[active])
        assert np.all(angles >= -180.0), f"Min angle: {angles.min()}"
        assert np.all(angles <= 180.0), f"Max angle: {angles.max()}"

    def test_step_length_bounded(self, kernel):
        """All output log_mov must be <= max_mov for active agents."""
        inputs = _make_inputs(n=1000, seed=1)
        out_angle, out_log_mov = kernel(**inputs, **CRW_PARAMS)
        active = inputs["mask"]
        log_movs = np.asarray(out_log_mov[active])
        assert np.all(log_movs <= CRW_PARAMS["max_mov"] + 1e-10), (
            f"Max log_mov: {log_movs.max()}"
        )

    def test_masked_agents_unchanged(self, kernel):
        """Inactive agents: angle=0, log_mov=prev_log_mov."""
        inputs = _make_inputs(n=200, seed=2)
        # Force some agents inactive
        mask = np.ones(200, dtype=bool)
        mask[10:20] = False
        inputs["mask"] = jnp.array(mask)

        out_angle, out_log_mov = kernel(**inputs, **CRW_PARAMS)
        inactive = ~mask

        np.testing.assert_array_equal(
            np.asarray(out_angle[inactive]),
            0.0,
            err_msg="Inactive agents should have angle=0",
        )
        np.testing.assert_array_almost_equal(
            np.asarray(out_log_mov[inactive]),
            np.asarray(inputs["prev_log_mov"][inactive]),
            err_msg="Inactive agents should keep prev_log_mov",
        )

    def test_distribution_comparable_to_numba(self, kernel):
        """Output distribution stats should be in reasonable range."""
        inputs = _make_inputs(n=2000, seed=3)
        # All active
        inputs["mask"] = jnp.ones(2000, dtype=bool)

        out_angle, out_log_mov = kernel(**inputs, **CRW_PARAMS)
        angles = np.asarray(out_angle)
        log_movs = np.asarray(out_log_mov)

        # Angles should be centered roughly around 0 with some spread
        assert abs(np.mean(angles)) < 30.0, f"Mean angle too far from 0: {np.mean(angles)}"
        assert 5.0 < np.std(angles) < 100.0, f"Angle std out of range: {np.std(angles)}"

        # Log step lengths should be positive and bounded
        assert 0.5 < np.mean(log_movs) < 3.5, f"Mean log_mov out of range: {np.mean(log_movs)}"
        assert np.std(log_movs) < 2.0, f"Log_mov std too large: {np.std(log_movs)}"

    def test_jit_compiles(self, kernel):
        """jax.jit(jax_crw_kernel) runs without error."""
        inputs = _make_inputs(n=50, seed=4)
        jitted = jax.jit(kernel)
        # First call triggers compilation
        out_angle, out_log_mov = jitted(**inputs, **CRW_PARAMS)
        # Second call uses cached compilation
        out_angle2, out_log_mov2 = jitted(**inputs, **CRW_PARAMS)
        # Same inputs + same key = same outputs (deterministic)
        np.testing.assert_array_equal(np.asarray(out_angle), np.asarray(out_angle2))
        np.testing.assert_array_equal(np.asarray(out_log_mov), np.asarray(out_log_mov2))

    def test_deterministic_with_same_key(self, kernel):
        """Same PRNG key produces identical results."""
        inputs = _make_inputs(n=100, seed=5)
        out1 = kernel(**inputs, **CRW_PARAMS)
        out2 = kernel(**inputs, **CRW_PARAMS)
        np.testing.assert_array_equal(np.asarray(out1[0]), np.asarray(out2[0]))
        np.testing.assert_array_equal(np.asarray(out1[1]), np.asarray(out2[1]))

    def test_different_key_different_results(self, kernel):
        """Different PRNG keys produce different results."""
        inputs = _make_inputs(n=100, seed=6)
        out1 = kernel(**inputs, **CRW_PARAMS)
        inputs["key"] = jax.random.PRNGKey(999)
        out2 = kernel(**inputs, **CRW_PARAMS)
        # Very unlikely to be identical with different keys
        assert not np.allclose(np.asarray(out1[0]), np.asarray(out2[0]))
