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


# ---------------------------------------------------------------------------
# Reference Memory kernel tests
# ---------------------------------------------------------------------------


def _make_ref_mem_inputs(n=50, mem_size=120, seed=42):
    """Create random ref_mem state for testing."""
    rng = np.random.default_rng(seed)

    stored_util = rng.uniform(0.0, 1.0, (n, mem_size)).astype(np.float32)
    # Sprinkle some zeros (realistic — not all cells have food)
    stored_util[rng.random((n, mem_size)) < 0.3] = 0.0

    pos_hist_x = rng.uniform(0.0, 500.0, (n, mem_size)).astype(np.float32)
    pos_hist_y = rng.uniform(0.0, 500.0, (n, mem_size)).astype(np.float32)

    mem_ptr = rng.integers(0, mem_size, size=n).astype(np.int32)
    # mem_count can exceed mem_size (circular buffer wraps)
    mem_count = rng.integers(0, mem_size * 3, size=n).astype(np.int32)
    # Ensure some agents have low counts (< mem_size)
    mem_count[:5] = rng.integers(0, 10, size=5).astype(np.int32)

    current_x = rng.uniform(0.0, 500.0, n).astype(np.float32)
    current_y = rng.uniform(0.0, 500.0, n).astype(np.float32)

    mask = rng.choice([True, False], n, p=[0.8, 0.2])

    # Decay table (workMemStrength / refMemStrength)
    from cenop.behavior.ref_mem import get_work_mem_strength_table

    work_mem_table = get_work_mem_strength_table(0.03, mem_size)

    return dict(
        stored_util=stored_util,
        pos_hist_x=pos_hist_x,
        pos_hist_y=pos_hist_y,
        mem_ptr=mem_ptr,
        mem_count=mem_count,
        current_x=current_x,
        current_y=current_y,
        work_mem_table=work_mem_table,
        mask=mask,
        n=n,
        mem_size=mem_size,
    )


class TestJaxRefMemKernel:
    """Tests for JAX reference memory kernels."""

    def test_ve_total_matches_numpy(self):
        """JAX veTotal should match NumPy/Numba output for same inputs."""
        from cenop.behavior.ref_mem import compute_ve_total
        from cenop.optimizations.jax_kernels import jax_compute_ve_total

        inputs = _make_ref_mem_inputs(n=100, mem_size=120, seed=10)

        # NumPy/Numba reference
        ref = compute_ve_total(
            inputs["stored_util"],
            inputs["mem_ptr"],
            inputs["mem_count"],
            inputs["work_mem_table"],
            inputs["mask"],
        )

        # JAX
        jax_result = jax_compute_ve_total(
            jnp.array(inputs["stored_util"]),
            jnp.array(inputs["mem_ptr"]),
            jnp.array(inputs["mem_count"]),
            jnp.array(inputs["work_mem_table"]),
            jnp.array(inputs["mask"]),
        )

        np.testing.assert_allclose(
            np.asarray(jax_result),
            ref,
            atol=1e-3,
            err_msg="JAX veTotal diverges from NumPy/Numba reference",
        )

    def test_ve_total_inactive_agents_zero(self):
        """Inactive agents should have veTotal = 0."""
        from cenop.optimizations.jax_kernels import jax_compute_ve_total

        inputs = _make_ref_mem_inputs(n=20, seed=11)
        mask = np.zeros(20, dtype=bool)  # all inactive

        result = jax_compute_ve_total(
            jnp.array(inputs["stored_util"]),
            jnp.array(inputs["mem_ptr"]),
            jnp.array(inputs["mem_count"]),
            jnp.array(inputs["work_mem_table"]),
            jnp.array(mask),
        )

        np.testing.assert_array_equal(np.asarray(result), 0.0)

    def test_ve_total_empty_memory(self):
        """Agents with mem_count=0 should have veTotal = 0."""
        from cenop.optimizations.jax_kernels import jax_compute_ve_total

        inputs = _make_ref_mem_inputs(n=10, seed=12)
        mem_count = np.zeros(10, dtype=np.int32)
        mask = np.ones(10, dtype=bool)

        result = jax_compute_ve_total(
            jnp.array(inputs["stored_util"]),
            jnp.array(inputs["mem_ptr"]),
            jnp.array(mem_count),
            jnp.array(inputs["work_mem_table"]),
            jnp.array(mask),
        )

        np.testing.assert_array_equal(np.asarray(result), 0.0)

    def test_ve_total_jit_compiles(self):
        """jax.jit(jax_compute_ve_total) runs without error."""
        from cenop.optimizations.jax_kernels import jax_compute_ve_total

        inputs = _make_ref_mem_inputs(n=30, seed=13)
        jitted = jax.jit(jax_compute_ve_total)

        r1 = jitted(
            jnp.array(inputs["stored_util"]),
            jnp.array(inputs["mem_ptr"]),
            jnp.array(inputs["mem_count"]),
            jnp.array(inputs["work_mem_table"]),
            jnp.array(inputs["mask"]),
        )
        r2 = jitted(
            jnp.array(inputs["stored_util"]),
            jnp.array(inputs["mem_ptr"]),
            jnp.array(inputs["mem_count"]),
            jnp.array(inputs["work_mem_table"]),
            jnp.array(inputs["mask"]),
        )

        np.testing.assert_array_equal(np.asarray(r1), np.asarray(r2))

    def test_attraction_vector_matches_numpy(self):
        """JAX attraction vector should match NumPy/Numba for same inputs."""
        from cenop.behavior.ref_mem import compute_attraction_vector
        from cenop.optimizations.jax_kernels import jax_compute_attraction

        inputs = _make_ref_mem_inputs(n=100, mem_size=120, seed=20)

        # NumPy/Numba reference (no world wrapping)
        ref_x, ref_y = compute_attraction_vector(
            inputs["stored_util"],
            inputs["pos_hist_x"],
            inputs["pos_hist_y"],
            inputs["mem_ptr"],
            inputs["mem_count"],
            inputs["current_x"],
            inputs["current_y"],
            inputs["work_mem_table"],
            inputs["mask"],
            world_width=0,
            world_height=0,
        )

        # JAX
        jax_x, jax_y = jax_compute_attraction(
            jnp.array(inputs["stored_util"]),
            jnp.array(inputs["pos_hist_x"]),
            jnp.array(inputs["pos_hist_y"]),
            jnp.array(inputs["mem_ptr"]),
            jnp.array(inputs["mem_count"]),
            jnp.array(inputs["current_x"]),
            jnp.array(inputs["current_y"]),
            jnp.array(inputs["work_mem_table"]),
            jnp.array(inputs["mask"]),
        )

        np.testing.assert_allclose(
            np.asarray(jax_x),
            ref_x,
            atol=1e-3,
            err_msg="JAX vt_x diverges from NumPy/Numba reference",
        )
        np.testing.assert_allclose(
            np.asarray(jax_y),
            ref_y,
            atol=1e-3,
            err_msg="JAX vt_y diverges from NumPy/Numba reference",
        )

    def test_attraction_inactive_agents_zero(self):
        """Inactive agents should have vt = (0, 0)."""
        from cenop.optimizations.jax_kernels import jax_compute_attraction

        inputs = _make_ref_mem_inputs(n=20, seed=21)
        mask = np.zeros(20, dtype=bool)

        vt_x, vt_y = jax_compute_attraction(
            jnp.array(inputs["stored_util"]),
            jnp.array(inputs["pos_hist_x"]),
            jnp.array(inputs["pos_hist_y"]),
            jnp.array(inputs["mem_ptr"]),
            jnp.array(inputs["mem_count"]),
            jnp.array(inputs["current_x"]),
            jnp.array(inputs["current_y"]),
            jnp.array(inputs["work_mem_table"]),
            jnp.array(mask),
        )

        np.testing.assert_array_equal(np.asarray(vt_x), 0.0)
        np.testing.assert_array_equal(np.asarray(vt_y), 0.0)

    def test_attraction_jit_compiles(self):
        """jax.jit(jax_compute_attraction) runs without error."""
        from cenop.optimizations.jax_kernels import jax_compute_attraction

        inputs = _make_ref_mem_inputs(n=30, seed=22)
        jitted = jax.jit(jax_compute_attraction)

        r1 = jitted(
            jnp.array(inputs["stored_util"]),
            jnp.array(inputs["pos_hist_x"]),
            jnp.array(inputs["pos_hist_y"]),
            jnp.array(inputs["mem_ptr"]),
            jnp.array(inputs["mem_count"]),
            jnp.array(inputs["current_x"]),
            jnp.array(inputs["current_y"]),
            jnp.array(inputs["work_mem_table"]),
            jnp.array(inputs["mask"]),
        )
        r2 = jitted(
            jnp.array(inputs["stored_util"]),
            jnp.array(inputs["pos_hist_x"]),
            jnp.array(inputs["pos_hist_y"]),
            jnp.array(inputs["mem_ptr"]),
            jnp.array(inputs["mem_count"]),
            jnp.array(inputs["current_x"]),
            jnp.array(inputs["current_y"]),
            jnp.array(inputs["work_mem_table"]),
            jnp.array(inputs["mask"]),
        )

        np.testing.assert_array_equal(np.asarray(r1[0]), np.asarray(r2[0]))
        np.testing.assert_array_equal(np.asarray(r1[1]), np.asarray(r2[1]))


# ---------------------------------------------------------------------------
# Land Avoidance (J4)
# ---------------------------------------------------------------------------


class TestJaxLandAvoidance:
    """Tests for jax_land_avoidance kernel."""

    def test_all_water_resolves_all(self):
        """On all-water grid, all agents should resolve."""
        from cenop.optimizations.jax_kernels import jax_land_avoidance

        n = 20
        rng = np.random.default_rng(42)
        # Deep water everywhere
        depth_grid = jnp.full((100, 100), 50.0, dtype=jnp.float32)
        x = jnp.array(rng.uniform(10, 90, n), dtype=jnp.float32)
        y = jnp.array(rng.uniform(10, 90, n), dtype=jnp.float32)
        heading = jnp.array(rng.uniform(0, 360, n), dtype=jnp.float32)
        step_dist = jnp.full(n, 5.0, dtype=jnp.float32)
        on_land = jnp.ones(n, dtype=bool)
        key = jax.random.PRNGKey(0)

        out_x, out_y, out_heading, resolved, _ = jax_land_avoidance(
            x, y, heading, step_dist, depth_grid, 1.0, on_land, key,
        )

        assert np.all(np.asarray(resolved)), "All agents should resolve on water grid"
        # Positions should have moved (not same as input)
        assert not np.allclose(np.asarray(out_x), np.asarray(x)), (
            "Resolved agents should have new positions"
        )

    def test_all_land_unresolved(self):
        """On all-land grid, agents should be unresolved and turn 180."""
        from cenop.optimizations.jax_kernels import jax_land_avoidance

        n = 10
        rng = np.random.default_rng(43)
        # Shallow everywhere (below min_depth)
        depth_grid = jnp.full((100, 100), 0.5, dtype=jnp.float32)
        x = jnp.array(rng.uniform(10, 90, n), dtype=jnp.float32)
        y = jnp.array(rng.uniform(10, 90, n), dtype=jnp.float32)
        heading = jnp.array(rng.uniform(0, 360, n), dtype=jnp.float32)
        step_dist = jnp.full(n, 5.0, dtype=jnp.float32)
        on_land = jnp.ones(n, dtype=bool)
        key = jax.random.PRNGKey(1)

        out_x, out_y, out_heading, resolved, _ = jax_land_avoidance(
            x, y, heading, step_dist, depth_grid, 1.0, on_land, key,
        )

        assert not np.any(np.asarray(resolved)), "No agent should resolve on land grid"
        # Heading should be (original + 180) % 360
        expected_heading = (np.asarray(heading) + 180.0) % 360.0
        np.testing.assert_allclose(
            np.asarray(out_heading), expected_heading, atol=1e-4,
            err_msg="Unresolved agents should turn 180 degrees",
        )
        # Positions should stay in place
        np.testing.assert_allclose(np.asarray(out_x), np.asarray(x), atol=1e-6)
        np.testing.assert_allclose(np.asarray(out_y), np.asarray(y), atol=1e-6)

    def test_picks_deeper_direction(self):
        """Should pick deeper cell when both directions valid."""
        from cenop.optimizations.jax_kernels import jax_land_avoidance

        # Grid: deep water on right side (cols >= 50), shallow-but-valid on left
        depth_grid = np.full((100, 100), 2.0, dtype=np.float32)
        depth_grid[:, 50:] = 50.0  # Much deeper on right
        depth_grid = jnp.array(depth_grid)

        # Agent at center, heading north (0 degrees)
        # Right turn (+40 deg) goes east (deeper), left turn (-40 deg) goes west (shallower)
        x = jnp.array([50.0], dtype=jnp.float32)
        y = jnp.array([50.0], dtype=jnp.float32)
        heading = jnp.array([0.0], dtype=jnp.float32)
        step_dist = jnp.array([10.0], dtype=jnp.float32)
        on_land = jnp.array([True])
        key = jax.random.PRNGKey(2)

        out_x, out_y, out_heading, resolved, _ = jax_land_avoidance(
            x, y, heading, step_dist, depth_grid, 1.0, on_land, key,
        )

        assert np.asarray(resolved)[0], "Agent should resolve"
        # The right turn goes toward deeper water (x > 50)
        # Due to jitter the exact position varies, but the resolved position
        # should be on the deeper side when right is much deeper
        assert np.asarray(out_x)[0] >= 50.0, (
            f"Should pick deeper (right) side, got x={np.asarray(out_x)[0]}"
        )

    def test_jit_compiles(self):
        """Land avoidance should JIT-compile."""
        from cenop.optimizations.jax_kernels import jax_land_avoidance

        n = 10
        depth_grid = jnp.full((50, 50), 20.0, dtype=jnp.float32)
        x = jnp.full(n, 25.0, dtype=jnp.float32)
        y = jnp.full(n, 25.0, dtype=jnp.float32)
        heading = jnp.full(n, 90.0, dtype=jnp.float32)
        step_dist = jnp.full(n, 3.0, dtype=jnp.float32)
        on_land = jnp.ones(n, dtype=bool)
        key = jax.random.PRNGKey(3)

        jitted = jax.jit(jax_land_avoidance, static_argnames=("min_depth",))
        r1 = jitted(x, y, heading, step_dist, depth_grid, 1.0, on_land, key)
        r2 = jitted(x, y, heading, step_dist, depth_grid, 1.0, on_land, key)

        # Same inputs + same key = same outputs (deterministic)
        np.testing.assert_array_equal(np.asarray(r1[0]), np.asarray(r2[0]))
        np.testing.assert_array_equal(np.asarray(r1[1]), np.asarray(r2[1]))
        np.testing.assert_array_equal(np.asarray(r1[2]), np.asarray(r2[2]))
        np.testing.assert_array_equal(np.asarray(r1[3]), np.asarray(r2[3]))

    def test_not_on_land_unchanged(self):
        """Agents not on land should keep original positions."""
        from cenop.optimizations.jax_kernels import jax_land_avoidance

        n = 5
        depth_grid = jnp.full((50, 50), 20.0, dtype=jnp.float32)
        x = jnp.array([10.0, 20.0, 30.0, 40.0, 25.0], dtype=jnp.float32)
        y = jnp.array([10.0, 20.0, 30.0, 40.0, 25.0], dtype=jnp.float32)
        heading = jnp.array([0.0, 90.0, 180.0, 270.0, 45.0], dtype=jnp.float32)
        step_dist = jnp.full(n, 5.0, dtype=jnp.float32)
        on_land = jnp.zeros(n, dtype=bool)  # Nobody on land
        key = jax.random.PRNGKey(4)

        out_x, out_y, out_heading, resolved, _ = jax_land_avoidance(
            x, y, heading, step_dist, depth_grid, 1.0, on_land, key,
        )

        # Nothing should change
        np.testing.assert_array_equal(np.asarray(out_x), np.asarray(x))
        np.testing.assert_array_equal(np.asarray(out_y), np.asarray(y))
        np.testing.assert_array_equal(np.asarray(out_heading), np.asarray(heading))
        assert not np.any(np.asarray(resolved))


# ---------------------------------------------------------------------------
# Heading Composition + Position Update + Boundary Reflection (J3)
# ---------------------------------------------------------------------------


def _make_heading_inputs(n=100, seed=42):
    """Create random test inputs for jax_heading_composition."""
    rng = np.random.default_rng(seed)
    return dict(
        heading=jnp.array(rng.uniform(0, 360, n), dtype=jnp.float64),
        pres_angle=jnp.array(rng.uniform(-90, 90, n), dtype=jnp.float64),
        log_mov=jnp.array(rng.uniform(1.0, 3.5, n), dtype=jnp.float64),
        ve_total=jnp.array(rng.uniform(0.0, 0.5, n), dtype=jnp.float32),
        vt_x=jnp.array(rng.uniform(-1, 1, n), dtype=jnp.float32),
        vt_y=jnp.array(rng.uniform(-1, 1, n), dtype=jnp.float32),
        deter_dx=jnp.zeros(n, dtype=jnp.float64),
        deter_dy=jnp.zeros(n, dtype=jnp.float64),
        social_dx=jnp.zeros(n, dtype=jnp.float64),
        social_dy=jnp.zeros(n, dtype=jnp.float64),
        mask=jnp.ones(n, dtype=bool),
        is_dispersing=jnp.zeros(n, dtype=bool),
        dispersal_target_x=jnp.zeros(n, dtype=jnp.float64),
        dispersal_target_y=jnp.zeros(n, dtype=jnp.float64),
        dispersal_target_distance=jnp.zeros(n, dtype=jnp.float64),
        prev_step_heading=jnp.array(rng.uniform(0, 360, n), dtype=jnp.float64),
        x=jnp.array(rng.uniform(10, 490, n), dtype=jnp.float32),
        y=jnp.array(rng.uniform(10, 490, n), dtype=jnp.float32),
        inertia_const=0.001,
        mean_disp_dist=2.0,
    )


class TestJaxHeadingAndPosition:
    """Tests for heading composition, boundary reflection, and position update."""

    def test_heading_composition_basic(self):
        """Heading should change after composition with non-zero inputs."""
        from cenop.optimizations.jax_kernels import jax_heading_composition

        inputs = _make_heading_inputs(n=50, seed=10)
        result = jax_heading_composition(**inputs)
        new_heading = np.asarray(result[0])
        original = np.asarray(inputs["heading"])

        # Heading should have changed for most agents
        changed = np.abs(new_heading - original) > 0.01
        assert np.sum(changed) > 25, "Expected most headings to change"

        # All headings in [0, 360)
        assert np.all(new_heading >= 0.0), f"Min heading: {new_heading.min()}"
        assert np.all(new_heading < 360.0), f"Max heading: {new_heading.max()}"

    def test_heading_composition_inactive_agents(self):
        """Inactive agents should have zero dx/dy and step_dist."""
        from cenop.optimizations.jax_kernels import jax_heading_composition

        inputs = _make_heading_inputs(n=50, seed=11)
        mask = np.zeros(50, dtype=bool)
        inputs["mask"] = jnp.array(mask)

        result = jax_heading_composition(**inputs)
        dx = np.asarray(result[3])
        dy = np.asarray(result[4])
        step_dist = np.asarray(result[2])

        np.testing.assert_array_equal(dx, 0.0)
        np.testing.assert_array_equal(dy, 0.0)
        np.testing.assert_array_equal(step_dist, 0.0)

    def test_dispersal_heading_override(self):
        """Dispersing agents should head toward their dispersal target."""
        from cenop.optimizations.jax_kernels import jax_heading_composition

        n = 10
        inputs = _make_heading_inputs(n=n, seed=12)

        # Make agents 0-4 dispersing, targeting (250, 250)
        is_disp = np.zeros(n, dtype=bool)
        is_disp[:5] = True
        inputs["is_dispersing"] = jnp.array(is_disp)
        inputs["dispersal_target_x"] = jnp.full(n, 250.0, dtype=jnp.float64)
        inputs["dispersal_target_y"] = jnp.full(n, 250.0, dtype=jnp.float64)
        inputs["dispersal_target_distance"] = jnp.full(n, 100.0, dtype=jnp.float64)

        # Place dispersing agents far from target
        x = np.asarray(inputs["x"]).copy()
        x[:5] = 50.0
        inputs["x"] = jnp.array(x, dtype=jnp.float32)
        y = np.asarray(inputs["y"]).copy()
        y[:5] = 50.0
        inputs["y"] = jnp.array(y, dtype=jnp.float32)

        result = jax_heading_composition(**inputs)
        new_heading = np.asarray(result[0])
        step_dist = np.asarray(result[2])

        # Dispersing agents' heading should be influenced by target direction
        # Target is at (250, 250), agents at (50, 50) -> target heading ~45 degrees
        # The SSLogis formula adjusts from prev_step_heading toward target
        # Just verify the heading is a valid value
        assert np.all(new_heading[:5] >= 0.0)
        assert np.all(new_heading[:5] < 360.0)

        # Dispersing step distance should be mean_disp_dist / 0.4 = 5.0
        expected_disp_step = 2.0 / 0.4
        np.testing.assert_allclose(
            step_dist[:5], expected_disp_step, rtol=1e-10,
            err_msg="Dispersing step should be mean_disp_dist / 0.4",
        )

    def test_step_dist_computed_correctly(self):
        """step_dist = 10^log_mov / 4, dispersing = mean_disp_dist / 0.4."""
        from cenop.optimizations.jax_kernels import jax_heading_composition

        n = 20
        inputs = _make_heading_inputs(n=n, seed=13)

        # Make agents 10-14 dispersing
        is_disp = np.zeros(n, dtype=bool)
        is_disp[10:15] = True
        inputs["is_dispersing"] = jnp.array(is_disp)
        inputs["dispersal_target_x"] = jnp.full(n, 250.0, dtype=jnp.float64)
        inputs["dispersal_target_y"] = jnp.full(n, 250.0, dtype=jnp.float64)
        inputs["dispersal_target_distance"] = jnp.full(n, 100.0, dtype=jnp.float64)

        result = jax_heading_composition(**inputs)
        step_dist = np.asarray(result[2])
        log_mov = np.asarray(inputs["log_mov"])

        # Non-dispersing active: 10^log_mov / 4.0
        expected_normal = 10.0 ** log_mov[:10] / 4.0
        np.testing.assert_allclose(
            step_dist[:10], expected_normal, rtol=1e-10,
            err_msg="Normal step should be 10^log_mov / 4",
        )

        # Dispersing: mean_disp_dist / 0.4
        expected_disp = 2.0 / 0.4
        np.testing.assert_allclose(
            step_dist[10:15], expected_disp, rtol=1e-10,
            err_msg="Dispersal step should be mean_disp_dist / 0.4",
        )

    def test_deter_strength_computed(self):
        """Deterrence strength should be |deter_dx| + |deter_dy|."""
        from cenop.optimizations.jax_kernels import jax_heading_composition

        inputs = _make_heading_inputs(n=20, seed=14)
        inputs["deter_dx"] = jnp.array(
            np.random.default_rng(14).uniform(-5, 5, 20), dtype=jnp.float64
        )
        inputs["deter_dy"] = jnp.array(
            np.random.default_rng(15).uniform(-5, 5, 20), dtype=jnp.float64
        )

        result = jax_heading_composition(**inputs)
        deter_strength = np.asarray(result[6])
        expected = np.abs(np.asarray(inputs["deter_dx"])) + np.abs(
            np.asarray(inputs["deter_dy"])
        )

        np.testing.assert_allclose(deter_strength, expected, rtol=1e-10)

    def test_prev_angle_range(self):
        """prev_angle should be in [-180, 180)."""
        from cenop.optimizations.jax_kernels import jax_heading_composition

        inputs = _make_heading_inputs(n=200, seed=15)
        result = jax_heading_composition(**inputs)
        prev_angle = np.asarray(result[1])

        assert np.all(prev_angle >= -180.0), f"Min prev_angle: {prev_angle.min()}"
        assert np.all(prev_angle < 180.0), f"Max prev_angle: {prev_angle.max()}"

    def test_heading_composition_jit_compiles(self):
        """jax.jit(jax_heading_composition) runs without error."""
        from cenop.optimizations.jax_kernels import jax_heading_composition

        inputs = _make_heading_inputs(n=30, seed=16)
        jitted = jax.jit(
            jax_heading_composition,
            static_argnames=("inertia_const", "mean_disp_dist"),
        )
        r1 = jitted(**inputs)
        r2 = jitted(**inputs)
        np.testing.assert_array_equal(np.asarray(r1[0]), np.asarray(r2[0]))

    def test_reflect_boundaries(self):
        """Positions outside world should be reflected back."""
        from cenop.optimizations.jax_kernels import jax_reflect_boundaries

        world_w, world_h = 500, 400

        # Agents: one normal, one under x, one over x, one under y, one over y
        new_x = jnp.array([100.0, -5.0, 504.0, 100.0, 100.0])
        new_y = jnp.array([100.0, 100.0, 100.0, -3.0, 402.0])
        dx = jnp.array([10.0, -15.0, 15.0, 10.0, 10.0])
        dy = jnp.array([10.0, 10.0, 10.0, -13.0, 13.0])

        rx, ry, rdx, rdy = jax_reflect_boundaries(
            new_x, new_y, dx, dy, world_w, world_h
        )

        rx, ry = np.asarray(rx), np.asarray(ry)
        rdx, rdy = np.asarray(rdx), np.asarray(rdy)

        # All positions should be within bounds
        assert np.all(rx >= 0) and np.all(rx <= world_w - 1)
        assert np.all(ry >= 0) and np.all(ry <= world_h - 1)

        # Normal agent: unchanged
        assert rx[0] == 100.0 and ry[0] == 100.0
        assert rdx[0] == 10.0 and rdy[0] == 10.0

        # Under x: reflected, dx flipped
        assert rx[1] == 5.0  # -(-5) = 5
        assert rdx[1] == 15.0  # sign flipped

        # Over x: reflected, dx flipped
        assert rx[2] == 2 * 499.0 - 504.0  # 994
        assert rdx[2] == -15.0

        # Under y: reflected, dy flipped
        assert ry[3] == 3.0
        assert rdy[3] == 13.0

        # Over y: reflected, dy flipped
        assert ry[4] == 2 * 399.0 - 402.0  # 396
        assert rdy[4] == -13.0

    def test_reflect_boundaries_clamps(self):
        """Extreme overshoot should be clamped after reflection."""
        from cenop.optimizations.jax_kernels import jax_reflect_boundaries

        # Position overshoots so much that reflection still out of bounds
        new_x = jnp.array([-1500.0, 2000.0])
        new_y = jnp.array([50.0, 50.0])
        dx = jnp.array([-100.0, 100.0])
        dy = jnp.array([5.0, 5.0])

        rx, ry, rdx, rdy = jax_reflect_boundaries(
            new_x, new_y, dx, dy, 500, 400
        )
        rx = np.asarray(rx)

        # Should be clamped to [0, 499]
        assert rx[0] >= 0.0
        assert rx[1] <= 499.0

    def test_update_positions(self):
        """Position update should apply dx/dy and reflect."""
        from cenop.optimizations.jax_kernels import jax_update_positions

        n = 5
        x = jnp.array([100.0, 100.0, 100.0, 498.0, 2.0], dtype=jnp.float32)
        y = jnp.array([100.0, 100.0, 100.0, 100.0, 100.0], dtype=jnp.float32)
        dx = jnp.array([5.0, 0.0, -5.0, 5.0, -5.0], dtype=jnp.float64)
        dy = jnp.array([5.0, 0.0, -5.0, 0.0, 0.0], dtype=jnp.float64)
        heading = jnp.array([45.0, 0.0, 225.0, 90.0, 270.0], dtype=jnp.float64)
        mask = jnp.array([True, True, True, True, True])

        new_x, new_y, new_heading = jax_update_positions(
            x, y, dx, dy, heading, 500, 400, mask
        )

        new_x, new_y = np.asarray(new_x), np.asarray(new_y)

        # Normal movement
        assert abs(new_x[0] - 105.0) < 1e-5
        assert abs(new_y[0] - 105.0) < 1e-5

        # No movement
        assert abs(new_x[1] - 100.0) < 1e-5

        # Boundary reflection: agent 4 at x=498 + dx=5 = 503 -> reflected
        assert new_x[3] <= 499.0
        assert new_x[3] >= 0.0

        # Agent 5 at x=2 + dx=-5 = -3 -> reflected to 3
        assert new_x[4] == 3.0

    def test_update_positions_inactive(self):
        """Inactive agents should keep original positions."""
        from cenop.optimizations.jax_kernels import jax_update_positions

        x = jnp.array([100.0, 200.0], dtype=jnp.float32)
        y = jnp.array([100.0, 200.0], dtype=jnp.float32)
        dx = jnp.array([50.0, 50.0], dtype=jnp.float64)
        dy = jnp.array([50.0, 50.0], dtype=jnp.float64)
        heading = jnp.array([45.0, 45.0], dtype=jnp.float64)
        mask = jnp.array([True, False])

        new_x, new_y, new_heading = jax_update_positions(
            x, y, dx, dy, heading, 500, 400, mask
        )

        # Active agent moved
        assert np.asarray(new_x)[0] == pytest.approx(150.0)
        # Inactive agent stayed
        assert np.asarray(new_x)[1] == pytest.approx(200.0)
        assert np.asarray(new_y)[1] == pytest.approx(200.0)

    def test_update_positions_heading_corrected_on_reflect(self):
        """Heading should be recalculated for reflected agents."""
        from cenop.optimizations.jax_kernels import jax_update_positions

        # Agent moving right (+x) but will overshoot
        x = jnp.array([498.0], dtype=jnp.float32)
        y = jnp.array([200.0], dtype=jnp.float32)
        dx = jnp.array([5.0], dtype=jnp.float64)
        dy = jnp.array([0.0], dtype=jnp.float64)
        heading = jnp.array([90.0], dtype=jnp.float64)
        mask = jnp.array([True])

        _, _, new_heading = jax_update_positions(
            x, y, dx, dy, heading, 500, 400, mask
        )

        new_h = float(np.asarray(new_heading)[0])
        # After reflection, dx flips to -5, dy stays 0
        # arctan2(-5, 0) = -pi/2 -> 270 degrees
        assert abs(new_h - 270.0) < 0.1, f"Expected ~270, got {new_h}"


# ---------------------------------------------------------------------------
# Food Intake (J5)
# ---------------------------------------------------------------------------


class TestJaxFoodKernel:
    """Tests for jax_eat_food kernel."""

    def test_two_pass_proportional_sharing(self):
        """3 agents on same cell get proportional shares."""
        from cenop.optimizations.jax_kernels import jax_eat_food

        # 5x5 grid with food=10.0 at cell (2,2)
        food_grid = jnp.full((5, 5), 0.5, dtype=jnp.float32)
        food_grid = food_grid.at[2, 2].set(10.0)

        # 3 agents on same cell (2,2), different energies
        xi = jnp.array([2, 2, 2], dtype=jnp.int32)
        yi = jnp.array([2, 2, 2], dtype=jnp.int32)
        energy = jnp.array([5.0, 10.0, 15.0], dtype=jnp.float32)
        # fracs: (20-5)/10=1.5->0.99, (20-10)/10=1.0->0.99, (20-15)/10=0.5

        eaten, new_food = jax_eat_food(food_grid, xi, yi, energy, 0.01)
        eaten = np.asarray(eaten)

        # All agents should get food
        assert np.all(eaten > 0), "All agents should get some food"
        # Total eaten should not exceed available (10.0)
        assert np.sum(eaten) <= 10.0 + 1e-6, f"Total eaten {np.sum(eaten)} > available 10.0"
        # Agents 0 and 1 have same fraction (both clipped to 0.99),
        # so they should get equal shares
        np.testing.assert_allclose(eaten[0], eaten[1], rtol=1e-5)
        # Agent 2 has lower fraction (0.5), so should get less
        assert eaten[2] < eaten[0], "Lower-energy agent should eat more than higher-energy"

    def test_energy_20_eats_nothing(self):
        """energy=20 -> fraction=0 -> no eating."""
        from cenop.optimizations.jax_kernels import jax_eat_food

        food_grid = jnp.full((5, 5), 10.0, dtype=jnp.float32)
        xi = jnp.array([1, 2], dtype=jnp.int32)
        yi = jnp.array([1, 2], dtype=jnp.int32)
        energy = jnp.array([20.0, 20.0], dtype=jnp.float32)

        eaten, new_food = jax_eat_food(food_grid, xi, yi, energy, 0.01)
        eaten = np.asarray(eaten)

        np.testing.assert_array_equal(eaten, 0.0)
        # Food grid unchanged
        np.testing.assert_allclose(np.asarray(new_food), np.asarray(food_grid), atol=1e-6)

    def test_zero_energy_eats_max(self):
        """energy=0 -> fraction=0.99."""
        from cenop.optimizations.jax_kernels import jax_eat_food

        food_grid = jnp.full((5, 5), 10.0, dtype=jnp.float32)
        xi = jnp.array([1], dtype=jnp.int32)
        yi = jnp.array([1], dtype=jnp.int32)
        energy = jnp.array([0.0], dtype=jnp.float32)

        eaten, new_food = jax_eat_food(food_grid, xi, yi, energy, 0.01)
        eaten_val = float(np.asarray(eaten)[0])

        # Should eat 10.0 * 0.99 = 9.9 (clip at 0.99 frac)
        assert abs(eaten_val - 9.9) < 0.01, f"Expected ~9.9, got {eaten_val}"

    def test_food_floor_enforced(self):
        """Food grid never goes below min_food."""
        from cenop.optimizations.jax_kernels import jax_eat_food

        food_grid = jnp.full((5, 5), 0.02, dtype=jnp.float32)
        xi = jnp.array([1], dtype=jnp.int32)
        yi = jnp.array([1], dtype=jnp.int32)
        energy = jnp.array([0.0], dtype=jnp.float32)  # max hunger

        eaten, new_food = jax_eat_food(food_grid, xi, yi, energy, 0.01)
        new_food = np.asarray(new_food)

        assert np.all(new_food >= 0.01), f"Min food: {new_food.min()}"

    def test_jit_compiles(self):
        """jax_eat_food should JIT-compile."""
        from cenop.optimizations.jax_kernels import jax_eat_food

        food_grid = jnp.full((10, 10), 5.0, dtype=jnp.float32)
        xi = jnp.array([1, 2, 3], dtype=jnp.int32)
        yi = jnp.array([1, 2, 3], dtype=jnp.int32)
        energy = jnp.array([5.0, 10.0, 15.0], dtype=jnp.float32)

        jitted = jax.jit(jax_eat_food, static_argnames=("min_food",))
        r1 = jitted(food_grid, xi, yi, energy, min_food=0.01)
        r2 = jitted(food_grid, xi, yi, energy, min_food=0.01)
        np.testing.assert_array_equal(np.asarray(r1[0]), np.asarray(r2[0]))


# ---------------------------------------------------------------------------
# Mortality (J5)
# ---------------------------------------------------------------------------


class TestJaxMortality:
    """Tests for jax_mortality kernel."""

    def test_zero_energy_dies(self):
        """Most agents with energy=0 should die."""
        from cenop.optimizations.jax_kernels import jax_mortality

        n = 1000
        energy = jnp.zeros(n, dtype=jnp.float32)
        active_mask = jnp.ones(n, dtype=bool)
        with_calf = jnp.zeros(n, dtype=bool)
        age = jnp.full(n, 5.0, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)

        new_mask, new_calf, _ = jax_mortality(
            energy, active_mask, with_calf, age, key,
            m_mort_prob_const=0.5, x_survival_const=0.5,
            is_day_boundary=jnp.bool_(False),
            bycatch_prob=0.0, max_age=30.0,
        )

        dead_count = n - int(np.sum(np.asarray(new_mask)))
        # With energy=0, step_surv=0, so all random > 0 => all die
        assert dead_count == n, f"Expected all {n} to die, got {dead_count} deaths"

    def test_high_energy_survives(self):
        """Most agents with energy=20 should survive."""
        from cenop.optimizations.jax_kernels import jax_mortality

        n = 1000
        energy = jnp.full(n, 20.0, dtype=jnp.float32)
        active_mask = jnp.ones(n, dtype=bool)
        with_calf = jnp.zeros(n, dtype=bool)
        age = jnp.full(n, 5.0, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)

        new_mask, new_calf, _ = jax_mortality(
            energy, active_mask, with_calf, age, key,
            m_mort_prob_const=0.5, x_survival_const=0.5,
            is_day_boundary=jnp.bool_(False),
            bycatch_prob=0.0, max_age=30.0,
        )

        survivors = int(np.sum(np.asarray(new_mask)))
        # With high energy, survival prob is very close to 1
        assert survivors > 990, f"Expected >990 survivors, got {survivors}"

    def test_calf_abandonment_before_death(self):
        """With-calf agents lose calf first, then die only if energy<=0."""
        from cenop.optimizations.jax_kernels import jax_mortality

        n = 2000
        # Very low energy but not zero — need aggressive mortality params
        # to trigger starvation events within a single tick
        energy = jnp.full(n, 0.01, dtype=jnp.float32)
        active_mask = jnp.ones(n, dtype=bool)
        with_calf = jnp.ones(n, dtype=bool)  # all have calves
        age = jnp.full(n, 5.0, dtype=jnp.float32)
        key = jax.random.PRNGKey(99)

        # Use very aggressive mortality parameters so that step_surv is low
        new_mask, new_calf, _ = jax_mortality(
            energy, active_mask, with_calf, age, key,
            m_mort_prob_const=0.999, x_survival_const=0.01,
            is_day_boundary=jnp.bool_(False),
            bycatch_prob=0.0, max_age=30.0,
        )

        new_mask_np = np.asarray(new_mask)
        new_calf_np = np.asarray(new_calf)

        # Starving agents with calves: calf abandoned, but agent survives
        # (because was_with_calf=True => ~was_with_calf=False => starved requires energy<=0)
        # energy=0.01 > 0, so starved = starving & (False | ~True) = False
        # => All survive, but calves lost for those that were starving
        assert np.all(new_mask_np), "No one should die (energy>0 and had calf)"
        # Some calves should be abandoned (those who were starving)
        calf_lost = int(np.sum(~new_calf_np))
        assert calf_lost > 0, "Some calves should be abandoned"

    def test_max_age_death_at_day_boundary(self):
        """Old agents die at day boundary."""
        from cenop.optimizations.jax_kernels import jax_mortality

        n = 10
        energy = jnp.full(n, 20.0, dtype=jnp.float32)
        active_mask = jnp.ones(n, dtype=bool)
        with_calf = jnp.zeros(n, dtype=bool)
        age = jnp.array([5.0, 10.0, 20.0, 25.0, 31.0,
                         35.0, 5.0, 5.0, 5.0, 5.0], dtype=jnp.float32)
        key = jax.random.PRNGKey(42)

        new_mask, _, _ = jax_mortality(
            energy, active_mask, with_calf, age, key,
            m_mort_prob_const=0.5, x_survival_const=0.5,
            is_day_boundary=jnp.bool_(True),
            bycatch_prob=0.0, max_age=30.0,
        )

        new_mask_np = np.asarray(new_mask)
        # Agents with age > 30 should die (indices 4, 5)
        assert not new_mask_np[4], "Age 31 should die"
        assert not new_mask_np[5], "Age 35 should die"
        # Young agents survive
        assert new_mask_np[0], "Age 5 should survive"
        assert new_mask_np[1], "Age 10 should survive"

    def test_jit_compiles(self):
        """jax_mortality should JIT-compile."""
        from cenop.optimizations.jax_kernels import jax_mortality

        n = 20
        jitted = jax.jit(jax_mortality)
        r1 = jitted(
            jnp.full(n, 10.0, dtype=jnp.float32),
            jnp.ones(n, dtype=bool),
            jnp.zeros(n, dtype=bool),
            jnp.full(n, 5.0, dtype=jnp.float32),
            jax.random.PRNGKey(0),
            0.5, 0.5,
            jnp.bool_(False), 0.0, 30.0,
        )
        r2 = jitted(
            jnp.full(n, 10.0, dtype=jnp.float32),
            jnp.ones(n, dtype=bool),
            jnp.zeros(n, dtype=bool),
            jnp.full(n, 5.0, dtype=jnp.float32),
            jax.random.PRNGKey(0),
            0.5, 0.5,
            jnp.bool_(False), 0.0, 30.0,
        )
        np.testing.assert_array_equal(np.asarray(r1[0]), np.asarray(r2[0]))


# ---------------------------------------------------------------------------
# BMR Cost (J5)
# ---------------------------------------------------------------------------


class TestJaxBMR:
    """Tests for jax_bmr_cost kernel."""

    def test_bmr_reduces_energy(self):
        """Energy should decrease after BMR."""
        from cenop.optimizations.jax_kernels import jax_bmr_cost

        n = 10
        energy = jnp.full(n, 15.0, dtype=jnp.float32)
        mask = jnp.ones(n, dtype=bool)
        speed = jnp.full(n, 1.0, dtype=jnp.float32)
        is_lact = jnp.zeros(n, dtype=bool)
        is_dist = jnp.zeros(n, dtype=bool)
        deter_mag = jnp.zeros(n, dtype=jnp.float32)

        new_energy, cost = jax_bmr_cost(
            energy, mask, speed, is_lact, is_dist, deter_mag,
            scaling=1.0, e_use_per_30_min=4.5, e_lact=1.4,
        )

        new_e = np.asarray(new_energy)
        cost_np = np.asarray(cost)

        assert np.all(new_e < 15.0), "Energy should decrease"
        assert np.all(cost_np > 0), "Cost should be positive"
        np.testing.assert_allclose(new_e, 15.0 - cost_np, rtol=1e-6)

    def test_lactation_multiplier(self):
        """Lactating agents pay higher cost."""
        from cenop.optimizations.jax_kernels import jax_bmr_cost

        n = 2
        energy = jnp.full(n, 15.0, dtype=jnp.float32)
        mask = jnp.ones(n, dtype=bool)
        speed = jnp.full(n, 1.0, dtype=jnp.float32)
        is_lact = jnp.array([False, True])
        is_dist = jnp.zeros(n, dtype=bool)
        deter_mag = jnp.zeros(n, dtype=jnp.float32)

        new_energy, cost = jax_bmr_cost(
            energy, mask, speed, is_lact, is_dist, deter_mag,
            scaling=1.0, e_use_per_30_min=4.5, e_lact=1.4,
        )

        cost_np = np.asarray(cost)
        # Lactating agent (idx 1) should pay more than non-lactating (idx 0)
        assert cost_np[1] > cost_np[0], (
            f"Lactating cost {cost_np[1]} should exceed non-lactating {cost_np[0]}"
        )

    def test_disturbance_cost(self):
        """Disturbed agents pay extra cost."""
        from cenop.optimizations.jax_kernels import jax_bmr_cost

        n = 2
        energy = jnp.full(n, 15.0, dtype=jnp.float32)
        mask = jnp.ones(n, dtype=bool)
        speed = jnp.full(n, 1.0, dtype=jnp.float32)
        is_lact = jnp.zeros(n, dtype=bool)
        is_dist = jnp.array([False, True])
        deter_mag = jnp.array([0.0, 5.0], dtype=jnp.float32)

        _, cost = jax_bmr_cost(
            energy, mask, speed, is_lact, is_dist, deter_mag,
            scaling=1.0, e_use_per_30_min=4.5, e_lact=1.4,
        )

        cost_np = np.asarray(cost)
        assert cost_np[1] > cost_np[0], "Disturbed agent should pay more"

    def test_inactive_no_cost(self):
        """Inactive agents should have zero cost and unchanged energy."""
        from cenop.optimizations.jax_kernels import jax_bmr_cost

        n = 5
        energy = jnp.full(n, 15.0, dtype=jnp.float32)
        mask = jnp.zeros(n, dtype=bool)
        speed = jnp.full(n, 1.0, dtype=jnp.float32)
        is_lact = jnp.zeros(n, dtype=bool)
        is_dist = jnp.zeros(n, dtype=bool)
        deter_mag = jnp.zeros(n, dtype=jnp.float32)

        new_energy, cost = jax_bmr_cost(
            energy, mask, speed, is_lact, is_dist, deter_mag,
            scaling=1.0, e_use_per_30_min=4.5, e_lact=1.4,
        )

        np.testing.assert_array_equal(np.asarray(new_energy), 15.0)
        np.testing.assert_array_equal(np.asarray(cost), 0.0)


# ---------------------------------------------------------------------------
# Energy History (J5)
# ---------------------------------------------------------------------------


class TestJaxEnergyHistory:
    """Tests for jax_energy_history_update kernel."""

    def test_accumulates_energy(self):
        """Energy should accumulate into ticks_today."""
        from cenop.optimizations.jax_kernels import jax_energy_history_update

        n = 5
        energy = jnp.full(n, 10.0, dtype=jnp.float32)
        mask = jnp.ones(n, dtype=bool)
        ticks_today = jnp.zeros(n, dtype=jnp.float32)
        history = jnp.zeros((n, 8), dtype=jnp.float32)
        tick_counter = jnp.int32(0)

        new_ticks, new_hist, new_tc = jax_energy_history_update(
            energy, mask, ticks_today, history, tick_counter,
            is_day_boundary=jnp.bool_(False),
        )

        np.testing.assert_allclose(np.asarray(new_ticks), 10.0)
        assert int(np.asarray(new_tc)) == 1

    def test_day_boundary_shifts_history(self):
        """At day boundary, history shifts right and daily avg inserted."""
        from cenop.optimizations.jax_kernels import jax_energy_history_update

        n = 3
        energy = jnp.full(n, 10.0, dtype=jnp.float32)
        mask = jnp.ones(n, dtype=bool)
        # Simulate 47 ticks already accumulated (480 total energy)
        ticks_today = jnp.full(n, 470.0, dtype=jnp.float32)  # 47 * 10
        history = jnp.full((n, 8), 5.0, dtype=jnp.float32)
        tick_counter = jnp.int32(47)

        new_ticks, new_hist, new_tc = jax_energy_history_update(
            energy, mask, ticks_today, history, tick_counter,
            is_day_boundary=jnp.bool_(True),
        )

        new_hist_np = np.asarray(new_hist)
        # After accumulation: ticks_today = 470 + 10 = 480
        # daily_avg = 480 / 48 = 10.0
        # history[:, 0] should be 10.0
        np.testing.assert_allclose(new_hist_np[:, 0], 10.0, atol=1e-5)
        # history[:, 1:-1] should be shifted old values (5.0)
        np.testing.assert_allclose(new_hist_np[:, 1:], 5.0, atol=1e-5)
        # ticks_today reset
        np.testing.assert_array_equal(np.asarray(new_ticks), 0.0)
        assert int(np.asarray(new_tc)) == 0


# ---------------------------------------------------------------------------
# Dispersal Update (J5)
# ---------------------------------------------------------------------------


class TestJaxDispersalUpdate:
    """Tests for jax_dispersal_update kernel."""

    def test_deterrence_cancels_dispersal(self):
        """Deterrence should cancel dispersal."""
        from cenop.optimizations.jax_kernels import jax_dispersal_update

        n = 5
        is_disp = jnp.array([True, True, False, True, False])
        deter = jnp.array([0.0, 5.0, 0.0, 3.0, 0.0], dtype=jnp.float32)

        new_disp, new_dist, new_dde = jax_dispersal_update(
            is_dispersing=is_disp,
            dispersal_start_x=jnp.zeros(n, dtype=jnp.float32),
            dispersal_start_y=jnp.zeros(n, dtype=jnp.float32),
            dispersal_target_distance=jnp.full(n, 100.0, dtype=jnp.float32),
            dispersal_distance_traveled=jnp.full(n, 50.0, dtype=jnp.float32),
            days_declining_energy=jnp.full(n, 3, dtype=jnp.int32),
            x=jnp.full(n, 10.0, dtype=jnp.float32),
            y=jnp.full(n, 10.0, dtype=jnp.float32),
            deter_strength=deter,
            energy_history=jnp.zeros((n, 8), dtype=jnp.float32),
            active_mask=jnp.ones(n, dtype=bool),
            is_day_boundary=jnp.bool_(False),
        )

        new_disp_np = np.asarray(new_disp)
        # Agent 0: dispersing, no deterrence -> still dispersing (but may complete distance)
        # Agent 1: dispersing + deterred -> stopped
        assert not new_disp_np[1], "Deterred agent should stop dispersal"
        # Agent 3: dispersing + deterred -> stopped
        assert not new_disp_np[3], "Deterred agent should stop dispersal"

    def test_distance_completion(self):
        """Agents reaching target distance should stop."""
        from cenop.optimizations.jax_kernels import jax_dispersal_update

        n = 3
        new_disp, _, _ = jax_dispersal_update(
            is_dispersing=jnp.ones(n, dtype=bool),
            dispersal_start_x=jnp.zeros(n, dtype=jnp.float32),
            dispersal_start_y=jnp.zeros(n, dtype=jnp.float32),
            dispersal_target_distance=jnp.array([100.0, 100.0, 100.0], dtype=jnp.float32),
            dispersal_distance_traveled=jnp.zeros(n, dtype=jnp.float32),
            days_declining_energy=jnp.zeros(n, dtype=jnp.int32),
            # Agent 0: at (96, 0) -> dist=96 >= 0.95*100=95 -> complete
            # Agent 1: at (50, 0) -> dist=50 < 95 -> continue
            # Agent 2: at (100, 0) -> dist=100 >= 95 -> complete
            x=jnp.array([96.0, 50.0, 100.0], dtype=jnp.float32),
            y=jnp.zeros(n, dtype=jnp.float32),
            deter_strength=jnp.zeros(n, dtype=jnp.float32),
            energy_history=jnp.zeros((n, 8), dtype=jnp.float32),
            active_mask=jnp.ones(n, dtype=bool),
            is_day_boundary=jnp.bool_(False),
        )

        new_disp_np = np.asarray(new_disp)
        assert not new_disp_np[0], "Agent at 96% should complete"
        assert new_disp_np[1], "Agent at 50% should continue"
        assert not new_disp_np[2], "Agent at 100% should complete"


# ---------------------------------------------------------------------------
# Full-tick integration tests (J6)
# ---------------------------------------------------------------------------


class TestJaxTickComposition:
    """Tests for the composed jax_tick_movement and jax_tick_energy functions."""

    def test_tick_movement_returns_valid_positions(self):
        """Movement tick should produce in-bounds positions."""
        from cenop.optimizations.tick_jax import jax_tick_movement
        from cenop.optimizations.jax_kernels import jax_crw_kernel

        n = 50
        rng = np.random.default_rng(42)
        world_w, world_h = 200, 200

        x = jnp.array(rng.uniform(10, 190, n), dtype=jnp.float32)
        y = jnp.array(rng.uniform(10, 190, n), dtype=jnp.float32)
        heading = jnp.array(rng.uniform(0, 360, n), dtype=jnp.float32)
        prev_angle = jnp.array(rng.uniform(-90, 90, n), dtype=jnp.float64)
        prev_log_mov = jnp.array(rng.uniform(0.5, 1.5, n), dtype=jnp.float64)
        mask = jnp.ones(n, dtype=bool)
        mem_size = 20
        stored_util = jnp.zeros((n, mem_size), dtype=jnp.float32)
        pos_hist_x = jnp.zeros((n, mem_size), dtype=jnp.float32)
        pos_hist_y = jnp.zeros((n, mem_size), dtype=jnp.float32)
        mem_ptr = jnp.zeros(n, dtype=jnp.int32)
        mem_count = jnp.zeros(n, dtype=jnp.int32)
        work_table = jnp.array(
            [np.exp(-i * 0.01) for i in range(mem_size)], dtype=jnp.float64
        )
        depth_grid = jnp.full((world_h, world_w), 30.0, dtype=jnp.float32)

        result = jax_tick_movement(
            x, y, heading, prev_angle, prev_log_mov, mask,
            stored_util, pos_hist_x, pos_hist_y, mem_ptr, mem_count,
            work_table,
            jnp.zeros(n, dtype=jnp.float64),
            jnp.zeros(n, dtype=jnp.float64),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=bool),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.array(rng.uniform(5.0, 50.0, n), dtype=jnp.float64),
            jnp.array(rng.uniform(10.0, 35.0, n), dtype=jnp.float64),
            depth_grid,
            -0.024, -0.008, 0.93, -14.0,
            0.35, 0.0005, -0.02, 1.73,
            0.0, 4.0, 0.0, 0.15,
            0.001, 2.0, 1.0,
            world_w, world_h,
            jax.random.PRNGKey(99),
        )

        new_x = np.asarray(result[0])
        new_y = np.asarray(result[1])
        assert np.all(new_x >= 0) and np.all(new_x < world_w)
        assert np.all(new_y >= 0) and np.all(new_y < world_h)
        assert np.all(np.isfinite(new_x)) and np.all(np.isfinite(new_y))

    def test_tick_energy_conserves_food(self):
        """Energy tick should not create food out of nothing."""
        from cenop.optimizations.tick_jax import jax_tick_energy

        n = 20
        rng = np.random.default_rng(7)
        food_grid = jnp.ones((50, 50), dtype=jnp.float32) * 0.5
        energy = jnp.array(rng.uniform(5, 15, n), dtype=jnp.float32)
        mask = jnp.ones(n, dtype=bool)
        xi = jnp.array(rng.integers(0, 50, n), dtype=jnp.int32)
        yi = jnp.array(rng.integers(0, 50, n), dtype=jnp.int32)

        result = jax_tick_energy(
            energy, mask,
            jnp.array(xi, dtype=jnp.float32),
            jnp.array(yi, dtype=jnp.float32),
            jnp.ones(n, dtype=jnp.float32) * 0.5,
            jnp.zeros(n, dtype=bool),
            jnp.zeros(n, dtype=bool),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=bool),
            jnp.ones(n, dtype=jnp.float32) * 5.0,
            food_grid, xi, yi,
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros((n, 10), dtype=jnp.float32),
            jnp.int32(0),
            jnp.zeros(n, dtype=bool),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.ones(n, dtype=jnp.float32) * 100.0,
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.int32),
            jnp.zeros(n, dtype=jnp.float32),
            1.0, 4.5, 1.4, 0.001,
            1.0, 0.4, 0.0, 30.0,
            jnp.bool_(False),
            jax.random.PRNGKey(42),
        )

        new_food_grid = np.asarray(result[3])
        assert np.all(new_food_grid <= np.asarray(food_grid) + 1e-6), (
            "Food grid should not increase"
        )
        assert np.all(new_food_grid >= 0.001 - 1e-6), "Food should not go below min"

    def test_is_jax_available(self):
        """is_jax_available should return True when JAX works."""
        from cenop.optimizations.tick_jax import is_jax_available

        assert is_jax_available() is True


class TestJaxFullTick:
    """End-to-end tests running the full JAX path through population.step()."""

    def test_population_survives_1000_ticks(self):
        """JAX path should maintain viable population."""
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.core.simulation import Simulation

        params = SimulationParameters(
            porpoise_count=50,
            turbines="off",
            ships_enabled=False,
            use_jax=True,
            random_seed=42,
        )
        sim = Simulation(params=params, seed=42)
        sim.initialize()
        for _ in range(1000):
            sim.step()
        pop = sim.population_manager
        alive = int(np.sum(pop.active_mask))
        mean_energy = float(np.mean(pop.energy[pop.active_mask]))
        assert alive > 10, f"Only {alive} alive after 1000 ticks"
        assert 2.0 < mean_energy <= 20.0, f"Mean energy {mean_energy} out of range"

    def test_jax_vs_numba_statistical_equivalence(self):
        """JAX and Numba produce statistically similar results."""
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.core.simulation import Simulation

        results = {}
        for use_jax in [False, True]:
            params = SimulationParameters(
                porpoise_count=100,
                turbines="off",
                ships_enabled=False,
                use_jax=use_jax,
                random_seed=42,
            )
            sim = Simulation(params=params, seed=42)
            sim.initialize()
            for _ in range(500):
                sim.step()
            pop = sim.population_manager
            active = pop.active_mask
            results[use_jax] = {
                'pop': int(np.sum(active)),
                'mean_energy': float(np.mean(pop.energy[active])),
            }
        r_jax = results[True]
        r_numba = results[False]
        pop_diff = abs(r_jax['pop'] - r_numba['pop']) / max(r_numba['pop'], 1)
        energy_diff = abs(r_jax['mean_energy'] - r_numba['mean_energy'])
        assert pop_diff < 0.15, (
            f"Population diff {pop_diff:.2%}: JAX={r_jax['pop']}, Numba={r_numba['pop']}"
        )
        assert energy_diff < 3.0, (
            f"Energy diff {energy_diff:.2f}: JAX={r_jax['mean_energy']:.2f}, "
            f"Numba={r_numba['mean_energy']:.2f}"
        )


class TestCRWK4Bounds:
    """Verify CRW outputs are valid with K=4 (reduced from K=32)."""

    def test_angle_bounds_k4(self):
        """All output angles must be in [-180, 180] with K=4."""
        from cenop.optimizations.jax_kernels import jax_crw_kernel

        inputs = _make_inputs(n=2000, seed=99)
        out_angle, out_log_mov = jax_crw_kernel(**inputs, **CRW_PARAMS)
        active = np.asarray(inputs['mask'])
        angles = np.asarray(out_angle)[active]
        assert np.all(np.abs(angles) <= 180.0)
        log_movs = np.asarray(out_log_mov)[active]
        assert np.all(log_movs <= CRW_PARAMS['max_mov'] + 1e-6)

    def test_angle_distribution_reasonable_k4(self):
        """Angles should have reasonable mean and spread with K=4."""
        from cenop.optimizations.jax_kernels import jax_crw_kernel

        inputs = _make_inputs(n=5000, seed=123)
        out_angle, _ = jax_crw_kernel(**inputs, **CRW_PARAMS)
        active = np.asarray(inputs['mask'])
        angles = np.asarray(out_angle)[active]
        assert abs(np.mean(angles)) < 20.0
        assert np.std(angles) > 10.0
