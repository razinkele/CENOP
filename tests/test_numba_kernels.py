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
        out_xi = np.zeros(n, dtype=np.int32)
        out_yi = np.zeros(n, dtype=np.int32)

        turn_position_kernel(
            x, y, heading, step_dist, 90.0, 20, 20,
            out_x, out_y, out_heading, out_xi, out_yi,
        )

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
        out_xi = np.zeros(n, dtype=np.int32)
        out_yi = np.zeros(n, dtype=np.int32)

        # Turn 90 degrees: dx = sin(pi/2)*4 = 4, new_x = 22 > 19 (max)
        turn_position_kernel(
            x, y, heading, step_dist, 90.0, 20, 20,
            out_x, out_y, out_heading, out_xi, out_yi,
        )

        # Should be reflected: 2*19 - 22 = 16
        assert out_x[0] == pytest.approx(16.0, abs=0.1)


class TestTurnPositionKernelIndices:
    """Test that turn_position_kernel outputs correct clamped int32 indices."""

    def test_outputs_clamped_indices(self):
        from cenop.optimizations.kernels import turn_position_kernel

        n = 3
        x = np.array([1.0, 25.0, 48.5], dtype=np.float64)
        y = np.array([1.0, 25.0, 48.5], dtype=np.float64)
        heading = np.zeros(n, dtype=np.float64)
        step = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        out_x = np.zeros(n, dtype=np.float64)
        out_y = np.zeros(n, dtype=np.float64)
        out_h = np.zeros(n, dtype=np.float64)
        out_xi = np.zeros(n, dtype=np.int32)
        out_yi = np.zeros(n, dtype=np.int32)
        turn_position_kernel(
            x, y, heading, step, 90.0, 50, 50,
            out_x, out_y, out_h, out_xi, out_yi,
        )
        assert out_xi.dtype == np.int32
        assert out_yi.dtype == np.int32
        assert np.all(out_xi >= 0) and np.all(out_xi <= 49)
        assert np.all(out_yi >= 0) and np.all(out_yi <= 49)

    def test_index_matches_float_truncation(self):
        from cenop.optimizations.kernels import turn_position_kernel

        n = 1
        x = np.array([10.7], dtype=np.float64)
        y = np.array([20.3], dtype=np.float64)
        heading = np.array([0.0], dtype=np.float64)
        step = np.array([0.0], dtype=np.float64)
        out_x = np.zeros(n, dtype=np.float64)
        out_y = np.zeros(n, dtype=np.float64)
        out_h = np.zeros(n, dtype=np.float64)
        out_xi = np.zeros(n, dtype=np.int32)
        out_yi = np.zeros(n, dtype=np.int32)
        turn_position_kernel(
            x, y, heading, step, 0.0, 50, 50,
            out_x, out_y, out_h, out_xi, out_yi,
        )
        assert out_xi[0] == int(out_x[0])
        assert out_yi[0] == int(out_y[0])


class TestEatFoodKernel:
    """Test eat_food_kernel with proportional-sharing semantics."""

    def test_basic_eating(self):
        """Single agent eats fraction of food in its cell."""
        from cenop.optimizations.kernels import eat_food_kernel

        food_grid = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        x = np.array([1], dtype=np.int32)  # column
        y = np.array([0], dtype=np.int32)  # row
        fraction = np.array([0.5], dtype=np.float32)
        food_eaten = np.zeros(1, dtype=np.float32)
        demand_grid = np.zeros_like(food_grid)

        eat_food_kernel(food_grid, x, y, fraction, food_eaten, 0.01, demand_grid)

        # Food at [0,1] was 20.0, ate 50% = 10.0
        assert food_eaten[0] == pytest.approx(10.0, abs=0.01)
        assert food_grid[0, 1] == pytest.approx(10.0, abs=0.01)

    def test_same_cell_proportional_sharing(self):
        """Two agents in same cell should get proportional shares (order-independent)."""
        from cenop.optimizations.kernels import eat_food_kernel

        food_grid = np.array([[100.0]], dtype=np.float32)
        x = np.array([0, 0], dtype=np.int32)
        y = np.array([0, 0], dtype=np.int32)
        # Both want 60%: total demand = 120% > available
        fraction = np.array([0.6, 0.6], dtype=np.float32)
        food_eaten = np.zeros(2, dtype=np.float32)
        demand_grid = np.zeros_like(food_grid)

        eat_food_kernel(food_grid, x, y, fraction, food_eaten, 0.01, demand_grid)

        # Total demand: 60 + 60 = 120, available: 100 (full cell food)
        # Each gets proportional share: 60/120 * 100 = 50.0
        assert food_eaten[0] == pytest.approx(food_eaten[1], abs=0.1), \
            "Equal-fraction agents should get equal shares (proportional)"
        assert food_eaten[0] + food_eaten[1] == pytest.approx(100.0, abs=0.1)
        # Grid floors to min_food after depletion
        assert food_grid[0, 0] == pytest.approx(0.01, abs=0.01)

    def test_same_cell_no_overdepletion(self):
        """When demand < supply, agents get exactly what they asked for."""
        from cenop.optimizations.kernels import eat_food_kernel

        food_grid = np.array([[100.0]], dtype=np.float32)
        x = np.array([0, 0], dtype=np.int32)
        y = np.array([0, 0], dtype=np.int32)
        fraction = np.array([0.2, 0.3], dtype=np.float32)
        food_eaten = np.zeros(2, dtype=np.float32)
        demand_grid = np.zeros_like(food_grid)

        eat_food_kernel(food_grid, x, y, fraction, food_eaten, 0.01, demand_grid)

        # Total demand: 20 + 30 = 50, available: 100, no competition
        assert food_eaten[0] == pytest.approx(20.0, abs=0.01)
        assert food_eaten[1] == pytest.approx(30.0, abs=0.01)
        assert food_grid[0, 0] == pytest.approx(50.0, abs=0.01)

    def test_order_independence(self):
        """Result should be the same regardless of agent ordering."""
        from cenop.optimizations.kernels import eat_food_kernel

        # Forward order
        grid1 = np.array([[100.0]], dtype=np.float32)
        x = np.array([0, 0], dtype=np.int32)
        y = np.array([0, 0], dtype=np.int32)
        frac_fwd = np.array([0.7, 0.5], dtype=np.float32)
        eaten_fwd = np.zeros(2, dtype=np.float32)
        dg = np.zeros_like(grid1)
        eat_food_kernel(grid1, x, y, frac_fwd, eaten_fwd, 0.01, dg)

        # Reverse order
        grid2 = np.array([[100.0]], dtype=np.float32)
        frac_rev = np.array([0.5, 0.7], dtype=np.float32)
        eaten_rev = np.zeros(2, dtype=np.float32)
        dg2 = np.zeros_like(grid2)
        eat_food_kernel(grid2, x, y, frac_rev, eaten_rev, 0.01, dg2)

        # Agent asking for 0.7 should get same amount in both orderings
        assert eaten_fwd[0] == pytest.approx(eaten_rev[1], abs=0.01), \
            "Proportional sharing should be order-independent"
        assert grid1[0, 0] == pytest.approx(grid2[0, 0], abs=0.01)

    def test_minimum_food_floor(self):
        """Food should never drop below the minimum floor."""
        from cenop.optimizations.kernels import eat_food_kernel

        food_grid = np.array([[0.05]], dtype=np.float32)
        x = np.array([0], dtype=np.int32)
        y = np.array([0], dtype=np.int32)
        fraction = np.array([0.99], dtype=np.float32)
        food_eaten = np.zeros(1, dtype=np.float32)
        demand_grid = np.zeros_like(food_grid)

        eat_food_kernel(food_grid, x, y, fraction, food_eaten, 0.01, demand_grid)

        assert food_grid[0, 0] >= 0.01


class TestDEPONSBmrCostKernel:
    """Test DEPONS BMR cost kernel matches Python implementation."""

    def test_basic_cost(self):
        """BMR + activity + disturbance should sum correctly."""
        from cenop.optimizations.kernels import depons_bmr_cost_kernel

        n = 2
        speed = np.array([0.5, 1.0], dtype=np.float32)
        scaling = np.array([1.0, 1.2], dtype=np.float32)
        is_lactating = np.array([False, True])
        is_disturbed = np.array([False, True])
        deter_magnitude = np.array([0.0, 0.5], dtype=np.float32)
        mask = np.array([True, True])
        out_cost = np.zeros(n, dtype=np.float32)

        depons_bmr_cost_kernel(
            speed, scaling, is_lactating, is_disturbed, deter_magnitude,
            mask, out_cost, 4.5, 1.4,
        )

        assert out_cost[0] > 0
        assert out_cost[1] > out_cost[0], "Lactating + disturbed should cost more"

    def test_mask_skips_inactive(self):
        """Masked agents should have zero cost."""
        from cenop.optimizations.kernels import depons_bmr_cost_kernel

        n = 2
        speed = np.array([1.0, 1.0], dtype=np.float32)
        scaling = np.ones(n, dtype=np.float32)
        is_lactating = np.array([False, False])
        is_disturbed = np.array([False, False])
        deter_magnitude = np.zeros(n, dtype=np.float32)
        mask = np.array([True, False])
        out_cost = np.zeros(n, dtype=np.float32)

        depons_bmr_cost_kernel(
            speed, scaling, is_lactating, is_disturbed, deter_magnitude,
            mask, out_cost, 4.5, 1.4,
        )

        assert out_cost[0] > 0
        assert out_cost[1] == 0.0

    def test_equivalence_with_python(self):
        """Numba kernel must match DEPONSEnergyModule.compute_bmr_cost output."""
        from cenop.optimizations.kernels import depons_bmr_cost_kernel
        from cenop.physiology.energy_budget import DEPONSEnergyModule, EnergyState, EnergyContext
        from cenop.parameters.simulation_params import SimulationParameters

        n = 50
        rng = np.random.default_rng(42)
        params = SimulationParameters()
        module = DEPONSEnergyModule(params)

        speed = rng.uniform(0, 2, n).astype(np.float32)
        is_lact = rng.choice([True, False], n)
        is_dist = rng.choice([True, False], n)
        deter_mag = (rng.uniform(0, 1, n) * is_dist).astype(np.float32)
        mask = np.ones(n, dtype=bool)

        # Python path
        state = EnergyState.create(n)
        context = EnergyContext(
            food_available=np.zeros(n, dtype=np.float32),
            food_quality=np.ones(n, dtype=np.float32),
            current_speed=speed,
            behavioral_state=np.ones(n, dtype=np.int32),
            water_temperature=np.full(n, 10.0, dtype=np.float32),
            current_month=6,
            is_disturbed=is_dist,
            deterrence_magnitude=deter_mag,
            is_lactating=is_lact,
            is_pregnant=np.zeros(n, dtype=bool),
        )
        py_cost = module.compute_bmr_cost(state, context, mask)

        # Get seasonal scaling that Python used
        scaling = np.full(n, float(module._get_seasonal_scaling(6, n)), dtype=np.float32)

        # Numba path
        nb_cost = np.zeros(n, dtype=np.float32)
        depons_bmr_cost_kernel(
            speed, scaling, is_lact, is_dist, deter_mag,
            mask, nb_cost, module.e_use_per_30_min, module.e_lact,
        )

        np.testing.assert_allclose(nb_cost[mask], py_cost[mask], atol=1e-6)


class TestSocialAccumulateKernel:
    """Test fused social vector accumulation kernel."""

    def test_basic_accumulation(self):
        """Two pairs of agents should accumulate correct social vectors."""
        from cenop.optimizations.kernels import social_accumulate_kernel

        count = 4
        # Pair (0,1) and (2,3)
        idx_i = np.array([0, 2], dtype=np.int64)
        idx_j = np.array([1, 3], dtype=np.int64)
        dx_ij = np.array([1.0, 0.0], dtype=np.float64)
        dy_ij = np.array([0.0, 1.0], dtype=np.float64)
        dist = np.array([1.0, 1.0], dtype=np.float64)  # already has eps added
        p_i = np.array([0.8, 0.6], dtype=np.float64)
        p_j = np.array([0.7, 0.5], dtype=np.float64)

        ux_total = np.zeros(count, dtype=np.float64)
        uy_total = np.zeros(count, dtype=np.float64)
        sw_total = np.zeros(count, dtype=np.float64)

        social_accumulate_kernel(idx_i, idx_j, dx_ij, dy_ij, dist, p_i, p_j,
                                ux_total, uy_total, sw_total)

        # Agent 0 hears agent 1: ux += (1/1)*0.8 = 0.8
        assert ux_total[0] == pytest.approx(0.8, abs=0.01)
        # Agent 1 hears agent 0: ux += (-1/1)*0.7 = -0.7
        assert ux_total[1] == pytest.approx(-0.7, abs=0.01)
        # Agent 2 hears agent 3: uy += (1/1)*0.6 = 0.6
        assert uy_total[2] == pytest.approx(0.6, abs=0.01)
        # sw_total: each agent accumulates the probability
        assert sw_total[0] == pytest.approx(0.8, abs=0.01)
        assert sw_total[1] == pytest.approx(0.7, abs=0.01)
        assert sw_total[2] == pytest.approx(0.6, abs=0.01)
        assert sw_total[3] == pytest.approx(0.5, abs=0.01)

    def test_multiple_neighbors(self):
        """Agent with multiple neighbors should accumulate all contributions."""
        from cenop.optimizations.kernels import social_accumulate_kernel

        count = 3
        # Agent 0 is paired with both agent 1 and agent 2
        idx_i = np.array([0, 0], dtype=np.int64)
        idx_j = np.array([1, 2], dtype=np.int64)
        dx_ij = np.array([1.0, 0.0], dtype=np.float64)
        dy_ij = np.array([0.0, 1.0], dtype=np.float64)
        dist = np.array([1.0, 1.0], dtype=np.float64)
        p_i = np.array([1.0, 1.0], dtype=np.float64)
        p_j = np.array([1.0, 1.0], dtype=np.float64)

        ux_total = np.zeros(count, dtype=np.float64)
        uy_total = np.zeros(count, dtype=np.float64)
        sw_total = np.zeros(count, dtype=np.float64)

        social_accumulate_kernel(idx_i, idx_j, dx_ij, dy_ij, dist, p_i, p_j,
                                ux_total, uy_total, sw_total)

        # Agent 0: ux from pair0=(1/1)*1=1, ux from pair1=(0/1)*1=0 → total=1
        assert ux_total[0] == pytest.approx(1.0, abs=0.01)
        # Agent 0: uy from pair0=(0/1)*1=0, uy from pair1=(1/1)*1=1 → total=1
        assert uy_total[0] == pytest.approx(1.0, abs=0.01)
        # Agent 0 heard 2 neighbors
        assert sw_total[0] == pytest.approx(2.0, abs=0.01)

    def test_equivalence_with_numpy(self):
        """Kernel should match the NumPy unit-vector + accumulation approach."""
        from cenop.optimizations.kernels import social_accumulate_kernel

        rng = np.random.default_rng(99)
        count = 100
        n_pairs = 200
        idx_i = rng.integers(0, count, n_pairs).astype(np.int64)
        idx_j = rng.integers(0, count, n_pairs).astype(np.int64)
        dx_ij = rng.uniform(-10, 10, n_pairs)
        dy_ij = rng.uniform(-10, 10, n_pairs)
        dist = np.hypot(dx_ij, dy_ij) + 1e-6
        p_i = rng.uniform(0, 1, n_pairs)
        p_j = rng.uniform(0, 1, n_pairs)

        # Kernel path
        ux_k = np.zeros(count, dtype=np.float64)
        uy_k = np.zeros(count, dtype=np.float64)
        sw_k = np.zeros(count, dtype=np.float64)
        social_accumulate_kernel(idx_i, idx_j, dx_ij, dy_ij, dist, p_i, p_j,
                                ux_k, uy_k, sw_k)

        # NumPy reference path (matching existing population.py logic)
        ux_ij = dx_ij / dist
        uy_ij = dy_ij / dist
        ux_contrib_i = ux_ij * p_i
        uy_contrib_i = uy_ij * p_i
        ux_contrib_j = -ux_ij * p_j
        uy_contrib_j = -uy_ij * p_j

        ux_ref = np.zeros(count, dtype=np.float64)
        uy_ref = np.zeros(count, dtype=np.float64)
        sw_ref = np.zeros(count, dtype=np.float64)
        np.add.at(ux_ref, idx_i, ux_contrib_i)
        np.add.at(ux_ref, idx_j, ux_contrib_j)
        np.add.at(uy_ref, idx_i, uy_contrib_i)
        np.add.at(uy_ref, idx_j, uy_contrib_j)
        np.add.at(sw_ref, idx_i, p_i)
        np.add.at(sw_ref, idx_j, p_j)

        np.testing.assert_allclose(ux_k, ux_ref, atol=1e-10)
        np.testing.assert_allclose(uy_k, uy_ref, atol=1e-10)
        np.testing.assert_allclose(sw_k, sw_ref, atol=1e-10)


class TestParallelEquivalence:
    """Verify parallel kernels produce same results across runs."""

    def test_reflect_parallel_deterministic(self):
        """Parallel reflect should produce consistent results."""
        from cenop.optimizations.kernels import reflect_boundaries_kernel

        rng = np.random.default_rng(123)
        n = 1000
        x1 = rng.uniform(-10, 30, n).astype(np.float64)
        y1 = rng.uniform(-10, 30, n).astype(np.float64)
        dx1 = rng.uniform(-5, 5, n).astype(np.float64)
        dy1 = rng.uniform(-5, 5, n).astype(np.float64)
        mask = rng.choice([True, False], n)

        x2, y2, dx2, dy2 = x1.copy(), y1.copy(), dx1.copy(), dy1.copy()

        reflect_boundaries_kernel(x1, y1, dx1, dy1, 20, 20, mask)
        reflect_boundaries_kernel(x2, y2, dx2, dy2, 20, 20, mask)

        np.testing.assert_allclose(x1, x2, atol=1e-10)
        np.testing.assert_allclose(y1, y2, atol=1e-10)

    def test_turn_position_parallel_deterministic(self):
        """Parallel turn_position should produce consistent results."""
        from cenop.optimizations.kernels import turn_position_kernel

        rng = np.random.default_rng(456)
        n = 1000
        x = rng.uniform(0, 19, n).astype(np.float64)
        y = rng.uniform(0, 19, n).astype(np.float64)
        heading = rng.uniform(0, 360, n).astype(np.float64)
        step_dist = rng.uniform(0.1, 5, n).astype(np.float64)

        ox1, oy1, oh1 = np.zeros(n, np.float64), np.zeros(n, np.float64), np.zeros(n, np.float64)
        ox2, oy2, oh2 = np.zeros(n, np.float64), np.zeros(n, np.float64), np.zeros(n, np.float64)
        oxi1, oyi1 = np.zeros(n, np.int32), np.zeros(n, np.int32)
        oxi2, oyi2 = np.zeros(n, np.int32), np.zeros(n, np.int32)

        turn_position_kernel(x, y, heading, step_dist, 45.0, 20, 20, ox1, oy1, oh1, oxi1, oyi1)
        turn_position_kernel(x, y, heading, step_dist, 45.0, 20, 20, ox2, oy2, oh2, oxi2, oyi2)

        np.testing.assert_allclose(ox1, ox2, atol=1e-10)
        np.testing.assert_allclose(oy1, oy2, atol=1e-10)
        np.testing.assert_array_equal(oxi1, oxi2)
        np.testing.assert_array_equal(oyi1, oyi2)

    def test_bmr_cost_parallel_deterministic(self):
        """Parallel BMR cost should produce consistent results."""
        from cenop.optimizations.kernels import depons_bmr_cost_kernel

        rng = np.random.default_rng(789)
        n = 1000
        speed = rng.uniform(0, 2, n).astype(np.float32)
        scaling = rng.uniform(0.8, 1.3, n).astype(np.float32)
        is_lact = rng.choice([True, False], n)
        is_dist = rng.choice([True, False], n)
        deter_mag = rng.uniform(0, 1, n).astype(np.float32)
        mask = rng.choice([True, False], n, p=[0.9, 0.1])

        out1 = np.zeros(n, dtype=np.float32)
        out2 = np.zeros(n, dtype=np.float32)

        depons_bmr_cost_kernel(speed, scaling, is_lact, is_dist, deter_mag, mask, out1, 4.5, 1.4)
        depons_bmr_cost_kernel(speed, scaling, is_lact, is_dist, deter_mag, mask, out2, 4.5, 1.4)

        np.testing.assert_allclose(out1, out2, atol=1e-10)


class TestLandAvoidanceKernel:
    """Tests for fused land avoidance kernel."""

    def test_all_agents_find_water_at_first_angle(self):
        """When 40 deg turn leads to water, all agents should resolve."""
        from cenop.optimizations.kernels import land_avoidance_kernel

        n = 3
        x = np.array([25.0, 30.0, 35.0], dtype=np.float64)
        y = np.array([25.0, 30.0, 35.0], dtype=np.float64)
        heading = np.array([0.0, 90.0, 180.0], dtype=np.float64)
        step_dist = np.array([2.0, 2.0, 2.0], dtype=np.float64)
        depth_grid = np.full((50, 50), 20.0, dtype=np.float32)
        min_depth = 1.0
        base_angles = np.array([40.0, 70.0, 120.0], dtype=np.float64)
        jitter = np.array([5.0, 5.0, 5.0], dtype=np.float64)
        out_x = np.zeros(n, dtype=np.float64)
        out_y = np.zeros(n, dtype=np.float64)
        out_heading = np.zeros(n, dtype=np.float64)
        resolved = np.zeros(n, dtype=np.bool_)
        land_avoidance_kernel(
            x, y, heading, step_dist, depth_grid, min_depth,
            base_angles, jitter, out_x, out_y, out_heading, resolved,
        )
        assert np.all(resolved)
        assert np.all(out_x >= 0) and np.all(out_x < 50)
        assert np.all(out_y >= 0) and np.all(out_y < 50)

    def test_picks_deeper_when_both_valid(self):
        """When both left and right turns lead to water, pick deeper."""
        from cenop.optimizations.kernels import land_avoidance_kernel

        depth_grid = np.full((100, 100), 5.0, dtype=np.float32)
        # Agent at (50, 50), heading 0, step 10, angle 45 (no jitter)
        # Right turn: heading=45, dx=sin(45)*10=7.07, dy=cos(45)*10=7.07
        #   -> (57.07, 57.07) -> int (57, 57)
        depth_grid[57, 57] = 30.0  # Right: deep
        # Left turn: heading=315, dx=sin(315)*10=-7.07, dy=cos(315)*10=7.07
        #   -> (42.93, 57.07) -> int (42, 57) but grid is [row,col]=[57,42]
        depth_grid[57, 42] = 10.0  # Left: shallower but valid
        n = 1
        x = np.array([50.0], dtype=np.float64)
        y = np.array([50.0], dtype=np.float64)
        heading = np.array([0.0], dtype=np.float64)
        step_dist = np.array([10.0], dtype=np.float64)
        base_angles = np.array([45.0, 70.0, 120.0], dtype=np.float64)
        jitter = np.zeros(3, dtype=np.float64)
        out_x = np.zeros(n, dtype=np.float64)
        out_y = np.zeros(n, dtype=np.float64)
        out_heading = np.zeros(n, dtype=np.float64)
        resolved = np.zeros(n, dtype=np.bool_)
        land_avoidance_kernel(
            x, y, heading, step_dist, depth_grid, 1.0,
            base_angles, jitter, out_x, out_y, out_heading, resolved,
        )
        assert resolved[0]
        # Should pick right (deeper: 30 > 10)
        assert out_heading[0] == pytest.approx(45.0, abs=0.1)

    def test_unresolved_when_all_land(self):
        """When all 6 directions are land, agent should be unresolved."""
        from cenop.optimizations.kernels import land_avoidance_kernel

        depth_grid = np.full((50, 50), 0.0, dtype=np.float32)
        n = 1
        x = np.array([25.0], dtype=np.float64)
        y = np.array([25.0], dtype=np.float64)
        heading = np.array([90.0], dtype=np.float64)
        step_dist = np.array([3.0], dtype=np.float64)
        base_angles = np.array([40.0, 70.0, 120.0], dtype=np.float64)
        jitter = np.zeros(3, dtype=np.float64)
        out_x = np.zeros(n, dtype=np.float64)
        out_y = np.zeros(n, dtype=np.float64)
        out_heading = np.zeros(n, dtype=np.float64)
        resolved = np.zeros(n, dtype=np.bool_)
        land_avoidance_kernel(
            x, y, heading, step_dist, depth_grid, 1.0,
            base_angles, jitter, out_x, out_y, out_heading, resolved,
        )
        assert not resolved[0]

    def test_tries_wider_angle_on_failure(self):
        """When 40 deg fails but 70 deg succeeds, uses 70 deg."""
        from cenop.optimizations.kernels import land_avoidance_kernel

        depth_grid = np.full((100, 100), 0.0, dtype=np.float32)
        # Agent at (50, 50), heading 0, step 10
        # 40 deg right: heading=40, dx=sin(40)*10=6.43, dy=cos(40)*10=7.66
        #   -> (56.43, 57.66) -> int (57, 56) — leave as land (0)
        # 40 deg left: heading=320, dx=sin(320)*10=-6.43, dy=cos(320)*10=7.66
        #   -> (43.57, 57.66) -> int (57, 43) — leave as land (0)
        # 70 deg right: heading=70, dx=sin(70)*10=9.40, dy=cos(70)*10=3.42
        #   -> (59.40, 53.42) -> int (53, 59)
        depth_grid[53, 59] = 20.0  # Only valid at ~70 deg right
        n = 1
        x = np.array([50.0], dtype=np.float64)
        y = np.array([50.0], dtype=np.float64)
        heading = np.array([0.0], dtype=np.float64)
        step_dist = np.array([10.0], dtype=np.float64)
        base_angles = np.array([40.0, 70.0, 120.0], dtype=np.float64)
        jitter = np.zeros(3, dtype=np.float64)
        out_x = np.zeros(n, dtype=np.float64)
        out_y = np.zeros(n, dtype=np.float64)
        out_heading = np.zeros(n, dtype=np.float64)
        resolved = np.zeros(n, dtype=np.bool_)
        land_avoidance_kernel(
            x, y, heading, step_dist, depth_grid, 1.0,
            base_angles, jitter, out_x, out_y, out_heading, resolved,
        )
        assert resolved[0]
        assert out_heading[0] == pytest.approx(70.0, abs=1.0)


class TestEatFoodKernelV2:
    """Test eat_food_kernel_v2 with inline fraction from energy."""

    def test_matches_original_kernel_single_agent(self):
        """V2 produces same result as v1 for single agent."""
        from cenop.optimizations.kernels import eat_food_kernel, eat_food_kernel_v2

        food_v1 = np.array([[100.0, 50.0], [75.0, 25.0]], dtype=np.float32)
        food_v2 = food_v1.copy()
        xi = np.array([0], dtype=np.int32)
        yi = np.array([0], dtype=np.int32)
        energy = np.array([10.0], dtype=np.float32)
        fraction = np.clip((20.0 - energy) / 10.0, 0.0, 0.99).astype(np.float32)
        eaten_v1 = np.zeros(1, dtype=np.float32)
        eaten_v2 = np.zeros(1, dtype=np.float32)
        dg1 = np.zeros((2, 2), dtype=np.float32)
        dg2 = np.zeros((2, 2), dtype=np.float32)
        eat_food_kernel(food_v1, xi, yi, fraction, eaten_v1, 0.01, dg1)
        eat_food_kernel_v2(food_v2, xi, yi, energy, eaten_v2, 0.01, dg2)
        np.testing.assert_allclose(eaten_v1, eaten_v2, rtol=1e-5)
        np.testing.assert_allclose(food_v1, food_v2, rtol=1e-5)

    def test_matches_original_competing_agents(self):
        """V2 proportional sharing matches v1."""
        from cenop.optimizations.kernels import eat_food_kernel, eat_food_kernel_v2

        food_v1 = np.full((3, 3), 50.0, dtype=np.float32)
        food_v2 = food_v1.copy()
        xi = np.array([1, 1, 1], dtype=np.int32)
        yi = np.array([1, 1, 1], dtype=np.int32)
        energy = np.array([5.0, 10.0, 18.0], dtype=np.float32)
        fracs = np.clip((20.0 - energy) / 10.0, 0.0, 0.99).astype(np.float32)
        eaten_v1 = np.zeros(3, dtype=np.float32)
        eaten_v2 = np.zeros(3, dtype=np.float32)
        dg1 = np.zeros((3, 3), dtype=np.float32)
        dg2 = np.zeros((3, 3), dtype=np.float32)
        eat_food_kernel(food_v1, xi, yi, fracs, eaten_v1, 0.01, dg1)
        eat_food_kernel_v2(food_v2, xi, yi, energy, eaten_v2, 0.01, dg2)
        np.testing.assert_allclose(eaten_v1, eaten_v2, rtol=1e-5)

    def test_high_energy_eats_nothing(self):
        """Agent with energy >= 20 should eat nothing."""
        from cenop.optimizations.kernels import eat_food_kernel_v2

        food = np.full((2, 2), 100.0, dtype=np.float32)
        eaten = np.zeros(1, dtype=np.float32)
        dg = np.zeros((2, 2), dtype=np.float32)
        eat_food_kernel_v2(
            food,
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([20.0], dtype=np.float32),
            eaten,
            0.01,
            dg,
        )
        assert eaten[0] == 0.0

    def test_zero_energy_eats_max(self):
        """Agent with energy 0 should eat at max fraction (0.99)."""
        from cenop.optimizations.kernels import eat_food_kernel_v2

        food = np.full((2, 2), 100.0, dtype=np.float32)
        eaten = np.zeros(1, dtype=np.float32)
        dg = np.zeros((2, 2), dtype=np.float32)
        eat_food_kernel_v2(
            food,
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0.0], dtype=np.float32),
            eaten,
            0.01,
            dg,
        )
        assert eaten[0] == pytest.approx(100.0 * 0.99, rel=1e-5)
