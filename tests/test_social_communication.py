"""Tests for social communication / social call system."""

import numpy as np
import pytest


@pytest.fixture
def make_params():
    """Factory fixture for SimulationParameters with communication settings."""
    from cenop.parameters import SimulationParameters

    def _make(
        comm_enabled=True,
        comm_range_km=5.0,
        source_level=130.0,
        threshold=80.0,
        slope=0.1,
        social_weight=0.3,
        count=10,
        **kw,
    ):
        return SimulationParameters(
            porpoise_count=count,
            sim_years=1,
            landscape="Homogeneous",
            communication_enabled=comm_enabled,
            communication_range_km=comm_range_km,
            communication_source_level=source_level,
            communication_threshold=threshold,
            communication_response_slope=slope,
            social_weight=social_weight,
            **kw,
        )

    return _make


@pytest.fixture
def small_population(make_params):
    """A 10-agent population at known positions for deterministic testing."""
    from cenop.agents.population import PorpoisePopulation

    params = make_params(comm_enabled=True, comm_range_km=5.0)
    pop = PorpoisePopulation(10, params)
    # Two clusters: agents 0-2 near origin, agents 3-5 near (100,100), rest far away
    pop.x[:] = np.array([1, 2, 3, 100, 101, 102, 500, 501, 502, 503], dtype=np.float32)
    pop.y[:] = np.array([1, 1, 1, 100, 100, 100, 500, 500, 500, 500], dtype=np.float32)
    pop.active_mask[:] = True
    return pop


class TestNumbaHelpersImport:
    """Verify the optimizations package structure is correct."""

    def test_numba_helpers_import(self):
        from cenop.optimizations.numba_helpers import accumulate_social_totals

        assert callable(accumulate_social_totals)

    def test_numba_helpers_weighted_direction_sum(self):
        from cenop.optimizations.numba_helpers import weighted_direction_sum

        assert callable(weighted_direction_sum)

    def test_optimizations_init_still_works(self):
        from cenop.optimizations import accumulate_psm_updates, vectorized_distance

        assert callable(accumulate_psm_updates)
        assert callable(vectorized_distance)


class TestSocialVectorsDisabled:
    """Social vectors should be zero when communication is disabled."""

    def test_returns_zeros(self, make_params):
        from cenop.agents.population import PorpoisePopulation

        params = make_params(comm_enabled=False)
        pop = PorpoisePopulation(5, params)
        pop.x[:] = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        pop.y[:] = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        pop.active_mask[:] = True

        mask = pop.active_mask
        soc_dx, soc_dy = pop._compute_social_vectors(mask)
        np.testing.assert_array_equal(soc_dx, 0.0)
        np.testing.assert_array_equal(soc_dy, 0.0)


class TestSocialVectorsWithinRange:
    """Two nearby agents should produce nonzero attraction vectors."""

    def test_two_agents_attract(self, make_params):
        from cenop.agents.population import PorpoisePopulation

        # 5 km range = 12.5 cells; place agents 5 cells apart
        params = make_params(comm_enabled=True, comm_range_km=5.0, count=2)
        pop = PorpoisePopulation(2, params)
        pop.x[:] = np.array([10.0, 15.0], dtype=np.float32)
        pop.y[:] = np.array([10.0, 10.0], dtype=np.float32)
        pop.active_mask[:] = True
        # Give them a nonzero step length for social vector scaling
        pop.prev_log_mov[:] = 0.8

        mask = pop.active_mask
        soc_dx, soc_dy = pop._compute_social_vectors(mask)

        # Agent 0 should be attracted toward agent 1 (positive dx)
        assert soc_dx[0] > 0, f"Agent 0 should move toward agent 1, got dx={soc_dx[0]}"
        # Agent 1 should be attracted toward agent 0 (negative dx)
        assert soc_dx[1] < 0, f"Agent 1 should move toward agent 0, got dx={soc_dx[1]}"
        # Symmetric magnitudes
        np.testing.assert_allclose(abs(soc_dx[0]), abs(soc_dx[1]), rtol=0.1)


class TestSocialVectorsOutOfRange:
    """Agents beyond communication range should produce zero vectors."""

    def test_far_agents_no_attraction(self, make_params):
        from cenop.agents.population import PorpoisePopulation

        # 1 km range = 2.5 cells; place agents 100 cells apart
        params = make_params(comm_enabled=True, comm_range_km=1.0, count=2)
        pop = PorpoisePopulation(2, params)
        pop.x[:] = np.array([10.0, 110.0], dtype=np.float32)
        pop.y[:] = np.array([10.0, 10.0], dtype=np.float32)
        pop.active_mask[:] = True

        mask = pop.active_mask
        soc_dx, soc_dy = pop._compute_social_vectors(mask)

        np.testing.assert_allclose(soc_dx, 0.0, atol=1e-6)
        np.testing.assert_allclose(soc_dy, 0.0, atol=1e-6)


class TestSocialWeightScaling:
    """Social vector magnitude should scale with social_weight."""

    def test_higher_weight_larger_vector(self, make_params):
        from cenop.agents.population import PorpoisePopulation

        results = []
        for weight in [0.1, 0.5, 0.9]:
            params = make_params(
                comm_enabled=True, comm_range_km=5.0, social_weight=weight, count=2
            )
            pop = PorpoisePopulation(2, params)
            pop.x[:] = np.array([10.0, 15.0], dtype=np.float32)
            pop.y[:] = np.array([10.0, 10.0], dtype=np.float32)
            pop.active_mask[:] = True
            pop.prev_log_mov[:] = 0.8

            mask = pop.active_mask
            soc_dx, soc_dy = pop._compute_social_vectors(mask)
            mag = np.sqrt(soc_dx[0] ** 2 + soc_dy[0] ** 2)
            results.append(mag)

        # Magnitude should increase with weight
        assert (
            results[0] < results[1] < results[2]
        ), f"Expected increasing magnitudes with weight, got {results}"


class TestSingleAgent:
    """A single agent should produce no social vectors."""

    def test_single_agent_zero_vectors(self, make_params):
        from cenop.agents.population import PorpoisePopulation

        params = make_params(comm_enabled=True, count=1)
        pop = PorpoisePopulation(1, params)
        pop.x[:] = np.array([50.0], dtype=np.float32)
        pop.y[:] = np.array([50.0], dtype=np.float32)
        pop.active_mask[:] = True

        mask = pop.active_mask
        soc_dx, soc_dy = pop._compute_social_vectors(mask)

        np.testing.assert_array_equal(soc_dx, 0.0)
        np.testing.assert_array_equal(soc_dy, 0.0)


class TestAccumulateSocialTotals:
    """Test the accumulate_social_totals helper directly."""

    def test_pairwise_accumulation(self):
        from cenop.optimizations.numba_helpers import accumulate_social_totals

        count = 4
        # One pair: agent 0 and agent 2
        idx_i = np.array([0], dtype=np.int64)
        idx_j = np.array([2], dtype=np.int64)
        ux_i = np.array([0.5], dtype=np.float64)
        uy_i = np.array([0.3], dtype=np.float64)
        ux_j = np.array([-0.5], dtype=np.float64)
        uy_j = np.array([-0.3], dtype=np.float64)
        p_i = np.array([0.8], dtype=np.float64)
        p_j = np.array([0.8], dtype=np.float64)

        ux_total = np.zeros(count, dtype=np.float64)
        uy_total = np.zeros(count, dtype=np.float64)
        sw_total = np.zeros(count, dtype=np.float64)

        accumulate_social_totals(
            count, idx_i, idx_j, ux_i, uy_i, ux_j, uy_j, p_i, p_j, ux_total, uy_total, sw_total
        )

        # Agent 0 should have i's contribution
        assert ux_total[0] == pytest.approx(0.5)
        assert uy_total[0] == pytest.approx(0.3)
        assert sw_total[0] == pytest.approx(0.8)
        # Agent 2 should have j's contribution
        assert ux_total[2] == pytest.approx(-0.5)
        assert uy_total[2] == pytest.approx(-0.3)
        assert sw_total[2] == pytest.approx(0.8)
        # Agents 1 and 3 untouched
        assert ux_total[1] == 0.0
        assert ux_total[3] == 0.0


class TestCombineRLs:
    """Test the combine_rls helper used for ambient noise."""

    def test_single_source(self):
        from cenop.behavior.sound import combine_rls

        rl = np.array([100.0, 90.0, 80.0])
        result = combine_rls(rl)
        np.testing.assert_array_equal(result, rl)

    def test_multiple_sources_takes_max(self):
        from cenop.behavior.sound import combine_rls

        rls = [
            np.array([100.0, 80.0, 70.0]),
            np.array([90.0, 95.0, 60.0]),
        ]
        result = combine_rls(rls)
        np.testing.assert_array_equal(result, [100.0, 95.0, 70.0])


class TestSocialBufferPreallocation:
    """D2: Verify social kernel buffer pre-allocation."""

    def test_population_sized_buffers_exist(self, small_population):
        """Check _social_ux, _social_out_dx etc exist with correct dtype/shape."""
        pop = small_population
        n = pop.count
        # Population-sized float64 accumulators
        for attr in ("_social_ux", "_social_uy", "_social_sw"):
            buf = getattr(pop, attr)
            assert buf.shape == (n,), f"{attr} shape mismatch"
            assert buf.dtype == np.float64, f"{attr} dtype mismatch"
        # Population-sized float32 output buffers
        for attr in ("_social_out_dx", "_social_out_dy"):
            buf = getattr(pop, attr)
            assert buf.shape == (n,), f"{attr} shape mismatch"
            assert buf.dtype == np.float32, f"{attr} dtype mismatch"
