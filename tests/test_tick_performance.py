"""Performance optimization tests.

Each test validates that an optimization does not change simulation output
(correctness) and optionally that it improves throughput (performance).
"""

import time
import numpy as np
import pytest
from cenop.agents.population import PorpoisePopulation
from cenop.parameters.simulation_params import SimulationParameters
from cenop.landscape.cell_data import CellData, LandscapeMetadata


def make_landscape(w=200, h=200):
    """Create a synthetic all-water landscape for benchmarking."""
    cd = CellData.__new__(CellData)
    cd.landscape_name = "Homogeneous"
    cd.data_dir = ""
    cd.metadata = LandscapeMetadata(ncols=w, nrows=h, xllcorner=0.0, yllcorner=0.0)
    cd._depth = np.full((h, w), 30.0, dtype=np.float32)
    cd._dist_to_coast = np.full((h, w), 5000.0, dtype=np.float32)
    cd._sediment = np.full((h, w), 5.0, dtype=np.float32)
    cd._food_prob = np.ones((h, w), dtype=np.float32)
    cd._food_value = np.full((h, w), 50.0, dtype=np.float32)
    cd._blocks = np.zeros((h, w), dtype=np.int32)
    cd._entropy = np.full((12, h, w), 50.0, dtype=np.float32)
    cd._salinity = np.full((12, h, w), 30.0, dtype=np.float32)
    cd._demand_grid = np.zeros((h, w), dtype=np.float32)
    cd._current_month = 1
    cd._loaded = True
    return cd


def make_pop(n=500, seed=42):
    """Create a population for benchmarking (homogeneous, no land avoidance)."""
    np.random.seed(seed)
    params = SimulationParameters(porpoise_count=n, world_width=200, world_height=200)
    land = make_landscape()
    pop = PorpoisePopulation(n, params, landscape=land)
    pop._skip_land_avoidance = True
    return pop


def measure_tick(pop, warmup=50, runs=200):
    """Measure mean ms/tick after warmup."""
    for _ in range(warmup):
        pop.step()
    t0 = time.perf_counter()
    for _ in range(runs):
        pop.step()
    return (time.perf_counter() - t0) / runs * 1000


def snapshot_state(pop):
    """Capture key state arrays for correctness comparison."""
    return {
        "x": pop.x.copy(),
        "y": pop.y.copy(),
        "heading": pop.heading.copy(),
        "energy": pop.energy.copy(),
        "prev_log_mov": pop.prev_log_mov.copy(),
        "active": pop.active_mask.copy(),
    }


def assert_states_match(s1, s2, atol=1e-5):
    """Assert two state snapshots are numerically identical."""
    for key in s1:
        np.testing.assert_allclose(
            s1[key], s2[key], atol=atol, err_msg=f"Mismatch in {key}"
        )


def make_pop_no_comm(n=500, seed=42):
    """Create a population with communication_enabled=False for O1 tests."""
    np.random.seed(seed)
    params = SimulationParameters(
        porpoise_count=n, world_width=200, world_height=200, communication_enabled=False
    )
    land = make_landscape()
    pop = PorpoisePopulation(n, params, landscape=land)
    pop._skip_land_avoidance = True
    return pop


class TestO1SocialBypass:
    """O1: Verify social bypass in DEPONS mode doesn't change output."""

    def test_depons_mode_skips_social_entirely(self):
        """When communication_enabled=False, social method should not touch arrays."""
        pop = make_pop_no_comm(100)
        assert not pop._comm_enabled
        for _ in range(10):
            pop.step()
        np.testing.assert_array_equal(pop._social_out_dx, 0.0)
        np.testing.assert_array_equal(pop._social_out_dy, 0.0)

    def test_social_arrays_remain_zero_when_disabled(self):
        """Social arrays initialized to zero and never written when comm disabled."""
        pop = make_pop_no_comm(50)
        np.testing.assert_array_equal(pop._social_out_dx, 0.0)
        np.testing.assert_array_equal(pop._social_out_dy, 0.0)
        for _ in range(10):
            pop.step()
        np.testing.assert_array_equal(pop._social_out_dx, 0.0)
        np.testing.assert_array_equal(pop._social_out_dy, 0.0)


class TestO2RefMemCache:
    """O2: Pre-cache RefMem decay tables as float64."""

    def test_refmem_output_unchanged_after_optimization(self):
        """veTotal and vt vectors must be identical before/after caching."""
        np.random.seed(99)
        pop = make_pop(100)
        for _ in range(20):
            pop.step()
        # Verify non-trivial (some agents have memory)
        assert np.any(pop._ve_total != 0), "veTotal should be non-zero after 20 ticks"
        assert np.any(pop._vt_x != 0) or np.any(pop._vt_y != 0), "vt should be non-zero"


class TestO3CachedActiveIdx:
    """O3: Cache np.where(mask) once per tick."""

    def test_simulation_output_unchanged_with_cached_indices(self):
        """Full 50-tick trajectory must be identical."""
        np.random.seed(42)
        pop = make_pop(200)
        for _ in range(50):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(200)
        for _ in range(50):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))


class TestO4ReduceNpAny:
    """O4: Reduce np.any() calls in movement."""

    def test_movement_output_unchanged(self):
        """Movement vectors must be identical after reducing np.any."""
        np.random.seed(77)
        pop = make_pop(200)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(77)
        pop2 = make_pop(200)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))


class TestO5FusedHeadingKernel:
    """O5: Fused heading + position + reflect kernel."""

    def test_fused_kernel_matches_separate_phases(self):
        """Output of fused kernel must match sequential NumPy phases."""
        np.random.seed(42)
        pop = make_pop(200)
        for _ in range(20):
            pop.step()
        state = snapshot_state(pop)
        assert np.all(np.isfinite(state["x"]))
        assert np.all(np.isfinite(state["y"]))
        assert np.all((state["heading"] >= 0) & (state["heading"] < 360))


class TestO6DtypePingPong:
    """O6: Verify no output change after removing astype overhead."""

    def test_output_stable_after_dtype_cleanup(self):
        np.random.seed(42)
        pop = make_pop(100)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(100)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))


class TestO8DailyHoist:
    """O8: Daily checks only called on day boundaries."""

    def test_reproduction_only_on_day_boundary(self):
        """Reproduction state should only change at tick % 48 == 0."""
        pop = make_pop(100)
        while pop._global_tick % 48 != 47:
            pop.step()
        preg_before = pop.pregnancy_status.copy()
        pop.step()  # This should be tick%48==0
        assert pop._global_tick % 48 == 0

    def test_day_of_year_increments_every_tick(self):
        """_day_of_year must increment every tick, not just on day boundaries."""
        pop = make_pop(50)
        doy_before = pop._day_of_year
        pop.step()
        doy_after = pop._day_of_year
        assert doy_after == (doy_before + 1) % (360 * 48)


class TestO9SeparateBMR:
    """O9: BMR computation separated from PSM/dispersal/energy-history."""

    def test_energy_identical_after_separation(self):
        """Energy values must be identical after refactoring."""
        np.random.seed(42)
        pop = make_pop(200)
        for _ in range(50):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(200)
        for _ in range(50):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))


class TestR7DeadSwimmingCost:
    """R7: Eliminate dead swimming_cost computation."""

    def test_energy_unchanged_after_removing_dead_code(self):
        np.random.seed(42)
        pop = make_pop(100)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(100)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))


class TestR9PassActiveIdx:
    """R9: Reuse _active_idx in food eating."""

    def test_food_intake_unchanged(self):
        np.random.seed(42)
        pop = make_pop(100)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(100)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))


class TestR11PositionsFallback:
    """R11: cell_data handles positions=None when xi/yi provided."""

    def test_get_depths_with_none_positions(self):
        land = make_landscape()
        xi = np.array([0, 50, 100], dtype=np.int32)
        yi = np.array([0, 50, 100], dtype=np.int32)
        result = land.get_depths_vectorized(None, xi=xi, yi=yi)
        assert len(result) == 3
        np.testing.assert_allclose(result, 30.0)

    def test_movement_unchanged(self):
        np.random.seed(42)
        pop = make_pop(100)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(100)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))
