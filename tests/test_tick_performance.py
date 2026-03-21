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
