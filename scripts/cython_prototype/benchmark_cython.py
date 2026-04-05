"""
Benchmark Cython fused tick vs current Numba+NumPy tick.

Measures heading+position+reflect+food+BMR+mortality as a single
fused C loop vs the current multi-phase Python orchestration.
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from cenop.parameters.simulation_params import SimulationParameters
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import CellData, LandscapeMetadata


def make_landscape(w=200, h=200):
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


def benchmark_current(n=500, warmup=100, runs=500):
    """Benchmark current Numba+NumPy tick."""
    params = SimulationParameters(
        porpoise_count=n, world_width=200, world_height=200,
        communication_enabled=False,
    )
    land = make_landscape()
    pop = PorpoisePopulation(n, params, landscape=land)
    pop._skip_land_avoidance = True

    for _ in range(warmup):
        pop.step()

    t0 = time.perf_counter()
    for _ in range(runs):
        pop.step()
    ms = (time.perf_counter() - t0) / runs * 1000
    return ms


def benchmark_cython_phases(n=500, warmup=50, runs=500):
    """
    Benchmark ONLY the phases that Cython replaces:
    heading composition + position + reflect + food + BMR + mortality.

    CRW angle/step and RefMem are NOT included (they stay as Numba kernels).
    """
    try:
        from tick_core import fused_depons_tick
    except ImportError:
        print("ERROR: Cython module not built. Run:")
        print("  cd scripts/cython_prototype && python setup.py build_ext --inplace")
        return None

    # Setup arrays matching population state
    np.random.seed(42)
    x = np.random.uniform(1, 199, n).astype(np.float32)
    y = np.random.uniform(1, 199, n).astype(np.float32)
    heading = np.random.uniform(0, 360, n).astype(np.float32)
    prev_angle = np.random.normal(0, 10, n).astype(np.float64)
    prev_log_mov = np.random.uniform(0.5, 1.5, n).astype(np.float64)
    energy = np.random.uniform(5, 15, n).astype(np.float32)
    active_mask = np.ones(n, dtype=np.uint8)
    is_dispersing = np.zeros(n, dtype=np.uint8)
    with_calf = (np.random.random(n) > 0.8).astype(np.uint8)

    # Pre-computed CRW output (simulating Numba kernel results)
    pres_angle = np.random.normal(0, 20, n).astype(np.float64)
    log_mov = np.random.uniform(0.5, 1.5, n).astype(np.float64)

    # RefMem output
    ve_total = np.random.uniform(0, 1, n).astype(np.float32)
    vt_x = np.random.normal(0, 0.1, n).astype(np.float32)
    vt_y = np.random.normal(0, 0.1, n).astype(np.float32)

    # Landscape
    depth_grid = np.full((200, 200), 30.0, dtype=np.float32)
    food_grid = np.full((200, 200), 50.0, dtype=np.float32)

    # Warmup
    for _ in range(warmup):
        food_grid[:] = 50.0
        energy[:] = np.random.uniform(5, 15, n).astype(np.float32)
        active_mask[:] = 1
        fused_depons_tick(
            x, y, heading, prev_angle, prev_log_mov, energy,
            active_mask, is_dispersing, with_calf,
            pres_angle, log_mov, ve_total, vt_x, vt_y,
            depth_grid, food_grid,
            0.001, 4.0, 4.5, 1.4, 1.0, 0.4, 1.0, 200, 200,
        )

    # Timed runs
    times = []
    for _ in range(runs):
        food_grid[:] = 50.0
        energy[:] = np.random.uniform(5, 15, n).astype(np.float32)
        active_mask[:] = 1
        t0 = time.perf_counter()
        fused_depons_tick(
            x, y, heading, prev_angle, prev_log_mov, energy,
            active_mask, is_dispersing, with_calf,
            pres_angle, log_mov, ve_total, vt_x, vt_y,
            depth_grid, food_grid,
            0.001, 4.0, 4.5, 1.4, 1.0, 0.4, 1.0, 200, 200,
        )
        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times)


def benchmark_numpy_phases(n=500, warmup=50, runs=500):
    """
    Benchmark the SAME phases using current NumPy code (for fair comparison).
    heading composition + position + reflect + food + BMR + mortality.
    """
    np.random.seed(42)
    x = np.random.uniform(1, 199, n).astype(np.float32)
    y = np.random.uniform(1, 199, n).astype(np.float32)
    heading = np.random.uniform(0, 360, n).astype(np.float32)
    prev_angle = np.random.normal(0, 10, n).astype(np.float64)
    prev_log_mov = np.random.uniform(0.5, 1.5, n).astype(np.float64)
    energy = np.random.uniform(5, 15, n).astype(np.float32)
    mask = np.ones(n, dtype=bool)
    with_calf = np.random.random(n) > 0.8

    pres_angle = np.random.normal(0, 20, n).astype(np.float64)
    log_mov = np.random.uniform(0.5, 1.5, n).astype(np.float64)
    ve_total = np.random.uniform(0, 1, n).astype(np.float32)
    vt_x = np.random.normal(0, 0.1, n).astype(np.float32)
    vt_y = np.random.normal(0, 0.1, n).astype(np.float32)

    food_grid = np.full((200, 200), 50.0, dtype=np.float32)

    # Warmup
    for _ in range(warmup):
        pass  # NumPy is already warm

    times = []
    for _ in range(runs):
        food_grid[:] = 50.0
        energy_t = np.random.uniform(5, 15, n).astype(np.float32)
        t0 = time.perf_counter()

        # Heading composition
        pres_mov = np.power(10.0, log_mov)
        h = (heading + pres_angle) % 360
        rads = np.radians(h)
        dx_crw = np.sin(rads)
        dy_crw = np.cos(rads)
        crw_c = 0.001 + pres_mov * ve_total
        total_dx = dx_crw * crw_c + vt_x
        total_dy = dy_crw * crw_c + vt_y
        new_h = np.degrees(np.arctan2(total_dx, total_dy)) % 360
        step = pres_mov / 4.0

        # Position + reflect
        rads2 = np.radians(new_h)
        ddx = np.sin(rads2) * step
        ddy = np.cos(rads2) * step
        nx = x + ddx
        ny = y + ddy
        neg_x = nx < 0
        nx[neg_x] = -nx[neg_x]
        over_x = nx > 199
        nx[over_x] = 2 * 199 - nx[over_x]
        np.clip(nx, 0, 199, out=nx)
        neg_y = ny < 0
        ny[neg_y] = -ny[neg_y]
        over_y = ny > 199
        ny[over_y] = 2 * 199 - ny[over_y]
        np.clip(ny, 0, 199, out=ny)

        # Food intake
        xi = np.clip(nx.astype(np.int32), 0, 199)
        yi = np.clip(ny.astype(np.int32), 0, 199)
        fract = np.clip((20.0 - energy_t) / 10.0, 0.0, 0.99)
        eaten = food_grid[yi, xi] * fract
        energy_t += eaten

        # BMR
        scaling = np.ones(n, dtype=np.float32)
        scaling[with_calf] *= 1.4
        bmr = 0.001 * scaling * 4.5
        energy_t -= bmr

        # Mortality
        yearly_surv = np.where(energy_t > 0,
            1.0 - 1.0 * np.exp(-energy_t * 0.4), 0.0)
        step_surv = np.where(energy_t > 0,
            np.exp(np.log(np.maximum(yearly_surv, 1e-10)) / 17280), 0.0)
        np.clip(energy_t, 0, 20, out=energy_t)

        times.append((time.perf_counter() - t0) * 1000)

    return np.median(times)


if __name__ == "__main__":
    N = 500
    print(f"=== Cython vs NumPy Phase Benchmark (N={N}) ===\n")

    # NumPy phases
    ms_numpy = benchmark_numpy_phases(N)
    print(f"NumPy phases (heading+pos+food+bmr+mort): {ms_numpy:.3f} ms")

    # Cython phases
    ms_cython = benchmark_cython_phases(N)
    if ms_cython is not None:
        print(f"Cython fused (same phases):               {ms_cython:.3f} ms")
        print(f"Speedup:                                  {ms_numpy/ms_cython:.2f}x")
        print()

        # Full tick comparison
        ms_full = benchmark_current(N)
        print(f"Current full tick (Numba+NumPy):           {ms_full:.3f} ms")

        # Estimate: replace NumPy phases with Cython
        # CRW kernel + RefMem kernel stay as Numba (~0.35ms)
        # NumPy phases take ~(ms_full - 0.35)ms, replace with Cython
        numba_part = 0.35  # CRW + RefMem (stays)
        numpy_phases = ms_full - numba_part
        projected = numba_part + ms_cython
        print(f"Numba kernels (CRW+RefMem, stays):        {numba_part:.3f} ms")
        print(f"Current non-kernel phases:                {numpy_phases:.3f} ms")
        print(f"Projected with Cython:                    {projected:.3f} ms")
        print(f"Projected speedup:                        {ms_full/projected:.2f}x")
        print(f"\nJava reference:                           0.795 ms")
        print(f"Projected gap to Java:                    {projected/0.795:.2f}x")
    else:
        print("\nSkipping Cython benchmark (module not built)")
        print()
        ms_full = benchmark_current(N)
        print(f"Current full tick:                        {ms_full:.3f} ms")
