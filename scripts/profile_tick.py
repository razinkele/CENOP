"""Deep tick profiler for CENOP population simulation.

Profiles step() and all sub-methods at N=500 for both DEPONS and JASMINE modes.
Reports phase-level timing, top functions by self-time, and allocation counts.
"""
import sys
import os
import time
import cProfile
import pstats
import io
import tracemalloc
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cenop.parameters.simulation_params import SimulationParameters
from cenop.landscape.cell_data import CellData
from cenop.agents.population import PorpoisePopulation

# Try to import Numba warmup
try:
    from cenop.optimizations.kernels import warmup_kernels
    warmup_kernels()
    print("[OK] Numba kernels warmed up")
except ImportError:
    print("[WARN] Numba kernels not available")


def create_test_setup(n_agents=500, comm_enabled=False, landscape_name='Kattegat'):
    """Create a population with a real landscape for profiling."""
    params = SimulationParameters()
    params.initial_population = n_agents

    # Communication settings
    params.communication_enabled = comm_enabled
    params.communication_range_km = 10.0
    params.communication_source_level = 160.0
    params.communication_threshold = 120.0
    params.communication_response_slope = 0.2
    params.social_weight = 0.3
    params.communication_recompute_interval = 4

    # Try to load real landscape
    landscape = None
    landscape_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'landscapes', landscape_name)
    if os.path.isdir(landscape_dir):
        try:
            landscape = CellData(landscape_dir, landscape_name=landscape_name)
            landscape.ensure_loaded()
            print(f"[OK] Loaded landscape: {landscape_name} ({landscape.width}x{landscape.height})")
            params.world_width = landscape.width
            params.world_height = landscape.height
        except Exception as e:
            print(f"[WARN] Could not load landscape: {e}")
    else:
        print(f"[WARN] Landscape dir not found: {landscape_dir}")
        params.world_width = 600
        params.world_height = 400

    pop = PorpoisePopulation(n_agents, params, landscape=landscape)
    return pop, params


def phase_timing(pop, n_ticks=200, warmup=50, label=""):
    """Time each phase of step() using manual instrumentation."""
    import types

    # Run warmup ticks
    for _ in range(warmup):
        pop.step()

    # Monkey-patch methods to time them
    method_names = [
        '_update_movement',
        '_handle_land_avoidance',
        '_apply_positions',
        '_apply_food_intake',
        '_check_mortality',
        '_apply_bmr_cost',
        '_update_psm',
        '_update_energy_history',
        '_update_dispersal',
        '_update_aging',
        '_update_reference_memory',
    ]

    timings = {name: [] for name in method_names}
    timings['_compute_social_vectors'] = []
    timings['step_overhead'] = []
    timings['step_total'] = []

    # Store originals
    originals = {}
    for name in method_names:
        if hasattr(pop, name):
            originals[name] = getattr(pop, name)

    # Also instrument social vectors if available
    if hasattr(pop, '_compute_social_vectors'):
        originals['_compute_social_vectors'] = pop._compute_social_vectors

    def make_wrapper(original, name):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter_ns()
            result = original(*args, **kwargs)
            elapsed = time.perf_counter_ns() - t0
            timings[name].append(elapsed)
            return result
        return wrapper

    for name, orig in originals.items():
        setattr(pop, name, make_wrapper(orig, name))

    # Run timed ticks
    for _ in range(n_ticks):
        t0 = time.perf_counter_ns()
        pop.step()
        elapsed = time.perf_counter_ns() - t0
        timings['step_total'].append(elapsed)

    # Restore originals
    for name, orig in originals.items():
        setattr(pop, name, orig)

    # Compute overhead
    for i in range(n_ticks):
        sum_phases = sum(
            timings[name][i] if i < len(timings[name]) else 0
            for name in method_names
            if name in timings and i < len(timings[name])
        )
        if '_compute_social_vectors' in timings and i < len(timings['_compute_social_vectors']):
            # social is called from within _update_movement, don't double count
            pass
        timings['step_overhead'].append(timings['step_total'][i] - sum_phases)

    # Print results
    print(f"\n{'='*70}")
    print(f"Phase Timing: {label} (N={pop.count}, {n_ticks} ticks)")
    print(f"{'='*70}")
    total_mean = np.mean(timings['step_total']) / 1e6
    print(f"{'TOTAL step()':40s} {total_mean:8.3f} ms  (100.0%)")
    print(f"{'-'*70}")

    for name in method_names + ['step_overhead']:
        if name in timings and len(timings[name]) > 0:
            mean_ns = np.mean(timings[name])
            mean_ms = mean_ns / 1e6
            pct = (mean_ns / np.mean(timings['step_total'])) * 100
            std_ms = np.std(timings[name]) / 1e6
            print(f"  {name:38s} {mean_ms:8.3f} ms  ({pct:5.1f}%)  +/- {std_ms:.3f}")

    # Social vectors sub-timing (inside _update_movement)
    if len(timings['_compute_social_vectors']) > 0:
        mean_ms = np.mean(timings['_compute_social_vectors']) / 1e6
        pct = (np.mean(timings['_compute_social_vectors']) / np.mean(timings['step_total'])) * 100
        print(f"    {'_compute_social_vectors (inside mvmt)':36s} {mean_ms:8.3f} ms  ({pct:5.1f}%)")

    return timings


def cprofile_analysis(pop, n_ticks=200, warmup=50, label=""):
    """Run cProfile for detailed function-level analysis."""
    for _ in range(warmup):
        pop.step()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_ticks):
        pop.step()
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(30)

    print(f"\n{'='*70}")
    print(f"cProfile Top 30 by tottime: {label} ({n_ticks} ticks)")
    print(f"{'='*70}")
    print(s.getvalue())

    # Also print by cumtime
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats('cumulative')
    ps2.print_stats(20)
    print(f"\nTop 20 by cumulative time:")
    print(s2.getvalue())

    return pr


def allocation_analysis(pop, n_ticks=5, label=""):
    """Track memory allocations during ticks."""
    # Warmup
    for _ in range(50):
        pop.step()

    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()

    for _ in range(n_ticks):
        pop.step()

    snap2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap2.compare_to(snap1, 'lineno')

    print(f"\n{'='*70}")
    print(f"Memory Allocations: {label} ({n_ticks} ticks)")
    print(f"{'='*70}")

    total_new = 0
    total_size = 0
    for stat in stats[:30]:
        if stat.size_diff > 0:
            total_new += stat.count_diff
            total_size += stat.size_diff
            print(f"  {stat}")

    print(f"\n  Total new allocations: {total_new}")
    print(f"  Total new memory: {total_size / 1024:.1f} KB over {n_ticks} ticks")
    print(f"  Per-tick: ~{total_new // max(n_ticks, 1)} allocs, ~{total_size / 1024 / max(n_ticks, 1):.1f} KB")


def numpy_dispatch_count(pop, n_ticks=10, label=""):
    """Count numpy array operations per tick (proxy for dispatch overhead)."""
    import numpy.core.multiarray as _ma

    # Count calls to numpy ufunc __call__
    original_where = np.where
    original_flatnonzero = np.flatnonzero
    counts = {'where': 0, 'flatnonzero': 0, 'random_normal': 0, 'copyto': 0}

    def counting_where(*args, **kwargs):
        counts['where'] += 1
        return original_where(*args, **kwargs)

    def counting_flatnonzero(*args, **kwargs):
        counts['flatnonzero'] += 1
        return original_flatnonzero(*args, **kwargs)

    original_copyto = np.copyto
    def counting_copyto(*args, **kwargs):
        counts['copyto'] += 1
        return original_copyto(*args, **kwargs)

    # Warmup
    for _ in range(50):
        pop.step()

    np.where = counting_where
    np.flatnonzero = counting_flatnonzero
    np.copyto = counting_copyto

    for _ in range(n_ticks):
        pop.step()

    np.where = original_where
    np.flatnonzero = original_flatnonzero
    np.copyto = original_copyto

    print(f"\n{'='*70}")
    print(f"NumPy Dispatch Counts: {label} ({n_ticks} ticks)")
    print(f"{'='*70}")
    for name, count in counts.items():
        print(f"  np.{name}: {count} total ({count / n_ticks:.1f}/tick)")


if __name__ == '__main__':
    N = 500

    # ===== DEPONS MODE (social OFF) =====
    print("\n" + "=" * 70)
    print("DEPONS MODE (social OFF)")
    print("=" * 70)
    pop_depons, _ = create_test_setup(N, comm_enabled=False)
    phase_timing(pop_depons, n_ticks=200, warmup=50, label="DEPONS")
    cprofile_analysis(pop_depons, n_ticks=200, warmup=50, label="DEPONS")
    allocation_analysis(pop_depons, n_ticks=5, label="DEPONS")
    numpy_dispatch_count(pop_depons, n_ticks=10, label="DEPONS")

    # ===== JASMINE MODE (social ON) =====
    print("\n" + "=" * 70)
    print("JASMINE MODE (social ON)")
    print("=" * 70)
    pop_jasmine, _ = create_test_setup(N, comm_enabled=True)
    phase_timing(pop_jasmine, n_ticks=200, warmup=50, label="JASMINE")
    cprofile_analysis(pop_jasmine, n_ticks=200, warmup=50, label="JASMINE")
    allocation_analysis(pop_jasmine, n_ticks=5, label="JASMINE")
    numpy_dispatch_count(pop_jasmine, n_ticks=10, label="JASMINE")
