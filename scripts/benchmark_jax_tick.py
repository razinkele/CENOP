"""
Benchmark JAX JIT tick performance.

C0 baseline: measures current JAX tick throughput vs 1.2 ms/tick target.

Run: cd /home/razinka/cenjas && python3 CENOP/scripts/benchmark_jax_tick.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# --- CRW parameters (DEPONS 3.2 defaults) ---
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

TARGET_MS = 1.2  # ms/tick target


def create_test_population(n=500, seed=42):
    """Create a minimal Simulation (and its population) for benchmarking.

    Returns (sim, pop) where pop is the PorpoisePopulation used by sim.
    No landscape is loaded — flat world, JAX path enabled.
    """
    from cenop.core.simulation import Simulation
    from cenop.parameters.simulation_params import SimulationParameters

    params = SimulationParameters(
        porpoise_count=n,
        turbines="off",
        ships_enabled=False,
        use_jax=True,
        random_seed=seed,
    )
    sim = Simulation(params=params, seed=seed)
    sim.initialize()
    return sim, sim.population_manager


def benchmark_full_tick(pop_sim, n_warmup=10, n_ticks=1000):
    """Benchmark full _step_jax via sim.step().

    Parameters
    ----------
    pop_sim : (Simulation, PorpoisePopulation)
        Tuple returned by create_test_population.
    n_warmup : int
        Number of ticks used for JIT compilation (not measured).
    n_ticks : int
        Number of measured ticks.

    Returns
    -------
    dict with keys: mean, median, p95, p99, min, max (all in ms)
    """
    import jax

    sim, _ = pop_sim

    print(f"  Warming up ({n_warmup} ticks for JIT compilation)...", flush=True)
    for _ in range(n_warmup):
        sim.step()
    # Flush any pending async work
    jax.effects_barrier()

    print(f"  Measuring {n_ticks} ticks...", flush=True)
    times_ms = np.empty(n_ticks)
    for i in range(n_ticks):
        t0 = time.perf_counter()
        sim.step()
        jax.effects_barrier()
        times_ms[i] = (time.perf_counter() - t0) * 1000.0

    return {
        'mean':   float(np.mean(times_ms)),
        'median': float(np.median(times_ms)),
        'p95':    float(np.percentile(times_ms, 95)),
        'p99':    float(np.percentile(times_ms, 99)),
        'min':    float(np.min(times_ms)),
        'max':    float(np.max(times_ms)),
    }


def benchmark_crw_kernel(n=500, n_runs=1000):
    """Benchmark the CRW kernel in isolation (JIT-compiled).

    Parameters
    ----------
    n : int
        Number of agents.
    n_runs : int
        Number of measured calls.

    Returns
    -------
    dict with keys: mean, median, p95, p99, min, max (all in ms)
    """
    import jax
    import jax.numpy as jnp
    from cenop.optimizations.jax_kernels import jax_crw_kernel

    rng = np.random.default_rng(42)

    prev_angle = jnp.array(rng.uniform(-90, 90, n), dtype=jnp.float64)
    prev_log_mov = jnp.array(rng.uniform(1.0, 3.5, n), dtype=jnp.float64)
    depths = jnp.array(rng.uniform(5.0, 100.0, n), dtype=jnp.float64)
    salinity = jnp.array(rng.uniform(10.0, 35.0, n), dtype=jnp.float64)
    mask = jnp.array(rng.choice([True, False], n, p=[0.9, 0.1]))
    key = jax.random.PRNGKey(42)

    jitted = jax.jit(jax_crw_kernel)

    # Warmup / JIT compilation
    print("  Warming up CRW kernel (JIT compilation)...", flush=True)
    for _ in range(5):
        out = jitted(
            prev_angle, prev_log_mov, depths, salinity, mask, key, **CRW_PARAMS
        )
        jax.block_until_ready(out)
        key, _ = jax.random.split(key)

    print(f"  Measuring CRW kernel ({n_runs} runs)...", flush=True)
    times_ms = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        out = jitted(
            prev_angle, prev_log_mov, depths, salinity, mask, key, **CRW_PARAMS
        )
        jax.block_until_ready(out)
        times_ms[i] = (time.perf_counter() - t0) * 1000.0
        key, _ = jax.random.split(key)

    return {
        'mean':   float(np.mean(times_ms)),
        'median': float(np.median(times_ms)),
        'p95':    float(np.percentile(times_ms, 95)),
        'p99':    float(np.percentile(times_ms, 99)),
        'min':    float(np.min(times_ms)),
        'max':    float(np.max(times_ms)),
    }


def _print_stats(label, stats):
    """Pretty-print a timing stats dict."""
    print(
        f"  {label}: "
        f"median={stats['median']:.3f}ms  mean={stats['mean']:.3f}ms  "
        f"p95={stats['p95']:.3f}ms  p99={stats['p99']:.3f}ms  "
        f"min={stats['min']:.3f}ms  max={stats['max']:.3f}ms"
    )


def main():
    # --- JAX availability check ---
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("ERROR: JAX not installed. Install with: pip install jax jaxlib")
        sys.exit(1)

    print("=" * 70)
    print("  JAX JIT Tick Benchmark  (C0 — baseline)")
    print("=" * 70)
    print(f"  JAX version : {jax.__version__}")
    print(f"  Backend     : {jax.default_backend()}")
    print(f"  Devices     : {jax.devices()}")
    print(f"  Target      : < {TARGET_MS:.1f} ms/tick (median)")
    print()

    # --- CRW kernel in isolation ---
    print("[ CRW Kernel (isolated) ]")
    crw_stats = benchmark_crw_kernel(n=500, n_runs=1000)
    _print_stats("crw_kernel (n=500, 1000 runs)", crw_stats)
    print()

    # --- Full tick ---
    print("[ Full JAX Tick (n=500, no landscape) ]")
    pop_sim = create_test_population(n=500, seed=42)
    tick_stats = benchmark_full_tick(pop_sim, n_warmup=10, n_ticks=1000)
    _print_stats("full_tick  (n=500, 1000 ticks)", tick_stats)
    print()

    # --- Pass/Fail ---
    median = tick_stats['median']
    passed = median < TARGET_MS
    status = "PASS" if passed else "FAIL"
    print(f"  Target {TARGET_MS:.1f} ms/tick: {status}  (median={median:.3f}ms)")
    print()

    if not passed:
        gap = median / TARGET_MS
        print(f"  Gap: {gap:.1f}x over target  (need {gap:.1f}x speedup)")
        print("  Expected at C0: ~11.7ms (bottlenecks: CRW kernel + np<->jax transfers)")
    print("=" * 70)


if __name__ == '__main__':
    main()
