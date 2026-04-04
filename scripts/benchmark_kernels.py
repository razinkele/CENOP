"""
Benchmark Numba kernels vs NumPy fallbacks.

Run: cd /home/razinka/cenjas && python3 CENOP/scripts/benchmark_kernels.py
"""
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cenop.optimizations.kernels import (
    reflect_boundaries_kernel,
    crw_angle_step_kernel,
    eat_food_kernel,
    depons_bmr_cost_kernel,
    social_accumulate_kernel,
    warmup_kernels,
    NUMBA_AVAILABLE,
)


def benchmark(name, func, n_runs=100):
    """Time a function over n_runs iterations, return mean ms."""
    # Warmup (2 calls to stabilize)
    func()
    func()

    start = time.perf_counter()
    for _ in range(n_runs):
        func()
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    print(f"  {name}: {elapsed:.3f} ms/call")
    return elapsed


def main():
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print("Warming up kernels...")
    warmup_kernels()

    n = 500  # Typical population size
    rng = np.random.default_rng(42)

    print(f"\nBenchmarks (n={n} agents, 100 iterations each):\n")

    # reflect_boundaries
    def bench_reflect():
        x = rng.uniform(-5, 25, n).astype(np.float64)
        y = rng.uniform(-5, 25, n).astype(np.float64)
        dx = rng.uniform(-3, 3, n).astype(np.float64)
        dy = rng.uniform(-3, 3, n).astype(np.float64)
        mask = np.ones(n, dtype=np.bool_)
        reflect_boundaries_kernel(x, y, dx, dy, 200, 200, mask)
    benchmark("reflect_boundaries", bench_reflect)

    # crw_angle_step
    def bench_crw():
        pa = rng.normal(0, 1, n).astype(np.float64)
        plm = rng.uniform(1, 3, n).astype(np.float64)
        d = np.full(n, 30.0, dtype=np.float64)
        s = np.full(n, 30.0, dtype=np.float64)
        ra = rng.normal(0, 4, n).astype(np.float64)
        rl = rng.normal(0, 1, n).astype(np.float64)
        m = np.ones(n, dtype=np.bool_)
        opa = np.zeros(n, dtype=np.float64)
        olm = np.zeros(n, dtype=np.float64)
        crw_angle_step_kernel(pa, plm, d, s, ra, rl, m, opa, olm,
                              0.5, 0.01, 0.01, 1.0, 0.5, 0.01, 0.01, 3.0,
                              0.0, 4.0, 0.0, 1.0, 0.00001)
    benchmark("crw_angle_step", bench_crw)

    # eat_food
    eat_demand = np.zeros((200, 200), dtype=np.float32)

    def bench_eat():
        grid = rng.uniform(0.1, 100, (200, 200)).astype(np.float32)
        x = rng.integers(0, 200, n).astype(np.int32)
        y = rng.integers(0, 200, n).astype(np.int32)
        frac = rng.uniform(0.1, 0.9, n).astype(np.float32)
        eaten = np.zeros(n, dtype=np.float32)
        eat_food_kernel(grid, x, y, frac, eaten, 0.01, eat_demand)
    benchmark("eat_food", bench_eat)

    # depons_bmr_cost
    def bench_bmr():
        speed = rng.uniform(0, 2, n).astype(np.float32)
        scale = np.ones(n, dtype=np.float32)
        is_lact = rng.choice([True, False], n)
        is_dist = rng.choice([True, False], n)
        deter = rng.uniform(0, 1, n).astype(np.float32)
        mask = np.ones(n, dtype=np.bool_)
        cost = np.zeros(n, dtype=np.float32)
        depons_bmr_cost_kernel(speed, scale, is_lact, is_dist, deter, mask, cost, 4.5, 1.4)
    benchmark("depons_bmr_cost", bench_bmr)

    # social_accumulate
    def bench_social():
        n_pairs = n * 3  # ~3 neighbors per agent
        idx_i = rng.integers(0, n, n_pairs).astype(np.int64)
        idx_j = rng.integers(0, n, n_pairs).astype(np.int64)
        dx = rng.normal(0, 1, n_pairs).astype(np.float64)
        dy = rng.normal(0, 1, n_pairs).astype(np.float64)
        dist = np.sqrt(dx**2 + dy**2) + 1e-6
        pi = rng.uniform(0, 1, n_pairs).astype(np.float64)
        pj = rng.uniform(0, 1, n_pairs).astype(np.float64)
        ux = np.zeros(n, dtype=np.float64)
        uy = np.zeros(n, dtype=np.float64)
        sw = np.zeros(n, dtype=np.float64)
        social_accumulate_kernel(idx_i, idx_j, dx, dy, dist, pi, pj, ux, uy, sw)
    benchmark("social_accumulate", bench_social)

    print("\nDone.")


if __name__ == '__main__':
    main()
