"""
Performance Profiler for CENOP simulations.

Provides tools for profiling simulation performance and identifying bottlenecks.
"""

from __future__ import annotations

import time
import cProfile
import pstats
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from functools import wraps
import numpy as np

from cenop.parameters.simulation_params import SimulationParameters
from cenop.core.simulation import Simulation


@dataclass
class TimingResult:
    """Result from timing a code section."""
    name: str
    total_time: float
    call_count: int
    avg_time: float = field(init=False)
    
    def __post_init__(self):
        self.avg_time = self.total_time / self.call_count if self.call_count > 0 else 0


class SimulationProfiler:
    """
    Profile simulation performance.
    
    Tracks time spent in different simulation phases:
    - Movement calculation
    - Deterrence calculation
    - Energy/feeding
    - Reproduction/mortality
    - Data collection
    
    Usage:
        profiler = SimulationProfiler()
        profiler.profile_simulation(sim, ticks=1000)
        profiler.print_report()
    """
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.total_time: float = 0.0
        self.tick_count: int = 0
        
    def _record(self, name: str, duration: float) -> None:
        """Record a timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
        
    def profile_simulation(
        self, 
        sim: Simulation, 
        ticks: int = 1000,
        verbose: bool = True
    ) -> Dict[str, TimingResult]:
        """
        Profile a simulation for a number of ticks.
        
        Args:
            sim: Initialized simulation
            ticks: Number of ticks to profile
            verbose: Print progress
            
        Returns:
            Dictionary of timing results by section
        """
        if not sim._is_initialized:
            sim.initialize()
            
        self.timings.clear()
        self.tick_count = ticks
        
        start_total = time.perf_counter()
        
        for tick in range(ticks):
            if verbose and tick % 100 == 0:
                print(f"Profiling tick {tick}/{ticks}...")
                
            # Profile each phase
            self._profile_tick(sim)
            
        self.total_time = time.perf_counter() - start_total
        
        return self.get_results()
    
    def _profile_tick(self, sim: Simulation) -> None:
        """Profile a single tick, measuring each phase."""
        pop = sim.population_manager
        
        # Movement phase
        t0 = time.perf_counter()
        # Movement is part of pop.step(), we'll measure the whole step
        t1 = time.perf_counter()
        
        # Full step timing
        t_step_start = time.perf_counter()
        sim.step()
        t_step_end = time.perf_counter()
        
        self._record("full_step", t_step_end - t_step_start)
        
    def profile_detailed(
        self,
        params: Optional[SimulationParameters] = None,
        ticks: int = 500
    ) -> Dict[str, TimingResult]:
        """
        Run detailed profiling with isolated component timing.
        
        Creates a fresh simulation and profiles each component separately.
        """
        if params is None:
            params = SimulationParameters(
                porpoise_count=200,
                sim_years=1,
                landscape="Homogeneous"
            )
            
        sim = Simulation(params=params)
        sim.initialize()
        
        pop = sim.population_manager
        n = pop.count
        
        # Profile individual operations
        for tick in range(ticks):
            # Profile numpy operations
            t0 = time.perf_counter()
            _ = np.random.normal(0, 1, n)
            self._record("random_normal", time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            _ = np.sqrt(pop.x**2 + pop.y**2)
            self._record("sqrt_distance", time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            _ = np.sin(np.radians(pop.heading))
            self._record("trig_ops", time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            mask = pop.active_mask.copy()
            _ = pop.x[mask]
            self._record("masked_indexing", time.perf_counter() - t0)
            
            # Full step
            t0 = time.perf_counter()
            sim.step()
            self._record("full_step", time.perf_counter() - t0)
            
        self.tick_count = ticks
        return self.get_results()
    
    def get_results(self) -> Dict[str, TimingResult]:
        """Get timing results as TimingResult objects."""
        results = {}
        for name, times in self.timings.items():
            results[name] = TimingResult(
                name=name,
                total_time=sum(times),
                call_count=len(times)
            )
        return results
    
    def print_report(self) -> None:
        """Print a formatted profiling report."""
        results = self.get_results()
        
        print("\n" + "="*60)
        print("CENOP Simulation Performance Report")
        print("="*60)
        print(f"Total ticks profiled: {self.tick_count}")
        print(f"Total time: {self.total_time:.3f}s")
        print(f"Average time per tick: {self.total_time/self.tick_count*1000:.3f}ms")
        print(f"Ticks per second: {self.tick_count/self.total_time:.1f}")
        print("-"*60)
        print(f"{'Section':<25} {'Total(s)':<12} {'Calls':<10} {'Avg(ms)':<12} {'%Total':<8}")
        print("-"*60)
        
        # Sort by total time
        sorted_results = sorted(results.values(), key=lambda x: x.total_time, reverse=True)
        
        for r in sorted_results:
            pct = r.total_time / self.total_time * 100 if self.total_time > 0 else 0
            print(f"{r.name:<25} {r.total_time:<12.4f} {r.call_count:<10} {r.avg_time*1000:<12.4f} {pct:<8.1f}")
            
        print("="*60)


def profile_with_cprofile(
    sim: Simulation,
    ticks: int = 100,
    top_n: int = 20
) -> str:
    """
    Profile simulation using Python's cProfile.
    
    Returns formatted statistics string.
    """
    profiler = cProfile.Profile()
    
    profiler.enable()
    for _ in range(ticks):
        sim.step()
    profiler.disable()
    
    # Format results
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)
    
    return stream.getvalue()


def benchmark_population_sizes(
    sizes: List[int] = [50, 100, 200, 500, 1000],
    ticks: int = 100
) -> Dict[int, float]:
    """
    Benchmark simulation performance at different population sizes.
    
    Returns dict mapping size to ticks per second.
    """
    results = {}
    
    for size in sizes:
        print(f"Benchmarking population size: {size}")
        
        params = SimulationParameters(
            porpoise_count=size,
            sim_years=1,
            landscape="Homogeneous"
        )
        
        sim = Simulation(params=params)
        sim.initialize()
        
        # Warm up
        for _ in range(10):
            sim.step()
            
        # Timed run
        start = time.perf_counter()
        for _ in range(ticks):
            sim.step()
        elapsed = time.perf_counter() - start
        
        ticks_per_sec = ticks / elapsed
        results[size] = ticks_per_sec
        
        print(f"  {ticks_per_sec:.1f} ticks/sec ({elapsed:.3f}s for {ticks} ticks)")
        
    return results


def estimate_realtime_capacity() -> int:
    """
    Estimate maximum population size for real-time simulation.
    
    Real-time requires ~10 ticks/sec for smooth UI updates.
    """
    target_tps = 10  # ticks per second
    
    # Binary search for max population
    low, high = 50, 2000
    best = low
    
    while low <= high:
        mid = (low + high) // 2
        
        params = SimulationParameters(
            porpoise_count=mid,
            sim_years=1,
            landscape="Homogeneous"
        )
        
        sim = Simulation(params=params)
        sim.initialize()
        
        # Quick benchmark
        start = time.perf_counter()
        for _ in range(20):
            sim.step()
        elapsed = time.perf_counter() - start
        
        tps = 20 / elapsed
        
        if tps >= target_tps:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
            
    return best


# Timing decorator for method profiling
def timed(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed*1000:.3f}ms")
        return result
    return wrapper


if __name__ == "__main__":
    # Quick performance check
    print("Running CENOP performance benchmark...")
    
    results = benchmark_population_sizes()
    
    print("\nSummary:")
    print("-" * 40)
    for size, tps in results.items():
        status = "✅" if tps >= 10 else "⚠️"
        print(f"{status} Population {size}: {tps:.1f} ticks/sec")
        
    max_realtime = estimate_realtime_capacity()
    print(f"\nEstimated max real-time population: {max_realtime}")
