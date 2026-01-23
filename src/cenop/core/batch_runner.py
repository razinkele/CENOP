"""
Batch Runner for CENOP simulations.

Runs multiple simulations with parameter variations for sensitivity analysis,
scenario comparison, and Monte Carlo studies.

Translates from: DEPONS batch/batch_params.xml and batch mode functionality.
"""

from __future__ import annotations

import itertools
import time
import json
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from cenop.parameters.simulation_params import SimulationParameters
from cenop.core.simulation import Simulation


@dataclass
class BatchResult:
    """Result from a single simulation run in a batch."""
    
    run_id: int
    parameters: Dict[str, Any]
    final_population: int
    initial_population: int
    total_births: int
    total_deaths: int
    total_ticks: int
    execution_time_seconds: float
    random_seed: int
    
    # Derived metrics
    population_change_pct: float = field(init=False)
    annual_growth_rate: float = field(init=False)
    
    def __post_init__(self):
        if self.initial_population > 0:
            self.population_change_pct = (
                (self.final_population - self.initial_population) 
                / self.initial_population * 100
            )
        else:
            self.population_change_pct = 0.0
            
        # Calculate annual growth rate (360 days per year, 48 ticks per day)
        years = self.total_ticks / (360 * 48) if self.total_ticks > 0 else 1
        if self.initial_population > 0 and years > 0:
            self.annual_growth_rate = (
                (self.final_population / self.initial_population) ** (1 / years) - 1
            ) * 100
        else:
            self.annual_growth_rate = 0.0


@dataclass
class BatchConfiguration:
    """Configuration for a batch of simulation runs."""
    
    # Base parameters that all runs share
    base_params: Dict[str, Any] = field(default_factory=dict)
    
    # Parameter variations: Dict[param_name, List[values]]
    variations: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Number of replicate runs per parameter combination
    replicates: int = 1
    
    # Random seeds for replicates (auto-generated if None)
    seeds: Optional[List[int]] = None
    
    # Output directory
    output_dir: str = "output/batch"
    
    # Whether to run in parallel
    parallel: bool = False
    max_workers: int = 4
    
    # Callback for progress updates
    progress_callback: Optional[Callable[[int, int, str], None]] = None
    
    def get_all_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        if not self.variations:
            return [self.base_params.copy()]
            
        keys = list(self.variations.keys())
        values = list(self.variations.values())
        
        combinations = []
        for combo in itertools.product(*values):
            params = self.base_params.copy()
            for key, val in zip(keys, combo):
                params[key] = val
            combinations.append(params)
            
        return combinations
    
    @property
    def total_runs(self) -> int:
        """Total number of simulation runs."""
        n_combos = len(self.get_all_combinations())
        return n_combos * self.replicates


class BatchRunner:
    """
    Run multiple simulations with parameter variations.
    
    Supports:
    - Parameter sweeps (e.g., vary porpoise_count from 100 to 500)
    - Multi-dimensional variation grids
    - Multiple replicates with different random seeds
    - Parallel execution
    - Result aggregation and export
    
    Example usage:
        config = BatchConfiguration(
            base_params={"sim_years": 5, "landscape": "Homogeneous"},
            variations={
                "porpoise_count": [100, 200, 300],
                "turbines": ["off", "construction", "operation"]
            },
            replicates=3,
            output_dir="output/sensitivity"
        )
        
        runner = BatchRunner(config)
        results = runner.run()
        runner.export_results(results, "batch_results.csv")
    """
    
    def __init__(self, config: BatchConfiguration):
        """
        Initialize the batch runner.
        
        Args:
            config: Batch configuration specifying parameters and variations
        """
        self.config = config
        self.results: List[BatchResult] = []
        
        # Ensure output directory exists
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate seeds if not provided
        if config.seeds is None:
            base_seed = np.random.randint(0, 100000)
            self.seeds = [base_seed + i for i in range(config.replicates)]
        else:
            self.seeds = config.seeds
            
    def run(self, progress: bool = True) -> List[BatchResult]:
        """
        Execute all simulation runs in the batch.
        
        Args:
            progress: Whether to show progress bar
            
        Returns:
            List of BatchResult objects for each run
        """
        combinations = self.config.get_all_combinations()
        total_runs = len(combinations) * self.config.replicates
        
        self.results = []
        run_id = 0
        
        if progress:
            print(f"Starting batch run: {total_runs} simulations")
            print(f"  - {len(combinations)} parameter combinations")
            print(f"  - {self.config.replicates} replicates each")
            
        start_time = time.time()
        
        if self.config.parallel and total_runs > 1:
            self.results = self._run_parallel(combinations)
        else:
            self.results = self._run_sequential(combinations, progress)
            
        elapsed = time.time() - start_time
        
        if progress:
            print(f"\nBatch complete: {len(self.results)} runs in {elapsed:.1f}s")
            print(f"  - Average time per run: {elapsed/len(self.results):.2f}s")
            
        return self.results
    
    def _run_sequential(
        self, 
        combinations: List[Dict[str, Any]],
        progress: bool = True
    ) -> List[BatchResult]:
        """Run simulations sequentially."""
        results = []
        run_id = 0
        total = len(combinations) * self.config.replicates
        
        for combo in combinations:
            for rep in range(self.config.replicates):
                seed = self.seeds[rep]
                
                if progress:
                    pct = (run_id + 1) / total * 100
                    param_str = ", ".join(f"{k}={v}" for k, v in combo.items() 
                                         if k in self.config.variations)
                    print(f"  [{run_id+1}/{total}] ({pct:.0f}%) {param_str}, rep={rep+1}")
                
                result = self._run_single(run_id, combo, seed)
                results.append(result)
                
                if self.config.progress_callback:
                    self.config.progress_callback(run_id + 1, total, param_str)
                    
                run_id += 1
                
        return results
    
    def _run_parallel(self, combinations: List[Dict[str, Any]]) -> List[BatchResult]:
        """Run simulations in parallel using process pool."""
        results = []
        tasks = []
        
        run_id = 0
        for combo in combinations:
            for rep in range(self.config.replicates):
                seed = self.seeds[rep]
                tasks.append((run_id, combo.copy(), seed))
                run_id += 1
        
        # Note: ProcessPoolExecutor requires picklable objects
        # For now, we'll run sequentially if parameters aren't picklable
        try:
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._run_single, *task): task[0] 
                    for task in tasks
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
        except Exception as e:
            print(f"Parallel execution failed: {e}, falling back to sequential")
            return self._run_sequential(combinations, progress=True)
            
        # Sort by run_id to maintain order
        results.sort(key=lambda r: r.run_id)
        return results
    
    def _run_single(
        self, 
        run_id: int, 
        params: Dict[str, Any], 
        seed: int
    ) -> BatchResult:
        """Run a single simulation and return results."""
        start_time = time.time()
        
        # Create parameters object
        sim_params = SimulationParameters(**params)
        
        # Create and run simulation
        sim = Simulation(params=sim_params, seed=seed)
        sim.initialize()
        
        initial_pop = sim.population_size
        
        # Run to completion
        sim.run(progress=False)
        
        execution_time = time.time() - start_time
        
        return BatchResult(
            run_id=run_id,
            parameters=params,
            final_population=sim.population_size,
            initial_population=initial_pop,
            total_births=sim.total_births,
            total_deaths=sim.total_deaths,
            total_ticks=sim.state.tick,
            execution_time_seconds=execution_time,
            random_seed=seed
        )
    
    def export_results(
        self, 
        results: Optional[List[BatchResult]] = None,
        filename: str = "batch_results.csv"
    ) -> Path:
        """
        Export batch results to CSV file.
        
        Args:
            results: Results to export (uses self.results if None)
            filename: Output filename
            
        Returns:
            Path to the exported file
        """
        if results is None:
            results = self.results
            
        if not results:
            raise ValueError("No results to export")
            
        output_file = self.output_path / filename
        
        # Flatten parameters dict for CSV
        rows = []
        for r in results:
            row = {
                "run_id": r.run_id,
                "random_seed": r.random_seed,
                "initial_population": r.initial_population,
                "final_population": r.final_population,
                "population_change_pct": r.population_change_pct,
                "annual_growth_rate": r.annual_growth_rate,
                "total_births": r.total_births,
                "total_deaths": r.total_deaths,
                "total_ticks": r.total_ticks,
                "execution_time_seconds": r.execution_time_seconds,
            }
            # Add parameter columns
            for k, v in r.parameters.items():
                row[f"param_{k}"] = v
            rows.append(row)
            
        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
        return output_file
    
    def export_summary(
        self,
        results: Optional[List[BatchResult]] = None,
        filename: str = "batch_summary.json"
    ) -> Path:
        """Export summary statistics as JSON."""
        if results is None:
            results = self.results
            
        if not results:
            raise ValueError("No results to summarize")
            
        # Calculate summary statistics
        final_pops = [r.final_population for r in results]
        growth_rates = [r.annual_growth_rate for r in results]
        
        summary = {
            "total_runs": len(results),
            "configuration": {
                "base_params": self.config.base_params,
                "variations": self.config.variations,
                "replicates": self.config.replicates
            },
            "results": {
                "population": {
                    "mean": float(np.mean(final_pops)),
                    "std": float(np.std(final_pops)),
                    "min": int(np.min(final_pops)),
                    "max": int(np.max(final_pops)),
                },
                "annual_growth_rate": {
                    "mean": float(np.mean(growth_rates)),
                    "std": float(np.std(growth_rates)),
                    "min": float(np.min(growth_rates)),
                    "max": float(np.max(growth_rates)),
                },
            },
            "execution": {
                "total_time_seconds": sum(r.execution_time_seconds for r in results),
                "mean_time_per_run": np.mean([r.execution_time_seconds for r in results]),
            }
        }
        
        output_file = self.output_path / filename
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return output_file
    
    def get_results_by_parameter(
        self, 
        param_name: str
    ) -> Dict[Any, List[BatchResult]]:
        """Group results by a specific parameter value."""
        grouped = {}
        for r in self.results:
            val = r.parameters.get(param_name)
            if val not in grouped:
                grouped[val] = []
            grouped[val].append(r)
        return grouped
    
    def calculate_sensitivity(
        self,
        param_name: str,
        metric: str = "final_population"
    ) -> Dict[str, Any]:
        """
        Calculate sensitivity of a metric to a parameter.
        
        Returns statistics for each parameter value.
        """
        grouped = self.get_results_by_parameter(param_name)
        
        sensitivity = {
            "parameter": param_name,
            "metric": metric,
            "values": {}
        }
        
        for val, results in grouped.items():
            metrics = [getattr(r, metric) for r in results]
            sensitivity["values"][str(val)] = {
                "mean": float(np.mean(metrics)),
                "std": float(np.std(metrics)),
                "n": len(metrics)
            }
            
        return sensitivity


def run_sensitivity_analysis(
    base_params: Dict[str, Any],
    param_name: str,
    param_values: List[Any],
    replicates: int = 5,
    output_dir: str = "output/sensitivity"
) -> Dict[str, Any]:
    """
    Convenience function to run a single-parameter sensitivity analysis.
    
    Args:
        base_params: Base simulation parameters
        param_name: Parameter to vary
        param_values: Values to test
        replicates: Number of replicates per value
        output_dir: Output directory
        
    Returns:
        Sensitivity analysis results
    """
    config = BatchConfiguration(
        base_params=base_params,
        variations={param_name: param_values},
        replicates=replicates,
        output_dir=output_dir
    )
    
    runner = BatchRunner(config)
    results = runner.run()
    
    # Export results
    runner.export_results(results)
    runner.export_summary(results)
    
    return runner.calculate_sensitivity(param_name)


def run_scenario_comparison(
    scenarios: Dict[str, Dict[str, Any]],
    replicates: int = 5,
    output_dir: str = "output/scenarios"
) -> Dict[str, List[BatchResult]]:
    """
    Compare multiple named scenarios.
    
    Args:
        scenarios: Dict mapping scenario name to parameters
        replicates: Number of replicates per scenario
        output_dir: Output directory
        
    Returns:
        Dict mapping scenario name to results
    """
    all_results = {}
    
    for name, params in scenarios.items():
        print(f"\n=== Running scenario: {name} ===")
        
        config = BatchConfiguration(
            base_params=params,
            variations={},
            replicates=replicates,
            output_dir=f"{output_dir}/{name}"
        )
        
        runner = BatchRunner(config)
        results = runner.run()
        runner.export_results(results, f"{name}_results.csv")
        
        all_results[name] = results
        
    return all_results
