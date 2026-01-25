"""
File Output Writer for CENOP simulations.

Generates DEPONS-compatible output files for analysis and comparison.

Output files match DEPONS format:
- Population.txt: Daily population counts
- PorpoiseStatistics.txt: Individual tracking data
- Dispersal.txt: Dispersal events log
- Mortality.txt: Death events with causes
- Energy.txt: Daily energy statistics
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Any, Optional, TextIO
import numpy as np

if TYPE_CHECKING:
    from cenop.core.simulation import Simulation
    from cenop.agents.population import PorpoisePopulation


@dataclass
class OutputConfig:
    """Configuration for output file generation."""
    
    # Output directory
    output_dir: str = "output"
    
    # Which outputs to generate
    population: bool = True
    porpoise_statistics: bool = True
    dispersal: bool = True
    mortality: bool = True
    energy: bool = True
    
    # Sampling intervals (in ticks)
    population_interval: int = 48  # Daily (48 ticks = 1 day)
    statistics_interval: int = 48  # Daily
    energy_interval: int = 48  # Daily
    
    # Run identifier for unique filenames
    run_id: Optional[str] = None
    
    # Whether to append timestamp to filenames
    timestamp: bool = False
    
    def get_output_path(self) -> Path:
        """Get output directory path, creating if needed."""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_filename(self, base_name: str, extension: str = "txt") -> str:
        """Generate filename with optional run_id and timestamp."""
        parts = [base_name]
        if self.run_id:
            parts.append(self.run_id)
        if self.timestamp:
            parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
        return f"{'_'.join(parts)}.{extension}"


@dataclass
class MortalityEvent:
    """Record of a porpoise death."""
    tick: int
    day: int
    porpoise_id: int
    age: float
    energy: float
    cause: str  # "starvation", "old_age", "bycatch"
    x: float
    y: float


@dataclass
class DispersalEvent:
    """Record of a dispersal behavior."""
    tick: int
    day: int
    porpoise_id: int
    start_x: float
    start_y: float
    target_x: float
    target_y: float
    target_distance_km: float
    dispersal_type: str


class OutputWriter:
    """
    Writes simulation output to DEPONS-compatible files.
    
    Usage:
        writer = OutputWriter(config)
        writer.open()
        
        for tick in simulation:
            sim.step()
            writer.record_tick(sim)
            
        writer.close()
    
    Or with context manager:
        with OutputWriter(config) as writer:
            for tick in simulation:
                sim.step()
                writer.record_tick(sim)
    """
    
    def __init__(self, config: Optional[OutputConfig] = None):
        """
        Initialize the output writer.
        
        Args:
            config: Output configuration (uses defaults if None)
        """
        self.config = config or OutputConfig()
        self.output_path = self.config.get_output_path()
        
        # File handles
        self._population_file: Optional[TextIO] = None
        self._statistics_file: Optional[TextIO] = None
        self._dispersal_file: Optional[TextIO] = None
        self._mortality_file: Optional[TextIO] = None
        self._energy_file: Optional[TextIO] = None
        
        # CSV writers
        self._population_writer: Optional[csv.DictWriter] = None
        self._statistics_writer: Optional[csv.DictWriter] = None
        self._dispersal_writer: Optional[csv.DictWriter] = None
        self._mortality_writer: Optional[csv.DictWriter] = None
        self._energy_writer: Optional[csv.DictWriter] = None
        
        # Event buffers
        self._mortality_events: List[MortalityEvent] = []
        self._dispersal_events: List[DispersalEvent] = []
        
        # Previous state for change detection
        self._prev_active_mask: Optional[np.ndarray] = None
        self._prev_dispersing: Optional[np.ndarray] = None
        
        self._is_open = False
        
    def __enter__(self) -> "OutputWriter":
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        
    def open(self) -> None:
        """Open all output files and write headers."""
        if self._is_open:
            return

        try:
            if self.config.population:
                self._open_population_file()

            if self.config.porpoise_statistics:
                self._open_statistics_file()

            if self.config.dispersal:
                self._open_dispersal_file()

            if self.config.mortality:
                self._open_mortality_file()

            if self.config.energy:
                self._open_energy_file()

            self._is_open = True
        except Exception:
            # Clean up any files that were opened before the failure
            self.close()
            raise
        
    def close(self) -> None:
        """Close all output files."""
        files = [
            self._population_file,
            self._statistics_file,
            self._dispersal_file,
            self._mortality_file,
            self._energy_file
        ]
        
        for f in files:
            if f is not None:
                f.close()
                
        self._is_open = False
        
    def _open_population_file(self) -> None:
        """Open Population.txt with header."""
        filename = self.config.get_filename("Population")
        filepath = self.output_path / filename

        f = open(filepath, 'w', newline='')
        try:
            writer = csv.DictWriter(
                f,
                fieldnames=["tick", "day", "month", "year", "population",
                           "births", "deaths", "deaths_starvation",
                           "deaths_old_age", "deaths_bycatch"],
                delimiter='\t'
            )
            writer.writeheader()
            self._population_file = f
            self._population_writer = writer
        except Exception:
            f.close()
            raise
        
    def _open_statistics_file(self) -> None:
        """Open PorpoiseStatistics.txt with header."""
        filename = self.config.get_filename("PorpoiseStatistics")
        filepath = self.output_path / filename

        f = open(filepath, 'w', newline='')
        try:
            writer = csv.DictWriter(
                f,
                fieldnames=["tick", "day", "id", "x", "y", "heading",
                           "age", "energy", "is_female", "with_calf",
                           "is_dispersing", "deter_strength"],
                delimiter='\t'
            )
            writer.writeheader()
            self._statistics_file = f
            self._statistics_writer = writer
        except Exception:
            f.close()
            raise
        
    def _open_dispersal_file(self) -> None:
        """Open Dispersal.txt with header."""
        filename = self.config.get_filename("Dispersal")
        filepath = self.output_path / filename

        f = open(filepath, 'w', newline='')
        try:
            writer = csv.DictWriter(
                f,
                fieldnames=["tick", "day", "porpoise_id", "start_x", "start_y",
                           "target_x", "target_y", "target_distance_km",
                           "dispersal_type"],
                delimiter='\t'
            )
            writer.writeheader()
            self._dispersal_file = f
            self._dispersal_writer = writer
        except Exception:
            f.close()
            raise
        
    def _open_mortality_file(self) -> None:
        """Open Mortality.txt with header."""
        filename = self.config.get_filename("Mortality")
        filepath = self.output_path / filename

        f = open(filepath, 'w', newline='')
        try:
            writer = csv.DictWriter(
                f,
                fieldnames=["tick", "day", "porpoise_id", "age", "energy",
                           "cause", "x", "y"],
                delimiter='\t'
            )
            writer.writeheader()
            self._mortality_file = f
            self._mortality_writer = writer
        except Exception:
            f.close()
            raise
        
    def _open_energy_file(self) -> None:
        """Open Energy.txt with header."""
        filename = self.config.get_filename("Energy")
        filepath = self.output_path / filename

        f = open(filepath, 'w', newline='')
        try:
            writer = csv.DictWriter(
                f,
                fieldnames=["tick", "day", "mean_energy", "std_energy",
                           "min_energy", "max_energy", "pct_hungry",
                           "pct_starving"],
                delimiter='\t'
            )
            writer.writeheader()
            self._energy_file = f
            self._energy_writer = writer
        except Exception:
            f.close()
            raise
        
    def record_tick(self, sim: Simulation) -> None:
        """
        Record data for the current simulation tick.
        
        Args:
            sim: The simulation instance
        """
        if not self._is_open:
            return
            
        tick = sim.state.tick
        day = sim.state.day
        
        # Check for mortality events (by comparing active masks)
        if self.config.mortality:
            self._detect_mortality(sim)
            
        # Check for new dispersal events
        if self.config.dispersal:
            self._detect_dispersal(sim)
            
        # Population output (at interval)
        if self.config.population and tick % self.config.population_interval == 0:
            self._write_population(sim)
            
        # Statistics output (at interval)
        if self.config.porpoise_statistics and tick % self.config.statistics_interval == 0:
            self._write_statistics(sim)
            
        # Energy output (at interval)
        if self.config.energy and tick % self.config.energy_interval == 0:
            self._write_energy(sim)
            
    def _detect_mortality(self, sim: Simulation) -> None:
        """Detect and record mortality events."""
        pop = sim.population_manager
        current_mask = pop.active_mask.copy()
        
        if self._prev_active_mask is not None:
            # Find newly dead porpoises
            newly_dead = self._prev_active_mask & ~current_mask
            dead_indices = np.where(newly_dead)[0]
            
            for idx in dead_indices:
                # Determine cause of death
                energy = pop.energy[idx]
                age = pop.age[idx]
                
                if energy <= 0:
                    cause = "starvation"
                elif age >= 24:  # Max age
                    cause = "old_age"
                else:
                    cause = "unknown"
                    
                event = MortalityEvent(
                    tick=sim.state.tick,
                    day=sim.state.day,
                    porpoise_id=idx,
                    age=float(age),
                    energy=float(energy),
                    cause=cause,
                    x=float(pop.x[idx]),
                    y=float(pop.y[idx])
                )
                
                self._write_mortality_event(event)
                
        self._prev_active_mask = current_mask
        
    def _detect_dispersal(self, sim: Simulation) -> None:
        """Detect and record new dispersal events."""
        pop = sim.population_manager
        
        if not hasattr(pop, 'is_dispersing'):
            return
            
        current_dispersing = pop.is_dispersing.copy()
        
        if self._prev_dispersing is not None:
            # Find newly dispersing porpoises
            newly_dispersing = ~self._prev_dispersing & current_dispersing
            dispersing_indices = np.where(newly_dispersing)[0]
            
            for idx in dispersing_indices:
                # Get dispersal target if available
                target_x = pop.dispersal_target_x[idx] if hasattr(pop, 'dispersal_target_x') else pop.x[idx]
                target_y = pop.dispersal_target_y[idx] if hasattr(pop, 'dispersal_target_y') else pop.y[idx]
                target_dist = pop.dispersal_target_distance[idx] if hasattr(pop, 'dispersal_target_distance') else 0
                
                event = DispersalEvent(
                    tick=sim.state.tick,
                    day=sim.state.day,
                    porpoise_id=idx,
                    start_x=float(pop.x[idx]),
                    start_y=float(pop.y[idx]),
                    target_x=float(target_x),
                    target_y=float(target_y),
                    target_distance_km=float(target_dist / 1000),  # m to km
                    dispersal_type="PSM"
                )
                
                self._write_dispersal_event(event)
                
        self._prev_dispersing = current_dispersing
        
    def _write_population(self, sim: Simulation) -> None:
        """Write population data for current tick."""
        if self._population_writer is None:
            return
            
        state = sim.state
        
        row = {
            "tick": state.tick,
            "day": state.day,
            "month": state.month,
            "year": state.year,
            "population": state.population,
            "births": state.births,
            "deaths": state.deaths,
            "deaths_starvation": state.deaths_starvation,
            "deaths_old_age": state.deaths_old_age,
            "deaths_bycatch": state.deaths_bycatch,
        }
        
        self._population_writer.writerow(row)
        self._population_file.flush()
        
    def _write_statistics(self, sim: Simulation) -> None:
        """Write individual porpoise statistics."""
        if self._statistics_writer is None:
            return
            
        pop = sim.population_manager
        tick = sim.state.tick
        day = sim.state.day
        
        # Get active porpoises
        active_indices = np.where(pop.active_mask)[0]
        
        for idx in active_indices:
            row = {
                "tick": tick,
                "day": day,
                "id": idx,
                "x": f"{pop.x[idx]:.2f}",
                "y": f"{pop.y[idx]:.2f}",
                "heading": f"{pop.heading[idx]:.1f}",
                "age": f"{pop.age[idx]:.2f}",
                "energy": f"{pop.energy[idx]:.2f}",
                "is_female": int(pop.is_female[idx]),
                "with_calf": int(pop.with_calf[idx]),
                "is_dispersing": int(pop.is_dispersing[idx]) if hasattr(pop, 'is_dispersing') else 0,
                "deter_strength": f"{pop.deter_strength[idx]:.4f}",
            }
            
            self._statistics_writer.writerow(row)
            
        self._statistics_file.flush()
        
    def _write_energy(self, sim: Simulation) -> None:
        """Write energy statistics for current tick."""
        if self._energy_writer is None:
            return
            
        pop = sim.population_manager
        stats = pop.get_energy_stats()
        
        row = {
            "tick": sim.state.tick,
            "day": sim.state.day,
            "mean_energy": f"{stats['mean']:.2f}",
            "std_energy": f"{stats['std']:.2f}",
            "min_energy": f"{stats['min']:.2f}",
            "max_energy": f"{stats['max']:.2f}",
            "pct_hungry": f"{stats['hungry'] / max(pop.population_size, 1) * 100:.1f}",
            "pct_starving": f"{stats['starving'] / max(pop.population_size, 1) * 100:.1f}",
        }
        
        self._energy_writer.writerow(row)
        self._energy_file.flush()
        
    def _write_mortality_event(self, event: MortalityEvent) -> None:
        """Write a mortality event."""
        if self._mortality_writer is None:
            return
            
        row = asdict(event)
        row["age"] = f"{event.age:.2f}"
        row["energy"] = f"{event.energy:.2f}"
        row["x"] = f"{event.x:.2f}"
        row["y"] = f"{event.y:.2f}"
        
        self._mortality_writer.writerow(row)
        self._mortality_file.flush()
        
    def _write_dispersal_event(self, event: DispersalEvent) -> None:
        """Write a dispersal event."""
        if self._dispersal_writer is None:
            return
            
        row = asdict(event)
        row["start_x"] = f"{event.start_x:.2f}"
        row["start_y"] = f"{event.start_y:.2f}"
        row["target_x"] = f"{event.target_x:.2f}"
        row["target_y"] = f"{event.target_y:.2f}"
        row["target_distance_km"] = f"{event.target_distance_km:.2f}"
        
        self._dispersal_writer.writerow(row)
        self._dispersal_file.flush()


def create_simulation_output(
    sim: Simulation,
    output_dir: str = "output",
    run_id: Optional[str] = None
) -> OutputWriter:
    """
    Convenience function to create an output writer for a simulation.
    
    Args:
        sim: The simulation instance
        output_dir: Output directory path
        run_id: Optional run identifier
        
    Returns:
        Configured OutputWriter instance
    """
    config = OutputConfig(
        output_dir=output_dir,
        run_id=run_id
    )
    
    return OutputWriter(config)


def run_with_output(
    sim: Simulation,
    output_dir: str = "output",
    progress: bool = True
) -> OutputWriter:
    """
    Run a simulation with file output enabled.
    
    Args:
        sim: Initialized simulation
        output_dir: Output directory
        progress: Whether to show progress
        
    Returns:
        OutputWriter with recorded data
    """
    config = OutputConfig(output_dir=output_dir)
    
    with OutputWriter(config) as writer:
        sim.initialize()
        
        max_ticks = sim.max_ticks
        
        for tick in range(max_ticks):
            sim.step()
            writer.record_tick(sim)
            
            if progress and tick % 1000 == 0:
                pct = tick / max_ticks * 100
                print(f"Progress: {pct:.1f}% (tick {tick}/{max_ticks})")
                
    return writer
