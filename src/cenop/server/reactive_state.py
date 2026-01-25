"""
Centralized Reactive State for CENOP Shiny App

All reactive values are defined here for easier management and testing.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from shiny import reactive
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported for type checking only to avoid runtime circular imports
    from cenop import Simulation


@dataclass
class SimulationState:
    """
    Centralized reactive state container for the simulation.
    
    All reactive values are initialized here to avoid scattered declarations
    throughout the server code.
    """
    
    # Core simulation state
    simulation: reactive.Value = field(default_factory=lambda: reactive.Value(None))
    running: reactive.Value = field(default_factory=lambda: reactive.Value(False))
    
    # Progress tracking
    progress: reactive.Value = field(default_factory=lambda: reactive.Value(0.0))
    progress_message: reactive.Value = field(default_factory=lambda: reactive.Value("Ready to run"))
    
    # History for charts
    population_history: reactive.Value = field(default_factory=lambda: reactive.Value([]))
    energy_history: reactive.Value = field(default_factory=lambda: reactive.Value([]))
    movement_history: reactive.Value = field(default_factory=lambda: reactive.Value([]))
    dispersal_history: reactive.Value = field(default_factory=lambda: reactive.Value([]))
    
    # Counters
    birth_count: reactive.Value = field(default_factory=lambda: reactive.Value(0))
    death_count: reactive.Value = field(default_factory=lambda: reactive.Value(0))
    map_update_counter: reactive.Value = field(default_factory=lambda: reactive.Value(0))

    # Latest lightweight porpoise positions snapshot for map rendering
    porpoise_positions: reactive.Value = field(default_factory=lambda: reactive.Value([]))
    
    # Landscape loading trigger
    landscape_load_counter: reactive.Value = field(default_factory=lambda: reactive.Value(0))
    landscape_loaded_name: reactive.Value = field(default_factory=lambda: reactive.Value(""))
    landscape_info: reactive.Value = field(default_factory=lambda: reactive.Value(""))  # e.g. "400x400 grid"
    
    # Turbine loading trigger
    turbine_load_counter: reactive.Value = field(default_factory=lambda: reactive.Value(0))
    turbine_loaded_name: reactive.Value = field(default_factory=lambda: reactive.Value(""))
    turbine_count: reactive.Value = field(default_factory=lambda: reactive.Value(0))  # Number of turbines loaded

    # UI / Refresh hooks
    last_refreshed: reactive.Value = field(default_factory=lambda: reactive.Value(None))  # ISO timestamp of last refresh
    selected_preview_file: reactive.Value = field(default_factory=lambda: reactive.Value(None))  # Format: {"landscape": "name", "file": "filename"}
    
    def reset(self):
        """Reset all state to initial values."""
        self.simulation.set(None)
        self.running.set(False)
        self.progress.set(0.0)
        self.progress_message.set("Ready to run")
        self.population_history.set([])
        self.energy_history.set([])
        self.movement_history.set([])
        self.dispersal_history.set([])
        self.birth_count.set(0)
        self.death_count.set(0)
        self.map_update_counter.set(0)


def create_state() -> SimulationState:
    """Factory function to create a new simulation state."""
    return SimulationState()
