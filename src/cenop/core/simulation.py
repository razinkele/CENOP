"""
Main simulation controller for CENOP.

This module contains the Simulation class which orchestrates the entire
agent-based model, managing agents, scheduling, and data collection.

Supports both DEPONS mode (fixed timestep, regulatory-compliant) and
JASMINE mode (flexible timestep, event-driven) via the TimeManager.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional, List, Dict, Any
from dataclasses import dataclass, field
from tqdm import tqdm

from cenop.core.time_manager import TimeManager, TimeMode

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters
    from cenop.landscape.cell_data import CellData
    from cenop.agents.porpoise import Porpoise
    from cenop.movement.base import MovementModule
    from cenop.behavior.hybrid_fsm import HybridBehaviorFSM
    from cenop.physiology.energy_budget import EnergyModule


@dataclass
class SimulationState:
    """Tracks the current state of the simulation."""
    
    tick: int = 0
    day: int = 0
    month: int = 1
    year: int = 1
    quarter: int = 0
    
    # Population statistics
    population: int = 0
    births: int = 0
    deaths: int = 0
    deaths_starvation: int = 0
    deaths_old_age: int = 0
    deaths_bycatch: int = 0
    
    @property
    def hour(self) -> int:
        """Current hour of day (0-23)."""
        # Each tick is 30 min, 48 ticks per day
        half_hour = self.tick % 48
        return half_hour // 2
    
    @property
    def is_daytime(self) -> bool:
        """Check if it's daytime (6:00 - 18:00)."""
        return 6 <= self.hour < 18
    
    def advance_tick(self) -> None:
        """Advance simulation by one tick (30 minutes)."""
        self.tick += 1
        
        # Update day (48 ticks per day) - only at day boundary
        if self.tick % 48 == 0:
            self.day += 1
            
            # Update month (30 days per month) - only when day advances
            if self.day % 30 == 0:
                self.month = (self.month % 12) + 1
                if self.month == 1:
                    self.year += 1
                
        # Update quarter
        self.quarter = (self.month - 1) // 3


class Simulation:
    """
    Main simulation controller.
    
    Orchestrates the CENOP agent-based model, managing:
    - Agent lifecycle (creation, stepping, removal)
    - Scheduling of tasks (daily, monthly, yearly)
    - Data collection and statistics
    - Landscape/environment interactions
    
    Translates from: PorpoiseSimBuilder.java
    """
    
    def __init__(
        self,
        params: SimulationParameters,
        cell_data: Optional[CellData] = None,
        seed: Optional[int] = None,
        time_manager: Optional[TimeManager] = None,
        time_mode: TimeMode = TimeMode.DEPONS,
        movement_module: Optional['MovementModule'] = None,
        behavior_fsm: Optional['HybridBehaviorFSM'] = None,
        energy_module: Optional['EnergyModule'] = None,
    ):
        """
        Initialize the simulation.

        Args:
            params: Simulation parameters configuration
            cell_data: Pre-loaded landscape data (optional)
            seed: Random seed for reproducibility
            time_manager: Pre-configured TimeManager (optional)
            time_mode: Time mode if creating new TimeManager (default: DEPONS)
            movement_module: Optional movement module for modular movement system
            behavior_fsm: Optional behavioral FSM for state transitions
            energy_module: Optional energy module for DEB calculations
        """
        self.params = params
        self.state = SimulationState()

        # Set random seed
        actual_seed = seed if seed is not None else params.random_seed
        if actual_seed is None:
            actual_seed = 42  # Default seed for reproducibility
        self._seed = actual_seed

        # Initialize TimeManager
        if time_manager is not None:
            self.time_manager = time_manager
        else:
            self.time_manager = TimeManager(
                mode=time_mode,
                base_seed=actual_seed,
                sim_years=params.sim_years
            )

        # Set initial random seed (will be updated per-tick)
        np.random.seed(actual_seed)
        
        # Initialize components (lazy loading or pre-provided)
        self._cell_data: Optional[CellData] = cell_data
        self._porpoises: List[Porpoise] = []
        
        # Turbine and ship managers
        from cenop.agents.turbine import TurbineManager
        from cenop.agents.ship import ShipManager
        self._turbine_manager = TurbineManager()
        self._ship_manager = ShipManager()
        
        # History for plotting
        self._history: List[Dict[str, Any]] = []

        # Max ticks from TimeManager (for backward compatibility)
        self.max_ticks = self.time_manager.max_ticks

        # Running state
        self._is_running = False
        self._is_initialized = False

        # Movement module (Phase 2: JASMINE integration)
        self._movement_module = movement_module

        # Behavior FSM (Phase 3: JASMINE integration)
        self._behavior_fsm = behavior_fsm

        # Energy module (Phase 4: JASMINE DEB integration)
        self._energy_module = energy_module

        # Auto-initialize if cell_data provided or using homogeneous landscape
        if cell_data is not None or params.landscape == "Homogeneous":
            self.initialize()
        
    def initialize(self) -> None:
        """
        Initialize the simulation environment and agents.
        
        This sets up:
        - Landscape data
        - Initial porpoise population
        - Turbines (if enabled)
        - Ships (if enabled)
        """
        if self._is_initialized:
            return
            
        # Load landscape data if not pre-provided
        if self._cell_data is None:
            from cenop.landscape.cell_data import (
                CellData, 
                create_homogeneous_landscape,
                create_landscape_from_depons
            )
            if self.params.is_homogeneous:
                # Try to load real DEPONS bathymetry first
                self._cell_data = create_landscape_from_depons()
                if self._cell_data.landscape_name == "Homogeneous":
                    # Fallback happened, create proper homogeneous
                    self._cell_data = create_homogeneous_landscape()
            else:
                self._cell_data = CellData(self.params.landscape)
        
        # Create initial porpoise population (Vectorized - Phase 3)
        from cenop.agents.population import PorpoisePopulation

        # Initialize vectorized population manager
        self.population_manager = PorpoisePopulation(
            count=self.params.porpoise_count,
            params=self.params,
            landscape=self._cell_data,
            movement_module=self._movement_module,
            behavior_fsm=self._behavior_fsm,
            energy_module=self._energy_module,
        )
        
        # Legacy list for backward compatibility (lazy loaded if accessed via property)
        self._porpoises = [] 
            
        # Set up turbines if enabled
        if self.params.turbines != "off":
            self._setup_turbines()
            
        # Set up ships if enabled
        if self.params.ships_enabled:
            self._setup_ships()
            
        self.state.population = self.population_manager.population_size
        self._is_initialized = True
        
    def _get_initial_age(self) -> float:
        """Get random initial age from the DEPONS age distribution."""
        # Age distribution moved to external configuration (Fix Phase 1.2)
        from cenop.parameters.demography import AGE_DISTRIBUTION_FREQUENCY
        return float(np.random.choice(AGE_DISTRIBUTION_FREQUENCY))
        
    def _get_valid_initial_position(self) -> tuple[float, float]:
        """Get a random initial position in valid water."""
        if self._cell_data is None:
            # Fallback for homogeneous landscape
            x = np.random.uniform(0, self.params.world_width)
            y = np.random.uniform(0, self.params.world_height)
            return x, y
            
        # Try to find valid position (depth > min_depth)
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(0, self._cell_data.width)
            y = np.random.uniform(0, self._cell_data.height)
            
            depth = self._cell_data.get_depth(x, y)
            if depth > self.params.min_depth:
                return x, y
                
        # Fallback to center if no valid position found
        return self._cell_data.width / 2, self._cell_data.height / 2
        
    def _setup_turbines(self) -> None:
        """
        Set up wind turbines from data files.
        
        Loads turbines from data/wind-farms/ directory based on the
        turbines parameter value (scenario name).
        
        File format:
        id  x.coordinate  y.coordinate  impact  tick.start  tick.end
        
        Coordinates are in UTM and converted to grid using landscape metadata.
        """
        from cenop.agents.turbine import TurbinePhase, Turbine
        from pathlib import Path
        
        # Handle "off" case
        if self.params.turbines == "off":
            self._turbine_manager.set_phase(TurbinePhase.OFF)
            return
            
        # Map turbine scenario names to files
        scenario_files = {
            "construction": "User-def.txt",
            "operation": "User-def.txt",
            "DanTysk": "DanTysk-construction.txt",
            "Gemini": "Gemini-construction.txt",
            "NorthSea_scenario1": "NorthSea_scenario1.txt",
            "NorthSea_scenario2": "NorthSea_scenario2.txt",
            "NorthSea_scenario3": "NorthSea_scenario3.txt",
            "User-def": "User-def.txt",
        }
        
        # Determine phase based on parameter
        # Scenarios ending with -construction or containing "construction" use CONSTRUCTION phase
        # Otherwise default to CONSTRUCTION (since most scenarios are pile-driving)
        if self.params.turbines == "operation":
            self._turbine_manager.set_phase(TurbinePhase.OPERATION)
        else:
            self._turbine_manager.set_phase(TurbinePhase.CONSTRUCTION)
        
        # Determine data file path using centralized config (Fix Phase 1.1)
        from cenop.config import get_wind_farm_file
        
        # Get the filename for this scenario
        scenario_key = self.params.turbines
        if scenario_key in scenario_files:
            filename = scenario_files[scenario_key]
        else:
            # Try using the scenario name directly as filename
            filename = f"{scenario_key}.txt"
            
        turbine_file = get_wind_farm_file(filename)
        
        # Get UTM origin from landscape metadata
        utm_origin_x = 0.0
        utm_origin_y = 0.0
        cell_size = 400.0
        
        if self._cell_data is not None and self._cell_data.metadata is not None:
            utm_origin_x = self._cell_data.metadata.xllcorner
            utm_origin_y = self._cell_data.metadata.yllcorner
            cell_size = self._cell_data.metadata.cellsize
        
        # Load turbines from file
        if turbine_file.exists():
            self._turbine_manager.load_from_file(
                str(turbine_file),
                utm_origin_x=utm_origin_x,
                utm_origin_y=utm_origin_y,
                cell_size=cell_size
            )
            
            # Set phase for all loaded turbines
            for turbine in self._turbine_manager.turbines:
                turbine.phase = self._turbine_manager.phase
                # Activate based on current tick
                turbine.update_phase(self.state.tick)
        else:
            # Fallback: create sample turbines in center for homogeneous landscape
            if self._cell_data is not None:
                center_x = self._cell_data.width / 2
                center_y = self._cell_data.height / 2
            else:
                center_x = self.params.world_width / 2
                center_y = self.params.world_height / 2
            
            # 3x3 grid of turbines (9 turbines total) with realistic impact
            for i in range(-1, 2):
                for j in range(-1, 2):
                    turbine = Turbine(
                        id=len(self._turbine_manager.turbines),
                        x=center_x + i * 5,  # 5 cells = 2km apart
                        y=center_y + j * 5,
                        heading=0.0,
                        name=f"Turbine_{i+2}_{j+2}",
                        impact=210.0,  # Realistic dB source level
                        start_tick=0,
                        end_tick=2147483647
                    )
                    turbine.phase = self._turbine_manager.phase
                    turbine._is_active = True
                    self._turbine_manager.turbines.append(turbine)
        
    def _setup_ships(self) -> None:
        """Set up ship traffic from DEPONS ships.json or create sample ships."""
        self._ship_manager.set_enabled(self.params.ships_enabled)
        
        if not self.params.ships_enabled:
            return
            
        from cenop.agents.ship import Ship, Route, Buoy, VesselClass
        import os
        
        # Determine landscape size
        if self._cell_data is not None:
            width = self._cell_data.width
            height = self._cell_data.height
            # Get UTM origin from metadata if available
            utm_origin_x = getattr(self._cell_data.metadata, 'xllcorner', 3976618.0)
            utm_origin_y = getattr(self._cell_data.metadata, 'yllcorner', 3363923.0)
            cell_size = getattr(self._cell_data.metadata, 'cellsize', 400.0)
        else:
            width = self.params.world_width
            height = self.params.world_height
            utm_origin_x = 3976618.0  # DEPONS UserDefined default
            utm_origin_y = 3363923.0
            cell_size = 400.0
        
        # Try to load ships from JSON file
        ships_loaded = False
        possible_paths = [
            "data/ships.json",
            "../data/ships.json",
            "cenop/data/ships.json",
            "../DEPONS-master/data/UserDefined/ships.json",
        ]
        
        for ships_path in possible_paths:
            if os.path.exists(ships_path):
                self._ship_manager.load_from_json(
                    ships_path,
                    utm_origin_x=utm_origin_x,
                    utm_origin_y=utm_origin_y,
                    cell_size=cell_size
                )
                if self._ship_manager.count > 0:
                    ships_loaded = True
                    print(f"[INFO] Loaded {self._ship_manager.count} ships from {ships_path}")
                    break
        
        # Fallback: Create sample ships if no JSON file found
        if not ships_loaded:
            print("[INFO] No ships.json found, creating sample ship route")
            
            # Create a simple route across the landscape
            route = Route(
                name="MainRoute",
                buoys=[
                    Buoy(x=10, y=height / 2, speed=12.0),
                    Buoy(x=width - 10, y=height / 2, speed=12.0),
                ]
            )
            
            # Create a sample ship
            ship = Ship(
                id=0,
                x=route.buoys[0].x,
                y=route.buoys[0].y,
                heading=90.0,
                name="CargoShip_1",
                vessel_type=VesselClass.CARGO,
                vessel_length=200.0,
                route=route
            )
            ship._is_active = True
            self._ship_manager.ships.append(ship)
        
    def step(self) -> None:
        """
        Execute one simulation step (30 minutes in DEPONS mode).

        Uses TimeManager for:
        - Deterministic per-tick seeding
        - Boundary detection (day/month/year)
        - Time advancement

        Vectorized implementation (Phase 3).
        """
        if not self._is_initialized:
            self.initialize()

        # 1. Set deterministic seed for this tick (CRITICAL for reproducibility)
        np.random.seed(self.time_manager.get_seed())

        # Debug first few steps
        if self.time_manager.tick < 5:
            print(f"[DEBUG] Simulation.step() tick={self.time_manager.tick}, pop={self.state.population}")

        # 2. Process scheduled events (JASMINE mode only, no-op in DEPONS mode)
        for event in self.time_manager.get_scheduled_events():
            event()

        # 3. Update turbines and ships for current tick
        self._turbine_manager.update(self.time_manager.tick)
        self._ship_manager.update(self.time_manager.tick)

        # 4. Calculate vectorized deterrence
        # Access arrays directly from population manager
        active_mask = self.population_manager.active_mask
        px = self.population_manager.x
        py = self.population_manager.y

        # Turbine deterrence (Vectorized)
        turb_dx, turb_dy = self._turbine_manager.calculate_aggregate_deterrence_vectorized(
            px, py, self.params, cell_size=400.0
        )

        # Ship deterrence (Vectorized)
        ship_dx, ship_dy = self._ship_manager.calculate_aggregate_deterrence_vectorized(
            px, py, self.params, is_day=self.time_manager.is_daytime, cell_size=400.0
        )

        # Combine
        total_dx = turb_dx + ship_dx
        total_dy = turb_dy + ship_dy

        # 5. Step population (Vectorized)
        self.population_manager.step(deterrence_vectors=(total_dx, total_dy))

        if self.time_manager.tick < 5:
            print(f"[DEBUG] Simulation.step() after pop_manager.step: active={self.population_manager.population_size}")

        # 6. Update Statistics
        current_pop = self.population_manager.population_size
        if current_pop != self.state.population:
            diff = current_pop - self.state.population
            if diff < 0:
                self.state.deaths += abs(diff)
            else:
                self.state.births += diff
            self.state.population = current_pop

        # 7. Advance time FIRST (so boundary checks work correctly)
        self.time_manager.advance()

        # 8. Sync legacy SimulationState from TimeManager
        self.state.tick = self.time_manager.tick
        self.state.day = self.time_manager.day
        self.state.month = self.time_manager.month
        self.state.year = self.time_manager.year
        self.state.quarter = self.time_manager.quarter

        # 9. Daily tasks (at day boundary)
        if self.time_manager.is_day_boundary():
            self._daily_tasks()

        # 10. Monthly tasks (at month boundary)
        if self.time_manager.is_month_boundary():
            self._monthly_tasks()

        # 11. Yearly tasks (at year boundary)
        if self.time_manager.is_year_boundary():
            self._yearly_tasks()

        # 12. Record history (daily)
        if self.time_manager.is_day_boundary():
            self._record_history()
            
    def _daily_tasks(self) -> None:
        """Execute daily tasks for all porpoises."""
        for porpoise in self._porpoises:
            if porpoise.alive:
                porpoise.daily_step(self._cell_data, self.params, self.state)
                
        # Create weaned calves as new agents
        new_calves = []
        next_id = max((p.id for p in self._porpoises), default=0) + 1
        for porpoise in self._porpoises:
            if hasattr(porpoise, '_calf_ready_to_wean') and porpoise._calf_ready_to_wean:
                from cenop.agents.porpoise import Porpoise
                # TRACE: 50% sex ratio for calves
                is_female = np.random.random() < 0.5
                calf = Porpoise(
                    id=next_id,
                    x=porpoise.x + np.random.uniform(-1, 1),
                    y=porpoise.y + np.random.uniform(-1, 1),
                    heading=np.random.uniform(0, 360),
                    age=0.0,
                    is_female=is_female
                )
                new_calves.append(calf)
                porpoise._calf_ready_to_wean = False
                next_id += 1
        self._porpoises.extend(new_calves)
                
        # Remove dead porpoises
        self._porpoises = [p for p in self._porpoises if p.alive]
        self.state.population = len(self._porpoises)
        
        # Replenish food across landscape
        if self._cell_data is not None:
            self._cell_data.replenish_food(self.params.r_u)
        
    def _monthly_tasks(self) -> None:
        """Execute monthly tasks."""
        # Reset monthly statistics
        self.state.births = 0
        self.state.deaths = 0
        
        # Update month in landscape data
        if self._cell_data is not None:
            self._cell_data.set_month(self.state.month)
        
    def _yearly_tasks(self) -> None:
        """Execute yearly tasks."""
        # Age all porpoises by 1 year
        for porpoise in self._porpoises:
            porpoise.age += 1.0
            
    def _record_history(self) -> None:
        """Record current state to history."""
        self._history.append({
            "tick": self.state.tick,
            "day": self.state.day,
            "year": self.state.year,
            "population": self.state.population,
            "births": self.state.births,
            "deaths": self.state.deaths,
        })
        
    def run(self, progress: bool = True) -> None:
        """
        Run the complete simulation.

        Uses TimeManager.is_finished() to determine completion.

        Args:
            progress: Show progress bar
        """
        if not self._is_initialized:
            self.initialize()

        self._is_running = True

        iterator = range(self.max_ticks)
        if progress:
            iterator = tqdm(iterator, desc="Simulating", unit="ticks")

        for _ in iterator:
            if not self._is_running or self.time_manager.is_finished():
                break
            self.step()

        self._is_running = False
        
    def stop(self) -> None:
        """Stop the simulation."""
        self._is_running = False
        
    def get_porpoise_positions(self) -> np.ndarray:
        """
        Get current positions and energy levels of all porpoises.
        
        Returns:
            Array of shape (N, 3) with [x, y, energy] for each porpoise
        """
        if not self._porpoises:
            return np.empty((0, 3))
            
        data = np.array([
            [p.x, p.y, p.energy_level]
            for p in self._porpoises
            if p.alive
        ])
        return data
        
    def get_population_history(self) -> Dict[str, List]:
        """Get population history as dictionary of lists."""
        if not self._history:
            return {"day": [], "population": [], "births": [], "deaths": []}
            
        return {
            "day": [h["day"] for h in self._history],
            "population": [h["population"] for h in self._history],
            "births": [h["births"] for h in self._history],
            "deaths": [h["deaths"] for h in self._history],
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get current simulation statistics."""
        return {
            "tick": self.state.tick,
            "day": self.state.day,
            "year": self.state.year,
            "population": self.state.population,
            "births_total": sum(h["births"] for h in self._history),
            "deaths_total": sum(h["deaths"] for h in self._history),
        }
        
    @property
    def cell_data(self) -> Optional[CellData]:
        """Get the landscape cell data."""
        return self._cell_data
        
    @property
    def porpoises(self) -> List[Any]:
        """Get list of all porpoises (Legacy compatibility)."""
        # Warning: This is slow if called repeatedly
        # Use population_manager or agents_df for performance
        if hasattr(self, 'population_manager'):
            # Return list of lightweight objects or named tuples
            from types import SimpleNamespace
            df = self.population_manager.to_dataframe()
            return [SimpleNamespace(**row) for row in df.to_dict('records')]
        return self._porpoises
    
    @property
    def agents(self) -> List[Any]:
        """Get list of all agents (alias for porpoises)."""
        return self.porpoises
        
    @property
    def agents_df(self) -> pd.DataFrame:
        """Get agents as DataFrame (Preferred for performance)."""
        if hasattr(self, 'population_manager'):
             return self.population_manager.to_dataframe()
        return pd.DataFrame()
    
    @property
    def population_size(self) -> int:
        """Get current population size."""
        if hasattr(self, 'population_manager'):
            return self.population_manager.population_size
        return len([p for p in self._porpoises if p.alive])
    
    @property
    def total_births(self) -> int:
        """Get total births across all history."""
        return sum(h.get("births", 0) for h in self._history)
    
    @property
    def total_deaths(self) -> int:
        """Get total deaths across all history."""
        return sum(h.get("deaths", 0) for h in self._history)
