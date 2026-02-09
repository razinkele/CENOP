"""
Simulation Controller for CENOP

Handles simulation creation, stepping, and lifecycle management.
"""

from typing import Optional
import numpy as np
import logging

from cenop import Simulation, SimulationParameters
from cenop.landscape import CellData, create_homogeneous_landscape, create_landscape_from_depons
from cenop.core.time_manager import TimeManager, TimeMode
from cenop.movement import MovementMode, create_movement_module
from cenop.behavior import FSMMode, create_behavior_fsm, MemoryMode, create_memory_module
from cenop.physiology import EnergyMode, create_energy_module

logger = logging.getLogger("CENOP")


def _safe_float(getter, default: float) -> float:
    """Safely get a float value from an input getter with a default."""
    try:
        val = getter()
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _get_time_mode(mode_str: str) -> TimeMode:
    """Convert string mode to TimeMode enum."""
    if mode_str.upper() == "JASMINE":
        return TimeMode.JASMINE
    return TimeMode.DEPONS


def _get_movement_mode(mode_str: str) -> MovementMode:
    """Convert string mode to MovementMode enum."""
    if mode_str.upper() == "JASMINE":
        return MovementMode.JASMINE_PHYSICS
    return MovementMode.DEPONS_CRW


def _get_fsm_mode(mode_str: str) -> FSMMode:
    """Convert string mode to FSMMode enum."""
    if mode_str.upper() == "JASMINE":
        return FSMMode.JASMINE
    return FSMMode.DEPONS


def _get_energy_mode(mode_str: str) -> EnergyMode:
    """Convert string mode to EnergyMode enum."""
    if mode_str.upper() == "JASMINE":
        return EnergyMode.JASMINE
    return EnergyMode.DEPONS


def _get_memory_mode(mode_str: str) -> MemoryMode:
    """Convert string mode to MemoryMode enum."""
    if mode_str.upper() == "JASMINE":
        return MemoryMode.JASMINE
    return MemoryMode.DEPONS


def create_simulation_from_inputs(input) -> Simulation:
    """
    Create a new simulation instance from Shiny input values.
    
    Args:
        input: Shiny input object with all form values
        
    Returns:
        Configured Simulation instance
    """
    seed_value = input.random_seed()
    
    # Parse PSM Dist string "N(300;100)"
    psm_dist_str = input.psm_dist()
    psm_dist_mean = 300.0
    psm_dist_sd = 100.0
    try:
        if psm_dist_str and psm_dist_str.startswith("N(") and psm_dist_str.endswith(")"):
            inner = psm_dist_str[2:-1]
            parts = inner.split(";")
            if len(parts) == 2:
                psm_dist_mean = float(parts[0])
                psm_dist_sd = float(parts[1])
    except (ValueError, TypeError, AttributeError):
        pass  # Use defaults for PSM dist parsing
    
    # Read and validate input values with safe defaults
    porpoise_count_val = input.porpoise_count()
    sim_years_val = input.sim_years()
    
    # Handle None or invalid values
    if porpoise_count_val is None or porpoise_count_val < 1:
        porpoise_count_val = 1000  # Default
        logger.warning("porpoise_count was None/invalid, using default: %d", porpoise_count_val)
    if sim_years_val is None or sim_years_val < 1:
        sim_years_val = 5  # Default
        logger.warning("sim_years was None/invalid, using default: %d", sim_years_val)
    
    # Ensure integer types
    porpoise_count_val = int(porpoise_count_val)
    sim_years_val = int(sim_years_val)
    
    logger.debug("create_simulation_from_inputs: porpoise_count=%d, sim_years=%d", porpoise_count_val, sim_years_val)
    
    # Calculate and log max_ticks for verification
    expected_max_ticks = sim_years_val * 360 * 48
    logger.debug("Expected max_ticks = %d years * 360 days * 48 ticks = %d", sim_years_val, expected_max_ticks)

    # Read JASMINE mode settings
    simulation_mode = input.simulation_mode() or "DEPONS"

    # Read subsystem mode overrides (empty string means follow main mode)
    time_mode_override = input.time_mode_override() or None
    movement_mode_override = input.movement_mode_override() or None
    fsm_mode_override = input.fsm_mode_override() or None
    energy_mode_override = input.energy_mode_override() or None
    memory_mode_override = input.memory_mode_override() or None

    # Read JASMINE-specific parameters with safe defaults
    jasmine_mass_kg = _safe_float(input.jasmine_mass_kg, 50.0)
    jasmine_drag_coeff = _safe_float(input.jasmine_drag_coeff, 0.01)
    jasmine_max_thrust = _safe_float(input.jasmine_max_thrust, 100.0)
    jasmine_current_weight = _safe_float(input.jasmine_current_weight, 0.5)
    jasmine_bmr_scale = _safe_float(input.jasmine_bmr_scale, 1.0)
    jasmine_activity_cost = _safe_float(input.jasmine_activity_cost, 2.0)
    jasmine_disturbance_cost = _safe_float(input.jasmine_disturbance_cost, 1.5)
    jasmine_memory_decay_rate = _safe_float(input.jasmine_memory_decay_rate, 0.001)
    jasmine_avoidance_strength = _safe_float(input.jasmine_avoidance_strength, 0.8)
    jasmine_avoidance_radius = _safe_float(input.jasmine_avoidance_radius, 20.0)

    params = SimulationParameters(
        porpoise_count=porpoise_count_val,
        sim_years=sim_years_val,
        landscape=input.landscape(),
        turbines=input.turbines(),
        ships_enabled=input.ships_enabled(),
        dispersal=input.dispersal(),
        random_seed=seed_value if seed_value > 0 else None,

        # JASMINE Mode Selection
        simulation_mode=simulation_mode,
        time_mode=time_mode_override,
        movement_mode=movement_mode_override,
        fsm_mode=fsm_mode_override,
        energy_mode=energy_mode_override,
        memory_mode=memory_mode_override,

        # JASMINE Physics Parameters
        jasmine_mass_kg=jasmine_mass_kg,
        jasmine_drag_coeff=jasmine_drag_coeff,
        jasmine_max_thrust=jasmine_max_thrust,
        jasmine_current_weight=jasmine_current_weight,

        # JASMINE DEB Parameters
        jasmine_bmr_scale=jasmine_bmr_scale,
        jasmine_activity_cost=jasmine_activity_cost,
        jasmine_disturbance_cost=jasmine_disturbance_cost,

        # JASMINE Memory Parameters
        jasmine_memory_decay_rate=jasmine_memory_decay_rate,
        jasmine_avoidance_strength=jasmine_avoidance_strength,
        jasmine_avoidance_radius=jasmine_avoidance_radius,

        # Advanced Parameters
        tracked_porpoise_count=input.tracked_porpoise_count(),
        t_disp=input.tdisp(),
        psm_log=input.psm_log(),
        psm_dist_mean=psm_dist_mean,
        psm_dist_sd=psm_dist_sd,
        psm_tol=input.psm_tol(),
        psm_angle=input.psm_angle(),
        
        # Memory & Energy
        r_s=input.param_rS(),
        r_r=input.param_rR(),
        r_u=input.param_rU(),
        
        # Survival
        bycatch_prob=input.bycatch_prob(),
        
        # Movement Coefficients (CRW)
        inertia_const=input.param_k(),
        corr_logmov_length=input.param_a0(),
        corr_logmov_bathy=input.param_a1(),
        corr_logmov_salinity=input.param_a2(),
        corr_angle_base=input.param_b0(),
        corr_angle_bathy=input.param_b1(),
        corr_angle_salinity=input.param_b2(),
        corr_angle_base_sd=input.param_b3(),

        # --- Communication / Social params ---
        communication_enabled=input.communication_enabled(),
        communication_range_km=input.communication_range_km(),
        communication_source_level=input.communication_source_level(),
        communication_threshold=input.communication_threshold(),
        communication_response_slope=input.communication_response_slope(),
        social_weight=input.social_weight(),
    )
    
    # Create landscape
    if params.is_homogeneous:
        landscape = create_homogeneous_landscape()
    elif params.landscape == "NorthSea":
        landscape = create_landscape_from_depons()
    else:
        landscape = CellData(params.landscape)

    # Determine effective modes for each subsystem
    effective_time_mode = _get_time_mode(params.get_effective_time_mode())
    effective_movement_mode = _get_movement_mode(params.get_effective_movement_mode())
    effective_fsm_mode = _get_fsm_mode(params.get_effective_fsm_mode())
    effective_energy_mode = _get_energy_mode(params.get_effective_energy_mode())
    effective_memory_mode = _get_memory_mode(params.get_effective_memory_mode())

    logger.info("Simulation modes: time=%s, movement=%s, fsm=%s, energy=%s, memory=%s",
                effective_time_mode.name, effective_movement_mode.name,
                effective_fsm_mode.name, effective_energy_mode.name, effective_memory_mode.name)

    # Create modules based on effective modes
    movement_module = create_movement_module(params, effective_time_mode, effective_movement_mode)
    behavior_fsm = create_behavior_fsm(params, effective_fsm_mode)
    energy_module = create_energy_module(params, effective_energy_mode)
    memory_module = create_memory_module(params, effective_memory_mode)

    # Create simulation with all modules
    sim = Simulation(
        params,
        cell_data=landscape,
        time_mode=effective_time_mode,
        movement_module=movement_module,
        behavior_fsm=behavior_fsm,
        energy_module=energy_module,
        memory_module=memory_module,
    )
    logger.info("Simulation created: %d porpoises, %d years, mode=%s",
                params.porpoise_count, params.sim_years, simulation_mode)
    return sim


class SimulationRunner:
    """
    Manages the simulation execution lifecycle.
    
    Tracks internal state for tick counting, birth/death tracking,
    and history accumulation.
    """
    
    def __init__(self, simulation: Simulation):
        self.sim = simulation
        self.tick = 0
        self.max_ticks = simulation.max_ticks
        self.history = []
        self.energy_history = []  # Track energy data separately
        self.total_births = 0
        self.total_deaths = 0
        self.last_pop = simulation.state.population
        self.update_count = 0
        self.ticks_per_update = 1  # Default to 1 tick per update for smooth animation
        logger.debug("SimulationRunner.__init__: last_pop=%d, max_ticks=%d", self.last_pop, self.max_ticks)
    
    def set_ticks_per_update(self, ticks: int):
        """Set the number of ticks to advance per update (1-48)."""
        self.ticks_per_update = max(1, min(48, ticks))
        
    def step_ticks(self) -> dict:
        """
        Advance the simulation by configured number of ticks.
        
        Returns:
            Dictionary with current state metrics
        """
        ticks_to_step = self.ticks_per_update
        
        if self.update_count < 3:
            logger.debug("step_ticks #%d: tick=%d, stepping %d ticks...", self.update_count, self.tick, ticks_to_step)
        
        for i in range(ticks_to_step):
            if self.tick >= self.max_ticks:
                break
            self.sim.step()
            self.tick += 1
        
        # Calculate metrics
        current_pop = self.sim.state.population
        
        if self.update_count < 3:
            logger.debug("step_ticks #%d: after stepping, pop=%d, tick=%d", self.update_count, current_pop, self.tick)
        
        # Count lactating porpoises with calves
        lact_calf_count = 0
        if hasattr(self.sim, 'population_manager'):
            pop = self.sim.population_manager
            if hasattr(pop, 'with_calf'):
                lact_calf_count = int(np.sum(pop.with_calf & pop.active_mask))
        elif self.sim._porpoises:
            lact_calf_count = sum(
                1 for p in self.sim._porpoises 
                if hasattr(p, 'with_lact_calf') and p.with_lact_calf
            )
        
        # Track births and deaths
        if current_pop < self.last_pop:
            inc = (self.last_pop - current_pop)
            self.total_deaths += inc
            if inc > max(1000, self.sim.population_size * 2):
                logger.warning("Large deaths increment: %d at tick %d, pop=%d", inc, self.tick, current_pop)
        if current_pop > self.last_pop:
            inc = (current_pop - self.last_pop)
            self.total_births += inc
            if inc > max(1000, self.sim.population_size * 2):
                logger.warning("Large births increment: %d at tick %d, pop=%d", inc, self.tick, current_pop)

        self.last_pop = current_pop
        
        # Calculate energy statistics
        avg_food_eaten = 0.0
        avg_energy_expended = 0.0
        if hasattr(self.sim, 'population_manager'):
            pm = self.sim.population_manager
            if hasattr(pm, 'energy') and hasattr(pm, 'active_mask'):
                active = pm.active_mask
                if np.any(active):
                    # Average energy level (as proxy for food eaten - energy balance)
                    avg_energy = float(np.mean(pm.energy[active]))
                    # Use energy level as food eaten proxy, energy use per step as expended
                    avg_food_eaten = avg_energy
                    # Default energy use is 4.5 per 30-min step, so per day (48 steps) = 216
                    avg_energy_expended = 4.5 * 48  # Daily energy expenditure estimate
        
        # Create energy history entry
        energy_entry = {
            'day': self.sim.state.day,
            'avg_food_eaten': avg_food_eaten,
            'avg_energy_expended': avg_energy_expended
        }
        self.energy_history.append(energy_entry)
        
        # Collect dispersal statistics
        dispersal_entry = None
        deterred_count = 0
        if hasattr(self.sim, 'population_manager'):
            pm = self.sim.population_manager
            # Get dispersal stats
            if hasattr(pm, 'get_dispersal_stats'):
                disp_stats = pm.get_dispersal_stats()
                dispersal_entry = {
                    'day': self.sim.state.day,
                    'dispersing_count': disp_stats.get('dispersing_count', 0),
                    'total_active': disp_stats.get('total_active', 0),
                    'max_declining_days': disp_stats.get('max_declining_days', 0)
                }
            # Get deterrence count
            if hasattr(pm, 'deter_strength'):
                deterred_count = int(np.sum(pm.deter_strength[pm.active_mask] > 0))
        
        # Create history entry
        entry = {
            'day': self.sim.state.day,
            'tick': self.sim.state.tick,
            'year': self.sim.state.year,
            'population': current_pop,
            'lact_calf': lact_calf_count,
            'births': self.total_births,
            'deaths': self.total_deaths,
            'energy_entry': energy_entry,  # Include energy data in entry
            'dispersal_entry': dispersal_entry,  # Include dispersal data
            'deterred_count': deterred_count  # Include deterrence count
        }
        self.history.append(entry)
        
        self.update_count += 1
        
        return entry
    
    @property
    def is_complete(self) -> bool:
        """Check if simulation has finished."""
        return self.tick >= self.max_ticks
    
    @property
    def progress_percent(self) -> float:
        """Get current progress as percentage."""
        return (self.tick / self.max_ticks) * 100 if self.max_ticks > 0 else 0
    
    @property
    def should_update_map(self) -> bool:
        """Check if map should be updated.

        To reduce overhead we avoid updating the map every tick. The UI
        visualization only needs to update at coarser intervals (e.g. daily).
        This property can be adjusted later for finer control.
        """
        # Update map approximately once per day (every 48 ticks)
        return (self.tick % 48) == 0
