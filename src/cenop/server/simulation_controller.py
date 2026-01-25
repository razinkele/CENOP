"""
Simulation Controller for CENOP

Handles simulation creation, stepping, and lifecycle management.
"""

from typing import Optional
import numpy as np
import logging

from cenop import Simulation, SimulationParameters
from cenop.landscape import CellData, create_homogeneous_landscape, create_landscape_from_depons

logger = logging.getLogger("CENOP")


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
    except Exception:
        pass  # Use defaults
    
    # Read and validate input values with safe defaults
    porpoise_count_val = input.porpoise_count()
    sim_years_val = input.sim_years()
    
    # Handle None or invalid values
    if porpoise_count_val is None or porpoise_count_val < 1:
        porpoise_count_val = 1000  # Default
        print(f"[WARNING] porpoise_count was None/invalid, using default: {porpoise_count_val}")
    if sim_years_val is None or sim_years_val < 1:
        sim_years_val = 5  # Default
        print(f"[WARNING] sim_years was None/invalid, using default: {sim_years_val}")
    
    # Ensure integer types
    porpoise_count_val = int(porpoise_count_val)
    sim_years_val = int(sim_years_val)
    
    print(f"[DEBUG] create_simulation_from_inputs: porpoise_count={porpoise_count_val}, sim_years={sim_years_val}")
    
    # Calculate and log max_ticks for verification
    expected_max_ticks = sim_years_val * 360 * 48
    print(f"[DEBUG] Expected max_ticks = {sim_years_val} years * 360 days * 48 ticks = {expected_max_ticks}")
        
    params = SimulationParameters(
        porpoise_count=porpoise_count_val,
        sim_years=sim_years_val,
        landscape=input.landscape(),
        turbines=input.turbines(),
        ships_enabled=input.ships_enabled(),
        dispersal=input.dispersal(),
        random_seed=seed_value if seed_value > 0 else None,
        
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
    
    sim = Simulation(params, cell_data=landscape)
    logger.info(f"Simulation created: {params.porpoise_count} porpoises, {params.sim_years} years")
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
        print(f"[DEBUG] SimulationRunner.__init__: last_pop={self.last_pop}, max_ticks={self.max_ticks}")
    
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
            print(f"[DEBUG] step_ticks #{self.update_count}: tick={self.tick}, stepping {ticks_to_step} ticks...")
        
        for i in range(ticks_to_step):
            if self.tick >= self.max_ticks:
                break
            self.sim.step()
            self.tick += 1
        
        # Calculate metrics
        current_pop = self.sim.state.population
        
        if self.update_count < 3:
            print(f"[DEBUG] step_ticks #{self.update_count}: after stepping, pop={current_pop}, tick={self.tick}")
        
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
                print(f"[WARNING] Large deaths increment: {inc} at tick {self.tick}, pop={current_pop}")
        if current_pop > self.last_pop:
            inc = (current_pop - self.last_pop)
            self.total_births += inc
            if inc > max(1000, self.sim.population_size * 2):
                print(f"[WARNING] Large births increment: {inc} at tick {self.tick}, pop={current_pop}")

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
