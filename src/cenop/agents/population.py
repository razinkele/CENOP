"""
Vectorized Porpoise Population Manager.

This module implements a Structure-of-Arrays (SoA) approach to managing
the porpoise population efficiently using NumPy.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

from cenop.parameters.simulation_params import SimulationParameters
from cenop.landscape.cell_data import CellData
from cenop.parameters.demography import AGE_DISTRIBUTION_FREQUENCY
from cenop.behavior.psm import PersistentSpatialMemory


class PorpoisePopulation:
    """
    Manages the entire population of porpoises using vectorized numpy arrays.
    Replaces the list of individual Porpoise objects for performance.
    """
    
    def __init__(self, count: int, params: SimulationParameters, landscape: Optional[CellData] = None):
        print(f"[DEBUG] PorpoisePopulation.__init__: count={count}")
        self.params = params
        self.landscape = landscape
        self.count = count # Initial count capacity
        
        # === Arrays (Structure of Arrays) ===
        # Use a dictionary of arrays or direct attributes? Direct attributes are faster.
        
        # Identity
        self.ids = np.arange(count, dtype=np.int32)
        self.active_mask = np.ones(count, dtype=bool) # True if alive/active slot
        
        # Position
        self.x = np.zeros(count, dtype=np.float32)
        self.y = np.zeros(count, dtype=np.float32)
        self.heading = np.zeros(count, dtype=np.float32)
        
        # Movement State
        self.prev_log_mov = np.full(count, 0.8, dtype=np.float32)
        self.prev_angle = np.full(count, 10.0, dtype=np.float32)
        
        # Demography
        self.is_female = np.zeros(count, dtype=bool)
        self.age = np.zeros(count, dtype=np.float32)
        
        # Energy
        self.energy = np.full(count, 10.0, dtype=np.float32)
        
        # Reproduction
        self.mating_day = np.full(count, -99, dtype=np.int16)
        self.days_since_mating = np.full(count, -99, dtype=np.int16)
        self.days_since_birth = np.full(count, -99, dtype=np.int16)
        self.with_calf = np.zeros(count, dtype=bool)
        
        # Deterrence status
        self.deter_strength = np.zeros(count, dtype=np.float32)
        
        # === PSM and Dispersal State (Phase 2) ===
        # Energy history for dispersal trigger (5 days = 5*48 ticks)
        self._energy_history = np.zeros((count, 5), dtype=np.float32)  # Last 5 daily averages
        self._energy_ticks_today = np.zeros(count, dtype=np.float32)   # Energy sum for current day
        self._tick_counter = 0  # Track ticks for daily updates
        
        # Dispersal state
        self.is_dispersing = np.zeros(count, dtype=bool)
        self.days_declining_energy = np.zeros(count, dtype=np.int16)
        self.dispersal_target_x = np.zeros(count, dtype=np.float32)
        self.dispersal_target_y = np.zeros(count, dtype=np.float32)
        self.dispersal_target_distance = np.zeros(count, dtype=np.float32)
        self.dispersal_distance_traveled = np.zeros(count, dtype=np.float32)
        self.dispersal_start_x = np.zeros(count, dtype=np.float32)
        self.dispersal_start_y = np.zeros(count, dtype=np.float32)
        
        # PSM instances - one per porpoise (list for object storage)
        world_w = self.params.world_width
        world_h = self.params.world_height
        if landscape is not None:
            world_w = landscape.width
            world_h = landscape.height
        self._psm_instances: List[PersistentSpatialMemory] = [
            PersistentSpatialMemory(world_w, world_h) for _ in range(count)
        ]
        
        # Initialize
        self._initialize_population()
        print(f"[DEBUG] PorpoisePopulation initialized: active={self.population_size}, x[0:3]={self.x[:3]}")
        
    @property
    def population_size(self) -> int:
        """Current number of living porpoises."""
        return np.sum(self.active_mask)
        
    def _initialize_population(self):
        """Vectorized initialization logic with land avoidance."""
        # Random positions - must place in water (depth > 0)
        world_w = self.params.world_width
        world_h = self.params.world_height
        
        if self.landscape is None:
            # No landscape - use simple random positions
            self.x = np.random.uniform(0, world_w, self.count).astype(np.float32)
            self.y = np.random.uniform(0, world_h, self.count).astype(np.float32)
        else:
            # Use landscape - place only in water cells (depth >= min_depth)
            lw = self.landscape.width
            lh = self.landscape.height
            min_depth = self.params.min_depth if self.params else 1.0
            
            if hasattr(self.landscape, '_depth') and self.landscape._depth is not None:
                # Find all valid water cells (depth >= min_depth AND not NaN)
                # NaN values indicate land (-9999 NODATA converted to NaN during loading)
                valid_mask = (self.landscape._depth >= min_depth) & ~np.isnan(self.landscape._depth)
                valid_y, valid_x = np.where(valid_mask)
                
                if len(valid_x) > 0:
                    # Randomly select from valid positions
                    indices = np.random.choice(len(valid_x), self.count, replace=True)
                    self.x = valid_x[indices].astype(np.float32) + np.random.uniform(0, 1, self.count).astype(np.float32)
                    self.y = valid_y[indices].astype(np.float32) + np.random.uniform(0, 1, self.count).astype(np.float32)
                else:
                    # Fallback - no valid water cells (shouldn't happen)
                    self.x = np.random.uniform(0, lw, self.count).astype(np.float32)
                    self.y = np.random.uniform(0, lh, self.count).astype(np.float32)
            else:
                # No depth data - use full area
                self.x = np.random.uniform(0, lw, self.count).astype(np.float32)
                self.y = np.random.uniform(0, lh, self.count).astype(np.float32)
            
        self.heading = np.random.uniform(0, 360, self.count).astype(np.float32)
        
        # Sex ratio 50%
        self.is_female = np.random.choice([True, False], self.count)
        
        # Ages from distribution
        self.age = np.random.choice(
            AGE_DISTRIBUTION_FREQUENCY, 
            size=self.count
        ).astype(np.float32)
        
        # Mating day (females only, N(225, 20))
        mating_days = np.random.normal(225, 20, self.count).astype(np.int16)
        # Apply only to females, others stay -99
        self.mating_day = np.where(self.is_female, mating_days, -99)
        
    def step(self, deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Main simulation step for the entire population.
        Args:
            deterrence_vectors: Tuple of (dx_array, dy_array) for deterrence vectors
        """
        # Slice only active agents for calculations (for performance)
        # Or use boolean masking for vector ops
        mask = self.active_mask
        if not np.any(mask):
            return

        cnt = np.sum(mask)
        
        # === 1. Movement Calculations ===
        # Calculate Turning Angle (Simplified CRW)
        # DEPONS: angleTmp = b0 * prevAngle + N(0,4)
        # presAngle = angleTmp * (b1*depth + b2*salinity + b3)
        # Assuming homogeneous environment for first pass of vectorization
        
        # Random component
        rand_angle = np.random.normal(self.params.r2_mean, self.params.r2_sd, self.count)
        
        # Base angle change
        angle_base = self.params.corr_angle_base * self.prev_angle
        
        # Environmental modifiers (Simplified: 1.0 if no landscape)
        env_mod = 1.0 
        
        pres_angle = (angle_base + rand_angle) * (self.params.corr_angle_base_sd) 
        
        # Wrap angles > 180 or < -180 (simplified wrapping)
        pres_angle = np.where(pres_angle > 180, 180, pres_angle) # Placeholder logic from original
        
        self.heading[mask] += pres_angle[mask]
        self.heading[mask] %= 360.0
        
        # === Apply dispersal heading override for dispersing porpoises ===
        # Dispersing porpoises maintain heading toward target with reduced turning
        dispersing = mask & self.is_dispersing
        if np.any(dispersing):
            # PSM-Type2: reduced random turning during dispersal
            # Calculate distance progress for logistic dampening
            dx_disp = self.x - self.dispersal_start_x
            dy_disp = self.y - self.dispersal_start_y
            dist_traveled = np.sqrt(dx_disp**2 + dy_disp**2)
            
            # Safe division: avoid divide by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                dist_perc = np.where(
                    self.dispersal_target_distance > 0,
                    dist_traveled / self.dispersal_target_distance,
                    0.0
                )
            dist_perc = np.nan_to_num(dist_perc, nan=0.0, posinf=1.0, neginf=0.0)
            dist_perc = np.clip(dist_perc, 0.0, 2.0)  # Prevent extreme values
            
            # Logistic dampening: as progress -> 1, turning -> 0
            dist_log_x = 3 * dist_perc - 1.5
            # Clip to prevent overflow in exp() (exp(700) overflows)
            dist_log_x = np.clip(dist_log_x, -100, 100)
            log_mult = 1.0 / (1.0 + np.exp(0.6 * dist_log_x))
            
            # Reduce turning angle for dispersing porpoises
            dampened_angle = pres_angle * log_mult * 0.3  # 70% reduction base
            self.heading[dispersing] -= pres_angle[dispersing]  # Remove normal turn
            self.heading[dispersing] += dampened_angle[dispersing]  # Add dampened
            self.heading[dispersing] %= 360.0
        
        self.prev_angle[mask] = pres_angle[mask]
        
        # Calculate Step Length
        # log_mov = R1 + a0*last
        rand_len = np.random.normal(self.params.r1_mean, self.params.r1_sd, self.count)
        log_mov = rand_len + (self.params.corr_logmov_length * self.prev_log_mov)
        
        # Clip max speed
        log_mov = np.minimum(log_mov, self.params.max_mov)
        self.prev_log_mov[mask] = log_mov[mask]
        
        # Convert to distance (10^log_mov) / 4.0 (for 400m cell adjustment?)
        step_dist = (10.0 ** log_mov) / 4.0
        
        # Determine new positions
        rads = np.radians(self.heading)
        dx = np.sin(rads) * step_dist
        dy = np.cos(rads) * step_dist
        
        # Apply deterrence if exists
        if deterrence_vectors is not None:
            d_dx = deterrence_vectors[0]
            d_dy = deterrence_vectors[1]
            
            # Calculate magnitude for deterrence status tracking
            # If d_dx or d_dy is non-zero, the animal is deterred
            magnitude = np.abs(d_dx) + np.abs(d_dy)
            
            # We need to map the full-sized d_dx/d_dy arrays to the active mask
            # deterrence_vectors from manager are already full-sized arrays matching x/y
            
            self.deter_strength[mask] = magnitude[mask]
            
            dx[mask] += d_dx[mask]
            dy[mask] += d_dy[mask]
        else:
            self.deter_strength[mask] = 0.0
        
        # === LAND AVOIDANCE (DEPONS Pattern) ===
        # Check if new position would be on land (depth < 0) and avoid
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Clamp to bounds first
        world_w = self.landscape.width if self.landscape else self.params.world_width
        world_h = self.landscape.height if self.landscape else self.params.world_height
        np.clip(new_x, 0, world_w - 1, out=new_x)
        np.clip(new_y, 0, world_h - 1, out=new_y)
        
        if self.landscape:
            # Check depth at new positions - vectorized lookup
            new_xi = new_x.astype(int)
            new_yi = new_y.astype(int)
            np.clip(new_xi, 0, world_w - 1, out=new_xi)
            np.clip(new_yi, 0, world_h - 1, out=new_yi)
            
            # Get depth at proposed new positions
            if hasattr(self.landscape, '_depth') and self.landscape._depth is not None:
                depths = self.landscape._depth[new_yi, new_xi]
            else:
                depths = np.full(self.count, 20.0)  # Default to water
            
            # Identify agents that would move onto land (depth < min_depth OR NaN)
            # DEPONS uses min_depth = 1.0m as minimum water depth for porpoises
            # NaN values indicate land (-9999 NODATA converted to NaN during loading)
            min_depth = self.params.min_depth if self.params else 1.0
            on_land = ((depths < min_depth) | np.isnan(depths)) & mask
            
            if np.any(on_land):
                # For agents hitting land, try turning left or right
                # DEPONS pattern: check 40°, 70°, 120° turns, pick deeper side
                for turn_angle in [40, 70, 120]:
                    still_blocked = on_land.copy()
                    
                    # Try right turn
                    right_heading = (self.heading + turn_angle) % 360
                    right_rads = np.radians(right_heading)
                    right_dx = np.sin(right_rads) * step_dist
                    right_dy = np.cos(right_rads) * step_dist
                    right_x = np.clip(self.x + right_dx, 0, world_w - 1)
                    right_y = np.clip(self.y + right_dy, 0, world_h - 1)
                    
                    # Try left turn
                    left_heading = (self.heading - turn_angle) % 360
                    left_rads = np.radians(left_heading)
                    left_dx = np.sin(left_rads) * step_dist
                    left_dy = np.cos(left_rads) * step_dist
                    left_x = np.clip(self.x + left_dx, 0, world_w - 1)
                    left_y = np.clip(self.y + left_dy, 0, world_h - 1)
                    
                    if hasattr(self.landscape, '_depth') and self.landscape._depth is not None:
                        right_xi = right_x.astype(int)
                        right_yi = right_y.astype(int)
                        left_xi = left_x.astype(int)
                        left_yi = left_y.astype(int)
                        np.clip(right_xi, 0, world_w - 1, out=right_xi)
                        np.clip(right_yi, 0, world_h - 1, out=right_yi)
                        np.clip(left_xi, 0, world_w - 1, out=left_xi)
                        np.clip(left_yi, 0, world_h - 1, out=left_yi)
                        
                        right_depths = self.landscape._depth[right_yi, right_xi]
                        left_depths = self.landscape._depth[left_yi, left_xi]
                        
                        # Pick deeper direction if either is valid water (>= min_depth and not NaN)
                        right_ok = ((right_depths >= min_depth) & ~np.isnan(right_depths)) & still_blocked
                        left_ok = ((left_depths >= min_depth) & ~np.isnan(left_depths)) & still_blocked
                        both_ok = right_ok & left_ok
                        
                        # If both OK, pick deeper
                        use_right = both_ok & (right_depths >= left_depths)
                        use_left = both_ok & (left_depths > right_depths)
                        
                        # If only one OK
                        use_right = use_right | (right_ok & ~left_ok)
                        use_left = use_left | (left_ok & ~right_ok)
                        
                        # Update positions for those who found water
                        new_x[use_right] = right_x[use_right]
                        new_y[use_right] = right_y[use_right]
                        self.heading[use_right] = right_heading[use_right]
                        
                        new_x[use_left] = left_x[use_left]
                        new_y[use_left] = left_y[use_left]
                        self.heading[use_left] = left_heading[use_left]
                        
                        # Mark as no longer blocked
                        on_land[use_right | use_left] = False
                
                # For any still blocked, don't move at all (stay in place)
                new_x[on_land] = self.x[on_land]
                new_y[on_land] = self.y[on_land]
                # Turn around (180°)
                self.heading[on_land] = (self.heading[on_land] + 180) % 360
        
        # Apply the final positions
        self.x[mask] = new_x[mask]
        self.y[mask] = new_y[mask]
             
        # === 2. Energetics (DEPONS Pattern) ===
        # Every step is 30 mins (48 steps/day)
        # Reference: Porpoise.java updEnergeticStatus()
        
        # === 2a. Food Consumption ===
        # DEPONS: fractOfFoodToEat = (20 - energyLevel) / 10, max 0.99
        # Hungry porpoises eat more
        fract_to_eat = np.clip((20.0 - self.energy) / 10.0, 0.0, 0.99)
        
        # Get food from landscape if available
        if self.landscape is not None and hasattr(self.landscape, 'eat_food'):
            food_gained = self._eat_food_vectorized(mask, fract_to_eat)
        else:
            # Simplified fallback: random food based on hunger
            food_gained = fract_to_eat * np.random.uniform(0.1, 0.5, self.count)
            
        self.energy[mask] += food_gained[mask]
        
        # === 2b. Energy Consumption (BMR + Swimming) ===
        # DEPONS formula: consumed = 0.001 * scaling * EUsePer30Min + movementCost
        # Calculate seasonal scaling factor
        current_month = self._get_current_month()
        scaling_factor = self._get_energy_scaling(current_month, mask)
        
        # Base metabolic rate (BMR)
        bmr_cost = 0.001 * scaling_factor * self.params.e_use_per_30_min
        
        # Swimming cost (depends on movement speed)
        # DEPONS: 10^prevLogMov * 0.001 * scaling * E_USE_PER_KM / 0.4
        # E_USE_PER_KM = 0 in default DEPONS, so we use simplified version
        swimming_cost = (10.0 ** self.prev_log_mov) * 0.001 * scaling_factor * 0.0
        
        total_cost = bmr_cost + swimming_cost
        self.energy[mask] -= total_cost[mask]
        
        # === 2c. Update PSM with food obtained ===
        self._update_psm(mask, food_gained)
        
        # === 2d. Update energy history and check dispersal triggers ===
        self._update_energy_history(mask)
        
        # === 2e. Update dispersal progress ===
        self._update_dispersal(mask)
        
        # Clamp energy to reasonable range
        np.clip(self.energy, 0, 20.0, out=self.energy)
        
        # === 3. Mortality (DEPONS Pattern) ===
        # DEPONS: yearly survival = 1 - (M_MORT_PROB_CONST * exp(-energy * xSurvivalProbConst))
        # Per-step survival = yearly ^ (1 / (360 * 48))
        
        # Energy-based starvation mortality
        m_mort_prob_const = 0.5  # DEPONS default
        x_survival_const = 0.15   # DEPONS default
        
        yearly_surv_prob = np.where(
            self.energy > 0,
            1.0 - (m_mort_prob_const * np.exp(-self.energy * x_survival_const)),
            0.0
        )
        step_surv_prob = np.where(
            self.energy > 0,
            np.exp(np.log(np.maximum(yearly_surv_prob, 1e-10)) / (360 * 48)),
            0.0
        )
        
        # Starvation death check
        starvation_check = np.random.random(self.count)
        starving = (starvation_check > step_surv_prob) & mask
        
        # Lactating mothers abandon calf before dying
        abandon_calf = starving & self.with_calf
        self.with_calf[abandon_calf] = False
        
        # Non-lactating mothers die from starvation
        starved = starving & ~abandon_calf
        
        # Age-dependent natural mortality 
        # Convert annual probability to per-tick probability
        # Formula: daily_prob = 1 - (1 - annual_prob)^(1/365), then per-tick = daily / 48
        # Simplified: per_tick ≈ annual_prob / 365 / 48
        annual_juvenile_mortality = 0.15  # 15% for juveniles (age < 1)
        annual_elderly_mortality = 0.15   # 15% for elderly (age > 20)
        annual_adult_mortality = 0.05     # 5% for adults (realistic harbor porpoise)
        
        per_tick_juvenile = annual_juvenile_mortality / 365.0 / 48.0
        per_tick_elderly = annual_elderly_mortality / 365.0 / 48.0
        per_tick_adult = annual_adult_mortality / 365.0 / 48.0
        
        daily_mortality_prob = np.where(
            self.age < 1, per_tick_juvenile,
            np.where(self.age > 20, per_tick_elderly,
                     per_tick_adult)
        )
        natural_death = (np.random.random(self.count) < daily_mortality_prob) & mask
        
        # Bycatch mortality (if enabled)
        bycatch_prob = getattr(self.params, 'bycatch_prob', 0.0) / 365.0 / 48.0  # Annual to per-tick
        bycatch = (np.random.random(self.count) < bycatch_prob) & mask
        
        # Mark all deaths
        all_deaths = starved | natural_death | bycatch
        if np.any(all_deaths):
            self.active_mask[all_deaths] = False
        
        # === 4. Aging (once per day = every 48 ticks) ===
        # We'll age continuously in small increments
        self.age[mask] += 1.0 / 365.0 / 48.0  # Age in years per tick
        
        # === 5. Reproduction (simplified) ===
        # Mature females (age 4-20) can give birth once per year
        # Breeding season around day 225 +/- 30
        if hasattr(self, '_day_of_year'):
            self._day_of_year = (self._day_of_year + 1) % (365 * 48)
        else:
            self._day_of_year = 0
        
        current_day = self._day_of_year // 48
        
        # Check for births during breeding season (days 195-255)
        if 195 <= current_day <= 255:
            # Eligible females: age 4-20, not already with calf
            eligible = mask & self.is_female & (self.age >= 4) & (self.age <= 20) & ~self.with_calf
            
            # Per-tick birth probability during the 60-day breeding season
            # Target: ~60% of eligible females give birth each year
            # Over 60 days * 48 ticks = 2880 ticks
            # Probability per tick to achieve 60% over season: 1 - (1-p)^2880 = 0.6
            # Solving: p ≈ 0.0003
            birth_prob = 0.0003  # Per tick during breeding
            giving_birth = (np.random.random(self.count) < birth_prob) & eligible
            
            if np.any(giving_birth):
                # Find inactive slots for new calves
                inactive_slots = np.where(~self.active_mask)[0]
                birth_count = np.sum(giving_birth)
                mother_indices = np.where(giving_birth)[0]
                
                slots_to_use = min(birth_count, len(inactive_slots))
                if slots_to_use > 0:
                    new_slots = inactive_slots[:slots_to_use]
                    mothers = mother_indices[:slots_to_use]
                    
                    # Activate new calves
                    self.active_mask[new_slots] = True
                    self.x[new_slots] = self.x[mothers]
                    self.y[new_slots] = self.y[mothers]
                    self.heading[new_slots] = self.heading[mothers]
                    self.age[new_slots] = 0.0
                    self.is_female[new_slots] = np.random.choice([True, False], size=int(slots_to_use))
                    self.energy[new_slots] = 10.0
                    self.with_calf[mothers] = True
                    self.with_calf[new_slots] = False

    def to_dataframe(self) -> pd.DataFrame:
        """Export active agents to DataFrame for UI helpers."""
        mask = self.active_mask
        n_active = np.sum(mask)
        return pd.DataFrame({
            'id': self.ids[mask],
            'x': self.x[mask],
            'y': self.y[mask],
            'age': self.age[mask],
            'is_female': self.is_female[mask],
            'energy': self.energy[mask],
            'alive': np.ones(n_active, dtype=bool)
        })

    # === PSM and Dispersal Methods (Phase 2) ===
    
    def _update_psm(self, mask: np.ndarray, food_gained: np.ndarray) -> None:
        """
        Update Persistent Spatial Memory for each porpoise.
        
        Records food obtained at current location for dispersal targeting.
        
        Args:
            mask: Active porpoise mask
            food_gained: Array of food gained this tick
        """
        active_indices = np.where(mask)[0]
        for idx in active_indices:
            self._psm_instances[idx].update(
                self.x[idx],
                self.y[idx],
                food_gained[idx]
            )
            
    def _update_energy_history(self, mask: np.ndarray) -> None:
        """
        Track energy history for dispersal trigger calculation.
        
        DEPONS triggers dispersal when energy declines for t_disp days (default 5).
        We track daily average energy and check for declining trends.
        """
        # Accumulate energy for current day
        self._energy_ticks_today[mask] += self.energy[mask]
        self._tick_counter += 1
        
        # At end of day (48 ticks), update history
        if self._tick_counter >= 48:
            self._tick_counter = 0
            
            # Calculate daily average
            daily_avg = self._energy_ticks_today / 48.0
            
            # Shift history and add new day
            self._energy_history[:, 1:] = self._energy_history[:, :-1]
            self._energy_history[:, 0] = daily_avg
            
            # Reset daily accumulator
            self._energy_ticks_today[:] = 0.0
            
            # Check for declining energy trend (5 consecutive days)
            self._check_dispersal_trigger(mask)
            
    def _check_dispersal_trigger(self, mask: np.ndarray) -> None:
        """
        Check if dispersal should trigger based on energy decline.
        
        DEPONS Pattern:
        - If energy has declined for t_disp consecutive days (default 5)
        - And porpoise has sufficient memory (50+ cells visited)
        - Then trigger dispersal to remembered high-food area
        """
        t_disp = getattr(self.params, 't_disp', 5)  # Days before dispersal triggers
        min_memory_cells = 50  # Minimum PSM cells for dispersal
        
        for idx in np.where(mask)[0]:
            if self.is_dispersing[idx]:
                continue  # Already dispersing
                
            # Check for declining energy over t_disp days
            history = self._energy_history[idx, :t_disp]
            if np.all(history[:-1] < history[1:]):  # Each day lower than previous
                # Check if enough memory
                psm = self._psm_instances[idx]
                if psm.visited_cell_count >= min_memory_cells:
                    self._start_dispersal(idx)
                    
    def _start_dispersal(self, idx: int) -> None:
        """
        Start dispersal behavior for a single porpoise.
        
        Uses PSM to find target cell at approximately preferred distance.
        """
        self.is_dispersing[idx] = True
        self.dispersal_start_x[idx] = self.x[idx]
        self.dispersal_start_y[idx] = self.y[idx]
        self.dispersal_distance_traveled[idx] = 0.0
        
        psm = self._psm_instances[idx]
        
        # Find target from PSM
        cell_size = 400.0  # meters per cell
        target = psm.get_target_cell_for_dispersal(
            self.x[idx], self.y[idx], 
            tolerance=5.0, cell_size=cell_size
        )
        
        if target is not None:
            self.dispersal_target_x[idx] = target[0]
            self.dispersal_target_y[idx] = target[1]
            dist_km = target[2]
            self.dispersal_target_distance[idx] = dist_km * 1000 / cell_size  # Convert to cells
        else:
            # Use random target at preferred distance
            random_target = psm.get_random_target(
                self.x[idx], self.y[idx], cell_size
            )
            self.dispersal_target_x[idx] = random_target[0]
            self.dispersal_target_y[idx] = random_target[1]
            self.dispersal_target_distance[idx] = psm.preferred_distance * 1000 / cell_size
            
        # Set heading toward target
        dx = self.dispersal_target_x[idx] - self.x[idx]
        dy = self.dispersal_target_y[idx] - self.y[idx]
        self.heading[idx] = np.degrees(np.arctan2(dx, dy))
        
    def _update_dispersal(self, mask: np.ndarray) -> None:
        """
        Update dispersal progress for dispersing porpoises.
        
        Check if target distance reached and end dispersal if so.
        """
        dispersing = mask & self.is_dispersing
        if not np.any(dispersing):
            return
            
        # Calculate distance from start
        dx = self.x - self.dispersal_start_x
        dy = self.y - self.dispersal_start_y
        distances = np.sqrt(dx**2 + dy**2)
        
        # Check for completion (95% of target distance - PSM-Type2 rule)
        completed = dispersing & (distances >= 0.95 * self.dispersal_target_distance)
        
        if np.any(completed):
            self.is_dispersing[completed] = False
            self.dispersal_distance_traveled[completed] = 0.0
            self.days_declining_energy[completed] = 0
            
    def get_psm(self, idx: int) -> PersistentSpatialMemory:
        """Get PSM instance for a specific porpoise."""
        return self._psm_instances[idx]
        
    def get_dispersal_stats(self) -> Dict[str, Any]:
        """Get statistics about dispersal behavior."""
        active = self.active_mask
        return {
            'dispersing_count': int(np.sum(self.is_dispersing & active)),
            'total_active': int(np.sum(active)),
            'avg_psm_cells': float(np.mean([
                self._psm_instances[i].visited_cell_count 
                for i in np.where(active)[0]
            ])) if np.any(active) else 0.0,
            'max_declining_days': int(np.max(self.days_declining_energy[active])) if np.any(active) else 0
        }

    # === Phase 3: Enhanced Energetics Methods ===
    
    def _eat_food_vectorized(self, mask: np.ndarray, fract_to_eat: np.ndarray) -> np.ndarray:
        """
        Eat food from landscape cells (vectorized).
        
        DEPONS Pattern: Each porpoise eats a fraction of available food
        based on their hunger level.
        
        Args:
            mask: Active porpoise mask
            fract_to_eat: Fraction of food to eat per porpoise
            
        Returns:
            Array of food eaten by each porpoise
        """
        food_eaten = np.zeros(self.count, dtype=np.float32)
        
        if self.landscape is None:
            return food_eaten
            
        # Get food at each porpoise position
        active_indices = np.where(mask)[0]
        
        for idx in active_indices:
            x, y = float(self.x[idx]), float(self.y[idx])
            food = self.landscape.eat_food(x, y, float(fract_to_eat[idx]))
            food_eaten[idx] = food
            
        return food_eaten
        
    def _get_current_month(self) -> int:
        """
        Get current month of simulation (1-12).
        
        Based on tick counter (48 ticks/day, ~30 days/month).
        """
        if not hasattr(self, '_day_of_year'):
            return 1
            
        day = self._day_of_year // 48
        # Approximate month (30 days each)
        month = (day // 30) % 12 + 1
        return month
        
    def _get_energy_scaling(self, month: int, mask: np.ndarray) -> np.ndarray:
        """
        Calculate energy scaling factor based on season and lactation.
        
        DEPONS Pattern:
        - Nov-Mar (cold): 1.0 (baseline)
        - Apr, Oct: 1.15 (transition)
        - May-Sep (warm): 1.3 (e_warm)
        - Lactating females: multiply by 1.4 (e_lact)
        
        Args:
            month: Current month (1-12)
            mask: Active porpoise mask
            
        Returns:
            Scaling factor array for each porpoise
        """
        scaling = np.ones(self.count, dtype=np.float32)
        
        # Seasonal scaling
        if month == 4 or month == 10:
            # April and October - transition months
            scaling[:] = 1.15
        elif 5 <= month <= 9:
            # May through September - warm months
            scaling[:] = self.params.e_warm
        # Nov-Mar stays at 1.0 (cold months, lower metabolism)
        
        # Lactation scaling (40% increase)
        lactating = self.with_calf & mask
        scaling[lactating] *= self.params.e_lact
        
        return scaling
        
    def get_energy_stats(self) -> Dict[str, Any]:
        """Get statistics about population energy levels."""
        active = self.active_mask
        if not np.any(active):
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'hungry': 0, 'starving': 0}
            
        active_energy = self.energy[active]
        return {
            'mean': float(np.mean(active_energy)),
            'std': float(np.std(active_energy)),
            'min': float(np.min(active_energy)),
            'max': float(np.max(active_energy)),
            'hungry': int(np.sum(active_energy < 10)),  # Below neutral
            'starving': int(np.sum(active_energy < 5))  # Critical
        }
