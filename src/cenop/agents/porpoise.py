"""
Porpoise agent implementation.

Main agent in the simulation representing individual harbour porpoises.
Translates from: Porpoise.java (1686 lines)
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List, Tuple, Deque
from enum import Enum

from cenop.agents.base import Agent

if TYPE_CHECKING:
    from cenop.landscape.cell_data import CellData
    from cenop.parameters.simulation_params import SimulationParameters
    from cenop.core.simulation import SimulationState


class PregnancyStatus(Enum):
    """Pregnancy status of a porpoise."""
    
    UNABLE_YOUNG = 0      # Unable to mate (too young or low energy)
    PREGNANT = 1          # Currently pregnant
    READY_TO_MATE = 2     # Ready to mate


# ── DEPONS Model Constants ──────────────────────────────────────────────────
# These constants match the original Java DEPONS implementation.
DAYS_PER_YEAR = 360                 # DEPONS uses a 360-day year
TICKS_PER_DAY = 48                  # 48 half-hour steps per day
TICKS_PER_YEAR = DAYS_PER_YEAR * TICKS_PER_DAY  # 17,280

MAX_CRW_ATTEMPTS = 200              # Max retries for CRW angle/step sampling
ANGLE_SENTINEL = 999.0              # Sentinel value indicating "not yet computed"
STEP_SENTINEL = 999.0               # Sentinel value for step length loop
NOT_PREGNANT_SENTINEL = -99         # Sentinel for days_since_mating / days_since_giving_birth

LAND_AVOIDANCE_ANGLES = [40, 70, 120]  # Degrees to try when avoiding land
FALLBACK_ANGLE_MIN = 90             # Fallback angle range if all CRW retries exhausted
FALLBACK_ANGLE_RANGE = 20           # Random offset added to fallback angle

TRANSITION_MONTH_MULTIPLIER = 1.15  # Energy multiplier for April/October (matches Java)
CALF_SEX_RATIO = 0.5                # 50% probability of female calf


class CauseOfDeath(Enum):
    """Cause of death for mortality tracking."""

    STARVATION = "starvation"
    OLD_AGE = "old_age"
    BYCATCH = "bycatch"


@dataclass
class Porpoise(Agent):
    """
    Main porpoise agent.
    
    Represents an individual harbour porpoise with movement behavior,
    energy dynamics, reproduction, and response to disturbances.
    
    Translates from: Porpoise.java
    """
    
    # === Identity (inherited from Agent: id, x, y, heading) ===
    is_female: bool = True              # Sex (True=female, False=male)
    
    # === Movement state ===
    prev_log_mov: float = 0.8       # Previous log10(movement/100) - DEPONS default
    pres_log_mov: float = 0.0       # Current log10(movement/100)
    prev_angle: float = 10.0        # Previous turning angle - DEPONS default
    pres_angle: float = 0.0         # Current turning angle
    
    # === Energy ===
    energy_level: float = field(default=10.0)
    energy_consumed_daily: float = 0.0
    energy_consumed_daily_temp: float = 0.0
    food_eaten_daily: float = 0.0
    food_eaten_daily_temp: float = 0.0
    
    # Daily energy buffer (last 10 days) - use deque for O(1) appendleft
    energy_level_daily: Deque[float] = field(default_factory=lambda: deque([0.0] * 10, maxlen=10))
    energy_level_sum: float = 0.0
    
    # === Age and life stage ===
    age: float = 0.0                    # Age in years
    age_of_maturity: float = 3.44       # Age when becoming receptive
    
    # === Reproduction ===
    pregnancy_status: PregnancyStatus = PregnancyStatus.UNABLE_YOUNG
    mating_day: int = 225               # Day of year for mating
    days_since_mating: int = NOT_PREGNANT_SENTINEL   # sentinel if not pregnant
    days_since_giving_birth: int = NOT_PREGNANT_SENTINEL  # sentinel if no calf
    with_lact_calf: bool = False        # With lactating calf
    calves_born: int = 0                # Counter
    calves_weaned: int = 0              # Counter
    
    # === Deterrence ===
    deter_vt: np.ndarray = field(default_factory=lambda: np.zeros(2))
    deter_strength: float = 0.0
    deter_time_left: int = 0
    ignore_deterrence: int = 0
    
    # === Memory ===
    vt: np.ndarray = field(default_factory=lambda: np.zeros(2))  # Attraction vector
    ve_total: float = 0.0              # Expected food value
    stored_util_list: List[float] = field(default_factory=list)
    
    # Position history - use deque for O(1) appendleft
    pos_list: Deque[tuple] = field(default_factory=lambda: deque(maxlen=120))
    pos_list_daily: Deque[tuple] = field(default_factory=lambda: deque([(0, 0)] * 10, maxlen=10))
    
    # === Dispersal ===
    is_dispersing: bool = False
    disp_num_ticks: int = 0
    dispersal_type: int = 0            # 0=off, 1-3=PSM types
    
    # === State ===
    alive: bool = True
    tick_move_adjust: float = 1.0      # Movement adjustment for partial steps
    _calf_ready_to_wean: bool = False  # Flag for simulation to create new agent
    
    def __post_init__(self):
        """Initialize computed fields."""
        # Initialize energy from random
        if self.energy_level == 10.0:
            self.energy_level = np.random.normal(10.0, 1.0)
            
        # Initialize mating day (only meaningful for females per TRACE)
        # Males don't need a mating day as only females can become pregnant
        if self.is_female:
            self.mating_day = int(np.random.normal(225, 20))
        else:
            self.mating_day = NOT_PREGNANT_SENTINEL  # Not applicable for males
        
        # Initialize position list
        if not self.pos_list:
            self.pos_list.append((self.x, self.y))
            
    def step(
        self,
        cell_data: Optional[CellData],
        params: SimulationParameters,
        state: SimulationState,
        deterrence: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Execute one simulation step (30 minutes).
        
        Args:
            cell_data: Landscape data
            params: Simulation parameters
            state: Current simulation state
            deterrence: Optional (dx, dy) deterrence vector from external sources
        """
        if not self.alive:
            return
            
        # Store external deterrence for use in movement
        self._external_deterrence = deterrence
            
        # Reset movement adjustment for new step
        self.tick_move_adjust = 1.0
        
        # Store current position (deque with maxlen auto-removes old entries)
        self.pos_list.appendleft((self.x, self.y))
            
        # Choose movement type
        if self.is_dispersing:
            self._dispersal_step(cell_data, params)
        else:
            self._standard_move(cell_data, params)
            
        # Update energetics (pass current month for seasonal energy calculation)
        self._update_energetic_status(cell_data, params, state.month)
        
    def _standard_move(
        self,
        cell_data: Optional[CellData],
        params: SimulationParameters
    ) -> None:
        """
        Perform correlated random walk movement.
        
        Translates from: Porpoise.stdMove()
        """
        # Calculate turning angle
        self.pres_angle = self._calculate_turning_angle(cell_data, params)
        
        # Apply turn
        self.heading += self.pres_angle
        self.heading = Agent.normalize_heading(self.heading)
        
        # Calculate step length
        self.pres_log_mov = self._calculate_step_length(cell_data, params)
        
        # Convert to actual distance (in 400m cells)
        pres_mov = 10 ** self.pres_log_mov
        move_distance = self.tick_move_adjust * (pres_mov / 4.0)
        
        # Check for land and avoid if necessary
        if cell_data is not None:
            if not self._check_water_ahead(cell_data, move_distance, params):
                self._avoid_land(cell_data, params)
                
        # Apply reference memory attraction
        if params.model >= 2:
            self._apply_memory_attraction(params)
            
        # Apply deterrence
        if params.model >= 3 and self.ignore_deterrence <= 0:
            self._apply_deterrence(params)
            
        # Move forward
        self.forward(move_distance)
        
        # Update previous values
        self.prev_angle = self.pres_angle
        self.prev_log_mov = self.pres_log_mov
        self.tick_move_adjust = 0.0
        
    def _calculate_turning_angle(
        self,
        cell_data: Optional[CellData],
        params: SimulationParameters
    ) -> float:
        """
        Calculate turning angle for CRW movement.
        
        Formula from DEPONS: 
        angleTmp = b0 * prevAngle + random(R2)
        presAngle = angleTmp * (b1 * depth + b2 * salinity + b3)
        
        R2 = N(0, 4) from parameters.xml
        """
        # Get environmental values
        if cell_data is not None:
            depth = cell_data.get_depth(self.x, self.y)
            salinity = cell_data.get_salinity(self.x, self.y)
        else:
            depth = 20.0  # Default depth
            salinity = 30.0  # Default salinity
            
        attempts = 0
        pres_angle = ANGLE_SENTINEL

        while abs(pres_angle) > 180 and attempts < MAX_CRW_ATTEMPTS:
            # R2 = N(0, 4) per TRACE Table 2 - turning angle random component
            random_angle = np.random.normal(params.r2_mean, params.r2_sd)
            
            angle_base = params.corr_angle_base * self.prev_angle
            angle_bathy = params.corr_angle_bathy * depth
            angle_salinity = params.corr_angle_salinity * salinity
            
            angle_tmp = angle_base + random_angle
            pres_angle = angle_tmp * (angle_bathy + angle_salinity + params.corr_angle_base_sd)
            attempts += 1
            
        if abs(pres_angle) > 180:
            pres_angle = np.sign(pres_angle) * 90
        
        # Second adjustment loop (based on prevMov and M)
        # DEPONS: Makes turning angle decrease linearly with movement distance
        # When prevMov <= M, angle is adjusted by: angle + rnd - (rnd * prevMov / M)
        sign = 1.0 if pres_angle >= 0 else -1.0
        pres_angle = abs(pres_angle)
        
        # M = 10^0.74 ≈ 5.495 in DEPONS (limit for when turning angles stop decreasing with speed)
        prev_mov = 10 ** self.prev_log_mov
        
        attempts = 0
        while pres_angle >= 180 and attempts < MAX_CRW_ATTEMPTS:
            # N(0, 1) for angle adjustment - hardcoded in DEPONS (R3)
            rnd = np.random.normal(0, 1)
            if prev_mov <= params.m:
                pres_angle = pres_angle + rnd - (rnd * prev_mov / params.m)
            attempts += 1
            
        if pres_angle >= 180:
            pres_angle = np.random.uniform(0, FALLBACK_ANGLE_RANGE) + FALLBACK_ANGLE_MIN
            
        return pres_angle * sign
        
    def _calculate_step_length(
        self,
        cell_data: Optional[CellData],
        params: SimulationParameters
    ) -> float:
        """
        Calculate log10 step length.
        
        TRACE Document Eqn A31:
        δt = R1 + a0*δt-1 + a1*wt-1 + a2*st-1
        where x = 10^δt * 100 (length in meters)
        
        R1 = N(1.25, 0.15) per TRACE Table 2
        """
        # Get environmental values
        if cell_data is not None:
            depth = cell_data.get_depth(self.x, self.y)
            salinity = cell_data.get_salinity(self.x, self.y)
        else:
            depth = 20.0  # Default depth
            salinity = 30.0  # Default salinity
            
        attempts = 0
        log_mov = STEP_SENTINEL

        while log_mov > params.max_mov and attempts < MAX_CRW_ATTEMPTS:
            # R1 random component - TRACE Table 2: R1 = N(1.25, 0.15)
            # This is the mean and SD for log10(d/100) per van Beest et al. 2018a
            random_length = np.random.normal(params.r1_mean, params.r1_sd)
            
            # TRACE Eqn A31: δt = R1 + a0*δt-1 + a1*wt-1 + a2*st-1
            log_mov_length = params.corr_logmov_length * self.prev_log_mov
            log_mov_bathy = params.corr_logmov_bathy * depth
            log_mov_salinity = params.corr_logmov_salinity * salinity
            
            log_mov = random_length + log_mov_length + log_mov_bathy + log_mov_salinity
            attempts += 1
            
        # Clamp to max if still too large
        if log_mov > params.max_mov:
            log_mov = params.max_mov
        
        return log_mov
        
    def _check_water_ahead(
        self,
        cell_data: CellData,
        distance: float,
        params: SimulationParameters
    ) -> bool:
        """Check if there is enough water depth ahead."""
        ahead_x, ahead_y = self.get_point_ahead(distance)
        
        # Check bounds
        if not cell_data.is_valid_position(ahead_x, ahead_y):
            return False
            
        depth = cell_data.get_depth(ahead_x, ahead_y)
        return depth >= params.min_depth
        
    def _avoid_land(
        self,
        cell_data: CellData,
        params: SimulationParameters
    ) -> None:
        """
        Avoid land by turning.
        
        Translates from: Porpoise.avoidLand()
        """
        pres_mov = 10 ** self.pres_log_mov
        
        # Try turning at increasing angles
        for angle in LAND_AVOIDANCE_ANGLES:
            rand_offset = np.random.uniform(0, 10)
            
            # Check right
            right_x, right_y = self.get_point_ahead(pres_mov, angle + rand_offset)
            right_ok = (
                cell_data.is_valid_position(right_x, right_y) and
                cell_data.get_depth(right_x, right_y) >= params.min_depth
            )
            
            # Check left
            left_x, left_y = self.get_point_ahead(pres_mov, -(angle + rand_offset))
            left_ok = (
                cell_data.is_valid_position(left_x, left_y) and
                cell_data.get_depth(left_x, left_y) >= params.min_depth
            )
            
            if right_ok or left_ok:
                if right_ok and left_ok:
                    # Turn towards deeper water
                    right_depth = cell_data.get_depth(right_x, right_y)
                    left_depth = cell_data.get_depth(left_x, left_y)
                    if right_depth >= left_depth:
                        self.heading += angle + rand_offset
                    else:
                        self.heading -= angle + rand_offset
                elif right_ok:
                    self.heading += angle + rand_offset
                else:
                    self.heading -= angle + rand_offset
                    
                self.heading = Agent.normalize_heading(self.heading)
                return
                
        # If all else fails, backtrack
        if self.pos_list:
            prev_x, prev_y = self.pos_list[0]
            self.face_point(prev_x, prev_y)
            
    def _apply_memory_attraction(self, params: SimulationParameters) -> None:
        """
        Apply reference memory attraction vector.
        
        TRACE Document Eqn A34:
        VM = Σ M[c] × i[c]
        
        where M is the remembered food value in patch c (weighted by travel cost),
        and i is a unity vector pointing toward patch c.
        
        The memory attraction is combined with CRW using Eqn A35.
        """
        # Check if there's any memory attraction
        if abs(self.vt[0]) < 0.001 and abs(self.vt[1]) < 0.001:
            return
            
        # CRW contribution: k + |VS| × VE (expected food value)
        vs_length = 10 ** self.pres_log_mov
        crw_contrib = params.inertia_const + vs_length * self.ve_total
        
        # Combine CRW direction with memory attraction (VM)
        # VS component: CRW direction scaled by contribution
        total_dx = self.get_dx() * crw_contrib + self.vt[0]
        total_dy = self.get_dy() * crw_contrib + self.vt[1]
        
        # Face the resultant direction
        resultant_length = np.sqrt(total_dx**2 + total_dy**2)
        if resultant_length > 0.001:
            new_heading = np.degrees(np.arctan2(total_dx, total_dy))
            self.heading = Agent.normalize_heading(new_heading)
            
    def _apply_deterrence(self, params: SimulationParameters) -> None:
        """
        Apply deterrence vector from noise sources.
        
        TRACE Document Eqn A35:
        V* = |VS| × (VS + VM + VD) / |VS + VM + VD|
        
        The resultant vector is normalized to have the same length as the
        original CRW step (VS), so noise affects direction but NOT step length.
        """
        # Get external deterrence (from turbines/ships) if provided
        ext_dx, ext_dy = 0.0, 0.0
        if hasattr(self, '_external_deterrence') and self._external_deterrence is not None:
            ext_dx, ext_dy = self._external_deterrence
            
        # Combine internal and external deterrence
        has_internal = self.deter_strength > 0
        has_external = abs(ext_dx) > 0.001 or abs(ext_dy) > 0.001
        
        if not has_internal and not has_external:
            return
        
        # Original CRW step length |VS| for normalization (TRACE Eqn A35)
        vs_length = 10 ** self.pres_log_mov
            
        crw_contrib = params.inertia_const + vs_length * self.ve_total
        
        # Internal deterrence vector
        int_dx = self.deter_vt[0] if has_internal else 0.0
        int_dy = self.deter_vt[1] if has_internal else 0.0
        
        # VS + VM + VD (unnormalized resultant)
        total_dx = self.get_dx() * crw_contrib + self.vt[0] + int_dx + ext_dx
        total_dy = self.get_dy() * crw_contrib + self.vt[1] + int_dy + ext_dy
        
        # TRACE Eqn A35: Normalize resultant to have same length as VS
        # V* = |VS| × (VS + VM + VD) / |VS + VM + VD|
        resultant_length = np.sqrt(total_dx**2 + total_dy**2)
        
        if resultant_length > 0.001:
            # Normalize and scale to original step length
            # This ensures deterrence affects direction but not step length
            normalized_dx = total_dx / resultant_length * vs_length
            normalized_dy = total_dy / resultant_length * vs_length
            
            new_heading = np.degrees(np.arctan2(normalized_dx, normalized_dy))
            self.heading = Agent.normalize_heading(new_heading)
            
    def _dispersal_step(
        self,
        cell_data: Optional[CellData],
        params: SimulationParameters
    ) -> None:
        """
        Perform dispersal movement.
        
        Translates from: DispersalPSMType2.disperse()
        """
        # Simplified implementation
        # Full version requires PSM (Persistent Spatial Memory) calculations
        
        self.disp_num_ticks += 1
        
        # Move in dispersal direction
        move_distance = params.mean_disp_dist / 4.0  # Convert km to cells
        
        # Add some randomness
        random_angle = np.random.normal(0, params.psm_angle)
        self.heading += random_angle
        self.heading = Agent.normalize_heading(self.heading)
        
        self.forward(move_distance)
        
    def _get_seasonal_energy_multiplier(self, month: int, params: SimulationParameters) -> float:
        """
        Get season-dependent energy multiplier.
        
        DEPONS Java (Porpoise.java lines 692-699) uses hardcoded values:
        - Months 5-9 (May-Sep): Ewarm (1.3)
        - Months 4, 10 (Apr, Oct): 1.15 (hardcoded in Java)
        - Other months (Nov-Mar): 1.0
        
        Note: TRACE document gives formula 0.5*(1-Ewarm)+1 = 0.85 for transition,
        but Java uses 1.15. We match Java for behavioral consistency.
        """
        if 5 <= month <= 9:  # May-September: warm water
            return params.e_warm  # 1.3
        elif month == 4 or month == 10:  # April, October: transition
            # DEPONS Java uses hardcoded 1.15 for transition months
            return TRANSITION_MONTH_MULTIPLIER
        else:  # November-March: cold water
            return 1.0
    
    def _update_energetic_status(
        self,
        cell_data: Optional[CellData],
        params: SimulationParameters,
        current_month: int = 6
    ) -> None:
        """
        Update energy based on food intake and consumption.
        
        TRACE Document Section 2.7.4:
        - Animals consume food when moving through patches
        - Energy consumption depends on season and lactation status
        
        Args:
            cell_data: Landscape data
            params: Simulation parameters
            current_month: Current month (1-12) for seasonal energy calculation
        """
        # Calculate food intake based on energy deficit
        food_eaten = 0.0
        fract_of_food_to_eat = 0.0
        
        if self.energy_level < 20:
            fract_of_food_to_eat = (20.0 - self.energy_level) / 10.0
        if fract_of_food_to_eat > 0.99:
            fract_of_food_to_eat = 0.99
            
        if cell_data is not None and fract_of_food_to_eat > 0:
            food_eaten = cell_data.eat_food(self.x, self.y, fract_of_food_to_eat)
                
        self.food_eaten_daily_temp += food_eaten
        self.energy_level += food_eaten
        
        # Calculate scaling factor based on season and lactation (TRACE Section 2.7.4)
        scaling_factor = self._get_seasonal_energy_multiplier(current_month, params)
        
        # Lactation multiplier
        if self.with_lact_calf:
            scaling_factor *= params.e_lact
            
        # Calculate energy consumed using DEPONS formula:
        # consumed = (0.001 * scalingFactor * Euse) + (10^prevLogMov * 0.001 * scalingFactor * E_USE_PER_KM / 0.4)
        # where E_USE_PER_KM = 0.0 (hardcoded in SimulationConstants)
        e_use_per_km = 0.0  # SimulationConstants.E_USE_PER_KM
        consumed = (0.001 * scaling_factor * params.e_use_per_30_min + 
                   (10 ** self.prev_log_mov) * 0.001 * scaling_factor * e_use_per_km / 0.4)
        
        self.energy_consumed_daily_temp += consumed
        self.energy_level -= consumed
        
        # Track for daily average
        self.energy_level_sum += self.energy_level
        
    def daily_step(
        self,
        cell_data: Optional[CellData],
        params: SimulationParameters,
        state: SimulationState
    ) -> None:
        """
        Perform daily updates.
        
        Translates from: Porpoise.performDailyStep()
        """
        if not self.alive:
            return
            
        # Update daily energy tracking (deque with maxlen auto-removes old entries)
        daily_avg = self.energy_level_sum / TICKS_PER_DAY
        self.energy_level_daily.appendleft(daily_avg)
        self.energy_level_sum = 0.0

        # Update daily position (deque with maxlen auto-removes old entries)
        self.pos_list_daily.appendleft((self.x, self.y))
            
        # Finalize daily consumption
        self.energy_consumed_daily = self.energy_consumed_daily_temp
        self.food_eaten_daily = self.food_eaten_daily_temp
        self.energy_consumed_daily_temp = 0.0
        self.food_eaten_daily_temp = 0.0
        
        # Check mortality
        self._check_mortality(params, state)
        
        # Update reproduction
        self._update_reproduction(params, state)
        
        # Check dispersal triggers
        self._check_dispersal(params)
        
        # Update deterrence decay
        self._update_deterrence(params)
        
        # Age by 1/DAYS_PER_YEAR year
        self.age += 1.0 / DAYS_PER_YEAR
        
    def _check_mortality(
        self,
        params: SimulationParameters,
        state: SimulationState
    ) -> None:
        """Check and apply mortality.
        
        TRACE Document Eqn A36-A37:
        - Yearly survival: sy = 1 / (1 + exp(-β * Ep))
        - Per-step survival: ss = sy^(1/17280)  where 17280 = 360 days * 48 ticks/day
        """
        # Check old age
        if self.age >= params.max_age:
            self._die(CauseOfDeath.OLD_AGE, state)
            return
            
        # Check starvation (DEPONS Porpoise.java lines 741-759)
        # DEPONS formula: yearlySurvProb = 1 - m_mort_prob_const * exp(-energyLevel * x_survival_const)
        # This differs from TRACE logistic formula but matches Java implementation
        m_mort_prob_const = getattr(params, 'm_mort_prob_const', 0.5)
        x_survival_const = getattr(params, 'x_survival_const', 0.15)
        yearly_survival = 1.0 - (m_mort_prob_const * np.exp(-self.energy_level * x_survival_const))

        # Convert to per-day survival (360 days per year, consistent with DEPONS)
        # DEPONS checks per-tick (48/day) but we check per-day for efficiency
        daily_survival = yearly_survival ** (1.0 / DAYS_PER_YEAR) if self.energy_level > 0 else 0.0
        
        if np.random.random() > daily_survival:
            # DEPONS: Nursing mothers can abandon calf to survive
            if self.with_lact_calf and self.energy_level > 0:
                # Abandon calf rather than dying
                self.with_lact_calf = False
                self.days_since_giving_birth = NOT_PREGNANT_SENTINEL
            else:
                self._die(CauseOfDeath.STARVATION, state)
                return
            
        # Check bycatch (DEPONS: uses compounded probability, not simple division)
        if params.bycatch_prob > 0:
            # DEPONS: dailySurvivalProb = exp(log(1 - bycatchProb) / DAYS_PER_YEAR)
            daily_bycatch_survival = np.exp(np.log(1.0 - params.bycatch_prob) / DAYS_PER_YEAR)
            if np.random.random() > daily_bycatch_survival:
                self._die(CauseOfDeath.BYCATCH, state)
                return
                
    def _die(self, cause: CauseOfDeath, state: SimulationState) -> None:
        """Mark porpoise as dead.

        Note: do NOT increment `state.deaths` here to avoid double-counting.
        The simulation detects population decreases and updates total deaths
        centrally. We still increment cause-specific counters for diagnostics.
        """
        self.alive = False
        
        # Increment cause-specific counters only
        if cause == CauseOfDeath.STARVATION:
            state.deaths_starvation += 1
        elif cause == CauseOfDeath.OLD_AGE:
            state.deaths_old_age += 1
        elif cause == CauseOfDeath.BYCATCH:
            state.deaths_bycatch += 1
            
    def _update_reproduction(
        self,
        params: SimulationParameters,
        state: SimulationState
    ) -> None:
        """Update reproduction state.
        
        TRACE: Only females can become pregnant and give birth.
        Males skip reproduction entirely.
        """
        # Males don't participate in reproduction
        if not self.is_female:
            return
            
        day_of_year = state.day % DAYS_PER_YEAR
        
        # Check if ready to mate
        if self.age >= self.age_of_maturity and self.pregnancy_status == PregnancyStatus.UNABLE_YOUNG:
            self.pregnancy_status = PregnancyStatus.READY_TO_MATE
            
        # Check mating - DEPONS uses exact day match
        if (self.pregnancy_status == PregnancyStatus.READY_TO_MATE and
            day_of_year == self.mating_day):
            if np.random.random() < params.conceive_prob:
                self.pregnancy_status = PregnancyStatus.PREGNANT
                self.days_since_mating = 0
                
        # Progress pregnancy
        if self.pregnancy_status == PregnancyStatus.PREGNANT:
            self.days_since_mating += 1
            
            # Give birth
            if self.days_since_mating >= params.gestation_time:
                self._give_birth(state)
                
        # Progress nursing
        if self.with_lact_calf:
            self.days_since_giving_birth += 1
            
            # Wean calf
            if self.days_since_giving_birth >= params.nursing_time:
                self.with_lact_calf = False
                self.days_since_giving_birth = NOT_PREGNANT_SENTINEL
                
                # DEPONS: 50% sex ratio - only female calves are tracked
                # Random check: if > 0.5, calf is female and added to population
                if np.random.random() > 0.5:
                    self.calves_weaned += 1
                    # Do NOT increment `state.births` here to avoid double-counting.
                    # The Simulation will account for population increases centrally
                    # when the active population size grows.
                    self._calf_ready_to_wean = True  # Flag for simulation to create new agent
                # else: calf is male, not tracked in simulation
                
    def _give_birth(self, state: SimulationState) -> None:
        """
        Give birth to a calf.
        
        DEPONS: After giving birth, pregnancy_status = 2 (READY_TO_MATE)
        so the female can mate again while still nursing.
        """
        self.pregnancy_status = PregnancyStatus.READY_TO_MATE  # Can mate while nursing
        self.days_since_mating = NOT_PREGNANT_SENTINEL
        self.with_lact_calf = True
        self.days_since_giving_birth = 0
        self.calves_born += 1
        self._calf_ready_to_wean = False
        
    def _check_dispersal(self, params: SimulationParameters) -> None:
        """
        Check if dispersal should be activated/deactivated.
        
        TRACE: Animals start large-scale movement (dispersal) when their
        energy has been declining for t_disp consecutive days.
        
        The energy_level_daily list stores daily energy levels with
        index 0 being the most recent day.
        """
        if len(self.energy_level_daily) < params.t_disp + 1:
            return
            
        # Check for t_disp consecutive days of declining energy
        # energy_level_daily[0] is today, [1] is yesterday, etc.
        declining_days = 0
        for i in range(params.t_disp):
            # Compare day i with day i+1 (i is more recent)
            if self.energy_level_daily[i] < self.energy_level_daily[i + 1]:
                declining_days += 1
            else:
                break  # Must be consecutive
                
        # Activate dispersal if energy has been declining for t_disp days
        if declining_days >= params.t_disp and not self.is_dispersing:
            self.is_dispersing = True
            self.dispersal_type = 2  # PSM-Type2
            self.disp_num_ticks = 0
            
        # Deactivate if energy is improving (not declining anymore)
        elif self.is_dispersing:
            # Check if the most recent day shows improvement
            if len(self.energy_level_daily) >= 2:
                if self.energy_level_daily[0] >= self.energy_level_daily[1]:
                    self.is_dispersing = False
                    self.dispersal_type = 0
            
    def _update_deterrence(self, params: SimulationParameters) -> None:
        """Update deterrence decay."""
        if self.deter_time_left > 0:
            self.deter_time_left -= 1
            self.deter_strength *= (100 - params.deter_decay) / 100.0
            self.deter_vt /= 2.0
        else:
            self.deter_strength = 0.0
            self.deter_vt[:] = 0.0
            
        if self.ignore_deterrence > 0:
            self.ignore_deterrence -= 1
            
    def deter(
        self,
        strength: float,
        source_x: float,
        source_y: float,
        params: SimulationParameters
    ) -> None:
        """
        Apply deterrence from a noise source.
        
        Args:
            strength: Deterrence strength
            source_x: X position of noise source
            source_y: Y position of noise source
            params: Simulation parameters
        """
        if strength > self.deter_strength:
            self.deter_strength = strength
            
            # Vector pointing away from source
            dx = self.x - source_x
            dy = self.y - source_y
            
            self.deter_vt[0] = strength * dx * params.deter_coeff
            self.deter_vt[1] = strength * dy * params.deter_coeff
            
            self.deter_time_left = params.deter_time
            
        # Stop dispersal when deterred
        if self.is_dispersing:
            self.is_dispersing = False
            self.dispersal_type = 0

    @property
    def energy(self) -> float:
        """Alias for energy_level for compatibility."""
        return self.energy_level
    
    @property
    def is_alive(self) -> bool:
        """Alias for alive for compatibility."""
        return self.alive
