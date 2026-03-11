"""
Energy Budget module for CENOP-JASMINE hybrid simulation.

This module provides energy tracking and metabolic calculations:
- DEPONS mode: Simple energy tracking (backward compatible)
- JASMINE mode: Full Dynamic Energy Budget (DEB) model

Key features of JASMINE DEB model:
- Body mass dependent metabolism
- Activity-specific metabolic rates
- Cost of transport (swimming speed dependent)
- Disturbance energy costs
- Cumulative impact tracking for fitness assessment

Reference:
- Nabe-Nielsen et al. (2018) - DEPONS energy model
- JASMINE-MB Technical Documentation - DEB model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple
import numpy as np

from cenop.parameters.constants import SimulationConstants

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters
    from cenop.behavior.states import BehaviorState


class EnergyMode(Enum):
    """Energy calculation modes."""
    DEPONS = auto()    # Simple energy tracking
    JASMINE = auto()   # Full DEB model
    HYBRID = auto()    # Context-dependent


@dataclass
class EnergyState:
    """
    Energy state for a population of agents.

    Tracks all energy-related variables for vectorized processing.
    """
    # Core energy (DEPONS compatible)
    energy: np.ndarray              # Current energy level (0-20 scale)

    # Body condition (JASMINE extension)
    body_mass: np.ndarray           # Body mass in kg
    body_condition: np.ndarray      # Body condition index (0-1)
    fat_reserve: np.ndarray         # Fat reserve in kg

    # Activity tracking
    activity_level: np.ndarray      # Current activity level (0-1)
    distance_traveled: np.ndarray   # Distance traveled this tick (m)

    # Disturbance impact tracking
    disturbance_energy_cost: np.ndarray  # Cumulative disturbance energy cost
    disturbance_events: np.ndarray       # Count of disturbance events

    # Fitness tracking
    cumulative_energy_deficit: np.ndarray  # Total energy shortfall

    @classmethod
    def create(cls, count: int, initial_energy: float = 10.0) -> 'EnergyState':
        """Create energy state for count agents."""
        return cls(
            energy=np.full(count, initial_energy, dtype=np.float32),
            body_mass=np.full(count, SimulationConstants.DEFAULT_BODY_MASS_KG, dtype=np.float32),
            body_condition=np.full(count, SimulationConstants.DEFAULT_BODY_CONDITION, dtype=np.float32),
            fat_reserve=np.full(count, SimulationConstants.DEFAULT_BODY_MASS_KG * SimulationConstants.DEFAULT_FAT_FRACTION, dtype=np.float32),
            activity_level=np.full(count, 0.5, dtype=np.float32),
            distance_traveled=np.zeros(count, dtype=np.float32),
            disturbance_energy_cost=np.zeros(count, dtype=np.float32),
            disturbance_events=np.zeros(count, dtype=np.int32),
            cumulative_energy_deficit=np.zeros(count, dtype=np.float32),
        )


@dataclass
class EnergyContext:
    """
    Environmental and behavioral context for energy calculations.

    Contains inputs needed for energy budget updates.
    """
    # Food availability
    food_available: np.ndarray      # Food available at current location
    food_quality: np.ndarray        # Food quality factor (0-1)

    # Activity context
    current_speed: np.ndarray       # Current swimming speed (m/s)
    behavioral_state: np.ndarray    # BehaviorState enum values

    # Environmental context
    water_temperature: np.ndarray   # Water temperature (°C)
    current_month: int              # Current month (1-12)

    # Disturbance context
    is_disturbed: np.ndarray        # Currently under disturbance
    deterrence_magnitude: np.ndarray  # Strength of deterrence

    # Reproduction context
    is_lactating: np.ndarray        # Currently lactating
    is_pregnant: np.ndarray         # Currently pregnant

    @classmethod
    def create_default(cls, count: int, month: int = 1) -> 'EnergyContext':
        """Create default context for count agents."""
        return cls(
            food_available=np.full(count, 0.5, dtype=np.float32),
            food_quality=np.ones(count, dtype=np.float32),
            current_speed=np.zeros(count, dtype=np.float32),
            behavioral_state=np.ones(count, dtype=np.int32),  # FORAGING
            water_temperature=np.full(count, 10.0, dtype=np.float32),
            current_month=month,
            is_disturbed=np.zeros(count, dtype=bool),
            deterrence_magnitude=np.zeros(count, dtype=np.float32),
            is_lactating=np.zeros(count, dtype=bool),
            is_pregnant=np.zeros(count, dtype=bool),
        )


@dataclass
class EnergyResult:
    """
    Result of energy budget calculation.

    Contains energy changes and derived values.
    """
    # Energy flows
    energy_intake: np.ndarray       # Energy gained from food
    energy_bmr: np.ndarray          # Basal metabolic cost
    energy_activity: np.ndarray     # Activity-related cost
    energy_thermoregulation: np.ndarray  # Thermoregulation cost
    energy_reproduction: np.ndarray  # Reproduction cost
    energy_disturbance: np.ndarray  # Disturbance-related cost

    # Net change
    net_energy_change: np.ndarray   # Total energy change

    # Derived metrics
    energy_balance: np.ndarray      # Positive/negative balance
    survival_probability: np.ndarray  # Current survival probability

    @property
    def total_cost(self) -> np.ndarray:
        """Total energy cost this tick."""
        return (self.energy_bmr + self.energy_activity +
                self.energy_thermoregulation + self.energy_reproduction +
                self.energy_disturbance)


class EnergyModule(ABC):
    """
    Abstract base class for energy budget modules.

    Defines the interface for energy calculations that can be
    implemented differently for DEPONS and JASMINE modes.
    """

    def __init__(self, params: 'SimulationParameters'):
        """
        Initialize energy module.

        Args:
            params: Simulation parameters
        """
        self.params = params

    @abstractmethod
    def compute_energy_update(
        self,
        state: EnergyState,
        context: EnergyContext,
        mask: np.ndarray,
        dt_seconds: int = 1800,
    ) -> EnergyResult:
        """
        Compute energy budget update.

        Args:
            state: Current energy state
            context: Environmental/behavioral context
            mask: Active agent mask
            dt_seconds: Timestep in seconds

        Returns:
            EnergyResult with all energy flows
        """
        pass

    @abstractmethod
    def apply_result(
        self,
        state: EnergyState,
        result: EnergyResult,
        mask: np.ndarray,
    ) -> None:
        """
        Apply energy result to state.

        Args:
            state: Energy state to update
            result: Computed energy changes
            mask: Active agent mask
        """
        pass

    @abstractmethod
    def compute_survival_probability(
        self,
        state: EnergyState,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute survival probability based on energy state.

        Args:
            state: Current energy state
            mask: Active agent mask

        Returns:
            Per-tick survival probability array
        """
        pass

    @abstractmethod
    def get_mode(self) -> EnergyMode:
        """Return the energy calculation mode."""
        pass

    def get_fitness_metrics(
        self,
        state: EnergyState,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Get fitness metrics for the population.

        Override in subclasses for mode-specific metrics.
        """
        active = mask
        if not np.any(active):
            return {}

        return {
            'mean_body_condition': float(np.mean(state.body_condition[active])),
            'total_disturbance_cost': float(np.sum(state.disturbance_energy_cost[active])),
            'agents_in_deficit': int(np.sum(state.cumulative_energy_deficit[active] > 0)),
            'mean_energy_deficit': float(np.mean(state.cumulative_energy_deficit[active])),
        }

    def get_statistics(self, state: EnergyState, mask: np.ndarray) -> Dict[str, Any]:
        """Get energy statistics for reporting."""
        active = mask
        if not np.any(active):
            return {}

        return {
            'mean_energy': float(np.mean(state.energy[active])),
            'min_energy': float(np.min(state.energy[active])),
            'max_energy': float(np.max(state.energy[active])),
            'std_energy': float(np.std(state.energy[active])),
            'mean_body_condition': float(np.mean(state.body_condition[active])),
            'cumulative_disturbance_cost': float(np.sum(state.disturbance_energy_cost[active])),
        }


class DEPONSEnergyModule(EnergyModule):
    """
    DEPONS energy module - simple energy tracking.

    Implements the original DEPONS energy model for regulatory compliance.
    Energy is tracked on a 0-20 scale with:
    - Food intake based on hunger
    - BMR cost with seasonal scaling
    - Swimming cost (optional)
    """

    # DEPONS constants
    ENERGY_MAX = 20.0
    ENERGY_MIN = 0.0

    def __init__(self, params: 'SimulationParameters'):
        super().__init__(params)

        # Extract parameters
        self.e_use_per_30_min = params.e_use_per_30_min
        self.e_lact = params.e_lact
        self.e_warm = params.e_warm

        # Mortality parameters
        self.m_mort_prob_const = params.m_mort_prob_const
        self.x_survival_const = params.x_survival_const

    def compute_energy_update(
        self,
        state: EnergyState,
        context: EnergyContext,
        mask: np.ndarray,
        dt_seconds: int = 1800,
    ) -> EnergyResult:
        """Compute DEPONS energy update."""
        count = len(state.energy)

        # Initialize all result arrays to zero
        energy_intake = np.zeros(count, dtype=np.float32)
        energy_bmr = np.zeros(count, dtype=np.float32)
        energy_activity = np.zeros(count, dtype=np.float32)
        energy_thermoregulation = np.zeros(count, dtype=np.float32)
        energy_reproduction = np.zeros(count, dtype=np.float32)
        energy_disturbance = np.zeros(count, dtype=np.float32)

        if np.any(mask):
            # Food intake — food_available is already hunger-weighted by eat_food_vectorized
            # (hunger fraction applied during food consumption, not here)
            energy_intake[mask] = context.food_available[mask]

            # Seasonal scaling
            scaling = self._get_seasonal_scaling(context.current_month, int(np.sum(mask)))

            # BMR cost
            bmr = 0.001 * scaling * self.e_use_per_30_min

            # Lactation multiplier
            bmr = np.where(context.is_lactating[mask], bmr * self.e_lact, bmr)

            energy_bmr[mask] = bmr

            # Activity cost (swimming) - currently minimal in DEPONS
            energy_activity[mask] = context.current_speed[mask] * 0.0001 * scaling

            # Disturbance cost (increased activity during deterrence)
            energy_disturbance[mask] = np.where(
                context.is_disturbed[mask],
                0.002 * context.deterrence_magnitude[mask] * scaling,
                0.0
            ).astype(np.float32)

        # Net change
        total_cost = energy_bmr + energy_activity + energy_thermoregulation + energy_reproduction + energy_disturbance
        net_change = energy_intake - total_cost

        # Energy balance
        energy_balance = np.where(net_change >= 0, 1, -1).astype(np.float32)

        # Survival probability
        survival_prob = self.compute_survival_probability(state, mask)

        return EnergyResult(
            energy_intake=energy_intake,
            energy_bmr=energy_bmr,
            energy_activity=energy_activity,
            energy_thermoregulation=energy_thermoregulation,
            energy_reproduction=energy_reproduction,
            energy_disturbance=energy_disturbance,
            net_energy_change=net_change.astype(np.float32),
            energy_balance=energy_balance,
            survival_probability=survival_prob,
        )

    def apply_result(
        self,
        state: EnergyState,
        result: EnergyResult,
        mask: np.ndarray,
    ) -> None:
        """Apply DEPONS energy result."""
        state.energy[mask] += result.net_energy_change[mask]
        np.clip(state.energy, self.ENERGY_MIN, self.ENERGY_MAX, out=state.energy)

        # Track disturbance costs
        state.disturbance_energy_cost[mask] += result.energy_disturbance[mask]

        # Track negative balance days
        negative = mask & (result.energy_balance < 0)
        state.cumulative_energy_deficit[negative] += np.abs(result.net_energy_change[negative])

    def compute_survival_probability(
        self,
        state: EnergyState,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Compute DEPONS survival probability."""
        # DEPONS formula: yearlySurvProb = 1 - (M * exp(-energy * X))
        yearly_surv = np.where(
            state.energy > 0,
            1.0 - (self.m_mort_prob_const * np.exp(-state.energy * self.x_survival_const)),
            0.0
        )
        # Convert to per-tick probability
        step_surv = np.where(
            state.energy > 0,
            np.exp(np.log(np.maximum(yearly_surv, 1e-10)) / (360 * 48)),
            0.0
        )
        return step_surv.astype(np.float32)

    def _get_seasonal_scaling(self, month: int, count: int) -> np.ndarray:
        """Get seasonal energy scaling factor.

        DEPONS 3-state step function:
        - Months 5-9 (May-Sep): e_warm (1.3)
        - Months 4, 10 (Apr, Oct): 1.15 (transition)
        - Months 1-3, 11-12: 1.0 (winter)
        """
        if 5 <= month <= 9:
            factor = self.e_warm
        elif month in (4, 10):
            factor = 1.15
        else:
            factor = 1.0
        return np.full(count, factor, dtype=np.float32)

    def get_mode(self) -> EnergyMode:
        return EnergyMode.DEPONS


class JASMINEEnergyModule(EnergyModule):
    """
    JASMINE Dynamic Energy Budget module.

    Implements a bioenergetics model with:
    - Body mass dependent metabolism
    - Activity-specific metabolic rates
    - Cost of transport (Kleiber scaling)
    - Thermoregulation costs
    - Disturbance energy costs with cumulative impact
    - Fitness tracking for population-level effects
    """

    # Bioenergetics constants (harbour porpoise)
    BODY_MASS_ADULT = 50.0          # Adult body mass (kg)
    BODY_MASS_CALF = 15.0           # Calf body mass (kg)
    BMR_COEFFICIENT = 3.4           # Kleiber coefficient (W/kg^0.75)
    BMR_EXPONENT = 0.75             # Kleiber exponent
    COT_COEFFICIENT = 0.0001        # Cost of transport coefficient (J/m/kg), scaled for 0-20 energy units

    # Activity multipliers (relative to BMR)
    ACTIVITY_MULTIPLIERS = {
        1: 1.0,   # FORAGING
        2: 1.5,   # TRAVELING
        3: 0.6,   # RESTING
        4: 1.2,   # DISPERSING
        5: 2.0,   # DISTURBED
    }

    # Temperature constants
    THERMONEUTRAL_LOWER = 5.0       # Lower critical temperature (°C)
    THERMONEUTRAL_UPPER = 20.0      # Upper critical temperature (°C)
    THERMAL_CONDUCTANCE = 0.02      # Thermal conductance (W/kg/°C)

    # Disturbance costs
    DISTURBANCE_BASE_COST = 0.1     # Base energy cost of disturbance response
    DISTURBANCE_SPEED_MULT = 2.0    # Speed multiplier during disturbance

    def __init__(self, params: 'SimulationParameters'):
        super().__init__(params)

        # JASMINE-specific parameters
        self.use_body_mass_scaling = params.jasmine_body_mass_scaling
        self.use_thermal_model = params.jasmine_thermal_model
        self.disturbance_cost_multiplier = params.jasmine_disturbance_cost_mult

        # Mortality parameters (shared with DEPONS formula)
        self.m_mort_prob_const = params.m_mort_prob_const
        self.x_survival_const = params.x_survival_const

    def compute_energy_update(
        self,
        state: EnergyState,
        context: EnergyContext,
        mask: np.ndarray,
        dt_seconds: int = 1800,
    ) -> EnergyResult:
        """Compute JASMINE DEB energy update."""
        count = len(state.energy)
        dt_hours = dt_seconds / 3600.0

        # Initialize all result arrays to zero
        energy_intake = np.zeros(count, dtype=np.float32)
        energy_bmr = np.zeros(count, dtype=np.float32)
        energy_activity = np.zeros(count, dtype=np.float32)
        energy_thermoregulation = np.zeros(count, dtype=np.float32)
        energy_reproduction = np.zeros(count, dtype=np.float32)
        energy_disturbance = np.zeros(count, dtype=np.float32)

        if np.any(mask):
            # === Energy Intake ===
            max_intake_rate = 0.05 * state.body_mass[mask]
            intake_efficiency = 0.8 * state.body_condition[mask]
            energy_density = 5.0 * context.food_quality[mask]

            intake = (
                max_intake_rate * context.food_available[mask] *
                intake_efficiency * energy_density * dt_hours
            ).astype(np.float32)
            # Scale to DEPONS energy units (0-20)
            energy_intake[mask] = intake * 0.5

            # === Basal Metabolic Rate ===
            if self.use_body_mass_scaling:
                bmr_watts = self.BMR_COEFFICIENT * np.power(
                    np.maximum(state.body_mass[mask], 1e-6), self.BMR_EXPONENT
                )
            else:
                bmr_watts = self.BMR_COEFFICIENT * np.power(self.BODY_MASS_ADULT, self.BMR_EXPONENT)

            energy_bmr[mask] = (bmr_watts * dt_hours * 0.0001).astype(np.float32)

            # === Activity Cost ===
            activity_mult = np.ones(int(np.sum(mask)), dtype=np.float32)
            for state_val, mult in self.ACTIVITY_MULTIPLIERS.items():
                in_state = context.behavioral_state[mask] == state_val
                activity_mult[in_state] = mult

            # Cost of transport: J/m/kg × kg × m = J
            distance_m = context.current_speed[mask] * dt_seconds
            cot = self.COT_COEFFICIENT * state.body_mass[mask] * distance_m
            energy_activity[mask] = (cot * 0.0001 * activity_mult).astype(np.float32)

            # === Thermoregulation ===
            if self.use_thermal_model:
                temp_m = context.water_temperature[mask]
                temp_diff = np.zeros(int(np.sum(mask)), dtype=np.float32)
                cold = temp_m < self.THERMONEUTRAL_LOWER
                temp_diff[cold] = self.THERMONEUTRAL_LOWER - temp_m[cold]
                hot = temp_m > self.THERMONEUTRAL_UPPER
                temp_diff[hot] = temp_m[hot] - self.THERMONEUTRAL_UPPER

                energy_thermoregulation[mask] = (
                    self.THERMAL_CONDUCTANCE * state.body_mass[mask] * temp_diff * dt_hours * 0.001
                ).astype(np.float32)

            # === Reproduction Cost ===
            lact_mask = mask & context.is_lactating
            energy_reproduction[lact_mask] += energy_bmr[lact_mask] * 0.4
            preg_mask = mask & context.is_pregnant
            energy_reproduction[preg_mask] += energy_bmr[preg_mask] * 0.2

            # === Disturbance Cost ===
            base_disturbance = self.DISTURBANCE_BASE_COST * context.deterrence_magnitude[mask]
            speed_penalty = self.DISTURBANCE_SPEED_MULT * context.current_speed[mask] * context.is_disturbed[mask].astype(float)
            energy_disturbance[mask] = (
                (base_disturbance + speed_penalty * 0.01) * self.disturbance_cost_multiplier
            ).astype(np.float32)

        # === Net Energy Change ===
        total_cost = (energy_bmr + energy_activity + energy_thermoregulation +
                      energy_reproduction + energy_disturbance)
        net_change = energy_intake - total_cost

        # Energy balance indicator
        energy_balance = np.where(net_change >= 0, 1, -1).astype(np.float32)

        # Survival probability
        survival_prob = self.compute_survival_probability(state, mask)

        return EnergyResult(
            energy_intake=energy_intake,
            energy_bmr=energy_bmr,
            energy_activity=energy_activity,
            energy_thermoregulation=energy_thermoregulation,
            energy_reproduction=energy_reproduction,
            energy_disturbance=energy_disturbance,
            net_energy_change=net_change.astype(np.float32),
            energy_balance=energy_balance,
            survival_probability=survival_prob,
        )

    def apply_result(
        self,
        state: EnergyState,
        result: EnergyResult,
        mask: np.ndarray,
    ) -> None:
        """Apply JASMINE energy result with fitness tracking."""
        # Update energy
        state.energy[mask] += result.net_energy_change[mask]
        np.clip(state.energy, 0, 20.0, out=state.energy)

        # Update body condition based on energy
        state.body_condition[mask] = np.clip(state.energy[mask] / 20.0, 0.1, 1.0)

        # Track disturbance costs
        state.disturbance_energy_cost[mask] += result.energy_disturbance[mask]

        # Track negative balance (fitness impact)
        negative = mask & (result.energy_balance < 0)
        state.cumulative_energy_deficit[negative] += np.abs(result.net_energy_change[negative])

        # Count disturbance events
        disturbed = mask & (result.energy_disturbance > 0.01)
        state.disturbance_events[disturbed] += 1

    def compute_survival_probability(
        self,
        state: EnergyState,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute JASMINE survival probability.

        Uses body condition and cumulative disturbance impact.
        """
        # Map body_condition (0.1-1.0) to effective energy (2-20)
        effective_energy = np.clip(state.body_condition, 0.1, 1.0) * 20.0

        # DEPONS starvation survival formula
        yearly_surv = np.where(
            effective_energy > 0,
            1.0 - (self.m_mort_prob_const * np.exp(-effective_energy * self.x_survival_const)),
            0.0,
        )

        # Disturbance impact (cumulative effect reduces survival)
        disturbance_impact = 1.0 - np.clip(state.disturbance_energy_cost * 0.001, 0, 0.5)
        yearly_surv *= disturbance_impact

        # Convert to per-tick
        step_surv = np.where(
            yearly_surv > 0,
            np.exp(np.log(np.maximum(yearly_surv, 1e-10)) / (360 * 48)),
            0.0,
        )

        return step_surv.astype(np.float32)

    def get_mode(self) -> EnergyMode:
        return EnergyMode.JASMINE

    def get_fitness_metrics(
        self,
        state: EnergyState,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Get JASMINE-specific fitness metrics."""
        active = mask
        if not np.any(active):
            return {}

        return {
            'mean_body_condition': float(np.mean(state.body_condition[active])),
            'mean_fat_reserve': float(np.mean(state.fat_reserve[active])),
            'total_disturbance_cost': float(np.sum(state.disturbance_energy_cost[active])),
            'mean_disturbance_events': float(np.mean(state.disturbance_events[active])),
            'agents_in_deficit': int(np.sum(state.cumulative_energy_deficit[active] > 0)),
            'mean_energy_deficit': float(np.mean(state.cumulative_energy_deficit[active])),
        }


def create_energy_module(
    params: 'SimulationParameters',
    mode: EnergyMode = EnergyMode.DEPONS,
) -> EnergyModule:
    """
    Factory function to create appropriate energy module.

    Args:
        params: Simulation parameters
        mode: Energy calculation mode

    Returns:
        Configured EnergyModule instance
    """
    if mode == EnergyMode.DEPONS:
        return DEPONSEnergyModule(params)
    elif mode == EnergyMode.JASMINE:
        return JASMINEEnergyModule(params)
    else:
        # Hybrid - use JASMINE as default
        return JASMINEEnergyModule(params)
