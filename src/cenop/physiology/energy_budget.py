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
    time_under_disturbance: np.ndarray   # Total ticks under disturbance
    disturbance_events: np.ndarray       # Count of disturbance events

    # Fitness tracking
    cumulative_energy_deficit: np.ndarray  # Total energy shortfall
    days_in_negative_balance: np.ndarray   # Days with energy deficit

    @classmethod
    def create(cls, count: int, initial_energy: float = 10.0) -> 'EnergyState':
        """Create energy state for count agents."""
        return cls(
            energy=np.full(count, initial_energy, dtype=np.float32),
            body_mass=np.full(count, 50.0, dtype=np.float32),  # ~50 kg adult
            body_condition=np.full(count, 0.5, dtype=np.float32),
            fat_reserve=np.full(count, 5.0, dtype=np.float32),  # ~10% body mass
            activity_level=np.full(count, 0.5, dtype=np.float32),
            distance_traveled=np.zeros(count, dtype=np.float32),
            disturbance_energy_cost=np.zeros(count, dtype=np.float32),
            time_under_disturbance=np.zeros(count, dtype=np.int32),
            disturbance_events=np.zeros(count, dtype=np.int32),
            cumulative_energy_deficit=np.zeros(count, dtype=np.float32),
            days_in_negative_balance=np.zeros(count, dtype=np.int32),
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
        self.e_use_per_30_min = getattr(params, 'e_use_per_30_min', 4.5)
        self.e_lact = getattr(params, 'e_lact', 1.4)
        self.e_warm = getattr(params, 'e_warm', 1.3)

        # Mortality parameters
        self.m_mort_prob_const = getattr(params, 'm_mort_prob_const', 0.5)
        self.x_survival_const = getattr(params, 'x_survival_const', 0.15)

    def compute_energy_update(
        self,
        state: EnergyState,
        context: EnergyContext,
        mask: np.ndarray,
        dt_seconds: int = 1800,
    ) -> EnergyResult:
        """Compute DEPONS energy update."""
        count = len(state.energy)

        # Food intake - hungry porpoises eat more
        hunger = np.clip((self.ENERGY_MAX - state.energy) / 10.0, 0.0, 0.99)
        energy_intake = hunger * context.food_available

        # Seasonal scaling
        scaling = self._get_seasonal_scaling(context.current_month, count)

        # BMR cost
        energy_bmr = 0.001 * scaling * self.e_use_per_30_min

        # Lactation multiplier
        energy_bmr = np.where(context.is_lactating, energy_bmr * self.e_lact, energy_bmr)

        # Warm water multiplier (June-October)
        if 6 <= context.current_month <= 10:
            energy_bmr = energy_bmr * self.e_warm

        # Activity cost (swimming) - currently minimal in DEPONS
        energy_activity = context.current_speed * 0.0001 * scaling

        # Thermoregulation (included in BMR for DEPONS)
        energy_thermoregulation = np.zeros(count, dtype=np.float32)

        # Reproduction cost (included in lactation multiplier)
        energy_reproduction = np.zeros(count, dtype=np.float32)

        # Disturbance cost (increased activity during deterrence)
        energy_disturbance = np.where(
            context.is_disturbed,
            0.002 * context.deterrence_magnitude * scaling,
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
            energy_intake=energy_intake.astype(np.float32),
            energy_bmr=energy_bmr.astype(np.float32),
            energy_activity=energy_activity.astype(np.float32),
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
        """Get seasonal energy scaling factor."""
        # DEPONS seasonal variation
        # Peak in summer (Jun-Aug), lower in winter
        seasonal_factors = {
            1: 1.0, 2: 1.0, 3: 1.0, 4: 1.1, 5: 1.15,
            6: 1.2, 7: 1.25, 8: 1.2, 9: 1.15, 10: 1.1,
            11: 1.0, 12: 1.0
        }
        return np.full(count, seasonal_factors.get(month, 1.0), dtype=np.float32)

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
    COT_COEFFICIENT = 0.1           # Cost of transport coefficient (J/m/kg)

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
        self.use_body_mass_scaling = getattr(params, 'jasmine_body_mass_scaling', True)
        self.use_thermal_model = getattr(params, 'jasmine_thermal_model', True)
        self.disturbance_cost_multiplier = getattr(params, 'jasmine_disturbance_cost_mult', 1.0)

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

        # === Energy Intake ===
        # Functional response based on food availability and body condition
        max_intake_rate = 0.05 * state.body_mass  # kg food per hour
        intake_efficiency = 0.8 * state.body_condition  # Assimilation efficiency
        energy_density = 5.0 * context.food_quality  # MJ/kg food

        energy_intake = (
            max_intake_rate * context.food_available *
            intake_efficiency * energy_density * dt_hours
        ).astype(np.float32)

        # Scale to DEPONS energy units (0-20)
        energy_intake = energy_intake * 0.5

        # === Basal Metabolic Rate ===
        # Kleiber scaling: BMR = a * M^0.75
        if self.use_body_mass_scaling:
            bmr_watts = self.BMR_COEFFICIENT * np.power(state.body_mass, self.BMR_EXPONENT)
        else:
            bmr_watts = self.BMR_COEFFICIENT * np.power(self.BODY_MASS_ADULT, self.BMR_EXPONENT)

        # Convert to energy units (scaled to be comparable to DEPONS 0.001-0.01 range)
        # DEPONS BMR is ~0.001-0.01 per 30min, so we scale accordingly
        energy_bmr = (bmr_watts * dt_hours * 0.0001).astype(np.float32)  # Much lower scaling

        # === Activity Cost ===
        # Get activity multiplier based on behavioral state
        activity_mult = np.ones(count, dtype=np.float32)
        for state_val, mult in self.ACTIVITY_MULTIPLIERS.items():
            in_state = context.behavioral_state == state_val
            activity_mult[in_state] = mult

        # Cost of transport
        cot = self.COT_COEFFICIENT * state.body_mass * context.current_speed * dt_seconds
        energy_activity = (cot * 0.001 * activity_mult).astype(np.float32)

        # === Thermoregulation ===
        if self.use_thermal_model:
            temp_diff = np.zeros(count, dtype=np.float32)
            # Below thermoneutral zone
            cold = context.water_temperature < self.THERMONEUTRAL_LOWER
            temp_diff[cold] = self.THERMONEUTRAL_LOWER - context.water_temperature[cold]
            # Above thermoneutral zone (less common)
            hot = context.water_temperature > self.THERMONEUTRAL_UPPER
            temp_diff[hot] = context.water_temperature[hot] - self.THERMONEUTRAL_UPPER

            energy_thermoregulation = (
                self.THERMAL_CONDUCTANCE * state.body_mass * temp_diff * dt_hours * 0.001
            ).astype(np.float32)
        else:
            energy_thermoregulation = np.zeros(count, dtype=np.float32)

        # === Reproduction Cost ===
        energy_reproduction = np.zeros(count, dtype=np.float32)
        # Lactation cost
        energy_reproduction[context.is_lactating] += energy_bmr[context.is_lactating] * 0.4
        # Pregnancy cost
        energy_reproduction[context.is_pregnant] += energy_bmr[context.is_pregnant] * 0.2

        # === Disturbance Cost ===
        # Additional energy cost during disturbance response
        base_disturbance = self.DISTURBANCE_BASE_COST * context.deterrence_magnitude
        speed_penalty = self.DISTURBANCE_SPEED_MULT * context.current_speed * context.is_disturbed.astype(float)
        energy_disturbance = (
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
        # Base survival from body condition
        condition_effect = np.clip(state.body_condition, 0.1, 1.0)

        # Disturbance impact (cumulative effect reduces survival)
        disturbance_impact = 1.0 - np.clip(
            state.disturbance_energy_cost * 0.001, 0, 0.5
        )

        # Combined yearly survival
        yearly_surv = 0.95 * condition_effect * disturbance_impact

        # Convert to per-tick
        step_surv = np.exp(np.log(np.maximum(yearly_surv, 1e-10)) / (360 * 48))

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
