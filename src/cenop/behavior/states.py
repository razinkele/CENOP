"""
Behavioral state definitions for CENOP-JASMINE hybrid simulation.

This module defines the behavioral states that agents can be in,
along with state containers and perception contexts.

State Mapping (DEPONS <-> JASMINE):
| DEPONS State | JASMINE State | Movement Model |
|--------------|---------------|----------------|
| Foraging     | Foraging      | DEPONS CRW     |
| Traveling    | Transit       | JASMINE if >threshold |
| Resting      | Resting       | JASMINE (reduced) |
| Dispersing   | Dispersing    | DEPONS CRW     |
| Disturbed    | Avoiding      | DEPONS deterrence |
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


class BehaviorState(Enum):
    """
    Behavioral states for porpoise agents.

    States determine movement patterns, energy consumption,
    and responsiveness to environmental stimuli.
    """
    FORAGING = auto()      # Default - searching for/consuming food
    TRAVELING = auto()     # Transit between areas (JASMINE: directed movement)
    RESTING = auto()       # Low activity state (JASMINE: energy recovery)
    DISPERSING = auto()    # PSM-driven dispersal to new area
    DISTURBED = auto()     # Response to disturbance (deterrence active)


class StateTransition(Enum):
    """
    Possible state transitions.

    Used for tracking transition statistics and debugging.
    """
    NONE = auto()                    # No transition
    FORAGING_TO_TRAVELING = auto()
    FORAGING_TO_RESTING = auto()
    FORAGING_TO_DISPERSING = auto()
    FORAGING_TO_DISTURBED = auto()
    TRAVELING_TO_FORAGING = auto()
    TRAVELING_TO_RESTING = auto()
    TRAVELING_TO_DISTURBED = auto()
    RESTING_TO_FORAGING = auto()
    RESTING_TO_DISTURBED = auto()
    DISPERSING_TO_FORAGING = auto()
    DISPERSING_TO_DISTURBED = auto()
    DISTURBED_TO_FORAGING = auto()
    DISTURBED_TO_TRAVELING = auto()
    DISTURBED_TO_DISPERSING = auto()


@dataclass
class BehaviorContext:
    """
    Context/perception data for state transitions.

    Contains environmental and internal state information
    needed to determine behavioral transitions.
    """
    # Required arrays (no defaults)
    deterrence_magnitude: np.ndarray    # Current deterrence strength
    time_since_disturbance: np.ndarray  # Ticks since last disturbance
    current_energy: np.ndarray          # Current energy level
    energy_declining_days: np.ndarray   # Days of declining energy
    current_speed: np.ndarray           # Current movement speed
    memory_cell_count: np.ndarray       # PSM memory cell count
    is_dispersing: np.ndarray           # Currently dispersing flag
    dispersal_complete: np.ndarray      # Dispersal target reached

    # Configuration parameters with defaults
    energy_threshold_low: float = 0.3   # Low energy threshold
    energy_threshold_high: float = 0.7  # High energy threshold
    speed_threshold: float = 2.0        # Speed threshold for traveling
    min_memory_cells: int = 50          # Minimum cells for PSM activation
    t_disp: int = 3                     # Days before dispersal triggers
    recovery_ticks: int = 48            # Ticks to recover from disturbance

    @classmethod
    def create_default(cls, count: int) -> 'BehaviorContext':
        """Create a default context for count agents."""
        return cls(
            deterrence_magnitude=np.zeros(count, dtype=np.float32),
            time_since_disturbance=np.full(count, 9999, dtype=np.int32),
            current_energy=np.full(count, 0.5, dtype=np.float32),
            energy_declining_days=np.zeros(count, dtype=np.int32),
            current_speed=np.zeros(count, dtype=np.float32),
            memory_cell_count=np.zeros(count, dtype=np.int32),
            is_dispersing=np.zeros(count, dtype=bool),
            dispersal_complete=np.zeros(count, dtype=bool),
        )


@dataclass
class BehaviorStateVector:
    """
    Vectorized behavioral state for population.

    Tracks current state and history for all agents.
    """
    # Current state for each agent
    state: np.ndarray  # BehaviorState enum values

    # State duration (ticks in current state)
    state_duration: np.ndarray

    # Previous state (for transition tracking)
    previous_state: np.ndarray

    # Statistics
    total_transitions: int = 0
    transition_counts: Dict[StateTransition, int] = field(default_factory=dict)

    @classmethod
    def create(cls, count: int) -> 'BehaviorStateVector':
        """Create a new state vector for count agents."""
        return cls(
            state=np.full(count, BehaviorState.FORAGING.value, dtype=np.int32),
            state_duration=np.zeros(count, dtype=np.int32),
            previous_state=np.full(count, BehaviorState.FORAGING.value, dtype=np.int32),
            transition_counts={t: 0 for t in StateTransition},
        )

    def get_state(self, idx: int) -> BehaviorState:
        """Get state for a single agent."""
        return BehaviorState(self.state[idx])

    def set_state(self, idx: int, new_state: BehaviorState) -> None:
        """Set state for a single agent."""
        old_state = self.state[idx]
        if old_state != new_state.value:
            self.previous_state[idx] = old_state
            self.state[idx] = new_state.value
            self.state_duration[idx] = 0
            self.total_transitions += 1

    def get_mask(self, state: BehaviorState) -> np.ndarray:
        """Get boolean mask for agents in a specific state."""
        return self.state == state.value

    def advance_duration(self) -> None:
        """Increment state duration for all agents."""
        self.state_duration += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get state statistics."""
        state_counts = {}
        for s in BehaviorState:
            state_counts[s.name] = int(np.sum(self.state == s.value))

        return {
            'state_counts': state_counts,
            'total_transitions': self.total_transitions,
            'mean_state_duration': float(np.mean(self.state_duration)),
        }


# State-specific parameters for movement and behavior
STATE_PARAMETERS = {
    BehaviorState.FORAGING: {
        'movement_mode': 'DEPONS_CRW',
        'speed_multiplier': 1.0,
        'turning_variance': 1.0,
        'energy_cost_multiplier': 1.0,
    },
    BehaviorState.TRAVELING: {
        'movement_mode': 'JASMINE_PHYSICS',  # Use physics for directed travel
        'speed_multiplier': 1.5,
        'turning_variance': 0.5,  # More directed
        'energy_cost_multiplier': 1.2,
    },
    BehaviorState.RESTING: {
        'movement_mode': 'JASMINE_PHYSICS',  # Minimal movement
        'speed_multiplier': 0.3,
        'turning_variance': 2.0,  # Random drifting
        'energy_cost_multiplier': 0.5,  # Lower energy use
    },
    BehaviorState.DISPERSING: {
        'movement_mode': 'DEPONS_CRW',  # Use validated dispersal
        'speed_multiplier': 1.2,
        'turning_variance': 0.3,  # PSM-modulated
        'energy_cost_multiplier': 1.1,
    },
    BehaviorState.DISTURBED: {
        'movement_mode': 'DEPONS_CRW',  # Use validated deterrence
        'speed_multiplier': 1.8,  # Fast escape
        'turning_variance': 0.2,  # Very directed
        'energy_cost_multiplier': 1.5,  # High energy cost
    },
}
