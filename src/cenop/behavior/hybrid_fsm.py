"""
Hybrid Behavioral Finite State Machine for CENOP-JASMINE integration.

This module provides a unified FSM that supports:
- DEPONS mode: Simple transitions for regulatory compliance
- JASMINE mode: Enhanced FSM with memory/sociality

Key design principle: DEPONS behavioral responses are preserved exactly
for regulatory use cases, while JASMINE features are opt-in for research.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any
from enum import Enum, auto
import numpy as np

from cenop.behavior.states import (
    BehaviorState,
    BehaviorContext,
    BehaviorStateVector,
    StateTransition,
    STATE_PARAMETERS,
)
from cenop.core.time_manager import TimeMode

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters


class FSMMode(Enum):
    """FSM operation modes."""
    DEPONS = auto()    # Simple transitions, regulatory-compliant
    JASMINE = auto()   # Enhanced transitions with memory/sociality
    HYBRID = auto()    # Context-dependent (DEPONS for disturbance, JASMINE otherwise)


class HybridBehaviorFSM:
    """
    Hybrid finite state machine combining DEPONS and JASMINE behaviors.

    The FSM manages behavioral state transitions for all agents:
    - DEPONS mode: Governs disturbance-driven transitions
    - JASMINE mode: Governs foraging, memory, sociality
    - Hybrid mode: Uses DEPONS for disturbance, JASMINE for normal behavior

    State Transition Rules:
    -----------------------
    DEPONS Mode (simple):
        FORAGING -> DISTURBED:    deterrence > threshold
        FORAGING -> DISPERSING:   energy declining for t_disp days
        DISTURBED -> FORAGING:    no deterrence for recovery_ticks
        DISPERSING -> FORAGING:   dispersal target reached

    JASMINE Mode (enhanced):
        All DEPONS transitions plus:
        FORAGING -> TRAVELING:    speed > threshold
        FORAGING -> RESTING:      energy < low_threshold
        TRAVELING -> FORAGING:    speed < threshold
        TRAVELING -> RESTING:     energy < low_threshold
        RESTING -> FORAGING:      energy > high_threshold

    Usage:
        fsm = HybridBehaviorFSM(params, TimeMode.DEPONS)
        state_vector = BehaviorStateVector.create(100)
        context = BehaviorContext.create_default(100)
        fsm.update_states(state_vector, context, active_mask)
    """

    # Thresholds
    DETERRENCE_THRESHOLD = 0.01    # Minimum deterrence to trigger DISTURBED
    SPEED_THRESHOLD = 2.0          # Speed threshold for TRAVELING
    ENERGY_LOW_THRESHOLD = 0.3     # Low energy threshold for RESTING
    ENERGY_HIGH_THRESHOLD = 0.7    # High energy threshold to exit RESTING

    def __init__(
        self,
        params: 'SimulationParameters',
        time_mode: TimeMode = TimeMode.DEPONS,
        fsm_mode: Optional[FSMMode] = None,
    ):
        """
        Initialize hybrid behavior FSM.

        Args:
            params: Simulation parameters
            time_mode: TimeMode from TimeManager
            fsm_mode: Explicit FSM mode (overrides time_mode inference)
        """
        self.params = params

        # Determine FSM mode from time_mode if not explicit
        if fsm_mode is not None:
            self.mode = fsm_mode
        elif time_mode == TimeMode.DEPONS:
            self.mode = FSMMode.DEPONS
        else:
            self.mode = FSMMode.JASMINE

        # Extract parameters
        self.t_disp = getattr(params, 't_disp', 3)
        self.recovery_ticks = getattr(params, 'disturbance_recovery_ticks', 48)
        self.min_memory_cells = getattr(params, 'min_memory_cells', 50)

        # Statistics
        self._stats: Dict[str, int] = {
            'total_updates': 0,
            'transitions_to_disturbed': 0,
            'transitions_to_dispersing': 0,
            'transitions_to_foraging': 0,
            'transitions_to_traveling': 0,
            'transitions_to_resting': 0,
        }

    def update_states(
        self,
        state_vector: BehaviorStateVector,
        context: BehaviorContext,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Update behavioral states for all agents.

        Args:
            state_vector: Current behavioral states
            context: Perception/environmental context
            mask: Boolean mask of active agents

        Returns:
            Boolean array indicating which agents changed state
        """
        self._stats['total_updates'] += 1
        count = len(state_vector.state)
        changed = np.zeros(count, dtype=bool)

        if self.mode == FSMMode.DEPONS:
            changed = self._update_depons(state_vector, context, mask)
        elif self.mode == FSMMode.JASMINE:
            changed = self._update_jasmine(state_vector, context, mask)
        else:  # HYBRID
            changed = self._update_hybrid(state_vector, context, mask)

        # Advance duration for all agents
        state_vector.advance_duration()

        return changed

    def _update_depons(
        self,
        state_vector: BehaviorStateVector,
        context: BehaviorContext,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Update states using DEPONS rules (simple transitions).

        DEPONS FSM Rules:
        1. Any state + deterrence > threshold -> DISTURBED
        2. DISTURBED + no deterrence for recovery_ticks -> FORAGING
        3. FORAGING + energy_declining >= t_disp + memory >= 50 -> DISPERSING
        4. DISPERSING + target reached -> FORAGING
        """
        count = len(state_vector.state)
        old_states = state_vector.state.copy()

        # === Rule 1: Transition to DISTURBED ===
        # Any active agent with significant deterrence becomes DISTURBED
        disturbed_trigger = mask & (context.deterrence_magnitude > self.DETERRENCE_THRESHOLD)
        state_vector.state[disturbed_trigger] = BehaviorState.DISTURBED.value
        state_vector.state_duration[disturbed_trigger] = 0

        # === Rule 2: DISTURBED -> FORAGING (recovery) ===
        # Agents in DISTURBED state with no deterrence for recovery_ticks
        currently_disturbed = mask & (state_vector.state == BehaviorState.DISTURBED.value)
        no_deterrence = context.deterrence_magnitude <= self.DETERRENCE_THRESHOLD
        recovered = currently_disturbed & no_deterrence & (context.time_since_disturbance > self.recovery_ticks)
        state_vector.state[recovered] = BehaviorState.FORAGING.value
        state_vector.state_duration[recovered] = 0

        # === Rule 3: FORAGING -> DISPERSING ===
        # Energy declining for t_disp days and sufficient memory
        currently_foraging = mask & (state_vector.state == BehaviorState.FORAGING.value)
        energy_declining = context.energy_declining_days >= self.t_disp
        sufficient_memory = context.memory_cell_count >= self.min_memory_cells
        start_dispersal = currently_foraging & energy_declining & sufficient_memory & ~context.is_dispersing
        state_vector.state[start_dispersal] = BehaviorState.DISPERSING.value
        state_vector.state_duration[start_dispersal] = 0

        # === Rule 4: DISPERSING -> FORAGING ===
        # Dispersal target reached
        currently_dispersing = mask & (state_vector.state == BehaviorState.DISPERSING.value)
        dispersal_done = currently_dispersing & context.dispersal_complete
        state_vector.state[dispersal_done] = BehaviorState.FORAGING.value
        state_vector.state_duration[dispersal_done] = 0

        # Track changes
        changed = state_vector.state != old_states
        self._update_stats(old_states, state_vector.state, changed)

        return changed

    def _update_jasmine(
        self,
        state_vector: BehaviorStateVector,
        context: BehaviorContext,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Update states using JASMINE rules (enhanced transitions).

        JASMINE FSM extends DEPONS with:
        - Speed-based TRAVELING state
        - Energy-based RESTING state
        - More nuanced transition conditions
        """
        count = len(state_vector.state)
        old_states = state_vector.state.copy()

        # === All DEPONS rules apply first ===
        # Priority: Disturbance > Dispersal > Energy states

        # Rule 1: Transition to DISTURBED (highest priority)
        disturbed_trigger = mask & (context.deterrence_magnitude > self.DETERRENCE_THRESHOLD)
        state_vector.state[disturbed_trigger] = BehaviorState.DISTURBED.value
        state_vector.state_duration[disturbed_trigger] = 0

        # Rule 2: DISTURBED -> FORAGING (or TRAVELING based on speed)
        currently_disturbed = mask & (state_vector.state == BehaviorState.DISTURBED.value)
        no_deterrence = context.deterrence_magnitude <= self.DETERRENCE_THRESHOLD
        recovered = currently_disturbed & no_deterrence & (context.time_since_disturbance > self.recovery_ticks)

        # High speed after recovery -> TRAVELING, else FORAGING
        high_speed = context.current_speed > context.speed_threshold
        state_vector.state[recovered & high_speed] = BehaviorState.TRAVELING.value
        state_vector.state[recovered & ~high_speed] = BehaviorState.FORAGING.value
        state_vector.state_duration[recovered] = 0

        # Rule 3: DISPERSING transitions (same as DEPONS)
        currently_dispersing = mask & (state_vector.state == BehaviorState.DISPERSING.value)
        dispersal_done = currently_dispersing & context.dispersal_complete
        state_vector.state[dispersal_done] = BehaviorState.FORAGING.value
        state_vector.state_duration[dispersal_done] = 0

        # === JASMINE-specific rules ===

        # Rule 4: FORAGING -> TRAVELING (speed threshold)
        currently_foraging = mask & (state_vector.state == BehaviorState.FORAGING.value)
        fast_moving = context.current_speed > context.speed_threshold
        to_traveling = currently_foraging & fast_moving
        state_vector.state[to_traveling] = BehaviorState.TRAVELING.value
        state_vector.state_duration[to_traveling] = 0

        # Rule 5: FORAGING -> RESTING (low energy)
        low_energy = context.current_energy < context.energy_threshold_low
        to_resting_from_foraging = currently_foraging & low_energy & ~fast_moving
        state_vector.state[to_resting_from_foraging] = BehaviorState.RESTING.value
        state_vector.state_duration[to_resting_from_foraging] = 0

        # Rule 6: FORAGING -> DISPERSING (energy declining)
        energy_declining = context.energy_declining_days >= self.t_disp
        sufficient_memory = context.memory_cell_count >= self.min_memory_cells
        start_dispersal = currently_foraging & energy_declining & sufficient_memory & ~context.is_dispersing
        state_vector.state[start_dispersal] = BehaviorState.DISPERSING.value
        state_vector.state_duration[start_dispersal] = 0

        # Rule 7: TRAVELING -> FORAGING (slow down)
        currently_traveling = mask & (state_vector.state == BehaviorState.TRAVELING.value)
        slow_moving = context.current_speed <= context.speed_threshold
        to_foraging = currently_traveling & slow_moving
        state_vector.state[to_foraging] = BehaviorState.FORAGING.value
        state_vector.state_duration[to_foraging] = 0

        # Rule 8: TRAVELING -> RESTING (low energy)
        to_resting_from_traveling = currently_traveling & low_energy
        state_vector.state[to_resting_from_traveling] = BehaviorState.RESTING.value
        state_vector.state_duration[to_resting_from_traveling] = 0

        # Rule 9: RESTING -> FORAGING (energy recovered)
        currently_resting = mask & (state_vector.state == BehaviorState.RESTING.value)
        high_energy = context.current_energy > context.energy_threshold_high
        to_foraging_from_rest = currently_resting & high_energy
        state_vector.state[to_foraging_from_rest] = BehaviorState.FORAGING.value
        state_vector.state_duration[to_foraging_from_rest] = 0

        # Track changes
        changed = state_vector.state != old_states
        self._update_stats(old_states, state_vector.state, changed)

        return changed

    def _update_hybrid(
        self,
        state_vector: BehaviorStateVector,
        context: BehaviorContext,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Update states using hybrid rules.

        Uses DEPONS rules when disturbance is present,
        JASMINE rules for normal behavior.
        """
        # Check if any disturbance present
        has_disturbance = np.any(context.deterrence_magnitude[mask] > self.DETERRENCE_THRESHOLD)

        if has_disturbance:
            # Use DEPONS for disturbance response
            return self._update_depons(state_vector, context, mask)
        else:
            # Use JASMINE for normal behavior
            return self._update_jasmine(state_vector, context, mask)

    def _update_stats(
        self,
        old_states: np.ndarray,
        new_states: np.ndarray,
        changed: np.ndarray,
    ) -> None:
        """Update transition statistics."""
        if not np.any(changed):
            return

        # Count transitions to each state
        for state in BehaviorState:
            to_this = changed & (new_states == state.value)
            count = int(np.sum(to_this))
            if count > 0:
                key = f'transitions_to_{state.name.lower()}'
                if key in self._stats:
                    self._stats[key] += count

    def get_movement_mode(
        self,
        state_vector: BehaviorStateVector,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Get movement mode for each agent based on state.

        Returns array of movement mode indices:
        - 0: DEPONS_CRW
        - 1: JASMINE_PHYSICS
        """
        count = len(state_vector.state)
        modes = np.zeros(count, dtype=np.int32)

        # States that use JASMINE physics
        jasmine_states = {BehaviorState.TRAVELING.value, BehaviorState.RESTING.value}

        for state_val in jasmine_states:
            in_state = mask & (state_vector.state == state_val)
            modes[in_state] = 1

        return modes

    def get_speed_multiplier(
        self,
        state_vector: BehaviorStateVector,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Get speed multiplier for each agent based on state."""
        count = len(state_vector.state)
        multipliers = np.ones(count, dtype=np.float32)

        for state in BehaviorState:
            in_state = mask & (state_vector.state == state.value)
            if np.any(in_state):
                mult = STATE_PARAMETERS[state]['speed_multiplier']
                multipliers[in_state] = mult

        return multipliers

    def get_energy_cost_multiplier(
        self,
        state_vector: BehaviorStateVector,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Get energy cost multiplier for each agent based on state."""
        count = len(state_vector.state)
        multipliers = np.ones(count, dtype=np.float32)

        for state in BehaviorState:
            in_state = mask & (state_vector.state == state.value)
            if np.any(in_state):
                mult = STATE_PARAMETERS[state]['energy_cost_multiplier']
                multipliers[in_state] = mult

        return multipliers

    def get_statistics(self) -> Dict[str, Any]:
        """Get FSM statistics."""
        return {
            'mode': self.mode.name,
            **self._stats,
        }

    def reset_statistics(self) -> None:
        """Reset FSM statistics."""
        for key in self._stats:
            self._stats[key] = 0


def create_behavior_fsm(
    params: 'SimulationParameters',
    time_mode: TimeMode = TimeMode.DEPONS,
) -> HybridBehaviorFSM:
    """
    Factory function to create appropriate behavior FSM.

    Args:
        params: Simulation parameters
        time_mode: TimeMode from TimeManager

    Returns:
        Configured HybridBehaviorFSM instance
    """
    return HybridBehaviorFSM(params, time_mode)
