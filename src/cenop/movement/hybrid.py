"""
Hybrid movement selector for CENOP-JASMINE integration.

This module provides a unified interface that selects between DEPONS CRW
and JASMINE physics-based movement based on:
- Simulation mode (DEPONS vs JASMINE)
- Behavioral state (disturbed vs normal)
- Agent-specific conditions

The hybrid approach allows:
- DEPONS mode: Full regulatory compliance with validated CRW
- JASMINE mode: Physics-based movement for research
- Hybrid mode: Context-dependent switching (e.g., DEPONS for disturbance)

Key design principle: DEPONS movement is preserved exactly for regulatory
use cases, while JASMINE features are opt-in for research applications.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Dict, Any

import numpy as np

from cenop.movement.base import (
    MovementModule,
    MovementMode,
    MovementState,
    EnvironmentContext,
    MovementResult,
)
from cenop.movement.depons_crw import DEPONSCRWMovement, DEPONSCRWMovementVectorized
from cenop.movement.jasmine_physics import JASMINEPhysicsMovement
from cenop.core.time_manager import TimeMode

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters


class HybridStrategy(Enum):
    """Strategy for hybrid movement selection."""
    DEPONS_ONLY = auto()          # Always use DEPONS CRW
    JASMINE_ONLY = auto()         # Always use JASMINE physics
    DISTURBANCE_AWARE = auto()    # DEPONS during disturbance, JASMINE otherwise
    STATE_BASED = auto()          # Select based on behavioral state


class HybridMovementSelector(MovementModule):
    """
    Hybrid movement module that switches between DEPONS and JASMINE.

    This class acts as a facade that delegates to the appropriate
    movement implementation based on configuration and runtime state.

    Usage:
        # Create selector with desired strategy
        selector = HybridMovementSelector(
            params=sim_params,
            time_mode=TimeMode.DEPONS,
            strategy=HybridStrategy.DEPONS_ONLY
        )

        # Compute step (automatically uses correct implementation)
        result = selector.compute_step(x, y, state, env, mask)

    Strategies:
        - DEPONS_ONLY: Always use DEPONS CRW (regulatory mode)
        - JASMINE_ONLY: Always use JASMINE physics (research mode)
        - DISTURBANCE_AWARE: Use DEPONS during disturbance events
        - STATE_BASED: Select based on agent behavioral state
    """

    def __init__(
        self,
        params: 'SimulationParameters',
        time_mode: TimeMode = TimeMode.DEPONS,
        strategy: HybridStrategy = HybridStrategy.DEPONS_ONLY,
        use_vectorized: bool = True,
    ):
        """
        Initialize hybrid movement selector.

        Args:
            params: Simulation parameters
            time_mode: TimeMode from TimeManager (DEPONS or JASMINE)
            strategy: Strategy for selecting movement implementation
            use_vectorized: Use vectorized DEPONS implementation
        """
        super().__init__(params)

        self.time_mode = time_mode
        self.strategy = strategy

        # Initialize movement implementations
        if use_vectorized:
            self._depons = DEPONSCRWMovementVectorized(params)
        else:
            self._depons = DEPONSCRWMovement(params)

        self._jasmine = JASMINEPhysicsMovement(params)

        # Current active module (for reporting)
        self._active_module: MovementModule = self._depons

        # Statistics
        self._stats: Dict[str, int] = {
            'depons_steps': 0,
            'jasmine_steps': 0,
            'hybrid_switches': 0,
        }

    @classmethod
    def from_time_mode(
        cls,
        params: 'SimulationParameters',
        time_mode: TimeMode,
    ) -> 'HybridMovementSelector':
        """
        Create selector with strategy based on TimeMode.

        DEPONS mode -> DEPONS_ONLY strategy (regulatory compliance)
        JASMINE mode -> JASMINE_ONLY strategy (research mode)

        Args:
            params: Simulation parameters
            time_mode: TimeMode from TimeManager

        Returns:
            Configured HybridMovementSelector
        """
        if time_mode == TimeMode.DEPONS:
            return cls(params, time_mode, HybridStrategy.DEPONS_ONLY)
        else:
            return cls(params, time_mode, HybridStrategy.JASMINE_ONLY)

    def compute_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: MovementState,
        environment: EnvironmentContext,
        mask: np.ndarray,
        deterrence_dx: Optional[np.ndarray] = None,
        deterrence_dy: Optional[np.ndarray] = None,
    ) -> MovementResult:
        """
        Compute movement step using selected strategy.

        Delegates to DEPONS or JASMINE implementation based on
        the configured strategy and current conditions.

        Args:
            x: Current X positions
            y: Current Y positions
            state: Movement state
            environment: Environmental context
            mask: Active agent mask
            deterrence_dx: Optional deterrence X
            deterrence_dy: Optional deterrence Y

        Returns:
            MovementResult from selected implementation
        """
        if self.strategy == HybridStrategy.DEPONS_ONLY:
            return self._compute_depons(
                x, y, state, environment, mask, deterrence_dx, deterrence_dy
            )

        elif self.strategy == HybridStrategy.JASMINE_ONLY:
            return self._compute_jasmine(
                x, y, state, environment, mask, deterrence_dx, deterrence_dy
            )

        elif self.strategy == HybridStrategy.DISTURBANCE_AWARE:
            return self._compute_disturbance_aware(
                x, y, state, environment, mask, deterrence_dx, deterrence_dy
            )

        elif self.strategy == HybridStrategy.STATE_BASED:
            return self._compute_state_based(
                x, y, state, environment, mask, deterrence_dx, deterrence_dy
            )

        else:
            # Default to DEPONS
            return self._compute_depons(
                x, y, state, environment, mask, deterrence_dx, deterrence_dy
            )

    def _compute_depons(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: MovementState,
        environment: EnvironmentContext,
        mask: np.ndarray,
        deterrence_dx: Optional[np.ndarray],
        deterrence_dy: Optional[np.ndarray],
    ) -> MovementResult:
        """Compute using DEPONS CRW."""
        self._active_module = self._depons
        self._stats['depons_steps'] += 1

        return self._depons.compute_step(
            x, y, state, environment, mask, deterrence_dx, deterrence_dy
        )

    def _compute_jasmine(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: MovementState,
        environment: EnvironmentContext,
        mask: np.ndarray,
        deterrence_dx: Optional[np.ndarray],
        deterrence_dy: Optional[np.ndarray],
    ) -> MovementResult:
        """Compute using JASMINE physics."""
        self._active_module = self._jasmine
        self._stats['jasmine_steps'] += 1

        return self._jasmine.compute_step(
            x, y, state, environment, mask, deterrence_dx, deterrence_dy
        )

    def _compute_disturbance_aware(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: MovementState,
        environment: EnvironmentContext,
        mask: np.ndarray,
        deterrence_dx: Optional[np.ndarray],
        deterrence_dy: Optional[np.ndarray],
    ) -> MovementResult:
        """
        Switch between DEPONS and JASMINE based on disturbance.

        When disturbance is detected (non-zero deterrence), use DEPONS
        validated response. Otherwise use JASMINE physics.
        """
        count = len(x)

        # Check if any agents are under disturbance
        has_disturbance = False
        if deterrence_dx is not None:
            disturbance_magnitude = np.abs(deterrence_dx) + np.abs(deterrence_dy)
            has_disturbance = np.any(disturbance_magnitude[mask] > 0.01)

        if has_disturbance:
            # Use DEPONS for disturbance response (validated)
            return self._compute_depons(
                x, y, state, environment, mask, deterrence_dx, deterrence_dy
            )
        else:
            # Use JASMINE for normal movement
            return self._compute_jasmine(
                x, y, state, environment, mask, deterrence_dx, deterrence_dy
            )

    def _compute_state_based(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: MovementState,
        environment: EnvironmentContext,
        mask: np.ndarray,
        deterrence_dx: Optional[np.ndarray],
        deterrence_dy: Optional[np.ndarray],
    ) -> MovementResult:
        """
        Select movement per-agent based on behavioral state.

        - Dispersing agents: DEPONS (validated dispersal behavior)
        - Disturbed agents: DEPONS (validated disturbance response)
        - Normal agents: JASMINE (physics-based exploration)

        This requires computing separate results and merging them.
        """
        count = len(x)

        # Determine which agents use which model
        use_depons = mask & (
            state.is_dispersing |
            (deterrence_dx is not None and np.abs(deterrence_dx) > 0.01)
        )
        use_jasmine = mask & ~use_depons

        # Initialize result arrays
        dx = np.zeros(count, dtype=np.float32)
        dy = np.zeros(count, dtype=np.float32)
        new_heading = state.heading.copy()
        step_distance = np.zeros(count, dtype=np.float32)
        turning_angle = np.zeros(count, dtype=np.float32)

        # Compute DEPONS for those agents
        if np.any(use_depons):
            depons_result = self._depons.compute_step(
                x, y, state, environment, use_depons, deterrence_dx, deterrence_dy
            )
            dx[use_depons] = depons_result.dx[use_depons]
            dy[use_depons] = depons_result.dy[use_depons]
            new_heading[use_depons] = depons_result.new_heading[use_depons]
            step_distance[use_depons] = depons_result.step_distance[use_depons]
            turning_angle[use_depons] = depons_result.turning_angle[use_depons]
            self._stats['depons_steps'] += np.sum(use_depons)

        # Compute JASMINE for remaining agents
        if np.any(use_jasmine):
            jasmine_result = self._jasmine.compute_step(
                x, y, state, environment, use_jasmine, deterrence_dx, deterrence_dy
            )
            dx[use_jasmine] = jasmine_result.dx[use_jasmine]
            dy[use_jasmine] = jasmine_result.dy[use_jasmine]
            new_heading[use_jasmine] = jasmine_result.new_heading[use_jasmine]
            step_distance[use_jasmine] = jasmine_result.step_distance[use_jasmine]
            turning_angle[use_jasmine] = jasmine_result.turning_angle[use_jasmine]
            self._stats['jasmine_steps'] += np.sum(use_jasmine)

        self._stats['hybrid_switches'] += 1

        return MovementResult(
            dx=dx,
            dy=dy,
            new_heading=new_heading,
            step_distance=step_distance,
            turning_angle=turning_angle,
        )

    def apply_dispersal_modulation(
        self,
        state: MovementState,
        turning_angle: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Delegate to active module."""
        return self._active_module.apply_dispersal_modulation(state, turning_angle, mask)

    def get_mode(self) -> MovementMode:
        """Return HYBRID mode."""
        return MovementMode.HYBRID

    def get_name(self) -> str:
        """Return descriptive name."""
        return f"Hybrid({self.strategy.name})"

    def get_statistics(self) -> Dict[str, Any]:
        """Get movement statistics."""
        return {
            'strategy': self.strategy.name,
            'time_mode': self.time_mode.name,
            'active_module': self._active_module.get_name(),
            **self._stats,
        }

    def reset_statistics(self) -> None:
        """Reset movement statistics."""
        self._stats = {
            'depons_steps': 0,
            'jasmine_steps': 0,
            'hybrid_switches': 0,
        }


def create_movement_module(
    params: 'SimulationParameters',
    time_mode: TimeMode = TimeMode.DEPONS,
    movement_mode: Optional[MovementMode] = None,
) -> MovementModule:
    """
    Factory function to create appropriate movement module.

    This is the recommended way to create movement modules, as it
    automatically selects the right implementation based on mode.

    Args:
        params: Simulation parameters
        time_mode: TimeMode from TimeManager
        movement_mode: Explicit movement mode (overrides time_mode)

    Returns:
        Configured MovementModule instance

    Example:
        # Let time mode determine movement
        movement = create_movement_module(params, TimeMode.DEPONS)

        # Explicitly request JASMINE physics
        movement = create_movement_module(
            params, time_mode, MovementMode.JASMINE_PHYSICS
        )
    """
    if movement_mode is not None:
        if movement_mode == MovementMode.DEPONS_CRW:
            return DEPONSCRWMovementVectorized(params)
        elif movement_mode == MovementMode.JASMINE_PHYSICS:
            return JASMINEPhysicsMovement(params)
        elif movement_mode == MovementMode.HYBRID:
            return HybridMovementSelector(
                params, time_mode, HybridStrategy.DISTURBANCE_AWARE
            )

    # Default: use time mode to determine
    if time_mode == TimeMode.DEPONS:
        return DEPONSCRWMovementVectorized(params)
    else:
        return HybridMovementSelector(
            params, time_mode, HybridStrategy.JASMINE_ONLY
        )
