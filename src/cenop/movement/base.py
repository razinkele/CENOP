"""
Abstract base class for movement modules.

This module defines the interface that all movement implementations must follow,
enabling seamless switching between DEPONS CRW and JASMINE physics-based movement.

The movement module architecture supports:
- DEPONS mode: Empirical step-length & turning-angle distributions
- JASMINE mode: Physics-based movement with advection and symplectic integration
- Hybrid mode: Context-dependent switching between approaches
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters
    from cenop.landscape.cell_data import CellData


class MovementMode(Enum):
    """Movement model modes."""
    DEPONS_CRW = auto()      # DEPONS Correlated Random Walk
    JASMINE_PHYSICS = auto()  # JASMINE physics-based movement
    HYBRID = auto()           # Context-dependent switching


@dataclass
class MovementState:
    """
    State variables for movement calculations.

    These are updated each step and carry information between steps
    for correlated movement models.
    """
    # Previous step values (for correlation)
    prev_heading: np.ndarray      # Previous heading (degrees)
    prev_log_mov: np.ndarray      # Previous log10(movement distance)
    prev_angle: np.ndarray        # Previous turning angle

    # Current step values
    heading: np.ndarray           # Current heading (degrees)
    step_distance: np.ndarray     # Distance to move this step
    dx: np.ndarray                # X displacement
    dy: np.ndarray                # Y displacement

    # Behavioral state
    is_dispersing: np.ndarray     # Whether agent is in dispersal mode
    dispersal_heading: np.ndarray  # Target heading for dispersal

    @classmethod
    def create(cls, count: int) -> 'MovementState':
        """Create a new MovementState for count agents."""
        return cls(
            prev_heading=np.zeros(count, dtype=np.float32),
            prev_log_mov=np.full(count, 1.25, dtype=np.float32),
            prev_angle=np.zeros(count, dtype=np.float32),
            heading=np.random.uniform(0, 360, count).astype(np.float32),
            step_distance=np.zeros(count, dtype=np.float32),
            dx=np.zeros(count, dtype=np.float32),
            dy=np.zeros(count, dtype=np.float32),
            is_dispersing=np.zeros(count, dtype=bool),
            dispersal_heading=np.zeros(count, dtype=np.float32),
        )


@dataclass
class EnvironmentContext:
    """
    Environmental context for movement calculations.

    Contains all environmental variables that can influence movement.
    """
    depth: np.ndarray              # Water depth at current position
    salinity: np.ndarray           # Salinity at current position
    temperature: Optional[np.ndarray] = None  # Temperature (JASMINE)
    current_u: Optional[np.ndarray] = None    # Current velocity U component (JASMINE)
    current_v: Optional[np.ndarray] = None    # Current velocity V component (JASMINE)
    prey_density: Optional[np.ndarray] = None  # Prey density (JASMINE)

    @classmethod
    def from_landscape(
        cls,
        landscape: 'CellData',
        x: np.ndarray,
        y: np.ndarray,
        month: int = 1
    ) -> 'EnvironmentContext':
        """Create environment context from landscape data."""
        positions = np.column_stack((x, y))

        depth = landscape.get_depths_vectorized(positions)
        salinity = landscape.get_salinities_vectorized(positions)

        return cls(
            depth=depth,
            salinity=salinity,
        )

    @classmethod
    def create_homogeneous(cls, count: int, depth: float = 30.0, salinity: float = 30.0) -> 'EnvironmentContext':
        """Create homogeneous environment context."""
        return cls(
            depth=np.full(count, depth, dtype=np.float32),
            salinity=np.full(count, salinity, dtype=np.float32),
        )


@dataclass
class MovementResult:
    """
    Result of movement calculation.

    Contains the computed displacements and updated state.
    """
    dx: np.ndarray                 # X displacement
    dy: np.ndarray                 # Y displacement
    new_heading: np.ndarray        # Updated heading
    step_distance: np.ndarray      # Distance moved
    turning_angle: np.ndarray      # Turning angle applied


class MovementModule(ABC):
    """
    Abstract base class for movement modules.

    All movement implementations (DEPONS CRW, JASMINE physics, etc.)
    must implement this interface to be interchangeable.

    The compute_step method is the core of the movement model, taking
    current positions, state, and environment to produce displacements.
    """

    def __init__(self, params: 'SimulationParameters'):
        """
        Initialize movement module.

        Args:
            params: Simulation parameters containing movement coefficients
        """
        self.params = params

    @abstractmethod
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
        Compute movement step for all agents.

        This is the core movement calculation. Implementations should:
        1. Calculate turning angles based on previous state and environment
        2. Calculate step lengths based on state and environment
        3. Apply any behavioral modifiers (dispersal, deterrence)
        4. Return the computed displacements

        Args:
            x: Current X positions
            y: Current Y positions
            state: Current movement state (heading, prev values, etc.)
            environment: Environmental context (depth, salinity, etc.)
            mask: Boolean mask of active agents to update
            deterrence_dx: Optional X deterrence vectors
            deterrence_dy: Optional Y deterrence vectors

        Returns:
            MovementResult with displacements and updated state values
        """
        pass

    @abstractmethod
    def apply_dispersal_modulation(
        self,
        state: MovementState,
        turning_angle: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply dispersal-specific movement modifications.

        During dispersal, turning is typically reduced to maintain
        directional persistence toward the target.

        Args:
            state: Current movement state
            turning_angle: Computed turning angles
            mask: Active agent mask

        Returns:
            Modified turning angles
        """
        pass

    def get_name(self) -> str:
        """Return the name of this movement module."""
        return self.__class__.__name__

    def get_mode(self) -> MovementMode:
        """Return the movement mode this module implements."""
        return MovementMode.DEPONS_CRW  # Default, override in subclasses
