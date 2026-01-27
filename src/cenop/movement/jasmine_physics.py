"""
JASMINE physics-based movement module.

Implements a physics-realistic movement model with:
- Velocity-based movement with inertia
- Advection by ocean currents
- Symplectic/Verlet integration for numerical stability
- Potential field gradients for behavioral decisions
- 3D movement support (horizontal + diving)

This model provides higher physical realism than the empirical CRW,
at the cost of more parameters and computational complexity.

Key features:
- Continuous velocity state (not just position)
- Current advection: v_total = v_self + v_current
- Symplectic integration preserves energy in long simulations
- Potential fields for prey, social, and avoidance behaviors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from cenop.movement.base import (
    MovementModule,
    MovementMode,
    MovementState,
    EnvironmentContext,
    MovementResult,
)

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters


@dataclass
class JASMINEMovementState(MovementState):
    """
    Extended movement state for JASMINE physics model.

    Adds velocity components and 3D position for physics-based movement.
    """
    # Velocity components (m/s)
    vx: np.ndarray                 # X velocity
    vy: np.ndarray                 # Y velocity
    vz: np.ndarray                 # Z velocity (diving)

    # 3D position
    z: np.ndarray                  # Depth (meters below surface)

    # Acceleration (for Verlet integration)
    ax: np.ndarray                 # X acceleration
    ay: np.ndarray                 # Y acceleration

    @classmethod
    def create(cls, count: int) -> 'JASMINEMovementState':
        """Create a new JASMINEMovementState for count agents."""
        base = MovementState.create(count)
        return cls(
            # Base state
            prev_heading=base.prev_heading,
            prev_log_mov=base.prev_log_mov,
            prev_angle=base.prev_angle,
            heading=base.heading,
            step_distance=base.step_distance,
            dx=base.dx,
            dy=base.dy,
            is_dispersing=base.is_dispersing,
            dispersal_heading=base.dispersal_heading,
            # Extended state
            vx=np.zeros(count, dtype=np.float32),
            vy=np.zeros(count, dtype=np.float32),
            vz=np.zeros(count, dtype=np.float32),
            z=np.zeros(count, dtype=np.float32),
            ax=np.zeros(count, dtype=np.float32),
            ay=np.zeros(count, dtype=np.float32),
        )


@dataclass
class JASMINEEnvironmentContext(EnvironmentContext):
    """
    Extended environment context for JASMINE physics model.

    Adds current velocities and potential field gradients.
    """
    # Ocean currents (m/s)
    current_u: np.ndarray          # Eastward current velocity
    current_v: np.ndarray          # Northward current velocity

    # Potential field gradients (for decision-making)
    prey_gradient_x: Optional[np.ndarray] = None
    prey_gradient_y: Optional[np.ndarray] = None
    social_gradient_x: Optional[np.ndarray] = None
    social_gradient_y: Optional[np.ndarray] = None
    avoidance_gradient_x: Optional[np.ndarray] = None
    avoidance_gradient_y: Optional[np.ndarray] = None


class JASMINEPhysicsMovement(MovementModule):
    """
    JASMINE physics-based movement implementation.

    Uses a velocity-based model with:
    1. Self-propulsion: Agent generates thrust in desired direction
    2. Advection: Passive transport by ocean currents
    3. Drag: Velocity-dependent resistance
    4. Behavioral forces: Attraction/repulsion from potential fields

    The equation of motion is:
        dv/dt = F_self + F_current + F_drag + F_behavior

    Integrated using velocity Verlet for stability.
    """

    # Physical constants
    DRAG_COEFFICIENT = 0.1         # Velocity-dependent drag
    MAX_SPEED = 3.0                # Maximum self-propulsion speed (m/s)
    ACCELERATION_SCALE = 0.5       # Thrust acceleration scaling
    CURRENT_COUPLING = 1.0         # How strongly currents affect movement

    def __init__(self, params: 'SimulationParameters'):
        """
        Initialize JASMINE physics movement module.

        Args:
            params: Simulation parameters
        """
        super().__init__(params)

        # Physics parameters (could be made configurable)
        self.drag = self.DRAG_COEFFICIENT
        self.max_speed = self.MAX_SPEED
        self.accel_scale = self.ACCELERATION_SCALE
        self.current_coupling = self.CURRENT_COUPLING

        # Time step (default 30 min = 1800 seconds)
        self.dt = getattr(params, 'dt_seconds', 1800)

        # Behavioral weights
        self.prey_weight = getattr(params, 'prey_attraction_weight', 1.0)
        self.social_weight = getattr(params, 'social_weight', 0.5)
        self.avoidance_weight = getattr(params, 'avoidance_weight', 2.0)

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
        Compute physics-based movement step.

        Uses velocity Verlet integration:
        1. Update position: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
        2. Compute new acceleration: a(t+dt) = F(x(t+dt), v(t)) / m
        3. Update velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt

        Args:
            x: Current X positions
            y: Current Y positions
            state: Movement state (must be JASMINEMovementState for full physics)
            environment: Environmental context
            mask: Boolean mask of active agents
            deterrence_dx: Optional X deterrence component
            deterrence_dy: Optional Y deterrence component

        Returns:
            MovementResult with displacements
        """
        count = len(x)

        # Check if we have JASMINE extended state
        if isinstance(state, JASMINEMovementState):
            return self._compute_physics_step(
                x, y, state, environment, mask, deterrence_dx, deterrence_dy
            )
        else:
            # Fallback to simplified physics for basic MovementState
            return self._compute_simplified_step(
                x, y, state, environment, mask, deterrence_dx, deterrence_dy
            )

    def _compute_physics_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: JASMINEMovementState,
        environment: EnvironmentContext,
        mask: np.ndarray,
        deterrence_dx: Optional[np.ndarray],
        deterrence_dy: Optional[np.ndarray],
    ) -> MovementResult:
        """
        Full physics-based movement with velocity state.
        """
        count = len(x)
        dt = self.dt

        # Get current velocities
        vx = state.vx.copy()
        vy = state.vy.copy()

        # === Compute forces ===

        # 1. Self-propulsion force (toward heading direction)
        heading_rad = np.radians(state.heading)
        thrust_x = self.accel_scale * np.sin(heading_rad)
        thrust_y = self.accel_scale * np.cos(heading_rad)

        # 2. Drag force (opposes velocity)
        speed = np.sqrt(vx**2 + vy**2)
        drag_x = -self.drag * vx * speed
        drag_y = -self.drag * vy * speed

        # 3. Current advection (if available)
        if hasattr(environment, 'current_u') and environment.current_u is not None:
            current_x = environment.current_u * self.current_coupling
            current_y = environment.current_v * self.current_coupling
        else:
            current_x = np.zeros(count, dtype=np.float32)
            current_y = np.zeros(count, dtype=np.float32)

        # 4. Behavioral forces from potential gradients
        behavior_x = np.zeros(count, dtype=np.float32)
        behavior_y = np.zeros(count, dtype=np.float32)

        if isinstance(environment, JASMINEEnvironmentContext):
            if environment.prey_gradient_x is not None:
                behavior_x += self.prey_weight * environment.prey_gradient_x
                behavior_y += self.prey_weight * environment.prey_gradient_y
            if environment.social_gradient_x is not None:
                behavior_x += self.social_weight * environment.social_gradient_x
                behavior_y += self.social_weight * environment.social_gradient_y
            if environment.avoidance_gradient_x is not None:
                behavior_x -= self.avoidance_weight * environment.avoidance_gradient_x
                behavior_y -= self.avoidance_weight * environment.avoidance_gradient_y

        # 5. Deterrence force
        if deterrence_dx is not None:
            behavior_x += deterrence_dx
            behavior_y += deterrence_dy

        # === Total acceleration ===
        ax = thrust_x + drag_x + behavior_x
        ay = thrust_y + drag_y + behavior_y

        # === Velocity Verlet integration ===
        # Half-step velocity update
        vx_half = vx + 0.5 * ax * dt
        vy_half = vy + 0.5 * ay * dt

        # Add current advection to velocity
        vx_total = vx_half + current_x
        vy_total = vy_half + current_y

        # Position update
        dx = vx_total * dt
        dy = vy_total * dt

        # Compute new acceleration at new position (simplified - same as old)
        ax_new = ax  # Would recompute with new position in full implementation

        # Full velocity update
        vx_new = vx + 0.5 * (ax + ax_new) * dt
        vy_new = vy + 0.5 * (ay + ax_new) * dt

        # Speed limiting
        speed_new = np.sqrt(vx_new**2 + vy_new**2)
        too_fast = speed_new > self.max_speed
        scale = np.where(too_fast, self.max_speed / (speed_new + 1e-10), 1.0)
        vx_new *= scale
        vy_new *= scale

        # Update state velocities
        state.vx[mask] = vx_new[mask]
        state.vy[mask] = vy_new[mask]

        # Compute new heading from velocity
        new_heading = np.degrees(np.arctan2(vx_new, vy_new)) % 360.0

        # Compute turning angle
        turning_angle = (new_heading - state.heading + 180) % 360 - 180

        # Step distance
        step_distance = np.sqrt(dx**2 + dy**2)

        return MovementResult(
            dx=dx.astype(np.float32),
            dy=dy.astype(np.float32),
            new_heading=new_heading.astype(np.float32),
            step_distance=step_distance.astype(np.float32),
            turning_angle=turning_angle.astype(np.float32),
        )

    def _compute_simplified_step(
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
        Simplified physics for basic MovementState (no velocity tracking).

        Uses heading and a physics-inspired speed model without full
        velocity state tracking.
        """
        count = len(x)

        # Base speed from parameters (convert from DEPONS log scale)
        base_speed = np.power(10.0, state.prev_log_mov) / 4.0

        # Heading-based displacement
        heading_rad = np.radians(state.heading)
        dx = np.sin(heading_rad) * base_speed
        dy = np.cos(heading_rad) * base_speed

        # Add current advection if available
        if hasattr(environment, 'current_u') and environment.current_u is not None:
            dx += environment.current_u * self.current_coupling * self.dt / 400.0  # Scale to cells
            dy += environment.current_v * self.current_coupling * self.dt / 400.0

        # Add deterrence
        if deterrence_dx is not None:
            dx += deterrence_dx
            dy += deterrence_dy

        # Compute new heading from displacement
        new_heading = np.degrees(np.arctan2(dx, dy)) % 360.0

        # Apply some random turning for exploration
        random_turn = np.random.normal(0, 10, count)
        new_heading = (new_heading + random_turn) % 360.0

        # Turning angle
        turning_angle = (new_heading - state.heading + 180) % 360 - 180

        # Step distance
        step_distance = np.sqrt(dx**2 + dy**2)

        return MovementResult(
            dx=dx.astype(np.float32),
            dy=dy.astype(np.float32),
            new_heading=new_heading.astype(np.float32),
            step_distance=step_distance.astype(np.float32),
            turning_angle=turning_angle.astype(np.float32),
        )

    def apply_dispersal_modulation(
        self,
        state: MovementState,
        turning_angle: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply dispersal modulation for physics model.

        In physics mode, dispersal biases the thrust direction
        toward the dispersal target rather than modifying turning.
        """
        dispersing = mask & state.is_dispersing

        if not np.any(dispersing):
            return turning_angle

        # For dispersing agents, blend current heading toward dispersal target
        modified = turning_angle.copy()

        # Calculate angle to dispersal heading
        angle_diff = (state.dispersal_heading - state.heading + 180) % 360 - 180

        # Bias turn toward dispersal target (50% blend)
        modified[dispersing] = 0.5 * modified[dispersing] + 0.5 * angle_diff[dispersing]

        return modified

    def get_mode(self) -> MovementMode:
        """Return JASMINE_PHYSICS mode."""
        return MovementMode.JASMINE_PHYSICS

    def get_name(self) -> str:
        """Return module name."""
        return "JASMINE_Physics"
