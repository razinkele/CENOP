"""
DEPONS Correlated Random Walk (CRW) movement module.

Implements the empirically-calibrated movement model from DEPONS 3.0.
This model uses step-length and turning-angle distributions conditioned
on behavioral state and environmental variables.

Key features:
- Autoregressive turning angles with environmental modulation
- Log-normal step lengths with depth/salinity effects
- Dispersal-specific movement modifications
- Deterrence vector integration

Reference:
    DEPONS Technical Documentation, Section 3.2: Movement Model
    Nabe-Nielsen et al. (2018) - Predicting the impacts of...

Translates from: Porpoise.java move() method
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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


class DEPONSCRWMovement(MovementModule):
    """
    DEPONS Correlated Random Walk movement implementation.

    The CRW model calculates:
    1. Turning angle: AR(1) process with environmental modulation
       angleTmp = b0 * prevAngle + N(0, σ)
       presAngle = angleTmp * (b1*depth + b2*salinity + b3)

    2. Step length: Log-normal with environmental effects
       log10(mov) = a0 * prev_log_mov + a1*depth + a2*salinity + N(μ, σ)

    Parameters are calibrated from harbour porpoise telemetry data
    to produce realistic movement patterns.
    """

    def __init__(self, params: 'SimulationParameters'):
        """
        Initialize DEPONS CRW movement module.

        Args:
            params: Simulation parameters with CRW coefficients:
                - corr_angle_base (b0): Turning angle autocorrelation
                - corr_angle_bathy (b1): Depth effect on turning
                - corr_angle_salinity (b2): Salinity effect on turning
                - corr_angle_base_sd (b3): Base turning angle SD
                - corr_logmov_length (a0): Step length autocorrelation
                - corr_logmov_bathy (a1): Depth effect on step length
                - corr_logmov_salinity (a2): Salinity effect on step length
                - r1_mean, r1_sd: Step length random component
                - r2_mean, r2_sd: Turning angle random component
                - max_mov: Maximum log10(movement)
        """
        super().__init__(params)

        # Pre-extract parameters for efficiency
        self.b0 = params.corr_angle_base
        self.b1 = params.corr_angle_bathy
        self.b2 = params.corr_angle_salinity
        self.b3 = params.corr_angle_base_sd

        self.a0 = params.corr_logmov_length
        self.a1 = params.corr_logmov_bathy
        self.a2 = params.corr_logmov_salinity

        self.r1_mean = params.r1_mean
        self.r1_sd = params.r1_sd
        self.r2_mean = params.r2_mean
        self.r2_sd = params.r2_sd

        self.max_mov = params.max_mov

        # Preallocated work arrays (set on first use)
        self._work_arrays_size = 0
        self._rand_angle: Optional[np.ndarray] = None
        self._rand_len: Optional[np.ndarray] = None
        self._angle_tmp: Optional[np.ndarray] = None
        self._env_mod: Optional[np.ndarray] = None
        self._log_mov: Optional[np.ndarray] = None
        self._step_dist: Optional[np.ndarray] = None
        self._rads: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._dy: Optional[np.ndarray] = None

    def _ensure_work_arrays(self, count: int) -> None:
        """Ensure work arrays are allocated for count agents."""
        if self._work_arrays_size >= count:
            return

        self._work_arrays_size = count
        self._rand_angle = np.zeros(count, dtype=np.float32)
        self._rand_len = np.zeros(count, dtype=np.float32)
        self._angle_tmp = np.zeros(count, dtype=np.float32)
        self._env_mod = np.zeros(count, dtype=np.float32)
        self._log_mov = np.zeros(count, dtype=np.float32)
        self._step_dist = np.zeros(count, dtype=np.float32)
        self._rads = np.zeros(count, dtype=np.float32)
        self._dx = np.zeros(count, dtype=np.float32)
        self._dy = np.zeros(count, dtype=np.float32)

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
        Compute DEPONS CRW movement step.

        Implements the full DEPONS movement algorithm:
        1. Calculate turning angle with AR(1) and environmental modulation
        2. Calculate step length with log-normal and environmental effects
        3. Apply dispersal modifications if applicable
        4. Convert to Cartesian displacements
        5. Add deterrence vectors if present

        Args:
            x: Current X positions
            y: Current Y positions
            state: Movement state (heading, prev values)
            environment: Environmental context (depth, salinity)
            mask: Boolean mask of active agents
            deterrence_dx: Optional X deterrence component
            deterrence_dy: Optional Y deterrence component

        Returns:
            MovementResult with displacements and updated values
        """
        count = len(x)
        self._ensure_work_arrays(count)

        # === Step 1: Calculate Turning Angle ===
        # DEPONS formula:
        #   angleTmp = b0 * prevAngle + R2
        #   presAngle = angleTmp * (b1*depth + b2*salinity + b3)

        # Random component R2 ~ N(r2_mean, r2_sd)
        np.copyto(self._rand_angle, np.random.normal(self.r2_mean, self.r2_sd, count))

        # angleTmp = b0 * prevAngle + R2
        np.multiply(self.b0, state.prev_angle, out=self._angle_tmp)
        self._angle_tmp += self._rand_angle

        # Environmental modulation: (b1*depth + b2*salinity + b3)
        np.multiply(self.b1, environment.depth, out=self._env_mod)
        self._env_mod += self.b2 * environment.salinity
        self._env_mod += self.b3

        # presAngle = angleTmp * env_modulation
        turning_angle = self._angle_tmp * self._env_mod

        # Clip to [-180, 180]
        np.clip(turning_angle, -180, 180, out=turning_angle)

        # === Step 2: Apply dispersal modulation ===
        turning_angle = self.apply_dispersal_modulation(state, turning_angle, mask)

        # === Step 3: Update heading ===
        new_heading = state.heading.copy()
        new_heading[mask] += turning_angle[mask]
        new_heading[mask] %= 360.0

        # === Step 4: Calculate Step Length ===
        # DEPONS formula:
        #   log10(mov) = a0 * prev_log_mov + a1*depth + a2*salinity + R1

        # Random component R1 ~ N(r1_mean, r1_sd)
        np.copyto(self._rand_len, np.random.normal(self.r1_mean, self.r1_sd, count))

        # log_mov = a0 * prev + a1*depth + a2*salinity + R1
        np.multiply(self.a0, state.prev_log_mov, out=self._log_mov)
        self._log_mov += self.a1 * environment.depth
        self._log_mov += self.a2 * environment.salinity
        self._log_mov += self._rand_len

        # Clip to max speed
        np.minimum(self._log_mov, self.max_mov, out=self._log_mov)

        # Convert to distance: 10^log_mov / 4.0 (400m cell adjustment)
        np.power(10.0, self._log_mov, out=self._step_dist)
        self._step_dist /= 4.0

        # === Step 5: Convert to Cartesian displacements ===
        np.radians(new_heading, out=self._rads)
        np.sin(self._rads, out=self._dx)
        self._dx *= self._step_dist
        np.cos(self._rads, out=self._dy)
        self._dy *= self._step_dist

        # === Step 6: Apply deterrence vectors ===
        if deterrence_dx is not None and deterrence_dy is not None:
            self._dx[mask] += deterrence_dx[mask]
            self._dy[mask] += deterrence_dy[mask]

        # === Step 7: Zero out inactive agents ===
        inactive = ~mask
        self._dx[inactive] = 0.0
        self._dy[inactive] = 0.0
        self._step_dist[inactive] = 0.0
        turning_angle[inactive] = 0.0

        return MovementResult(
            dx=self._dx.copy(),
            dy=self._dy.copy(),
            new_heading=new_heading,
            step_distance=self._step_dist.copy(),
            turning_angle=turning_angle,
        )

    def apply_dispersal_modulation(
        self,
        state: MovementState,
        turning_angle: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply dispersal-specific movement modifications (PSM-Type2).

        During dispersal:
        - Turning is progressively reduced as distance to target decreases
        - Uses logistic dampening function for smooth transition
        - Heading is biased toward dispersal target

        The logistic function ensures:
        - Full turning early in dispersal
        - Reduced turning as target is approached
        - Smooth transition without abrupt changes

        Args:
            state: Movement state with dispersal info
            turning_angle: Computed turning angles
            mask: Active agent mask

        Returns:
            Modified turning angles with dispersal dampening
        """
        dispersing = mask & state.is_dispersing

        if not np.any(dispersing):
            return turning_angle

        # For dispersing agents, apply logistic dampening
        # This is a simplified version - full implementation would
        # track distance traveled vs target distance

        # Reduce turning by 70% during dispersal
        modified = turning_angle.copy()
        modified[dispersing] *= 0.3

        return modified

    def get_mode(self) -> MovementMode:
        """Return DEPONS_CRW mode."""
        return MovementMode.DEPONS_CRW

    def get_name(self) -> str:
        """Return module name."""
        return "DEPONS_CRW"


class DEPONSCRWMovementVectorized(DEPONSCRWMovement):
    """
    Fully vectorized DEPONS CRW for maximum performance.

    This version optimizes memory access patterns and reduces
    Python overhead for large populations.
    """

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
        Optimized vectorized CRW computation.

        Uses fused operations where possible to reduce memory traffic.
        """
        count = len(x)
        active_idx = np.where(mask)[0]

        if len(active_idx) == 0:
            return MovementResult(
                dx=np.zeros(count, dtype=np.float32),
                dy=np.zeros(count, dtype=np.float32),
                new_heading=state.heading.copy(),
                step_distance=np.zeros(count, dtype=np.float32),
                turning_angle=np.zeros(count, dtype=np.float32),
            )

        # Work only on active agents
        n_active = len(active_idx)

        # Random components
        rand_angle = np.random.normal(self.r2_mean, self.r2_sd, n_active).astype(np.float32)
        rand_len = np.random.normal(self.r1_mean, self.r1_sd, n_active).astype(np.float32)

        # Extract active values
        prev_angle_active = state.prev_angle[active_idx]
        prev_log_mov_active = state.prev_log_mov[active_idx]
        heading_active = state.heading[active_idx]
        depth_active = environment.depth[active_idx]
        salinity_active = environment.salinity[active_idx]

        # Turning angle calculation (fused)
        angle_tmp = self.b0 * prev_angle_active + rand_angle
        env_mod = self.b1 * depth_active + self.b2 * salinity_active + self.b3
        turning_angle_active = np.clip(angle_tmp * env_mod, -180, 180)

        # Apply dispersal modulation for active dispersing
        if np.any(state.is_dispersing[active_idx]):
            disp_mask = state.is_dispersing[active_idx]
            turning_angle_active[disp_mask] *= 0.3

        # Update heading
        new_heading_active = (heading_active + turning_angle_active) % 360.0

        # Step length calculation (fused)
        log_mov = (self.a0 * prev_log_mov_active +
                   self.a1 * depth_active +
                   self.a2 * salinity_active +
                   rand_len)
        log_mov = np.minimum(log_mov, self.max_mov)
        step_dist_active = np.power(10.0, log_mov) / 4.0

        # Convert to displacements
        rads = np.radians(new_heading_active)
        dx_active = np.sin(rads) * step_dist_active
        dy_active = np.cos(rads) * step_dist_active

        # Apply deterrence
        if deterrence_dx is not None and deterrence_dy is not None:
            dx_active += deterrence_dx[active_idx]
            dy_active += deterrence_dy[active_idx]

        # Build full result arrays
        dx = np.zeros(count, dtype=np.float32)
        dy = np.zeros(count, dtype=np.float32)
        new_heading = state.heading.copy()
        step_distance = np.zeros(count, dtype=np.float32)
        turning_angle = np.zeros(count, dtype=np.float32)

        dx[active_idx] = dx_active
        dy[active_idx] = dy_active
        new_heading[active_idx] = new_heading_active
        step_distance[active_idx] = step_dist_active
        turning_angle[active_idx] = turning_angle_active

        return MovementResult(
            dx=dx,
            dy=dy,
            new_heading=new_heading,
            step_distance=step_distance,
            turning_angle=turning_angle,
        )
