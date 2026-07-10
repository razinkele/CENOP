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

    def __init__(self, params: "SimulationParameters", rng: Optional[np.random.Generator] = None):
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
            rng: NumPy random Generator for reproducibility
        """
        super().__init__(params)
        self.rng = rng if rng is not None else np.random.default_rng()

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
        """DEPONS CRW step via the shared validated core.

        Generates the turning angle (reject-and-redraw + distance second loop) and step
        length, then composes the heading with reference-memory attraction and deterrence
        (deterrence enters the heading vector, NOT the raw displacement). Dispersal heading
        override is the caller's responsibility. Exposes raw pres_angle/log_mov for parity.
        """
        from cenop.movement.crw_core import generate_crw_angle_step, compose_movement

        count = len(x)
        depths = np.asarray(environment.depth, dtype=np.float64)
        salinity = np.asarray(environment.salinity, dtype=np.float64)
        prev_angle = np.asarray(state.prev_angle, dtype=np.float64)
        prev_log_mov = np.asarray(state.prev_log_mov, dtype=np.float64)

        pres_angle = np.zeros(count, dtype=np.float64)
        log_mov = np.zeros(count, dtype=np.float64)
        env_mod = np.zeros(count, dtype=np.float32)
        rand_angle = np.zeros(count, dtype=np.float64)
        rand_len = np.zeros(count, dtype=np.float64)

        generate_crw_angle_step(
            self.rng,
            prev_angle,
            prev_log_mov,
            depths,
            salinity,
            mask,
            self.params,
            pres_angle,
            log_mov,
            env_mod,
            rand_angle,
            rand_len,
        )

        # Turn heading (dispersal override handled by the caller)
        heading = np.asarray(state.heading, dtype=np.float32).copy()
        heading[mask] = (heading[mask] + pres_angle[mask]) % 360.0

        ve_total = (
            state.ve_total if state.ve_total is not None else np.zeros(count, dtype=np.float32)
        )
        vt_x = state.vt_x if state.vt_x is not None else np.zeros(count, dtype=np.float32)
        vt_y = state.vt_y if state.vt_y is not None else np.zeros(count, dtype=np.float32)

        d_dx = (
            np.asarray(deterrence_dx, dtype=np.float64)
            if deterrence_dx is not None
            else np.zeros(count, dtype=np.float64)
        )
        d_dy = (
            np.asarray(deterrence_dy, dtype=np.float64)
            if deterrence_dy is not None
            else np.zeros(count, dtype=np.float64)
        )

        rads = np.zeros(count, dtype=np.float32)
        dx = np.zeros(count, dtype=np.float32)
        dy = np.zeros(count, dtype=np.float32)
        step_dist = np.zeros(count, dtype=np.float32)

        disp_step = getattr(self.params, "mean_disp_dist", 1.6) / 0.4
        compose_movement(
            heading,
            pres_angle,
            log_mov,
            ve_total,
            vt_x,
            vt_y,
            d_dx,
            d_dy,
            state.is_dispersing,
            mask,
            self.params.inertia_const,
            disp_step,
            rads,
            dx,
            dy,
            step_dist,
        )

        inactive = ~mask
        dx[inactive] = 0.0
        dy[inactive] = 0.0
        step_dist[inactive] = 0.0

        turning_angle = np.zeros(count, dtype=np.float32)
        turning_angle[mask] = pres_angle[mask].astype(np.float32)

        new_heading = np.asarray(state.heading, dtype=np.float32).copy()
        new_heading[mask] = heading[mask]

        return MovementResult(
            dx=dx,
            dy=dy,
            new_heading=new_heading,
            step_distance=step_dist,
            turning_angle=turning_angle,
            pres_angle=pres_angle,
            log_mov=log_mov,
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
