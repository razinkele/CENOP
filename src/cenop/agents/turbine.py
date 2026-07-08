"""
Wind turbine agent implementation.

Turbines generate noise during construction and operation phases
that deters porpoises within a certain radius.
Translates from: Turbine.java (258 lines)
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List, Tuple
from pathlib import Path

from enum import Enum

from cenop.agents.base import Agent
from cenop.behavior.sound import (
    TurbineNoise,
    calculate_received_level,
    calculate_deterrence_vector,
    response_probability_from_rl,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters
    from cenop.core.simulation import SimulationState


class TurbinePhase(str, Enum):
    """Turbine operational phases."""
    OFF = "off"
    PLANNED = "planned"
    CONSTRUCTION = "construction"
    OPERATION = "operation"


@dataclass
class Turbine(Agent):
    """
    Wind turbine agent that generates noise deterrence.
    
    Turbines can be in different phases:
    - Construction: High-intensity pile driving noise
    - Operation: Lower-intensity operational noise
    
    Translates from: Turbine.java
    """
    
    # Turbine identification
    name: str = ""
    
    # Impact factor (relative to reference Roedsand turbine)
    # Values > 1 mean louder, < 1 mean quieter
    impact: float = 1.0
    
    # Timing (in ticks)
    start_tick: int = 0
    end_tick: int = 2147483647  # Integer.MAX_VALUE equivalent
    
    # Current phase
    phase: str = TurbinePhase.CONSTRUCTION
    
    # Noise characteristics
    noise: TurbineNoise = field(default_factory=TurbineNoise)
    
    # Track if turbine is active
    _is_active: bool = False
    
    def __post_init__(self):
        """Initialize turbine noise with impact factor."""
        self.noise = TurbineNoise(impact=self.impact)
        
    def is_active(self, tick: int = None) -> bool:
        """Check if turbine is actively producing noise (construction or operational)."""
        if tick is not None:
            # Active means construction OR operational (not planned)
            return tick >= self.start_tick
        return self._is_active
        
    def update_phase(self, current_tick: int) -> None:
        """Update turbine phase and active status based on current tick.

        Lifecycle: PLANNED → CONSTRUCTION → OPERATION
        - tick < start_tick  → PLANNED   (visible, no noise)
        - start_tick <= tick < end_tick → CONSTRUCTION (pile-driving noise)
        - tick >= end_tick   → OPERATION  (operational noise)
        """
        if current_tick < self.start_tick:
            self.phase = TurbinePhase.PLANNED
            self._is_active = False
        elif current_tick < self.end_tick:
            self.phase = TurbinePhase.CONSTRUCTION
            self._is_active = True
        else:
            self.phase = TurbinePhase.OPERATION
            self._is_active = True
        
    def get_source_level(self) -> float:
        """Get the current source level based on lifecycle phase.

        - CONSTRUCTION: impact value IS the source level (pile-driving, ~200 dB)
        - OPERATION: fixed operational noise level (145 dB)
        - PLANNED/OFF: 0 (no noise, turbine inactive)
        """
        if self.phase == TurbinePhase.CONSTRUCTION:
            return self.impact  # In DEPONS, impact IS the SL in dB
        elif self.phase == TurbinePhase.OPERATION:
            return self.noise.source_level_operation  # 145 dB
        else:
            return 0.0
        
    def get_received_level(
        self,
        porpoise_x: float,
        porpoise_y: float,
        source_level: float = None,
        alpha: float = 0.0,
        beta: float = 20.0,
        cell_size: float = 400.0
    ) -> float:
        """
        Calculate received sound level at porpoise position.
        
        Args:
            porpoise_x, porpoise_y: Porpoise position
            source_level: Source level (uses phase-based if None)
            alpha: Absorption coefficient
            beta: Spreading loss factor
            cell_size: Cell size in meters
            
        Returns:
            Received level in dB
        """
        if source_level is None:
            # In DEPONS, impact IS the source level in dB directly
            source_level = self.impact
            
        # Calculate distance in meters
        dx = (porpoise_x - self.x) * cell_size
        dy = (porpoise_y - self.y) * cell_size
        distance_m = np.sqrt(dx**2 + dy**2)
        
        if distance_m < 1.0:
            distance_m = 1.0
            
        return calculate_received_level(source_level, distance_m, alpha, beta)
        
    def should_deter(
        self,
        target_x: float,
        target_y: float,
        params: SimulationParameters,
        cell_size: float = 400.0
    ) -> Tuple[bool, float, float, float]:
        """
        Check if this turbine should deter a porpoise at the given location.
        
        In DEPONS, the 'impact' field IS the source level (SL) in dB.
        The deterrence strength is: strength = RL - threshold
        where RL = impact - (β*log10(dist) + α*dist)
        
        Args:
            target_x, target_y: Porpoise position
            params: Simulation parameters
            cell_size: Cell size in meters
            
        Returns:
            (should_deter, received_level, distance_m, strength)
        """
        if not self._is_active:
            return (False, 0.0, 0.0, 0.0)
            
        # Calculate distance in meters
        dx = (target_x - self.x) * cell_size
        dy = (target_y - self.y) * cell_size
        distance_m = np.sqrt(dx**2 + dy**2)
        
        # Check max distance (km to m)
        max_dist_m = params.deter_max_distance * 1000
        if distance_m > max_dist_m:
            return (False, 0.0, distance_m, 0.0)
            
        # Minimum distance to avoid log(0)
        if distance_m < 1.0:
            distance_m = 1.0
            
        # DEPONS formula: RL = SL - (β*log10(dist) + α*dist)
        # Source level depends on lifecycle phase (construction ~200 dB, operation ~145 dB)
        source_level = self.get_source_level()
        transmission_loss = (
            params.beta_hat * np.log10(distance_m) +
            params.alpha_hat * distance_m
        )
        received_level = source_level - transmission_loss
        
        # Deterrence strength = RL - threshold (DEPONS Turbine.java line 227)
        strength = received_level - params.deter_threshold
        
        # Only deter if strength > 0
        if strength <= 0:
            return (False, received_level, distance_m, 0.0)
            
        return (True, received_level, distance_m, strength)
        
    def get_deterrence_vector(
        self,
        porpoise_x: float,
        porpoise_y: float,
        strength: float,
        deter_coeff: float = 0.07
    ) -> Tuple[float, float]:
        """Calculate deterrence vector for a porpoise."""
        return calculate_deterrence_vector(
            porpoise_x, porpoise_y,
            self.x, self.y,
            strength, deter_coeff
        )
        
    @classmethod
    def load_from_file(
        cls,
        filepath: str,
        utm_origin_x: float = 0.0,
        utm_origin_y: float = 0.0,
        cell_size: float = 400.0
    ) -> List[Turbine]:
        """
        Load turbines from a data file.
        
        File format (tab/space separated):
        name  utm_x  utm_y  impact  [start_tick]  [end_tick]
        
        Args:
            filepath: Path to turbine data file
            utm_origin_x, utm_origin_y: UTM origin for coordinate conversion
            cell_size: Grid cell size in meters
            
        Returns:
            List of Turbine objects
        """
        turbines = []
        filepath = Path(filepath)
        
        if not filepath.exists():
            return turbines
            
        with open(filepath, 'r') as f:
            # Read header to detect time unit
            header = next(f, "").strip().lower()
            # If header contains "tick.start" / "tick.end", values are in ticks.
            # If header says just "start" / "end", values are in days → multiply by 48.
            header_cols = header.split()
            uses_ticks = any("tick" in col for col in header_cols)
            day_to_tick = 1 if uses_ticks else 48

            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                cols = line.split()
                if len(cols) < 4:
                    logger.warning("Turbine file line %d: expected >= 4 columns, got %d — skipping", i + 2, len(cols))
                    continue

                try:
                    name = cols[0]
                    utm_x = float(cols[1])
                    utm_y = float(cols[2])
                    impact = float(cols[3])
                except ValueError as e:
                    logger.warning("Turbine file line %d: invalid numeric value (%s) — skipping", i + 2, e)
                    continue

                # Convert UTM to grid coordinates
                grid_x = (utm_x - utm_origin_x) / cell_size
                grid_y = (utm_y - utm_origin_y) / cell_size

                try:
                    raw_start = int(cols[4]) if len(cols) > 4 else 0
                    raw_end = int(cols[5]) if len(cols) > 5 else 2147483647
                except ValueError as e:
                    logger.warning("Turbine file line %d: invalid tick value (%s) — using defaults", i + 2, e)
                    raw_start = 0
                    raw_end = 2147483647

                start_tick = raw_start * day_to_tick
                end_tick = raw_end * day_to_tick

                turbine = cls(
                    id=i,
                    x=grid_x,
                    y=grid_y,
                    heading=0.0,
                    name=name,
                    impact=impact,
                    start_tick=start_tick,
                    end_tick=end_tick
                )
                turbines.append(turbine)
                
        return turbines


class TurbineManager:
    """
    Manages all turbines in the simulation.
    
    Handles dynamic creation/removal based on timing and
    calculates aggregate deterrence effects.
    """
    
    def __init__(self, turbines: Optional[List[Turbine]] = None):
        self.turbines: List[Turbine] = turbines or []
        self.phase: str = TurbinePhase.OFF
        
    def set_phase(self, phase: str) -> None:
        """Set the manager-level phase (used only for OFF to disable all turbines)."""
        self.phase = phase
            
    def update(self, current_tick: int) -> None:
        """Update all turbines for the current tick."""
        for turbine in self.turbines:
            turbine.update_phase(current_tick)
            
    def get_active_turbines(self) -> List[Turbine]:
        """Get list of currently active turbines."""
        return [t for t in self.turbines if t._is_active]
        
    def calculate_aggregate_deterrence(
        self,
        porpoise_x: float,
        porpoise_y: float,
        params: SimulationParameters,
        cell_size: float = 400.0
    ) -> Tuple[float, float, float]:
        """
        Calculate aggregate deterrence from all turbines.
        
        Args:
            porpoise_x, porpoise_y: Porpoise position
            params: Simulation parameters
            cell_size: Cell size in meters
            
        Returns:
            (max_strength, total_dx, total_dy)
        """
        if self.phase == TurbinePhase.OFF:
            return (0.0, 0.0, 0.0)
            
        max_strength = 0.0
        total_dx = 0.0
        total_dy = 0.0
        
        for turbine in self.get_active_turbines():
            should_deter, _, _, strength = turbine.should_deter(
                porpoise_x, porpoise_y, params, cell_size
            )
            
            if should_deter and strength > 0:
                dx, dy = turbine.get_deterrence_vector(
                    porpoise_x, porpoise_y,
                    strength, params.deter_coeff
                )
                
                if strength > max_strength:
                    max_strength = strength
                    
                total_dx += dx
                total_dy += dy
                
        return (max_strength, total_dx, total_dy)

    def calculate_aggregate_deterrence_vectorized(
        self,
        porpoise_x: np.ndarray,
        porpoise_y: np.ndarray,
        params: SimulationParameters,
        cell_size: float = 400.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate aggregate deterrence vector from all turbines for a population.
        
        Args:
            porpoise_x, porpoise_y: Arrays of Porpoise positions
            params: Simulation parameters
            cell_size: Cell size in meters
            
        Returns:
            (total_dx, total_dy) as numpy arrays
        """
        if self.phase == TurbinePhase.OFF:
            zeros = np.zeros_like(porpoise_x)
            return (zeros, zeros)
            
        total_dx = np.zeros_like(porpoise_x)
        total_dy = np.zeros_like(porpoise_y)
        
        active_turbines = self.get_active_turbines()
        if not active_turbines:
            return (total_dx, total_dy)
            
        # Constants
        max_dist_m = params.deter_max_distance * 1000.0
        threshold = params.deter_threshold
        beta = params.beta_hat
        alpha = params.alpha_hat
        deter_coeff = params.deter_coeff
        
        for turbine in active_turbines:
            # 1. Calculate distances (Vectorized)
            # using broadcast or direct subtraction since x/y are 1D arrays of size N
            # turbine.x is scalar
            dx_m = (porpoise_x - turbine.x) * cell_size
            dy_m = (porpoise_y - turbine.y) * cell_size
            dist_sq = dx_m**2 + dy_m**2
            dist_m = np.sqrt(dist_sq)
            
            # Avoid log(0)
            np.maximum(dist_m, 1.0, out=dist_m)
            
            # 2. Check range mask
            in_range_mask = dist_m < max_dist_m
            if not np.any(in_range_mask):
                continue
                
            # 3. Calculate sound level for potential candidates
            # RL = SL - (beta*log10(dist) + alpha*dist)
            source_level = turbine.get_source_level()  # Phase-dependent SL
            
            # Compute transmission loss only for in-range
            # Make copies to work on
            d_masked = dist_m[in_range_mask]
            
            tl = beta * np.log10(d_masked) + alpha * d_masked
            rl = source_level - tl
            str_val = rl - threshold
            
            # 4. Filter positive strength
            # Create sub-mask relative to in_range_mask
            deter_mask_local = str_val > 0
            
            if not np.any(deter_mask_local):
                continue
                
            # DEPONS parity: turbine deterrence is deterministic — full strength once
            # RL > threshold (only ships draw a Bernoulli reaction). JASMINE may opt into
            # logistic response-probability scaling via params.deter_probabilistic (default False).
            if params.deter_probabilistic:
                # Compute response probability for masked distances
                p = response_probability_from_rl(
                    rl, threshold, params.deter_response_slope
                )
                # p has same shape as d_masked
            else:
                p = None
            
            # 5. Calculate vectors (DEPONS logic)
            # vector X = ((porpX - turbX) / dist) * strength * coeff
            # We need to map back to original indices
            
            # Indices where both conditions met
            # We can use boolean indexing on the original arrays
            full_mask = np.zeros_like(in_range_mask)
            full_mask[in_range_mask] = deter_mask_local
            
            # Strength for full mask
            s_final = np.zeros_like(dist_m)
            s_final[in_range_mask] = str_val
            
            # Apply probabilistic scaling where appropriate
            if p is not None:
                # p corresponds to all in-range distances (d_masked), so we need to mask it
                p_full = np.zeros_like(dist_m)
                p_full[in_range_mask] = p
                s_final = s_final * p_full
            
            s = s_final[full_mask]
            # DEPONS 3.2 (Porpoise.java:1290-1292): raw GRID displacement (cell
            # units), NOT metres. dx_m/dy_m are metres (needed above for TL/range),
            # so divide by cell_size to recover grid displacement — matching the
            # scalar calculate_deterrence_vector path (grid units, no *cell_size).
            grid_dx = dx_m[full_mask] / cell_size
            grid_dy = dy_m[full_mask] / cell_size
            vec_x = grid_dx * s * deter_coeff
            vec_y = grid_dy * s * deter_coeff
            
            total_dx[full_mask] += vec_x
            total_dy[full_mask] += vec_y
            
        return (total_dx, total_dy)

    def ambient_received_level_at_positions(
        self,
        porpoise_x: np.ndarray,
        porpoise_y: np.ndarray,
        params: SimulationParameters,
        cell_size: float = 400.0
    ) -> np.ndarray:
        """
        Compute ambient received level (dB) at porpoise positions from all active turbines.
        Returns an array of RL in dB (same length as porpoise_x). If no sources, returns very low values (-999).
        """
        n = len(porpoise_x)
        if self.phase == TurbinePhase.OFF:
            return np.full(n, -999.0, dtype=np.float32)
        active_turbines = self.get_active_turbines()
        if not active_turbines:
            return np.full(n, -999.0, dtype=np.float32)

        # Accumulate linear power (10^(RL/10)) per porpoise
        lin_power = np.zeros(n, dtype=np.float64)
        max_dist_m = params.deter_max_distance * 1000.0
        for turbine in active_turbines:
            dx_m = (porpoise_x - turbine.x) * cell_size
            dy_m = (porpoise_y - turbine.y) * cell_size
            dist_m = np.sqrt(dx_m**2 + dy_m**2)
            # Avoid zero
            dist_m = np.maximum(dist_m, 1.0)
            mask = dist_m < max_dist_m
            if not np.any(mask):
                continue
            rl_all = np.full(n, -999.0, dtype=np.float32)
            dmask = dist_m[mask]
            tl = params.beta_hat * np.log10(dmask) + params.alpha_hat * dmask
            rl_mask = turbine.get_source_level() - tl
            rl_all[mask] = rl_mask
            # Convert dB to linear (power) and accumulate
            lin_power[mask] += 10.0 ** (rl_mask / 10.0)

        # Convert linear power back to dB, handle zeros
        rl_combined = np.zeros(n, dtype=np.float32)
        nonzero = lin_power > 0
        rl_combined[~nonzero] = -999.0
        rl_combined[nonzero] = 10.0 * np.log10(lin_power[nonzero])
        return rl_combined

    def load_from_file(
        self,
        filepath: str,
        utm_origin_x: float = 0.0,
        utm_origin_y: float = 0.0,
        cell_size: float = 400.0
    ) -> None:
        """Load turbines from a file."""
        self.turbines = Turbine.load_from_file(
            filepath, utm_origin_x, utm_origin_y, cell_size
        )
        
    @property
    def count(self) -> int:
        """Number of turbines."""
        return len(self.turbines)
        
    @property
    def active_count(self) -> int:
        """Number of active turbines."""
        return len(self.get_active_turbines())
