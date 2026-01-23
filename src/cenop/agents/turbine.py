"""
Wind turbine agent implementation.

Turbines generate noise during construction and operation phases
that deters porpoises within a certain radius.
Translates from: Turbine.java (258 lines)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, List, Tuple
from pathlib import Path

from cenop.agents.base import Agent
from cenop.behavior.sound import (
    TurbineNoise,
    calculate_received_level,
    calculate_deterrence_vector
)

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters
    from cenop.core.simulation import SimulationState


class TurbinePhase:
    """Turbine operational phases."""
    OFF = "off"
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
        """Check if turbine is actively producing noise."""
        if tick is not None:
            return self.start_tick <= tick < self.end_tick
        return self._is_active
        
    def update_phase(self, current_tick: int) -> None:
        """Update turbine active status based on current tick."""
        self._is_active = self.start_tick <= current_tick < self.end_tick
        
    def get_source_level(self) -> float:
        """Get the current source level based on phase."""
        is_construction = (self.phase == TurbinePhase.CONSTRUCTION)
        return self.noise.get_source_level(is_construction)
        
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
            source_level = self.get_source_level()
            
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
            
        # DEPONS formula: RL = impact - (β*log10(dist) + α*dist)
        # Where 'impact' IS the source level (SL) in dB
        transmission_loss = (
            params.beta_hat * np.log10(distance_m) +
            params.alpha_hat * distance_m
        )
        received_level = self.impact - transmission_loss
        
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
            # Skip header
            next(f, None)
            
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                cols = line.split()
                if len(cols) < 4:
                    continue
                
                name = cols[0]
                utm_x = float(cols[1])
                utm_y = float(cols[2])
                impact = float(cols[3])
                
                # Convert UTM to grid coordinates
                grid_x = (utm_x - utm_origin_x) / cell_size
                grid_y = (utm_y - utm_origin_y) / cell_size
                
                start_tick = int(cols[4]) if len(cols) > 4 else 0
                end_tick = int(cols[5]) if len(cols) > 5 else 2147483647
                    
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
        """Set the operational phase for all turbines."""
        self.phase = phase
        for turbine in self.turbines:
            turbine.phase = phase
            
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
            source_level = turbine.impact # In DEPONS impact IS source level
            
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
            s = s_final[full_mask]
            d = dist_m[full_mask]
            
            # Normalized direction vector * strength * coeff
            # (dx_m / dist_m) is unit vector X component
            vec_x = (dx_m[full_mask] / d) * s * deter_coeff
            vec_y = (dy_m[full_mask] / d) * s * deter_coeff
            
            # Accumulate
            # Converts grid meters back to grid cells? 
            # calculate_deterrence_vector returns grid cell units usually.
            # Let's check calculate_deterrence_vector in sound.py
            # Original code:
            # return calculate_deterrence_vector(
            #    porpoise_x, porpoise_y,
            #    self.x, self.y,
            #    strength, deter_coeff
            # )
            # 
            # In sound.py (implied):
            # dx = (target_x - source_x)
            # dy = (target_y - source_y)
            # dist = sqrt(dx*dx + dy*dy)
            # unit_x = dx/dist
            # unit_y = dy/dist
            # return unit_x * strength * deter_coeff, unit_y * strength * deter_coeff
            #
            # My calc vec_x above uses dx_m (meters). 
            # dx_m = (px - tx) * cell_size.
            # dist_m = dist * cell_size.
            # dx_m / dist_m = dx / dist. Unit vector is identical.
            # So vec_x is in "strength units".
            # The movement logic adds this to grid coordinates.
            # So strength * coeff must result in grid cells.
            
            total_dx[full_mask] += vec_x
            total_dy[full_mask] += vec_y
            
        return (total_dx, total_dy)
        
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
