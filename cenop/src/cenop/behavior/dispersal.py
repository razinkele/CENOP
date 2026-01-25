"""
Dispersal behavior implementations.

Different dispersal strategies when porpoises are stressed.
Translates from: Dispersal.java, AbstractPSMDispersal.java, 
                 DispersalPSMType2.java, DispersalPSMType3.java

CRITICAL FORMULAS (from DEPONS Java):
- SSLogis: phi1 / (1 + exp((phi2 - x) / phi3))
- PSM-Type2 input transform: distLogX = (3 * distPerc) - 1.5
- PSM-Type2: angleDelta = random(-maxAngle, +maxAngle) * SSLogis(distLogX)
- PSM-Type3: angleDelta = maxAngle / (1 + exp(-psmLog * (dist - x0)))
           where x0 = targetDistance / 2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


class DispersalType(Enum):
    """Available dispersal behavior types."""
    OFF = "off"
    PSM_TYPE1 = "PSM-Type1"
    PSM_TYPE2 = "PSM-Type2"
    PSM_TYPE3 = "PSM-Type3"


def sslogis(x: float, phi1: float = 1.0, phi2: float = 0.0, phi3: float = 0.6) -> float:
    """
    Simple Logistic Model (SSLogis) function.
    
    Translates from: LogisticDecreaseSSLogis.java
    
    Formula: phi1 / (1 + exp((phi2 - x) / phi3))
    
    Args:
        x: Input value
        phi1: Asymptote (default 1.0)
        phi2: Inflection point (default 0.0)
        phi3: Scale parameter / psm_log (default 0.6)
        
    Returns:
        Logistic function output
    """
    return phi1 / (1.0 + np.exp((phi2 - x) / phi3))


@dataclass
class DispersalParams:
    """Parameters for dispersal behavior."""
    psm_log: float = 0.6             # PSM logistic parameter (phi3 in SSLogis)
    dist_mean: float = 300.0         # Mean dispersal distance (km)
    dist_sd: float = 100.0           # SD of dispersal distance
    tolerance: float = 5.0           # Tolerance band (km)
    psm_type2_random_angle: float = 20.0  # Max random angle for PSM-Type2/3
    t_disp: int = 3                  # Days before dispersal triggers
    q1: float = 0.001                # PSM-Type3 cost parameter
    min_memory_cells: int = 50       # Minimum cells in memory for PSM activation


class DispersalBehavior(ABC):
    """
    Abstract base class for dispersal behaviors.
    
    Translates from: Dispersal.java
    """
    
    def __init__(self, params: DispersalParams):
        self.params = params
        self._target_distance: Optional[float] = None
        self._dispersing: bool = False
        self._distance_traveled: float = 0.0
        
    @property
    def is_dispersing(self) -> bool:
        """Check if currently in dispersal mode."""
        return self._dispersing
        
    @abstractmethod
    def should_start_dispersal(
        self,
        days_declining_energy: int,
        current_energy: float,
        memory_cell_count: int = 0
    ) -> bool:
        """
        Check if dispersal should begin.
        
        Args:
            days_declining_energy: Days of declining energy
            current_energy: Current energy level
            memory_cell_count: Number of cells in persistent spatial memory
            
        Returns:
            True if dispersal should start
        """
        pass
        
    @abstractmethod
    def get_dispersal_move(
        self,
        x: float,
        y: float,
        heading: float,
        rng: np.random.Generator
    ) -> Tuple[float, float, float]:
        """
        Get dispersal movement.
        
        Returns:
            (new_heading, distance, turning_adjustment)
        """
        pass
        
    def start_dispersal(self, rng: np.random.Generator) -> None:
        """Start dispersal behavior."""
        self._dispersing = True
        self._distance_traveled = 0.0
        # Draw target distance from normal distribution
        self._target_distance = abs(rng.normal(
            self.params.dist_mean,
            self.params.dist_sd
        ))
        
    def update_dispersal(self, distance_moved: float) -> None:
        """Update dispersal progress."""
        if self._dispersing:
            self._distance_traveled += distance_moved
            
    def check_dispersal_complete(self) -> bool:
        """Check if dispersal target has been reached."""
        if not self._dispersing or self._target_distance is None:
            return False
            
        lower = self._target_distance - self.params.tolerance
        upper = self._target_distance + self.params.tolerance
        
        if lower <= self._distance_traveled <= upper:
            return True
        if self._distance_traveled > upper:
            return True
            
        return False
        
    def end_dispersal(self) -> None:
        """End dispersal behavior."""
        self._dispersing = False
        self._target_distance = None
        self._distance_traveled = 0.0


class NoDispersal(DispersalBehavior):
    """No dispersal behavior (disabled)."""
    
    def should_start_dispersal(
        self,
        days_declining_energy: int,
        current_energy: float,
        memory_cell_count: int = 0
    ) -> bool:
        return False
        
    def get_dispersal_move(
        self,
        x: float,
        y: float,
        heading: float,
        rng: np.random.Generator
    ) -> Tuple[float, float, float]:
        return (heading, 0.0, 0.0)


class PSMType1Dispersal(DispersalBehavior):
    """
    PSM Type 1 dispersal.
    
    Straight-line dispersal with no turning until target reached.
    Requires at least 50 cells in memory to activate.
    """
    
    def should_start_dispersal(
        self,
        days_declining_energy: int,
        current_energy: float,
        memory_cell_count: int = 0
    ) -> bool:
        # DEPONS requires at least 50 cells in memory for PSM activation
        if memory_cell_count < self.params.min_memory_cells:
            return False
        return days_declining_energy >= self.params.t_disp
        
    def get_dispersal_move(
        self,
        x: float,
        y: float,
        heading: float,
        rng: np.random.Generator
    ) -> Tuple[float, float, float]:
        # Keep same heading during dispersal
        return (heading, 1.0, 0.0)


class PSMType2Dispersal(DispersalBehavior):
    """
    PSM Type 2 dispersal.
    
    Logistic DECREASE in random turning as dispersal progresses.
    
    Translates from: DispersalPSMType2.java
    
    Key formulas:
    - distLogX = (3 * distPerc) - 1.5  (input transformation)
    - logDistPerc = SSLogis(distLogX)  (logistic decrease)
    - angleDelta = random(-maxAngle, +maxAngle) * logDistPerc
    - newHeading = previousStepHeading + angleDelta
    - Target = 95% of target distance (PSM-Type2 specific)
    """
    
    def __init__(self, params: DispersalParams):
        super().__init__(params)
        self._previous_step_heading: Optional[float] = None
        self._start_position: Optional[Tuple[float, float]] = None
    
    def should_start_dispersal(
        self,
        days_declining_energy: int,
        current_energy: float,
        memory_cell_count: int = 0
    ) -> bool:
        # DEPONS requires at least 50 cells in memory for PSM activation
        if memory_cell_count < self.params.min_memory_cells:
            return False
        return days_declining_energy >= self.params.t_disp
    
    def start_dispersal(
        self,
        rng: np.random.Generator,
        target_heading: Optional[float] = None,
        start_position: Optional[Tuple[float, float]] = None
    ) -> None:
        """Start dispersal - track previous heading for PSM-Type2."""
        super().start_dispersal(rng)
        # PSM-Type2 uses 95% of target distance
        if self._target_distance is not None:
            self._target_distance = self._target_distance * 0.95
        # Initialize previous heading to target heading
        self._previous_step_heading = target_heading
        self._start_position = start_position
        
    def get_dispersal_move(
        self,
        x: float,
        y: float,
        heading: float,
        rng: np.random.Generator
    ) -> Tuple[float, float, float]:
        """
        Calculate PSM-Type2 dispersal move.
        
        DEPONS formula:
        - angleDelta = (2 * maxAngle * random) - maxAngle  # Uniform(-maxAngle, +maxAngle)
        - distPerc = distanceTravelled / targetDistance
        - distLogX = (3 * distPerc) - 1.5
        - logDistPerc = SSLogis(distLogX, phi1=1.0, phi2=0.0, phi3=psm_log)
        - angleDelta = angleDelta * logDistPerc
        - newHeading = previousStepHeading + angleDelta
        """
        if self._target_distance is None or self._target_distance == 0:
            return (heading, 1.0, 0.0)
        
        # Use previous step heading (or current if not set)
        prev_heading = self._previous_step_heading if self._previous_step_heading is not None else heading
            
        # Random angle in range [-maxAngle, +maxAngle]
        max_angle = self.params.psm_type2_random_angle
        angle_delta = (2 * max_angle * rng.random()) - max_angle
        
        # Calculate logistic factor using SSLogis
        dist_perc = self._distance_traveled / self._target_distance
        dist_log_x = (3 * dist_perc) - 1.5
        log_dist_perc = sslogis(dist_log_x, phi1=1.0, phi2=0.0, phi3=self.params.psm_log)
        
        # Scale angle by logistic output (decreasing as we travel)
        angle_delta = angle_delta * log_dist_perc
        
        # New heading based on PREVIOUS step heading
        new_heading = (prev_heading + angle_delta) % 360
        
        # Update previous step heading for next iteration
        self._previous_step_heading = new_heading
        
        return (new_heading, 1.0, angle_delta)


class PSMType3Dispersal(DispersalBehavior):
    """
    PSM Type 3 dispersal.
    
    Logistic INCREASE in random turning as dispersal progresses.
    Uses distance from start (not cumulative travel) for stop condition.
    
    Translates from: DispersalPSMType3.java
    
    Key formulas:
    - x0 = targetDistance / 2  (inflection point at halfway)
    - z = -psm_log * (distanceTravelled - x0)
    - angleDelta = maxAngle / (1 + exp(z))
    - angleDelta = randomSign * angleDelta  (random +1 or -1)
    - newHeading = currentHeading + angleDelta
    - Stop condition: distance from start >= target (not cumulative travel)
    """
    
    def __init__(self, params: DispersalParams):
        super().__init__(params)
        self._start_position: Optional[Tuple[float, float]] = None
    
    def should_start_dispersal(
        self,
        days_declining_energy: int,
        current_energy: float,
        memory_cell_count: int = 0
    ) -> bool:
        # DEPONS requires at least 50 cells in memory for PSM activation
        if memory_cell_count < self.params.min_memory_cells:
            return False
        return days_declining_energy >= self.params.t_disp
    
    def start_dispersal(
        self,
        rng: np.random.Generator,
        target_heading: Optional[float] = None,
        start_position: Optional[Tuple[float, float]] = None
    ) -> None:
        """Start dispersal - track start position for PSM-Type3."""
        super().start_dispersal(rng)
        self._start_position = start_position
        
    def get_dispersal_move(
        self,
        x: float,
        y: float,
        heading: float,
        rng: np.random.Generator
    ) -> Tuple[float, float, float]:
        """
        Calculate PSM-Type3 dispersal move.
        
        DEPONS formula:
        - x0 = targetDistance / 2
        - z = -psm_log * (distanceTravelled - x0)
        - angleDelta = maxAngle / (1 + exp(z))
        - angleDelta = randomPlusMinusOne * angleDelta
        - newHeading = currentHeading + angleDelta
        """
        if self._target_distance is None or self._target_distance == 0:
            return (heading, 1.0, 0.0)
            
        max_angle = self.params.psm_type2_random_angle
        
        # Calculate logistic increase formula (more turning as we travel)
        x0 = self._target_distance / 2  # Inflection point at halfway
        z = -self.params.psm_log * (self._distance_traveled - x0)
        angle_delta = max_angle / (1 + np.exp(z))
        
        # Random sign (+1 or -1)
        random_sign = 1 if rng.random() < 0.5 else -1
        angle_delta = random_sign * angle_delta
        
        # New heading based on current heading (not previous)
        new_heading = (heading + angle_delta) % 360
        
        return (new_heading, 1.0, angle_delta)
    
    def should_stop_dispersing(
        self,
        current_x: float,
        current_y: float
    ) -> bool:
        """
        PSM-Type3 specific stop condition.
        
        Stop when distance from start position >= target distance
        (not when cumulative travel distance >= target).
        """
        if not self._dispersing or self._target_distance is None:
            return False
            
        if self._start_position is None:
            # Fall back to default behavior
            return self.check_dispersal_complete()
            
        # Distance from start position
        dx = current_x - self._start_position[0]
        dy = current_y - self._start_position[1]
        distance_from_start = np.sqrt(dx * dx + dy * dy)
        
        return distance_from_start >= self._target_distance


def create_dispersal_behavior(
    dispersal_type: DispersalType | str,
    params: Optional[DispersalParams] = None
) -> DispersalBehavior:
    """
    Create a dispersal behavior instance.
    
    Args:
        dispersal_type: Type of dispersal behavior
        params: Optional parameters
        
    Returns:
        DispersalBehavior instance
    """
    if params is None:
        params = DispersalParams()
        
    if isinstance(dispersal_type, str):
        dispersal_type = DispersalType(dispersal_type)
        
    if dispersal_type == DispersalType.OFF:
        return NoDispersal(params)
    elif dispersal_type == DispersalType.PSM_TYPE1:
        return PSMType1Dispersal(params)
    elif dispersal_type == DispersalType.PSM_TYPE2:
        return PSMType2Dispersal(params)
    elif dispersal_type == DispersalType.PSM_TYPE3:
        return PSMType3Dispersal(params)
    else:
        return NoDispersal(params)
