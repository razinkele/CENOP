"""
Base agent class for all simulation agents.

Provides common functionality for position, movement, and grid operations.
"""

from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from cenop.landscape.cell_data import CellData


@dataclass
class Agent:
    """
    Base class for all agents in the simulation.
    
    Provides position management and basic spatial operations.
    
    Translates from: Agent.java
    """
    
    id: int
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # degrees, 0 = North, clockwise
    
    def get_position(self) -> Tuple[float, float]:
        """Get current position as (x, y) tuple."""
        return (self.x, self.y)
        
    def set_position(self, x: float, y: float) -> None:
        """Set position to given coordinates."""
        self.x = x
        self.y = y
        
    def get_grid_position(self) -> Tuple[int, int]:
        """Get grid cell indices for current position."""
        return (int(self.x), int(self.y))
        
    def distance_to(self, other: Agent) -> float:
        """Calculate Euclidean distance to another agent."""
        return math.hypot(other.x - self.x, other.y - self.y)

    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate Euclidean distance to a point."""
        return math.hypot(x - self.x, y - self.y)

    def heading_to(self, other: Agent) -> float:
        """Calculate heading towards another agent in degrees."""
        return self.heading_to_point(other.x, other.y)

    def heading_to_point(self, x: float, y: float) -> float:
        """Calculate heading towards a point in degrees."""
        dx = x - self.x
        dy = y - self.y
        angle_rad = math.atan2(dx, dy)  # Note: atan2(dx, dy) for North=0
        return math.degrees(angle_rad)

    def get_dx(self) -> float:
        """Get x-component of unit vector in heading direction."""
        return math.sin(math.radians(self.heading))

    def get_dy(self) -> float:
        """Get y-component of unit vector in heading direction."""
        return math.cos(math.radians(self.heading))

    def forward(self, distance: float) -> None:
        """Move forward in current heading direction."""
        rad = math.radians(self.heading)
        self.x += distance * math.sin(rad)
        self.y += distance * math.cos(rad)
        
    def move(self, distance: float) -> None:
        """Move forward by given distance. Alias for forward()."""
        self.forward(distance)
        
    def turn_right(self, degrees: float) -> None:
        """Turn right (clockwise) by given degrees."""
        self.heading = (self.heading + degrees) % 360
        
    def turn_left(self, degrees: float) -> None:
        """Turn left (counter-clockwise) by given degrees."""
        self.heading = (self.heading - degrees) % 360
        
    def face_point(self, x: float, y: float) -> None:
        """Turn to face the given point."""
        self.heading = self.heading_to_point(x, y)
        
    def face_agent(self, other: Agent) -> None:
        """Turn to face another agent."""
        self.face_point(other.x, other.y)
        
    def get_point_ahead(self, distance: float, angle_offset: float = 0.0) -> Tuple[float, float]:
        """
        Get point at given distance ahead with optional angle offset.
        
        Args:
            distance: Distance ahead
            angle_offset: Offset from current heading in degrees (positive = right)
            
        Returns:
            (x, y) tuple of the point
        """
        rad = math.radians(self.heading + angle_offset)
        x = self.x + distance * math.sin(rad)
        y = self.y + distance * math.cos(rad)
        return (x, y)
        
    @staticmethod
    def normalize_heading(heading: float) -> float:
        """Normalize heading to [0, 360) range."""
        heading = heading % 360
        if heading < 0:
            heading += 360
        return heading
        
    @staticmethod
    def subtract_headings(h1: float, h2: float) -> float:
        """
        Calculate difference between two headings.
        
        Returns value in [-180, 180] range.
        Positive = clockwise from h2 to h1.
        """
        diff = h1 - h2
        if diff <= -180:
            diff += 360
        elif diff > 180:
            diff -= 360
        return diff
