"""
Reference memory implementation.

Manages porpoise memory of food patches.
Translates from: RefMem.java
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class MemoryEntry:
    """A single memory entry of a food patch location."""
    x: float
    y: float
    strength: float  # How well the patch is remembered
    steps_ago: int = 0


class RefMem:
    """
    Reference memory for food patches.
    
    Translates from: RefMem.java
    
    Porpoises remember locations where they found food.
    Memory strength decays over time.
    """
    
    def __init__(
        self,
        max_entries: int = 120,
        decay_satiation: float = 0.04,
        decay_reference: float = 0.04
    ):
        """
        Initialize reference memory.
        
        Args:
            max_entries: Maximum number of memory entries
            decay_satiation: Satiation memory decay rate (r_s)
            decay_reference: Reference memory decay rate (r_r)
        """
        self.max_entries = max_entries
        self.decay_satiation = decay_satiation
        self.decay_reference = decay_reference
        
        self._entries: List[MemoryEntry] = []
        self._satiation: float = 0.0  # Current satiation level
        
    @property
    def satiation(self) -> float:
        """Current satiation level (0-1)."""
        return self._satiation
        
    def add(self, x: float, y: float, food_amount: float) -> None:
        """
        Add or update a memory entry.
        
        Args:
            x: X position of food
            y: Y position of food
            food_amount: Amount of food found
        """
        # Increase satiation
        self._satiation = min(1.0, self._satiation + food_amount)
        
        # Check if we already have memory of this location
        for entry in self._entries:
            if abs(entry.x - x) < 0.5 and abs(entry.y - y) < 0.5:
                entry.strength = min(1.0, entry.strength + food_amount)
                entry.steps_ago = 0
                return
                
        # Add new entry
        entry = MemoryEntry(x=x, y=y, strength=food_amount, steps_ago=0)
        self._entries.append(entry)
        
        # Remove oldest if over limit
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
            
    def update(self) -> None:
        """Update memory (decay strengths and age entries)."""
        # Decay satiation
        self._satiation *= (1 - self.decay_satiation)
        
        # Decay and age entries
        for entry in self._entries:
            entry.strength *= (1 - self.decay_reference)
            entry.steps_ago += 1
            
        # Remove weak entries
        self._entries = [
            e for e in self._entries 
            if e.strength > 0.001
        ]
        
    def get_best_memory(self) -> Optional[Tuple[float, float]]:
        """
        Get the strongest memory location.
        
        Returns:
            (x, y) of best remembered location, or None
        """
        if not self._entries:
            return None
            
        best = max(self._entries, key=lambda e: e.strength)
        return (best.x, best.y)
        
    def get_nearest_memory(
        self,
        x: float,
        y: float,
        min_strength: float = 0.1
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get nearest remembered location.
        
        Args:
            x: Current x position
            y: Current y position
            min_strength: Minimum strength to consider
            
        Returns:
            (x, y, strength) of nearest location, or None
        """
        nearest = None
        min_dist_sq = float('inf')
        
        for entry in self._entries:
            if entry.strength < min_strength:
                continue
            dist_sq = (entry.x - x)**2 + (entry.y - y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest = (entry.x, entry.y, entry.strength)
                
        return nearest
        
    def get_memories_in_direction(
        self,
        x: float,
        y: float,
        heading: float,
        cone_angle: float = 90.0
    ) -> List[Tuple[float, float, float]]:
        """
        Get memories within a directional cone.
        
        Args:
            x: Current x position
            y: Current y position
            heading: Current heading in degrees
            cone_angle: Half-angle of cone in degrees
            
        Returns:
            List of (x, y, strength) tuples
        """
        result = []
        heading_rad = np.radians(heading)
        
        for entry in self._entries:
            dx = entry.x - x
            dy = entry.y - y
            
            if abs(dx) < 0.001 and abs(dy) < 0.001:
                continue
                
            angle_to = np.degrees(np.arctan2(dy, dx))
            angle_diff = abs((angle_to - heading + 180) % 360 - 180)
            
            if angle_diff <= cone_angle:
                result.append((entry.x, entry.y, entry.strength))
                
        return result
        
    def clear(self) -> None:
        """Clear all memories."""
        self._entries.clear()
        self._satiation = 0.0
        
    def __len__(self) -> int:
        return len(self._entries)
