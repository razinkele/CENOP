"""
Random number generation for the simulation.

Provides consistent random number generation that can be seeded
for reproducibility or replayed from recorded values.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class RandomSource(ABC):
    """Abstract base class for random number sources."""
    
    @abstractmethod
    def next_uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Get next uniform random number."""
        pass
        
    @abstractmethod
    def next_normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Get next normal random number."""
        pass
        
    @abstractmethod
    def next_int(self, low: int, high: int) -> int:
        """Get next random integer in [low, high)."""
        pass


class GeneratedRandomSource(RandomSource):
    """
    Random source using NumPy random generation.
    
    Translates from: GeneratedRandomSource.java
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize with optional seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self._rng = np.random.default_rng(seed)
        
    def next_uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Get next uniform random number."""
        return self._rng.uniform(low, high)
        
    def next_normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Get next normal random number."""
        return self._rng.normal(mean, std)
        
    def next_int(self, low: int, high: int) -> int:
        """Get next random integer in [low, high)."""
        return self._rng.integers(low, high)
        
    def next_crw_angle(self) -> float:
        """Get random angle for CRW movement. N(0, 38)"""
        return self.next_normal(0, 38)
        
    def next_crw_angle_with_m(self) -> float:
        """Get random angle adjustment. N(96, 28)"""
        return self.next_normal(96, 28)
        
    def next_step_length(self) -> float:
        """Get random step length. N(1.25, 0.15)"""
        return self.next_normal(1.25, 0.15)
        
    def next_energy_initial(self) -> float:
        """Get random initial energy. N(10.0, 1.0)"""
        return self.next_normal(10.0, 1.0)
        
    def next_mating_day(self) -> int:
        """Get random mating day. N(225, 20)"""
        return int(self.next_normal(225, 20))
        
    def next_dispersal_distance(self, mean: float, std: float) -> float:
        """Get random dispersal distance. N(mean, std)"""
        return self.next_normal(mean, std)


class ReplayedRandomSource(RandomSource):
    """
    Random source that replays recorded random values.
    
    Used for debugging and validation against Java reference.
    
    Translates from: ReplayedRandomSource.java
    """
    
    def __init__(self, values: list[float]):
        """
        Initialize with list of values to replay.
        
        Args:
            values: Pre-recorded random values
        """
        self._values = values
        self._index = 0
        
    def _next_value(self) -> float:
        """Get next value from replay list."""
        if self._index >= len(self._values):
            raise RuntimeError("Ran out of replayed random values")
        value = self._values[self._index]
        self._index += 1
        return value
        
    def next_uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Get next uniform random number (scaled from replay)."""
        value = self._next_value()
        return low + value * (high - low)
        
    def next_normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Get next normal random number (from replay)."""
        return self._next_value()
        
    def next_int(self, low: int, high: int) -> int:
        """Get next random integer."""
        return int(self._next_value())
        
    def reset(self) -> None:
        """Reset replay index to start."""
        self._index = 0
