"""
Simulation constants.

Fixed values that should not be changed during simulation.
Translates from: SimulationConstants.java
"""

from __future__ import annotations


class SimulationConstants:
    """
    Fixed simulation constants.
    
    Translates from: SimulationConstants.java
    """
    
    # Cell size in meters
    CELL_SIZE: int = 400
    
    # Time steps
    TICKS_PER_HOUR: int = 2
    TICKS_PER_DAY: int = 48
    DAYS_PER_MONTH: int = 30
    DAYS_PER_YEAR: int = 360
    TICKS_PER_YEAR: int = TICKS_PER_DAY * DAYS_PER_YEAR  # 17,280
    
    # Memory
    MEMORY_MAX: int = 120  # Max steps for food memory (2.5 days)
    
    # Food behavior
    ADD_ARTIFICIAL_FOOD: bool = True  # Invent food if porpoise eats all
    
    # Time adjustments
    SHIFT_QUARTER: bool = True  # NetLogo compatibility
    OFFSET_MONTH: bool = True   # NetLogo compatibility
    
    # Mortality
    MORTALITY_ENABLED: bool = True
    M_MORT_PROB_CONST: float = 1.0
    E_USE_PER_KM: float = 0.0
    
    # Emergency behavior when stuck and deterred
    IGNORE_DETER_MIN_IMPACT: float = 0.0
    IGNORE_DETER_MIN_DISTANCE: float = 0.0
    IGNORE_DETER_STUCK_TIME: int = 0
    IGNORE_DETER_NUMBER_OF_STEPS_IGNORE: int = 0
    
    # Coordinate conversion
    @staticmethod
    def utm_to_grid(utm_coord: float, origin: float) -> float:
        """Convert UTM coordinate to grid coordinate."""
        return (utm_coord - origin) / SimulationConstants.CELL_SIZE - 0.5
        
    @staticmethod
    def grid_to_utm(grid_coord: float, origin: float) -> float:
        """Convert grid coordinate to UTM coordinate."""
        return (grid_coord + 0.5) * SimulationConstants.CELL_SIZE + origin
        
    @staticmethod
    def utm_distance_to_grid(utm_distance: float) -> float:
        """Convert UTM distance to grid distance."""
        return utm_distance / SimulationConstants.CELL_SIZE
        
    @staticmethod
    def grid_distance_to_utm(grid_distance: float) -> float:
        """Convert grid distance to UTM distance."""
        return grid_distance * SimulationConstants.CELL_SIZE
