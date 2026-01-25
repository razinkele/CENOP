"""
Persistent Spatial Memory (PSM) implementation.

Porpoises maintain a memory of areas where they have found food,
used for dispersal behavior targeting.
Translates from: PersistentSpatialMemory.java (185 lines)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters


# Memory cell size (5 world grid units = 2 km for 400m cells)
MEM_CELL_SIZE = 5


@dataclass
class MemCellData:
    """
    Data stored for each memory cell.
    
    Tracks time spent and food obtained in each 2km x 2km area.
    """
    
    ticks_spent: int = 0
    food_obtained: float = 0.0
    
    def update(self, food_eaten: float) -> None:
        """Update memory cell with food eaten during this tick."""
        self.food_obtained += food_eaten
        self.ticks_spent += 1
        
    @property
    def energy_expectation(self) -> float:
        """Calculate expected energy per tick in this cell."""
        if self.ticks_spent == 0:
            return 0.0
        return self.food_obtained / self.ticks_spent


class PersistentSpatialMemory:
    """
    Persistent Spatial Memory for porpoise dispersal.
    
    The memory uses coarser-grained cells (5x5 grid cells = 2km)
    to track where porpoises have found food over their lifetime.
    
    This is inherited from mother to calf.
    
    Translates from: PersistentSpatialMemory.java
    """
    
    def __init__(
        self,
        world_width: int,
        world_height: int,
        preferred_distance: Optional[float] = None,
        mem_cell_size: int = MEM_CELL_SIZE
    ):
        """
        Initialize PSM.
        
        Args:
            world_width: World width in grid cells
            world_height: World height in grid cells
            preferred_distance: Preferred dispersal distance in km
            mem_cell_size: Size of memory cells in grid units
        """
        self.world_width = world_width
        self.world_height = world_height
        self.mem_cell_size = mem_cell_size
        
        # Calculate memory grid dimensions
        self.cells_per_row = world_width // mem_cell_size
        self.cells_per_col = world_height // mem_cell_size
        
        # Memory data stored as dict for efficiency (only visited cells)
        self._mem_cells: Dict[int, MemCellData] = {}
        
        # Preferred dispersal distance (generated from distribution)
        if preferred_distance is None:
            self.preferred_distance = self.generate_preferred_distance()
        else:
            self.preferred_distance = preferred_distance
            
    @staticmethod
    def generate_preferred_distance(mean: float = 300.0, sd: float = 100.0) -> float:
        """
        Generate preferred dispersal distance from normal distribution.
        
        Returns:
            Preferred distance in km (minimum 1.0 km)
        """
        distance = np.random.normal(mean, sd)
        return max(1.0, distance)  # Minimum 1 km
        
    def _position_to_cell_number(self, x: float, y: float) -> int:
        """
        Convert grid position to memory cell number.
        
        Args:
            x, y: Position in grid cells
            
        Returns:
            Memory cell ID
        """
        mem_x = int(x) // self.mem_cell_size
        mem_y = int(y) // self.mem_cell_size
        
        # Clamp to valid range
        mem_x = max(0, min(mem_x, self.cells_per_row - 1))
        mem_y = max(0, min(mem_y, self.cells_per_col - 1))
        
        return mem_y * self.cells_per_row + mem_x
        
    def _cell_number_to_position(self, cell_number: int) -> Tuple[float, float]:
        """
        Convert memory cell number to grid position (center of cell).
        
        Args:
            cell_number: Memory cell ID
            
        Returns:
            (x, y) center position in grid cells
        """
        mem_x = cell_number % self.cells_per_row
        mem_y = cell_number // self.cells_per_row
        
        # Return center of memory cell
        x = (mem_x + 0.5) * self.mem_cell_size
        y = (mem_y + 0.5) * self.mem_cell_size
        
        return (x, y)
        
    def update(self, x: float, y: float, food_eaten: float) -> None:
        """
        Update memory for current position.
        
        Called every tick when porpoise eats food.
        
        Args:
            x, y: Current position
            food_eaten: Amount of food eaten this tick
        """
        cell_num = self._position_to_cell_number(x, y)
        
        if cell_num not in self._mem_cells:
            self._mem_cells[cell_num] = MemCellData()
            
        self._mem_cells[cell_num].update(food_eaten)
        
    def get_cell_data(self, x: float, y: float) -> Optional[MemCellData]:
        """Get memory data for position."""
        cell_num = self._position_to_cell_number(x, y)
        return self._mem_cells.get(cell_num)
        
    def get_best_remembered_cell(self) -> Optional[Tuple[float, float, float]]:
        """
        Get the cell with highest energy expectation.
        
        Returns:
            (x, y, energy_expectation) or None if no memory
        """
        if not self._mem_cells:
            return None
            
        best_cell = max(
            self._mem_cells.items(),
            key=lambda item: item[1].energy_expectation
        )
        
        x, y = self._cell_number_to_position(best_cell[0])
        return (x, y, best_cell[1].energy_expectation)
        
    def get_target_cell_for_dispersal(
        self,
        current_x: float,
        current_y: float,
        tolerance: float = 5.0,  # km
        cell_size: float = 400.0  # meters
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find a target cell for dispersal at approximately preferred distance.
        
        Searches for the best cell (highest energy expectation) that is
        approximately at the preferred dispersal distance.
        
        Args:
            current_x, current_y: Current position
            tolerance: Tolerance band around preferred distance (km)
            cell_size: Grid cell size in meters
            
        Returns:
            (x, y, distance_km) or None if no suitable cell found
        """
        if not self._mem_cells:
            return None
            
        preferred_dist_cells = self.preferred_distance * 1000 / cell_size  # km to cells
        tolerance_cells = tolerance * 1000 / cell_size
        
        min_dist = preferred_dist_cells - tolerance_cells
        max_dist = preferred_dist_cells + tolerance_cells
        
        candidates = []
        
        for cell_num, cell_data in self._mem_cells.items():
            x, y = self._cell_number_to_position(cell_num)
            
            # Calculate distance from current position
            dx = x - current_x
            dy = y - current_y
            dist = np.sqrt(dx**2 + dy**2)
            
            if min_dist <= dist <= max_dist:
                candidates.append((x, y, dist, cell_data.energy_expectation))
                
        if not candidates:
            # No cells in target range - find closest to preferred distance
            for cell_num, cell_data in self._mem_cells.items():
                x, y = self._cell_number_to_position(cell_num)
                dx = x - current_x
                dy = y - current_y
                dist = np.sqrt(dx**2 + dy**2)
                candidates.append((x, y, dist, cell_data.energy_expectation))
                
        if not candidates:
            return None
            
        # Sort by energy expectation (descending)
        candidates.sort(key=lambda c: c[3], reverse=True)
        
        # Return best candidate
        best = candidates[0]
        dist_km = best[2] * cell_size / 1000
        return (best[0], best[1], dist_km)
        
    def get_random_target(
        self,
        current_x: float,
        current_y: float,
        cell_size: float = 400.0
    ) -> Tuple[float, float, float]:
        """
        Generate a random target at approximately preferred distance.
        
        Used when no suitable remembered cell is found.
        
        Args:
            current_x, current_y: Current position
            cell_size: Grid cell size in meters
            
        Returns:
            (x, y, heading_to_target)
        """
        # Random direction
        heading = np.random.uniform(0, 360)
        heading_rad = np.radians(heading)
        
        # Distance in grid cells
        dist_cells = self.preferred_distance * 1000 / cell_size
        
        # Calculate target position
        target_x = current_x + dist_cells * np.sin(heading_rad)
        target_y = current_y + dist_cells * np.cos(heading_rad)
        
        return (target_x, target_y, heading)
        
    def copy_for_calf(self) -> PersistentSpatialMemory:
        """
        Create a copy of this PSM for a newborn calf.
        
        Calves inherit their mother's spatial memory.
        
        Returns:
            New PSM with copied data
        """
        new_psm = PersistentSpatialMemory(
            world_width=self.world_width,
            world_height=self.world_height,
            preferred_distance=self.generate_preferred_distance(),
            mem_cell_size=self.mem_cell_size
        )
        
        # Copy memory cells
        for cell_num, cell_data in self._mem_cells.items():
            new_psm._mem_cells[cell_num] = MemCellData(
                ticks_spent=cell_data.ticks_spent,
                food_obtained=cell_data.food_obtained
            )
            
        return new_psm
        
    @property
    def visited_cell_count(self) -> int:
        """Number of cells with memory data."""
        return len(self._mem_cells)
        
    @property
    def total_ticks(self) -> int:
        """Total ticks spent across all cells."""
        return sum(c.ticks_spent for c in self._mem_cells.values())
        
    @property
    def total_food(self) -> float:
        """Total food obtained across all cells."""
        return sum(c.food_obtained for c in self._mem_cells.values())


class PSMDispersalType2:
    """
    PSM-Type2 dispersal behavior.
    
    Porpoise disperses towards a target cell at approximately the
    preferred dispersal distance, with turning angle that decreases
    as the porpoise approaches the target.
    
    Translates from: DispersalPSMType2.java
    """
    
    def __init__(
        self,
        psm: PersistentSpatialMemory,
        random_angle: float = 20.0,  # Max random turning angle
        logistic_param: float = 0.6  # Logistic decrease steepness
    ):
        """
        Initialize PSM-Type2 dispersal.
        
        Args:
            psm: Persistent spatial memory
            random_angle: Maximum random turning angle during dispersal
            logistic_param: Steepness of logistic decrease function
        """
        self.psm = psm
        self.random_angle = random_angle
        self.logistic_param = logistic_param
        
        # Dispersal state
        self._is_dispersing = False
        self._target_x = 0.0
        self._target_y = 0.0
        self._target_distance = 0.0
        self._distance_traveled = 0.0
        self._start_x = 0.0
        self._start_y = 0.0
        self._previous_heading = 0.0
        
    @property
    def is_dispersing(self) -> bool:
        """Check if currently dispersing."""
        return self._is_dispersing
        
    def start_dispersal(
        self,
        current_x: float,
        current_y: float,
        current_heading: float,
        cell_size: float = 400.0
    ) -> float:
        """
        Start dispersal behavior.
        
        Args:
            current_x, current_y: Current position
            current_heading: Current heading
            cell_size: Grid cell size in meters
            
        Returns:
            Initial heading towards target
        """
        self._is_dispersing = True
        self._start_x = current_x
        self._start_y = current_y
        self._distance_traveled = 0.0
        self._previous_heading = current_heading
        
        # Find target from PSM
        target = self.psm.get_target_cell_for_dispersal(
            current_x, current_y, cell_size=cell_size
        )
        
        if target is not None:
            self._target_x, self._target_y, _ = target
        else:
            # Use random target
            self._target_x, self._target_y, _ = self.psm.get_random_target(
                current_x, current_y, cell_size
            )
            
        # Calculate target distance
        dx = self._target_x - current_x
        dy = self._target_y - current_y
        self._target_distance = np.sqrt(dx**2 + dy**2)
        
        # For PSM-Type2, stop when 95% of distance covered
        self._target_distance *= 0.95
        
        # Calculate heading to target
        heading = np.degrees(np.arctan2(dx, dy))
        self._previous_heading = heading
        
        return heading
        
    def calculate_new_heading(self, current_x: float, current_y: float) -> float:
        """
        Calculate new heading during dispersal.
        
        The random turning angle decreases as the porpoise approaches
        the target using a logistic function.
        
        Args:
            current_x, current_y: Current position
            
        Returns:
            New heading in degrees
        """
        if not self._is_dispersing:
            return self._previous_heading
            
        # Calculate distance traveled from start
        dx = current_x - self._start_x
        dy = current_y - self._start_y
        self._distance_traveled = np.sqrt(dx**2 + dy**2)
        
        # Random angle delta
        angle_delta = np.random.uniform(-self.random_angle, self.random_angle)
        
        # Apply logistic decrease based on distance progress
        if self._target_distance > 0:
            dist_perc = self._distance_traveled / self._target_distance
            dist_log_x = 3 * dist_perc - 1.5
            
            # Logistic decrease: as dist_perc -> 1, multiplier -> 0
            log_mult = 1.0 / (1.0 + np.exp(self.logistic_param * dist_log_x))
            angle_delta *= log_mult
            
        new_heading = self._previous_heading + angle_delta
        self._previous_heading = new_heading
        
        return new_heading
        
    def check_complete(self, current_x: float, current_y: float) -> bool:
        """
        Check if dispersal is complete.
        
        Args:
            current_x, current_y: Current position
            
        Returns:
            True if dispersal should end
        """
        if not self._is_dispersing:
            return True
            
        # Check if target distance reached
        dx = current_x - self._start_x
        dy = current_y - self._start_y
        distance = np.sqrt(dx**2 + dy**2)
        
        return distance >= self._target_distance
        
    def end_dispersal(self) -> None:
        """End dispersal behavior."""
        self._is_dispersing = False
        self._distance_traveled = 0.0
