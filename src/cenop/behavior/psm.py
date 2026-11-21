"""
Persistent Spatial Memory (PSM) implementation.

Porpoises maintain a memory of areas where they have found food,
used for dispersal behavior targeting.
Translates from: PersistentSpatialMemory.java (185 lines)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


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
        preferred_distance: float | None = None,
        mem_cell_size: int = MEM_CELL_SIZE,
        rng: np.random.Generator | None = None,
        pref_dist_mean: float = 350.0,
        pref_dist_sd: float = 100.0,
    ):
        """
        Initialize PSM.

        Args:
            world_width: World width in grid cells
            world_height: World height in grid cells
            preferred_distance: Preferred dispersal distance in km
            mem_cell_size: Size of memory cells in grid units
            rng: NumPy random Generator for reproducibility
            pref_dist_mean: Mean of the preferred-distance distribution (km)
            pref_dist_sd: Std dev of the preferred-distance distribution (km)
        """
        self.world_width = world_width
        self.world_height = world_height
        self.mem_cell_size = mem_cell_size
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pref_dist_mean = pref_dist_mean
        self.pref_dist_sd = pref_dist_sd

        # Calculate memory grid dimensions
        self.cells_per_row = world_width // mem_cell_size
        self.cells_per_col = world_height // mem_cell_size

        # Memory data stored as dict for efficiency (only visited cells)
        self._mem_cells: dict[int, MemCellData] = {}

        # Preferred dispersal distance (generated from distribution)
        if preferred_distance is None:
            self.preferred_distance = self.generate_preferred_distance(
                mean=self.pref_dist_mean, sd=self.pref_dist_sd, rng=self.rng
            )
        else:
            self.preferred_distance = preferred_distance

    @staticmethod
    def generate_preferred_distance(
        mean: float = 350.0,
        sd: float = 100.0,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Generate preferred dispersal distance from normal distribution.

        Returns:
            Preferred distance in km (minimum 1.0 km)
        """
        _rng = rng if rng is not None else np.random.default_rng()
        distance = _rng.normal(mean, sd)
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

    def _cell_number_to_position(self, cell_number: int) -> tuple[float, float]:
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

    def get_cell_data(self, x: float, y: float) -> MemCellData | None:
        """Get memory data for position."""
        cell_num = self._position_to_cell_number(x, y)
        return self._mem_cells.get(cell_num)

    def get_best_remembered_cell(self) -> tuple[float, float, float] | None:
        """
        Get the cell with highest energy expectation.

        Returns:
            (x, y, energy_expectation) or None if no memory
        """
        if not self._mem_cells:
            return None

        best_cell = max(self._mem_cells.items(), key=lambda item: item[1].energy_expectation)

        x, y = self._cell_number_to_position(best_cell[0])
        return (x, y, best_cell[1].energy_expectation)

    def get_target_cell_for_dispersal(
        self,
        current_x: float,
        current_y: float,
        tolerance: float = 5.0,  # km
        cell_size: float = 400.0,  # meters
        q1: float = 0.0,  # PSM-Type3 distance-cost coefficient
    ) -> tuple[float, float, float] | None:
        """
        Find a target cell for dispersal at approximately preferred distance.

        Searches for the best cell (highest fitness) that is approximately
        at the preferred dispersal distance.

        When q1 > 0 (PSM-Type3), applies distance-cost:
            cost = energy * (1 - exp(-distance * q1))
            fitness = energy - cost = energy * exp(-distance * q1)
        This penalizes far-away cells proportionally to their distance.
        (Translates DispersalPSMType3.findMostAttractiveMemCell)

        Args:
            current_x, current_y: Current position
            tolerance: Tolerance band around preferred distance (km)
            cell_size: Grid cell size in meters
            q1: Distance-cost coefficient (0 = no cost, >0 = penalize distance)

        Returns:
            (x, y, distance_km) or None if no suitable cell found
        """
        if not self._mem_cells:
            return None

        preferred_dist_cells = self.preferred_distance * 1000 / cell_size  # km to cells
        tolerance_cells = tolerance * 1000 / cell_size

        min_dist = preferred_dist_cells - tolerance_cells
        max_dist = preferred_dist_cells + tolerance_cells

        # Extract all cell data into arrays at once
        n = len(self._mem_cells)
        cell_nums = np.fromiter(self._mem_cells.keys(), dtype=np.int32, count=n)
        energies = np.array(
            [cd.energy_expectation for cd in self._mem_cells.values()],
            dtype=np.float64,
        )

        # Vectorized cell_number_to_position
        mem_x = cell_nums % self.cells_per_row
        mem_y = cell_nums // self.cells_per_row
        xs = (mem_x + 0.5) * self.mem_cell_size
        ys = (mem_y + 0.5) * self.mem_cell_size

        # Vectorized distance computation
        dx = xs - current_x
        dy = ys - current_y
        dists = np.sqrt(dx * dx + dy * dy)

        # Apply PSM-Type3 distance-cost if q1 > 0
        # Java: cost = energy * (1 - exp(-dist * q1)); fitness = energy - cost
        # Simplifies to: fitness = energy * exp(-dist * q1)
        if q1 > 0:
            fitness = energies * np.exp(-dists * q1)
        else:
            fitness = energies

        # Find candidates in target distance range
        in_range = (dists >= min_dist) & (dists <= max_dist)

        if np.any(in_range):
            # Best candidate in range by fitness
            range_fitness = fitness.copy()
            range_fitness[~in_range] = -np.inf
            best_idx = int(np.argmax(range_fitness))
        else:
            # No cells in target range - pick highest fitness overall
            best_idx = int(np.argmax(fitness))

        dist_km = float(dists[best_idx]) * cell_size / 1000
        return (float(xs[best_idx]), float(ys[best_idx]), dist_km)

    def get_random_target(
        self, current_x: float, current_y: float, cell_size: float = 400.0
    ) -> tuple[float, float, float]:
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
        heading = self.rng.uniform(0, 360)
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
            preferred_distance=self.generate_preferred_distance(
                mean=self.pref_dist_mean, sd=self.pref_dist_sd, rng=self.rng
            ),
            mem_cell_size=self.mem_cell_size,
            rng=self.rng,
            pref_dist_mean=self.pref_dist_mean,
            pref_dist_sd=self.pref_dist_sd,
        )

        # Copy memory cells
        for cell_num, cell_data in self._mem_cells.items():
            new_psm._mem_cells[cell_num] = MemCellData(
                ticks_spent=cell_data.ticks_spent, food_obtained=cell_data.food_obtained
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
