"""
Landscape cell data management.

Manages all spatial data layers for the simulation environment.
Translates from: CellData.java
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List


@dataclass
class LandscapeMetadata:
    """Metadata from ASC file headers."""
    
    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float = 400.0
    nodata_value: float = -9999.0
    
    @property
    def width(self) -> int:
        return self.ncols
        
    @property
    def height(self) -> int:
        return self.nrows


class CellData:
    """
    Manages all spatial data layers for the simulation.
    
    Translates from: CellData.java
    
    Data layers:
    - depth: Water depth (bathymetry)
    - dist_to_coast: Distance to coastline
    - sediment: Sediment type
    - food_prob: Probability of food (patches)
    - food_value: Current food level
    - blocks: Block identifiers
    - entropy: Monthly MaxEnt values
    - salinity: Monthly salinity values
    """
    
    def __init__(self, landscape_name: str, data_dir: str = None):
        """
        Initialize cell data for a landscape.
        
        Args:
            landscape_name: Name of landscape (e.g., 'Lithuania')
            data_dir: Base data directory. If None, uses cenop/data.
        """
        self.landscape_name = landscape_name
        
        # Resolve data directory - if not specified, use cenop/data relative to this package
        if data_dir is None:
            # Get the cenop package root (3 levels up from this file)
            package_root = Path(__file__).parent.parent.parent.parent  # src/cenop/landscape -> cenop
            self.data_dir = package_root / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.metadata: Optional[LandscapeMetadata] = None
        
        # Data arrays
        self._depth: Optional[np.ndarray] = None
        self._dist_to_coast: Optional[np.ndarray] = None
        self._sediment: Optional[np.ndarray] = None
        self._food_prob: Optional[np.ndarray] = None
        self._food_value: Optional[np.ndarray] = None
        self._blocks: Optional[np.ndarray] = None
        self._entropy: Optional[np.ndarray] = None  # Shape: (12, height, width)
        self._salinity: Optional[np.ndarray] = None  # Shape: (12, height, width)
        
        self._demand_grid: Optional[np.ndarray] = None
        self._current_month: int = 1
        self._loaded: bool = False
        
    def load(self) -> None:
        """Load all data layers from files."""
        from cenop.landscape.loader import LandscapeLoader
        
        loader = LandscapeLoader(self.landscape_name, self.data_dir)
        data = loader.load_all()
        
        self.metadata = data['metadata']
        self._depth = data['depth']
        self._dist_to_coast = data['dist_to_coast']
        self._sediment = data['sediment']
        self._food_prob = data['food_prob']
        self._blocks = data['blocks']
        self._entropy = data['entropy']
        self._salinity = data['salinity']
        
        # Initialize food from maxEnt (Java CellData.java:256-268)
        if self._entropy is not None:
            month_idx = 0  # Start at January
            max_ent = self._entropy[month_idx]
            max_u = 1.0
            mean_max_ent = 1.0
            self._food_value = np.where(
                (self._food_prob > 0) & (max_ent > 0),
                max_u * max_ent / mean_max_ent,
                0.0
            ).astype(np.float32)
        else:
            self._food_value = self._food_prob.copy()
        
        self._loaded = True
        
    def _ensure_loaded(self) -> None:
        """Ensure data is loaded."""
        if not self._loaded:
            self.load()
            
    @property
    def width(self) -> int:
        """Grid width in cells."""
        self._ensure_loaded()
        return self.metadata.ncols if self.metadata else 0
        
    @property
    def height(self) -> int:
        """Grid height in cells."""
        self._ensure_loaded()
        return self.metadata.nrows if self.metadata else 0
        
    def is_valid_position(self, x: float, y: float) -> bool:
        """Check if position is within grid bounds."""
        self._ensure_loaded()
        return 0 <= x < self.width and 0 <= y < self.height
        
    def _get_indices(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert continuous position to grid indices.
        
        Note: The loader flips Y when loading ASC files (using np.flipud),
        so array[0, :] = SOUTH and array[height-1, :] = NORTH, matching DEPONS.
        Grid coordinates: y=0 is SOUTH, y=height-1 is NORTH.
        Direct indexing works because the array is already flipped.
        """
        # Direct mapping since array is pre-flipped during loading
        i = int(np.clip(y, 0, self.height - 1))
        j = int(np.clip(x, 0, self.width - 1))
        return (i, j)
        
    def get_depth(self, x: float, y: float) -> float:
        """Get water depth at position."""
        self._ensure_loaded()
        if self._depth is None:
            return 20.0  # Default depth
        i, j = self._get_indices(x, y)
        return float(self._depth[i, j])
        
    def get_dist_to_coast(self, x: float, y: float) -> float:
        """Get distance to coast at position."""
        self._ensure_loaded()
        if self._dist_to_coast is None:
            return 10000.0  # Default distance
        i, j = self._get_indices(x, y)
        return float(self._dist_to_coast[i, j])
        
    def get_sediment(self, x: float, y: float) -> float:
        """Get sediment type at position."""
        self._ensure_loaded()
        if self._sediment is None:
            return 1.0
        i, j = self._get_indices(x, y)
        return float(self._sediment[i, j])
        
    def get_food_prob(self, x: float, y: float) -> float:
        """Get food probability at position."""
        self._ensure_loaded()
        if self._food_prob is None:
            return 0.5
        i, j = self._get_indices(x, y)
        return float(self._food_prob[i, j])
        
    def get_food_level(self, x: float, y: float) -> float:
        """Get current food level at position."""
        self._ensure_loaded()
        if self._food_value is None:
            return 0.5
        i, j = self._get_indices(x, y)
        return float(self._food_value[i, j])

    def get_food_levels_vectorized(self, positions=None, xi=None, yi=None):
        """Get food levels for multiple positions at once.

        Args:
            positions: (N, 2) array of (x, y) positions (used if xi/yi not provided)
            xi: Optional pre-computed int column indices
            yi: Optional pre-computed int row indices
        """
        self._ensure_loaded()
        if self._food_value is None:
            n = len(xi) if xi is not None else len(positions)
            return np.full(n, 0.5, dtype=np.float32)
        if xi is not None and yi is not None:
            xi = np.clip(xi, 0, self.width - 1)
            yi = np.clip(yi, 0, self.height - 1)
            return self._food_value[yi, xi].astype(np.float32)
        j = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
        i = np.clip(positions[:, 1].astype(int), 0, self.height - 1)
        return self._food_value[i, j].astype(np.float32)
        
    def remove_food(self, x: float, y: float, amount: float) -> None:
        """Remove food from a cell."""
        self._ensure_loaded()
        if self._food_value is None:
            return
        i, j = self._get_indices(x, y)
        self._food_value[i, j] = max(0.0, self._food_value[i, j] - amount)
        
    def eat_food(self, x: float, y: float, fraction: float) -> float:
        """
        Eat a fraction of the food in a cell.
        
        Translates from: CellData.eatFood() in DEPONS
        
        Args:
            x, y: Position
            fraction: Fraction of available food to eat (0-1)
            
        Returns:
            Amount of food eaten
        """
        self._ensure_loaded()
        if self._food_value is None or self._food_prob is None:
            return 0.0
            
        i, j = self._get_indices(x, y)
        
        # Get current food
        current_food = self._food_value[i, j]
        
        if current_food <= 0:
            return 0.0
            
        # Calculate food to eat
        food_eaten = current_food * fraction
        
        # Update food value
        self._food_value[i, j] = max(0.0, current_food - food_eaten)
        
        # ADD_ARTIFICIAL_FOOD: If food drops below 0.01, set to 0.01
        # DEPONS Java: if (foodValue < 0.01) foodValue = 0.01
        if self._food_value[i, j] < 0.01:
            self._food_value[i, j] = 0.01

        return food_eaten

    def eat_food_vectorized(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fraction: np.ndarray,
        xi: Optional[np.ndarray] = None,
        yi: Optional[np.ndarray] = None,
        energy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Eat food from multiple cells (vectorized).

        Args:
            x: Array of x positions
            y: Array of y positions
            fraction: Array of fractions to eat (0-1) for each position
            xi: Pre-computed clamped x indices (int32). Skips recomputation.
            yi: Pre-computed clamped y indices (int32). Skips recomputation.

        Returns:
            Array of food amounts eaten at each position
        """
        self._ensure_loaded()

        n = len(x)
        food_eaten = np.zeros(n, dtype=np.float32)

        if self._food_value is None or self._food_prob is None:
            return food_eaten

        # Get grid indices for all positions
        if xi is not None and yi is not None:
            j_arr, i_arr = xi, yi
        else:
            i_arr = np.clip(y.astype(np.int32), 0, self.height - 1)
            j_arr = np.clip(x.astype(np.int32), 0, self.width - 1)

        # Try v2 kernel (inline fraction from energy)
        if energy is not None:
            try:
                from cenop.optimizations.kernels import eat_food_kernel_v2
                if (
                    self._demand_grid is None
                    or self._demand_grid.shape != self._food_value.shape
                ):
                    self._demand_grid = np.zeros_like(self._food_value)
                eat_food_kernel_v2(
                    self._food_value,
                    j_arr,
                    i_arr,
                    energy.astype(np.float32),
                    food_eaten,
                    0.01,
                    self._demand_grid,
                )
                return food_eaten
            except ImportError:
                pass

        # Fallback to v1 kernel
        try:
            from cenop.optimizations.kernels import eat_food_kernel
            if self._demand_grid is None or self._demand_grid.shape != self._food_value.shape:
                self._demand_grid = np.zeros_like(self._food_value)
            eat_food_kernel(
                self._food_value, j_arr, i_arr, fraction.astype(np.float32),
                food_eaten, 0.01, self._demand_grid,
            )
            return food_eaten
        except ImportError:
            pass

        # Get current food at each position
        current_food = self._food_value[i_arr, j_arr]

        # Calculate food to eat
        food_eaten = current_food * fraction

        # Aggregate consumption for agents in the same cell using np.add.at
        # This prevents the last-write-wins race condition when multiple
        # agents occupy the same cell
        total_consumed = np.zeros_like(self._food_value)
        np.add.at(total_consumed, (i_arr, j_arr), food_eaten)

        # Update food values with aggregated consumption
        new_food = np.maximum(0.0, self._food_value - total_consumed)

        # ADD_ARTIFICIAL_FOOD: minimum 0.01
        new_food = np.maximum(new_food, 0.01)

        # Write back to food grid
        self._food_value[:] = new_food

        # Recompute actual food eaten per agent (may be less if cell was depleted)
        actual_available = self._food_value[i_arr, j_arr] + food_eaten
        ratio = np.where(total_consumed[i_arr, j_arr] > 0,
                         food_eaten / total_consumed[i_arr, j_arr],
                         0.0)
        # Each agent gets its proportional share of the actual depletion
        actual_eaten = np.minimum(food_eaten, actual_available * ratio)

        return actual_eaten

    def replenish_food(
        self,
        rate: float,
        max_u: float = 1.0,
        regrowth_qualifier: float = 0.001,
        max_ent: Optional[np.ndarray] = None,
        mean_max_ent_in_quarter: float = 1.0,
    ) -> None:
        """Replenish food across all cells using DEPONS 3.2 logistic regrowth.

        Formula (per active cell): F += rU * F * (1 - F/K)
        where K = max_u * max_ent / mean_max_ent_in_quarter.

        The first iteration is always applied.  If the resulting delta
        exceeds *regrowth_qualifier* the step is repeated 47 more times
        (48 total), matching the DEPONS 3.2 inner-loop behaviour.

        Args:
            rate: Per-step growth rate (rU in DEPONS).
            max_u: Scaling factor for carrying capacity (default 1.0).
            regrowth_qualifier: Delta threshold that triggers 47 extra
                iterations (default 0.001).
            max_ent: Optional 2-D array of MaxEnt values to use as the
                spatial capacity numerator.  If *None* the current-month
                entropy layer is used; if that is also absent, food_prob
                is used instead.
            mean_max_ent_in_quarter: Denominator for the capacity
                calculation (default 1.0).
        """
        self._ensure_loaded()
        if self._food_value is None or self._food_prob is None:
            return

        # --- Determine max_ent layer (2-D) --------------------------------
        if max_ent is None:
            if self._entropy is not None:
                month_idx = (self._current_month - 1) % 12
                ent_layer = self._entropy[month_idx]
            else:
                ent_layer = self._food_prob
        else:
            ent_layer = max_ent

        # --- Carrying capacity K (per cell) --------------------------------
        # Guard against zero denominator
        safe_denom = mean_max_ent_in_quarter if mean_max_ent_in_quarter > 0.0 else 1.0
        k_vals = max_u * ent_layer / safe_denom

        # --- Active mask: cells where food_prob > 0 and K > 0 -------------
        active = (self._food_prob > 0.0) & (k_vals > 0.0)

        food = self._food_value.copy()

        # --- Floor: bump very-low cells to 0.01 before regrowth -----------
        floor_mask = active & (food < 0.01)
        food[floor_mask] = 0.01

        # --- Guard: skip cells already at or above capacity ---------------
        active = active & (food < k_vals)

        if not np.any(active):
            self._food_value[:] = food
            return

        # --- First logistic iteration -------------------------------------
        delta = np.zeros_like(food)
        delta[active] = rate * food[active] * (
            1.0 - food[active] / k_vals[active]
        )
        food[active] += delta[active]

        # --- Per-cell delta check -----------------------------------------
        delta_mask = active & (np.abs(delta) > regrowth_qualifier)

        # --- 47 extra iterations where delta was large --------------------
        if np.any(delta_mask):
            f_sub = food[delta_mask].copy().astype(np.float64)
            k_sub = k_vals[delta_mask].astype(np.float64)
            try:
                from cenop.optimizations.kernels import regrow_food_kernel
                regrow_food_kernel(f_sub, k_sub, float(rate), 47)
            except ImportError:
                for _ in range(47):
                    f_sub = f_sub + rate * f_sub * (1.0 - f_sub / k_sub)
            food[delta_mask] = f_sub

        # --- Clip to capacity and write back ------------------------------
        food[active | delta_mask] = np.minimum(
            food[active | delta_mask], k_vals[active | delta_mask]
        )
        self._food_value[:] = food
        
    def get_block(self, x: float, y: float) -> int:
        """Get block ID at position."""
        self._ensure_loaded()
        if self._blocks is None:
            return 0
        i, j = self._get_indices(x, y)
        return int(self._blocks[i, j])
        
    def get_salinity(self, x: float, y: float, month: Optional[int] = None) -> float:
        """Get salinity at position for given month."""
        self._ensure_loaded()
        if self._salinity is None:
            return 30.0  # Default salinity
        if month is None:
            month = self._current_month
        month_idx = (month - 1) % 12
        i, j = self._get_indices(x, y)
        return float(self._salinity[month_idx, i, j])
        
    def get_max_ent(self, x: float, y: float, month: Optional[int] = None) -> float:
        """Get MaxEnt (entropy/prey) value at position for given month."""
        self._ensure_loaded()
        if self._entropy is None:
            return 0.5
        if month is None:
            month = self._current_month
        month_idx = (month - 1) % 12
        i, j = self._get_indices(x, y)
        return float(self._entropy[month_idx, i, j])
        
    def set_month(self, month: int) -> None:
        """Set the current month for lookups."""
        self._current_month = max(1, min(12, month))

    def get_quarter_of_year(self, tick: int, shift_quarter: bool = True) -> int:
        """Compute quarter index (0-3) from tick, matching Java SimulationTime.java:88-95."""
        effective_tick = tick + (30 * 48 if shift_quarter else 0)
        return int((effective_tick / (3 * 30 * 48)) % 4)

    def get_current_max_ent(self) -> Optional[np.ndarray]:
        """Get the maxEnt array for the current month."""
        if self._entropy is None:
            return None
        month_idx = (self._current_month - 1) % 12
        return self._entropy[month_idx]
        
    def get_depths_vectorized(
        self,
        positions: np.ndarray,
        xi: Optional[np.ndarray] = None,
        yi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get depths for multiple positions at once.

        Args:
            positions: Array of shape (N, 2) with [x, y] positions
            xi: Pre-computed clamped x indices (int32). Skips recomputation.
            yi: Pre-computed clamped y indices (int32). Skips recomputation.

        Returns:
            Array of depths with shape (N,)
        """
        self._ensure_loaded()
        if self._depth is None:
            n = len(xi) if xi is not None else len(positions)
            return np.full(n, 20.0)

        if xi is not None and yi is not None:
            x, y = xi, yi
        else:
            x = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
            y = np.clip(positions[:, 1].astype(int), 0, self.height - 1)

        return self._depth[y, x]
        
    def get_salinities_vectorized(
        self,
        positions: np.ndarray,
        month: Optional[int] = None,
        xi: Optional[np.ndarray] = None,
        yi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get salinities for multiple positions at once.

        Args:
            positions: Array of shape (N, 2) with [x, y] positions
            month: Month (1-12) or None for current month
            xi: Pre-computed clamped x indices (int32). Skips recomputation.
            yi: Pre-computed clamped y indices (int32). Skips recomputation.

        Returns:
            Array of salinities with shape (N,)
        """
        self._ensure_loaded()
        if self._salinity is None:
            n = len(xi) if xi is not None else len(positions)
            return np.full(n, 30.0)

        if month is None:
            month = self._current_month
        month_idx = (month - 1) % 12

        if xi is not None and yi is not None:
            x, y = xi, yi
        else:
            x = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
            y = np.clip(positions[:, 1].astype(int), 0, self.height - 1)

        return self._salinity[month_idx, y, x]

    def get_sediments_vectorized(self, positions: np.ndarray) -> np.ndarray:
        """
        Get sediment grain sizes for multiple positions at once.

        Args:
            positions: Array of shape (N, 2) with [x, y] positions

        Returns:
            Array of sediment grain sizes with shape (N,).
            Returns 1.0 (fine sand) for all positions if no sediment data loaded.
        """
        self._ensure_loaded()
        if self._sediment is None:
            return np.full(len(positions), 1.0)

        x = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
        y = np.clip(positions[:, 1].astype(int), 0, self.height - 1)

        return self._sediment[y, x]


def load_bathymetry_from_asc(filepath: str) -> Tuple[np.ndarray, LandscapeMetadata]:
    """
    Load bathymetry data from a DEPONS ASC file.
    
    Args:
        filepath: Path to the .asc file
        
    Returns:
        Tuple of (depth array, metadata)
    """
    with open(filepath, 'r') as f:
        # Read header with validation
        expected_headers = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize']
        header_values = {}

        for expected in expected_headers:
            line = f.readline()
            if not line:
                raise ValueError(
                    f"ASC header incomplete: missing '{expected}' "
                    f"(file: {filepath})"
                )
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"ASC header malformed: expected '{expected} <value>', "
                    f"got '{line.strip()}' (file: {filepath})"
                )
            header_values[expected] = parts[1]

        try:
            ncols = int(header_values['ncols'])
            nrows = int(header_values['nrows'])
            xllcorner = float(header_values['xllcorner'])
            yllcorner = float(header_values['yllcorner'])
            cellsize = float(header_values['cellsize'])
        except ValueError as e:
            raise ValueError(
                f"ASC header has non-numeric value: {e} (file: {filepath})"
            ) from e

        # NODATA line (optional)
        nodata_line = f.readline().split()
        nodata_value = float(nodata_line[1]) if len(nodata_line) > 1 else -9999.0

        # Read data with validation
        data = []
        for line_num, line in enumerate(f, start=7):
            try:
                values = [float(v) for v in line.split()]
            except ValueError as e:
                raise ValueError(
                    f"ASC data has non-numeric value at line {line_num}: {e} "
                    f"(file: {filepath})"
                ) from e
            data.extend(values)

        expected_count = nrows * ncols
        if len(data) != expected_count:
            raise ValueError(
                f"ASC data values count mismatch: expected {expected_count} "
                f"({nrows}x{ncols}), got {len(data)} (file: {filepath})"
            )

        depth_array = np.array(data).reshape((nrows, ncols))
        
        # Replace nodata values with land indicator (-10)
        depth_array = np.where(depth_array == nodata_value, -10.0, depth_array)
        
        # Flip Y-axis to match DEPONS convention:
        # DEPONS stores array[x][height-1-y], so row 0 = SOUTH, row max = NORTH
        # ASC file has row 0 at NORTH, so we flip vertically
        depth_array = np.flipud(depth_array)
        
    metadata = LandscapeMetadata(
        ncols=ncols,
        nrows=nrows,
        xllcorner=xllcorner,
        yllcorner=yllcorner,
        cellsize=cellsize,
        nodata_value=nodata_value
    )
    
    return depth_array, metadata


def create_landscape_from_depons(
    depons_data_dir: str = None,
    food_prob: float = 0.5
) -> CellData:
    """
    Create a landscape using real DEPONS bathymetry data.

    This loads bathymetry from external DEPONS-master data files (not shipped
    with CENOP). The DEPONS grid is 400x400 cells at 400m resolution.

    Args:
        depons_data_dir: Path to a DEPONS data folder containing bathy.asc.
                        If None, will search common locations.
        food_prob: Uniform food probability

    Returns:
        CellData with DEPONS bathymetry, or homogeneous fallback
    """
    import os
    import logging
    logger = logging.getLogger("CENOP")

    # Search for DEPONS data directory
    if depons_data_dir is None:
        possible_paths = [
            "../DEPONS-master/data",
            "../../DEPONS-master/data",
            "DEPONS-master/data",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                depons_data_dir = path
                break

    if depons_data_dir is None or not os.path.exists(depons_data_dir):
        logger.info("DEPONS data directory not found, falling back to homogeneous landscape")
        return create_homogeneous_landscape()
    
    bathy_file = os.path.join(depons_data_dir, "bathy.asc")
    if not os.path.exists(bathy_file):
        logger.info("Bathymetry file not found: %s", bathy_file)
        return create_homogeneous_landscape()

    # Load bathymetry
    logger.info("Loading DEPONS bathymetry from %s...", bathy_file)
    depth_array, metadata = load_bathymetry_from_asc(bathy_file)
    
    # In DEPONS, depth values are positive (meters below sea level)
    # We need to convert so that negative = land, positive = water
    # If depth < 0 or very small, it's likely land
    # DEPONS uses depth > 0 for water, but we treat depth <= 0 as land
    # Actually in the file, all values are positive depths
    # We just need to mark land where there's no water
    
    # The DEPONS bathy.asc has all positive values for water depths
    # For land avoidance, we check if depth > 0 (water)
    # Values of 0 or negative would be land
    
    logger.info("Loaded bathymetry: %dx%d, depth range: %.1f to %.1fm",
                metadata.nrows, metadata.ncols, depth_array.min(), depth_array.max())
    
    cell_data = CellData.__new__(CellData)
    cell_data.landscape_name = "DEPONS_external"
    cell_data.data_dir = Path(depons_data_dir)
    cell_data.metadata = metadata
    
    cell_data._depth = depth_array
    cell_data._dist_to_coast = np.full((metadata.nrows, metadata.ncols), 10000.0)
    cell_data._sediment = np.ones((metadata.nrows, metadata.ncols))
    cell_data._food_prob = np.full((metadata.nrows, metadata.ncols), food_prob)
    cell_data._food_value = np.full((metadata.nrows, metadata.ncols), food_prob)
    cell_data._blocks = np.zeros((metadata.nrows, metadata.ncols), dtype=int)
    cell_data._entropy = np.full((12, metadata.nrows, metadata.ncols), 0.5)
    cell_data._salinity = np.full((12, metadata.nrows, metadata.ncols), 30.0)
    cell_data._demand_grid = None

    cell_data._current_month = 1
    cell_data._loaded = True

    return cell_data


def create_homogeneous_landscape(
    width: int = 400,
    height: int = 400,
    depth: float = 30.0,
    food_prob: float = 0.5
) -> CellData:
    """
    Create a homogeneous (uniform) landscape for testing.
    Now uses DEPONS-compatible dimensions (400x400 cells).
    Includes land boundaries at edges to simulate a coastal area.
    
    Args:
        width: Grid width (default 400 to match DEPONS)
        height: Grid height (default 400 to match DEPONS)
        depth: Uniform depth value (for water cells)
        food_prob: Uniform food probability
        
    Returns:
        CellData with homogeneous values and coastal boundaries
    """
    cell_data = CellData.__new__(CellData)
    cell_data.landscape_name = "Homogeneous"
    cell_data.data_dir = Path(".")
    cell_data.metadata = LandscapeMetadata(
        ncols=width,
        nrows=height,
        xllcorner=0.0,
        yllcorner=0.0,
        cellsize=400.0
    )
    
    # Create depth array with land at edges (simulating North Sea coastline)
    depth_array = np.full((height, width), depth)
    
    # Add land (depth = -10) at edges to keep porpoises in water
    # Southern edge (simulating continental coast - thicker)
    land_thickness_s = int(height * 0.05)  # 5% of height
    depth_array[:land_thickness_s, :] = -10.0
    
    # Eastern edge (simulating some coast)  
    land_thickness_e = int(width * 0.03)
    depth_array[:, -land_thickness_e:] = -10.0
    
    # Western edge (simulating UK coast)
    land_thickness_w = int(width * 0.03)
    depth_array[:, :land_thickness_w] = -10.0
    
    # Northern edge (open sea but with some islands)
    land_thickness_n = int(height * 0.02)
    depth_array[-land_thickness_n:, :] = -10.0
    
    cell_data._depth = depth_array
    cell_data._dist_to_coast = np.full((height, width), 10000.0)
    cell_data._sediment = np.ones((height, width))
    cell_data._food_prob = np.full((height, width), food_prob)
    cell_data._food_value = np.full((height, width), food_prob)
    cell_data._blocks = np.zeros((height, width), dtype=int)
    cell_data._entropy = np.full((12, height, width), 0.5)
    cell_data._salinity = np.full((12, height, width), 30.0)
    cell_data._demand_grid = None

    cell_data._current_month = 1
    cell_data._loaded = True
    
    return cell_data
