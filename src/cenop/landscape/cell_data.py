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
            landscape_name: Name of landscape (e.g., 'NorthSea')
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
        
        # Initialize food values from food probability
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
        fraction: np.ndarray
    ) -> np.ndarray:
        """
        Eat food from multiple cells (vectorized).

        Args:
            x: Array of x positions
            y: Array of y positions
            fraction: Array of fractions to eat (0-1) for each position

        Returns:
            Array of food amounts eaten at each position
        """
        self._ensure_loaded()

        n = len(x)
        food_eaten = np.zeros(n, dtype=np.float32)

        if self._food_value is None or self._food_prob is None:
            return food_eaten

        # Get grid indices for all positions
        i_arr = np.clip(y.astype(np.int32), 0, self.height - 1)
        j_arr = np.clip(x.astype(np.int32), 0, self.width - 1)

        # Get current food at each position
        current_food = self._food_value[i_arr, j_arr]

        # Calculate food to eat
        food_eaten = current_food * fraction

        # Update food values
        new_food = np.maximum(0.0, current_food - food_eaten)

        # ADD_ARTIFICIAL_FOOD: minimum 0.01
        new_food = np.maximum(new_food, 0.01)

        # Write back to food grid
        # Note: This handles duplicate positions by last-write-wins
        # For accurate multi-agent eating, would need aggregation
        self._food_value[i_arr, j_arr] = new_food

        return food_eaten

    def replenish_food(self, rate: float) -> None:
        """Replenish food across all cells."""
        self._ensure_loaded()
        if self._food_value is None or self._food_prob is None:
            return
        # Food regenerates towards food_prob level
        diff = self._food_prob - self._food_value
        self._food_value += rate * diff
        
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
        
    def get_depths_vectorized(self, positions: np.ndarray) -> np.ndarray:
        """
        Get depths for multiple positions at once.
        
        Args:
            positions: Array of shape (N, 2) with [x, y] positions
            
        Returns:
            Array of depths with shape (N,)
        """
        self._ensure_loaded()
        if self._depth is None:
            return np.full(len(positions), 20.0)
            
        x = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
        y = np.clip(positions[:, 1].astype(int), 0, self.height - 1)
        
        return self._depth[y, x]
        
    def get_salinities_vectorized(
        self,
        positions: np.ndarray,
        month: Optional[int] = None
    ) -> np.ndarray:
        """
        Get salinities for multiple positions at once.
        
        Args:
            positions: Array of shape (N, 2) with [x, y] positions
            month: Month (1-12) or None for current month
            
        Returns:
            Array of salinities with shape (N,)
        """
        self._ensure_loaded()
        if self._salinity is None:
            return np.full(len(positions), 30.0)
            
        if month is None:
            month = self._current_month
        month_idx = (month - 1) % 12
        
        x = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
        y = np.clip(positions[:, 1].astype(int), 0, self.height - 1)
        
        return self._salinity[month_idx, y, x]


def load_bathymetry_from_asc(filepath: str) -> Tuple[np.ndarray, LandscapeMetadata]:
    """
    Load bathymetry data from a DEPONS ASC file.
    
    Args:
        filepath: Path to the .asc file
        
    Returns:
        Tuple of (depth array, metadata)
    """
    with open(filepath, 'r') as f:
        # Read header
        ncols = int(f.readline().split()[1])
        nrows = int(f.readline().split()[1])
        xllcorner = float(f.readline().split()[1])
        yllcorner = float(f.readline().split()[1])
        cellsize = float(f.readline().split()[1])
        nodata_line = f.readline().split()
        nodata_value = float(nodata_line[1]) if len(nodata_line) > 1 else -9999.0
        
        # Read data - each row contains space-separated values
        data = []
        for line in f:
            values = [float(v) for v in line.split()]
            data.extend(values)
        
        # Reshape to 2D array (nrows x ncols)
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
    
    This loads the actual North Sea bathymetry from the DEPONS-master data files.
    The DEPONS grid is 400x400 cells at 400m resolution (160km x 160km area).
    
    Args:
        depons_data_dir: Path to DEPONS-master/data/UserDefined folder
                        If None, will search common locations
        food_prob: Uniform food probability
        
    Returns:
        CellData with real North Sea bathymetry
    """
    import os
    
    # Search for DEPONS data directory
    if depons_data_dir is None:
        possible_paths = [
            "../DEPONS-master/data/UserDefined",
            "../../DEPONS-master/data/UserDefined",
            "../../../DEPONS-master/data/UserDefined",
            "DEPONS-master/data/UserDefined",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                depons_data_dir = path
                break
    
    if depons_data_dir is None or not os.path.exists(depons_data_dir):
        print("DEPONS data directory not found, falling back to homogeneous landscape")
        return create_homogeneous_landscape()
    
    bathy_file = os.path.join(depons_data_dir, "bathy.asc")
    if not os.path.exists(bathy_file):
        print(f"Bathymetry file not found: {bathy_file}")
        return create_homogeneous_landscape()
    
    # Load bathymetry
    print(f"Loading DEPONS bathymetry from {bathy_file}...")
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
    
    print(f"Loaded bathymetry: {metadata.nrows}x{metadata.ncols}, depth range: {depth_array.min():.1f} to {depth_array.max():.1f}m")
    
    cell_data = CellData.__new__(CellData)
    cell_data.landscape_name = "NorthSea_DEPONS"
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
    
    cell_data._current_month = 1
    cell_data._loaded = True
    
    return cell_data
