"""
Landscape data file loader.

Loads ASCII grid files and other data formats used by DEPONS.
Translates from: LandscapeLoader.java
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

from cenop.landscape.cell_data import LandscapeMetadata


# File names used by DEPONS
BATHY_FILE = "bathy.asc"
DISTTOCOAST_FILE = "disttocoast.asc"
SEDIMENT_FILE = "sediment.asc"
PATCHES_FILE = "patches.asc"
BLOCKS_FILE = "blocks.asc"
PREY_FILE_PREFIX = "prey"
SALINITY_FILE_PREFIX = "salinity"


class LandscapeLoader:
    """
    Loads landscape data from files.
    
    Translates from: LandscapeLoader.java
    """
    
    def __init__(self, landscape_name: str, data_dir: Path | str = "data"):
        """
        Initialize loader for a landscape.
        
        Args:
            landscape_name: Name of landscape folder
            data_dir: Base data directory
        """
        self.landscape_name = landscape_name
        self.data_dir = Path(data_dir)
        self.landscape_path = self.data_dir / landscape_name
        
    def load_all(self) -> Dict[str, Any]:
        """
        Load all data files for the landscape.
        
        Returns:
            Dictionary containing all loaded data arrays and metadata
        """
        # Load core files
        depth, metadata = self._load_asc(BATHY_FILE)
        dist_to_coast, _ = self._load_asc(DISTTOCOAST_FILE)
        sediment, _ = self._load_asc(SEDIMENT_FILE)
        food_prob, _ = self._load_asc(PATCHES_FILE)
        blocks, _ = self._load_asc(BLOCKS_FILE)
        
        # Load monthly files
        entropy = self._load_monthly(PREY_FILE_PREFIX)
        salinity = self._load_monthly(SALINITY_FILE_PREFIX)
        
        return {
            'metadata': metadata,
            'depth': depth,
            'dist_to_coast': dist_to_coast,
            'sediment': sediment,
            'food_prob': food_prob,
            'blocks': blocks.astype(int),
            'entropy': entropy,
            'salinity': salinity,
        }
        
    def _load_asc(self, filename: str) -> tuple[np.ndarray, LandscapeMetadata]:
        """
        Load an ASCII grid file.
        
        Args:
            filename: Name of file to load
            
        Returns:
            Tuple of (data array, metadata)
        """
        filepath = self.landscape_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Landscape file not found: {filepath}")
            
        # Parse header
        header = {}
        data_lines = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this is a header line
                parts = line.split()
                if len(parts) == 2 and parts[0].lower() in [
                    'ncols', 'nrows', 'xllcorner', 'yllcorner', 
                    'cellsize', 'nodata_value'
                ]:
                    key = parts[0].lower()
                    value = parts[1]
                    if key in ['ncols', 'nrows']:
                        header[key] = int(value)
                    else:
                        header[key] = float(value)
                else:
                    # Data line
                    data_lines.append(line)
                    
        # Create metadata
        metadata = LandscapeMetadata(
            ncols=header.get('ncols', 0),
            nrows=header.get('nrows', 0),
            xllcorner=header.get('xllcorner', 0.0),
            yllcorner=header.get('yllcorner', 0.0),
            cellsize=header.get('cellsize', 400.0),
            nodata_value=header.get('nodata_value', -9999.0),
        )
        
        # Parse data
        data = []
        for line in data_lines:
            row = [float(x) for x in line.split()]
            data.append(row)
            
        data_array = np.array(data)
        
        # DEPONS Compatibility: Keep NODATA as -9999, do NOT convert to NaN
        # NaN comparisons always return False, breaking land detection
        # -9999 is always < min_depth, so land detection works correctly
        
        # Flip Y-axis to match DEPONS convention:
        # DEPONS stores array[x][height-1-y], so row 0 = SOUTH, row max = NORTH
        # ASC file has row 0 at NORTH, so we flip vertically
        data_array = np.flipud(data_array)
        
        return data_array, metadata
        
    def _load_monthly(self, prefix: str) -> np.ndarray:
        """
        Load 12 monthly data files.
        
        Args:
            prefix: File prefix (e.g., 'prey' for prey01.asc, prey02.asc, ...)
            
        Returns:
            Array of shape (12, height, width)
        """
        monthly_data = []
        
        for month in range(1, 13):
            filename = f"{prefix}{month:02d}.asc"
            filepath = self.landscape_path / filename
            
            if filepath.exists():
                data, _ = self._load_asc(filename)
                monthly_data.append(data)
            else:
                # If file doesn't exist, use previous month or zeros
                if monthly_data:
                    monthly_data.append(monthly_data[-1].copy())
                else:
                    # Need to get dimensions from another file
                    raise FileNotFoundError(
                        f"Monthly file not found: {filepath}"
                    )
                    
        return np.stack(monthly_data)
        
    def file_exists(self, filename: str) -> bool:
        """Check if a data file exists."""
        return (self.landscape_path / filename).exists()
        
    @staticmethod
    def list_landscapes(data_dir: Path | str = "data") -> List[str]:
        """
        List available landscapes in the data directory.
        
        Args:
            data_dir: Base data directory
            
        Returns:
            List of landscape names
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            return []
            
        landscapes = []
        for item in data_path.iterdir():
            if item.is_dir():
                # Check if it has required files
                if (item / BATHY_FILE).exists():
                    landscapes.append(item.name)
                    
        return sorted(landscapes)
