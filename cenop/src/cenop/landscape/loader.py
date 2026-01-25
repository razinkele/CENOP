"""
Landscape data file loader.

Loads ASCII grid files and other data formats used by DEPONS.
Translates from: LandscapeLoader.java
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from cenop.landscape.cell_data import LandscapeMetadata

logger = logging.getLogger(__name__)


# File names used by DEPONS
BATHY_FILE = "bathy.asc"
DISTTOCOAST_FILE = "disttocoast.asc"
SEDIMENT_FILE = "sediment.asc"
PATCHES_FILE = "patches.asc"
BLOCKS_FILE = "blocks.asc"
PREY_FILE_PREFIX = "prey"
SALINITY_FILE_PREFIX = "salinity"


def get_default_data_dir() -> Path:
    """Get the default data directory path relative to the package."""
    # Get the directory containing this module (cenop/landscape/)
    module_dir = Path(__file__).parent
    # Go up to cenop package root, then to workspace root, then find data
    # Structure: workspace/cenop/src/cenop/landscape/loader.py
    # Data is at: workspace/cenop/data/
    package_root = module_dir.parent.parent.parent  # Up to workspace/cenop/
    data_dir = package_root / "data"
    logger.debug(f"Default data directory: {data_dir} (exists: {data_dir.exists()})")
    return data_dir


class LandscapeLoader:
    """
    Loads landscape data from files.
    
    Translates from: LandscapeLoader.java
    """
    
    def __init__(self, landscape_name: str, data_dir: Path | str | None = None):
        """
        Initialize loader for a landscape.
        
        Args:
            landscape_name: Name of landscape folder
            data_dir: Base data directory (defaults to package data directory)
        """
        self.landscape_name = landscape_name
        if data_dir is None:
            self.data_dir = get_default_data_dir()
        else:
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
        
        # Load monthly files (may produce warnings if fallbacks used)
        entropy, entropy_warnings = self._load_monthly(PREY_FILE_PREFIX)
        salinity, salinity_warnings = self._load_monthly(SALINITY_FILE_PREFIX)
        warnings = []
        warnings.extend(entropy_warnings)
        warnings.extend(salinity_warnings)
        
        return {
            'metadata': metadata,
            'depth': depth,
            'dist_to_coast': dist_to_coast,
            'sediment': sediment,
            'food_prob': food_prob,
            'blocks': blocks.astype(int),
            'entropy': entropy,
            'salinity': salinity,
            'warnings': warnings,
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
        
        warnings = []
        for month in range(1, 13):
            filename = f"{prefix}{month:02d}.asc"
            filepath = self.landscape_path / filename

            if filepath.exists():
                data, _ = self._load_asc(filename)
                monthly_data.append(data)
            else:
                # If file doesn't exist, use previous month if available
                if monthly_data:
                    monthly_data.append(monthly_data[-1].copy())
                else:
                    # First month missing: try to infer dimensions from bathy or any ASC file
                    msg = f"Monthly file missing for landscape '{self.landscape_name}': {filepath} - falling back to zeros"
                    logger.warning(msg)
                    warnings.append(msg)
                    # Try bathy file
                    try:
                        bathy_data, _ = self._load_asc(BATHY_FILE)
                        fallback_shape = bathy_data.shape
                        zeros = np.zeros(fallback_shape, dtype=float)
                        monthly_data.append(zeros)
                    except FileNotFoundError:
                        # Try any other asc file in directory
                        asc_files = list(self.landscape_path.glob('*.asc'))
                        if asc_files:
                            try:
                                sample_data, _ = self._load_asc(asc_files[0].name)
                                fallback_shape = sample_data.shape
                                zeros = np.zeros(fallback_shape, dtype=float)
                                monthly_data.append(zeros)
                            except Exception:
                                msg2 = f"Unable to load sample asc file for fallback in landscape '{self.landscape_name}'; using 1x1 zero"
                                logger.exception(msg2)
                                warnings.append(msg2)
                                monthly_data.append(np.zeros((1, 1), dtype=float))
                        else:
                            msg3 = f"No asc files found in landscape '{self.landscape_name}'; using 1x1 zero fallback"
                            logger.warning(msg3)
                            warnings.append(msg3)
                            monthly_data.append(np.zeros((1, 1), dtype=float))

        return np.stack(monthly_data), warnings
        
    def file_exists(self, filename: str) -> bool:
        """Check if a data file exists."""
        return (self.landscape_path / filename).exists()

    def list_files(self) -> dict:
        """List expected landscape files and monthly coverage.

        Returns a dictionary with:
        - presence booleans for core files
        - months_found: list of ints for available monthly files for 'prey' and 'salinity'
        """
        info = {}
        core_files = [BATHY_FILE, DISTTOCOAST_FILE, SEDIMENT_FILE, PATCHES_FILE, BLOCKS_FILE]
        for f in core_files:
            info[f] = self.file_exists(f)

        # Monthly files - check multiple naming patterns
        prey_months = []
        sal_months = []
        for m in range(1, 13):
            # Try multiple naming patterns:
            # 1. prey01.asc, prey02.asc, etc.
            # 2. prey0000_01.asc, prey0000_02.asc, etc.
            prey_patterns = [
                f"{PREY_FILE_PREFIX}{m:02d}.asc",
                f"{PREY_FILE_PREFIX}0000_{m:02d}.asc"
            ]
            sal_patterns = [
                f"{SALINITY_FILE_PREFIX}{m:02d}.asc",
                f"{SALINITY_FILE_PREFIX}0000_{m:02d}.asc"
            ]
            
            # Check if any prey pattern exists
            if any(self.file_exists(pattern) for pattern in prey_patterns):
                prey_months.append(m)
            
            # Check if any salinity pattern exists
            if any(self.file_exists(pattern) for pattern in sal_patterns):
                sal_months.append(m)

        info['prey_months'] = prey_months
        info['salinity_months'] = sal_months
        # Convenience summary
        info['has_full_prey'] = len(prey_months) == 12
        info['has_full_salinity'] = len(sal_months) == 12
        return info

    @staticmethod
    def list_landscapes(data_dir: Path | str | None = None) -> List[str]:
        """
        List available landscapes in the data directory.

        Args:
            data_dir: Base data directory (defaults to package data directory)

        Returns:
            List of landscape names
        """
        if data_dir is None:
            data_path = get_default_data_dir()
        else:
            data_path = Path(data_dir)
            
        if not data_path.exists():
            logger.warning(f"Data directory does not exist: {data_path}")
            return []

        landscapes = []
        for item in data_path.iterdir():
            if item.is_dir():
                # Check if it has required files
                if (item / BATHY_FILE).exists():
                    landscapes.append(item.name)

        return sorted(landscapes)
