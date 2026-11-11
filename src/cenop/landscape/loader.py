"""
Landscape data file loader.

Loads ASCII grid files and other data formats used by DEPONS.
Translates from: LandscapeLoader.java
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

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


def nodata_on_water_counts(
    depth: np.ndarray, layers: dict[str, np.ndarray], nodata: float
) -> dict[str, int]:
    """Count water cells (depth > 0) that have no value in each layer.

    A value is missing when it equals ``nodata`` or is non-finite. A monthly layer of shape
    (12, H, W) counts a cell once if any month is missing there.
    """
    water = depth > 0
    counts = {}
    for name, arr in layers.items():
        a = np.asarray(arr, dtype=np.float64)
        missing = np.isclose(a, nodata) | ~np.isfinite(a)
        if missing.ndim == 3:
            missing = missing.any(axis=0)
        counts[name] = int((missing & water).sum())
    return counts


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
        if (
            not landscape_name
            or "/" in landscape_name
            or "\\" in landscape_name
            or landscape_name in (".", "..")
        ):
            raise ValueError(f"Invalid landscape name: {landscape_name!r}")
        self.landscape_name = landscape_name
        self.data_dir = Path(data_dir)
        self.landscape_path = self.data_dir / landscape_name

    def load_all(self) -> dict[str, Any]:
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

        # NODATA on a water cell silently breaks the model there (salinity -9999 forces every
        # CRW step to max_mov; patches/prey -9999 mean no food), so report it loudly.
        gaps = nodata_on_water_counts(
            depth,
            {
                "disttocoast": dist_to_coast,
                "sediment": sediment,
                "patches": food_prob,
                "blocks": blocks,
                "prey": entropy,
                "salinity": salinity,
            },
            metadata.nodata_value,
        )
        gaps = {name: n for name, n in gaps.items() if n}
        if gaps:
            logger.warning(
                "Landscape %s: water cells with NODATA (of %d water cells): %s",
                self.landscape_name,
                int((depth > 0).sum()),
                ", ".join(f"{name} {n}" for name, n in gaps.items()),
            )

        return {
            "metadata": metadata,
            "depth": depth,
            "dist_to_coast": dist_to_coast,
            "sediment": sediment,
            "food_prob": food_prob,
            "blocks": blocks.astype(int),
            "entropy": entropy,
            "salinity": salinity,
        }

    def _load_asc(self, filename: str) -> tuple[np.ndarray, LandscapeMetadata]:
        """
        Load an ASCII grid file.

        Optimized: uses np.loadtxt for data parsing instead of
        pure-Python line-by-line float conversion.

        Args:
            filename: Name of file to load

        Returns:
            Tuple of (data array, metadata)
        """
        filepath = self.landscape_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Landscape file not found: {filepath}")

        # Parse header (typically 6 lines)
        header = {}
        header_line_count = 0

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    header_line_count += 1
                    continue

                # Check if this is a header line
                parts = line.split()
                if len(parts) == 2 and parts[0].lower() in [
                    "ncols",
                    "nrows",
                    "xllcorner",
                    "yllcorner",
                    "cellsize",
                    "nodata_value",
                ]:
                    key = parts[0].lower()
                    value = parts[1]
                    if key in ["ncols", "nrows"]:
                        header[key] = int(value)
                    else:
                        header[key] = float(value)
                    header_line_count += 1
                else:
                    break  # First data line found

        # Validate required header fields
        required_fields = ["ncols", "nrows", "xllcorner", "yllcorner", "cellsize"]
        missing = [f for f in required_fields if f not in header]
        if missing:
            raise ValueError(
                f"ASC file {filepath} missing required header fields: {missing}. "
                f"Found: {list(header.keys())}"
            )

        # Create metadata
        metadata = LandscapeMetadata(
            ncols=header["ncols"],
            nrows=header["nrows"],
            xllcorner=header["xllcorner"],
            yllcorner=header["yllcorner"],
            cellsize=header["cellsize"],
            nodata_value=header.get("nodata_value", -9999.0),
        )

        # Parse data with numpy (much faster than pure Python)
        data_array = np.loadtxt(str(filepath), skiprows=header_line_count)

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

        Supports two naming conventions:
          - CENOP short:  prey01.asc, prey02.asc, ...
          - DEPONS long:  prey0000_01.asc, prey0000_02.asc, ...

        Args:
            prefix: File prefix (e.g., 'prey' or 'salinity')

        Returns:
            Array of shape (12, height, width)
        """
        monthly_data = []

        for month in range(1, 13):
            # Try short name first, then DEPONS long name
            short = f"{prefix}{month:02d}.asc"
            long = f"{prefix}0000_{month:02d}.asc"

            if (self.landscape_path / short).exists():
                data, _ = self._load_asc(short)
                monthly_data.append(data)
            elif (self.landscape_path / long).exists():
                data, _ = self._load_asc(long)
                monthly_data.append(data)
            else:
                # If file doesn't exist, duplicate previous month with warning
                if monthly_data:
                    logger.warning(
                        "Monthly %s file missing for month %d (%s / %s). "
                        "Duplicating month %d data as fallback.",
                        prefix,
                        month,
                        short,
                        long,
                        month - 1,
                    )
                    monthly_data.append(monthly_data[-1].copy())
                else:
                    raise FileNotFoundError(
                        f"Monthly file not found: {self.landscape_path / short} "
                        f"or {self.landscape_path / long}"
                    )

        return np.stack(monthly_data)

    def file_exists(self, filename: str) -> bool:
        """Check if a data file exists."""
        return (self.landscape_path / filename).exists()

    @staticmethod
    def list_landscapes(data_dir: Path | str = "data") -> list[str]:
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
