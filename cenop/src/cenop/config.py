"""
Configuration and path management for CENOP.

This module provides centralized access to project paths and resources,
ensuring robustness across different execution environments.
"""

from pathlib import Path
import logging

logger = logging.getLogger("CENOP")

# Determine project root
# We assume this file is in src/cenop/config.py
# Root should be 3 levels up: src/cenop/ -> src/ -> cenop/ (project root containing detailed data)
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Define standard paths
DATA_DIR = PROJECT_ROOT / "data"
WIND_FARMS_DIR = DATA_DIR / "wind-farms"
STATIC_DIR = PROJECT_ROOT / "static"

def get_data_file(filename: str) -> Path:
    """Resolve a path to a data file, checking existence."""
    path = DATA_DIR / filename
    if not path.exists():
        logger.warning(f"Data file not found: {path}")
    return path

def get_wind_farm_file(filename: str) -> Path:
    """Resolve a path to a wind farm definition file."""
    path = WIND_FARMS_DIR / filename
    # Don't check existence here as the caller might handle it
    return path
