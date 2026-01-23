"""
CENOP - CETacean Noise-Population Model

A Python translation of the DEPONS agent-based model for simulating
harbour porpoise population dynamics under disturbance.
"""

__version__ = "0.1.0"
__author__ = "AI4WIND Project Team"

from cenop.core.simulation import Simulation
from cenop.parameters.simulation_params import SimulationParameters
from cenop.parameters.constants import SimulationConstants
from cenop.landscape.cell_data import (
    CellData, 
    create_homogeneous_landscape,
    create_landscape_from_depons
)

__all__ = [
    "Simulation",
    "SimulationParameters",
    "SimulationConstants",
    "CellData",
    "create_homogeneous_landscape",
    "create_landscape_from_depons",
]
