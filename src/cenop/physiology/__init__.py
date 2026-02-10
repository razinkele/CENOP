"""
Physiology module for CENOP-JASMINE hybrid simulation.

This package provides energy budget and metabolic calculations that can be
switched between DEPONS (simple) and JASMINE (DEB model) implementations.

Module Structure:
    energy_budget.py - Energy tracking and metabolic calculations

Quick Start:
    from cenop.physiology import create_energy_module, EnergyMode

    # Create DEPONS energy module (regulatory)
    energy = create_energy_module(params, EnergyMode.DEPONS)

    # Create JASMINE energy module (research)
    energy = create_energy_module(params, EnergyMode.JASMINE)

Available Classes:
    EnergyModule       - Abstract base class
    EnergyState        - State container for energy variables
    EnergyContext      - Environmental context for calculations
    EnergyResult       - Result of energy budget computation
    DEPONSEnergyModule - DEPONS simple energy model
    JASMINEEnergyModule - JASMINE DEB model

Factory Functions:
    create_energy_module() - Create appropriate module for mode
"""

from cenop.physiology.energy_budget import (
    EnergyMode,
    EnergyState,
    EnergyContext,
    EnergyResult,
    EnergyModule,
    DEPONSEnergyModule,
    JASMINEEnergyModule,
    create_energy_module,
)

__all__ = [
    # Enums
    "EnergyMode",
    # Data classes
    "EnergyState",
    "EnergyContext",
    "EnergyResult",
    # Base class
    "EnergyModule",
    # Implementations
    "DEPONSEnergyModule",
    "JASMINEEnergyModule",
    # Factory
    "create_energy_module",
]
