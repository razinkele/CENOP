"""
Movement module for CENOP-JASMINE hybrid simulation.

This package provides modular movement implementations that can be
switched between DEPONS (empirical) and JASMINE (physics-based) models.

Module Structure:
    base.py          - Abstract MovementModule interface
    depons_crw.py    - DEPONS Correlated Random Walk implementation
    jasmine_physics.py - JASMINE physics-based movement
    hybrid.py        - Hybrid selector for context-dependent switching

Quick Start:
    from cenop.movement import create_movement_module
    from cenop.core.time_manager import TimeMode

    # Create movement module based on simulation mode
    movement = create_movement_module(params, TimeMode.DEPONS)

    # Compute movement step
    result = movement.compute_step(x, y, state, environment, mask)

    # Apply result
    x += result.dx
    y += result.dy

Available Classes:
    MovementModule      - Abstract base class
    MovementState       - State container for movement variables
    EnvironmentContext  - Environmental variables for movement
    MovementResult      - Result of movement computation
    DEPONSCRWMovement   - DEPONS CRW implementation
    JASMINEPhysicsMovement - JASMINE physics implementation
    HybridMovementSelector - Hybrid mode selector

Factory Functions:
    create_movement_module() - Create appropriate module for mode
"""

from cenop.movement.base import (
    MovementModule,
    MovementMode,
    MovementState,
    EnvironmentContext,
    MovementResult,
)

from cenop.movement.depons_crw import (
    DEPONSCRWMovement,
    DEPONSCRWMovementVectorized,
)

from cenop.movement.jasmine_physics import (
    JASMINEPhysicsMovement,
    JASMINEMovementState,
    JASMINEEnvironmentContext,
)

from cenop.movement.hybrid import (
    HybridMovementSelector,
    HybridStrategy,
    create_movement_module,
)

__all__ = [
    # Base classes
    "MovementModule",
    "MovementMode",
    "MovementState",
    "EnvironmentContext",
    "MovementResult",
    # DEPONS implementation
    "DEPONSCRWMovement",
    "DEPONSCRWMovementVectorized",
    # JASMINE implementation
    "JASMINEPhysicsMovement",
    "JASMINEMovementState",
    "JASMINEEnvironmentContext",
    # Hybrid selector
    "HybridMovementSelector",
    "HybridStrategy",
    # Factory function
    "create_movement_module",
]
