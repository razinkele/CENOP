"""Behavior modules for movement, memory, sound, and state machines."""

from cenop.behavior.memory import RefMem
from cenop.behavior.dispersal import DispersalType, DispersalBehavior
from cenop.behavior.sound import (
    calculate_received_level,
    calculate_transmission_loss,
    calculate_deterrence_vector,
    TurbineNoise,
    ShipNoise,
    ShipDeterrenceModel
)
from cenop.behavior.psm import (
    PersistentSpatialMemory,
    PSMDispersalType2,
    MemCellData
)
from cenop.behavior.states import (
    BehaviorState,
    BehaviorContext,
    BehaviorStateVector,
    StateTransition,
    STATE_PARAMETERS,
)
from cenop.behavior.hybrid_fsm import (
    HybridBehaviorFSM,
    FSMMode,
    create_behavior_fsm,
)
from cenop.behavior.disturbance_memory import (
    MemoryMode,
    DisturbanceMemoryState,
    DisturbanceMemoryContext,
    AvoidanceResult,
    DisturbanceMemoryModule,
    DEPONSMemoryModule,
    JASMINEMemoryModule,
    create_memory_module,
)

__all__ = [
    # Memory
    "RefMem",
    # Dispersal
    "DispersalType",
    "DispersalBehavior",
    # Sound/Deterrence
    "calculate_received_level",
    "calculate_transmission_loss",
    "calculate_deterrence_vector",
    "TurbineNoise",
    "ShipNoise",
    "ShipDeterrenceModel",
    # PSM
    "PersistentSpatialMemory",
    "PSMDispersalType2",
    "MemCellData",
    # Behavioral States
    "BehaviorState",
    "BehaviorContext",
    "BehaviorStateVector",
    "StateTransition",
    "STATE_PARAMETERS",
    # Hybrid FSM
    "HybridBehaviorFSM",
    "FSMMode",
    "create_behavior_fsm",
    # Disturbance Memory (Phase 5)
    "MemoryMode",
    "DisturbanceMemoryState",
    "DisturbanceMemoryContext",
    "AvoidanceResult",
    "DisturbanceMemoryModule",
    "DEPONSMemoryModule",
    "JASMINEMemoryModule",
    "create_memory_module",
]
