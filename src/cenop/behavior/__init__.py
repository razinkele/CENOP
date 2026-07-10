"""Behavior modules for movement, memory, sound, and state machines."""

from cenop.behavior.dispersal import DispersalBehavior, DispersalType
from cenop.behavior.disturbance_memory import (
    AvoidanceResult,
    DEPONSMemoryModule,
    DisturbanceMemoryContext,
    DisturbanceMemoryModule,
    DisturbanceMemoryState,
    JASMINEMemoryModule,
    MemoryMode,
    create_memory_module,
)
from cenop.behavior.hybrid_fsm import (
    FSMMode,
    HybridBehaviorFSM,
    create_behavior_fsm,
)
from cenop.behavior.memory import RefMem
from cenop.behavior.psm import MemCellData, PersistentSpatialMemory, PSMDispersalType2
from cenop.behavior.sound import (
    ShipDeterrenceModel,
    ShipNoise,
    TurbineNoise,
    calculate_deterrence_vector,
    calculate_received_level,
    calculate_transmission_loss,
)
from cenop.behavior.states import (
    STATE_PARAMETERS,
    BehaviorContext,
    BehaviorState,
    BehaviorStateVector,
    StateTransition,
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
