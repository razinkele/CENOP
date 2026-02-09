"""
Disturbance Memory module for CENOP-JASMINE hybrid simulation.

This module provides enhanced memory systems for tracking disturbance events:
- DEPONS mode: No disturbance memory (stateless deterrence)
- JASMINE mode: Full disturbance memory with learned avoidance

Key features of JASMINE memory model:
- Spatial memory of disturbance zones
- Configurable memory decay rates
- Learned avoidance behavior
- Cumulative disturbance impact tracking

Reference:
- JASMINE-MB Technical Documentation - Memory and Cognition
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Dict, Any, Tuple, List
import numpy as np

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters


class MemoryMode(Enum):
    """Memory system modes."""
    DEPONS = auto()    # No disturbance memory (stateless)
    JASMINE = auto()   # Full memory with decay and avoidance


@dataclass
class DisturbanceEvent:
    """
    Record of a single disturbance event.

    Tracks when and where a disturbance occurred.
    """
    x: float                    # X position in grid cells
    y: float                    # Y position in grid cells
    intensity: float            # Disturbance intensity (0-1)
    tick: int                   # Tick when disturbance occurred
    duration: int = 1           # Duration in ticks
    source_type: str = "unknown"  # turbine, ship, etc.


@dataclass
class DisturbanceMemoryState:
    """
    State container for disturbance memory system.

    Stores per-agent disturbance memory in vectorized arrays.
    """
    # Memory grid per agent (flattened: agent_idx * grid_size + cell_idx)
    # For efficiency, we store as sparse dict per agent
    memory_grids: List[Dict[int, float]]  # List of {cell_id: remembered_intensity}

    # Summary statistics per agent
    total_disturbance_exposure: np.ndarray  # Cumulative disturbance exposure
    disturbance_event_count: np.ndarray     # Number of disturbance events remembered
    last_disturbance_tick: np.ndarray       # Tick of most recent disturbance

    # Avoidance state
    avoidance_heading_bias: np.ndarray      # Bias direction away from remembered disturbances
    avoidance_strength: np.ndarray          # Current avoidance strength (0-1)

    @classmethod
    def create(cls, count: int) -> 'DisturbanceMemoryState':
        """Create memory state for count agents."""
        return cls(
            memory_grids=[{} for _ in range(count)],
            total_disturbance_exposure=np.zeros(count, dtype=np.float32),
            disturbance_event_count=np.zeros(count, dtype=np.int32),
            last_disturbance_tick=np.full(count, -9999, dtype=np.int32),
            avoidance_heading_bias=np.zeros(count, dtype=np.float32),
            avoidance_strength=np.zeros(count, dtype=np.float32),
        )


@dataclass
class DisturbanceMemoryContext:
    """
    Context for disturbance memory updates.

    Contains information about current disturbances.
    """
    # Current disturbance state
    is_disturbed: np.ndarray          # Currently under disturbance
    disturbance_intensity: np.ndarray  # Current disturbance intensity
    disturbance_x: np.ndarray         # X position of disturbance source
    disturbance_y: np.ndarray         # Y position of disturbance source

    # Position
    agent_x: np.ndarray               # Agent X positions
    agent_y: np.ndarray               # Agent Y positions

    # Time
    current_tick: int                 # Current simulation tick

    @classmethod
    def create_default(cls, count: int, tick: int = 0) -> 'DisturbanceMemoryContext':
        """Create default context."""
        return cls(
            is_disturbed=np.zeros(count, dtype=bool),
            disturbance_intensity=np.zeros(count, dtype=np.float32),
            disturbance_x=np.zeros(count, dtype=np.float32),
            disturbance_y=np.zeros(count, dtype=np.float32),
            agent_x=np.zeros(count, dtype=np.float32),
            agent_y=np.zeros(count, dtype=np.float32),
            current_tick=tick,
        )


@dataclass
class AvoidanceResult:
    """
    Result of avoidance calculation.

    Contains movement bias from remembered disturbances.
    """
    avoidance_dx: np.ndarray      # X component of avoidance vector
    avoidance_dy: np.ndarray      # Y component of avoidance vector
    avoidance_strength: np.ndarray  # Strength of avoidance (0-1)
    cells_avoided: np.ndarray     # Number of cells being avoided


class DisturbanceMemoryModule(ABC):
    """
    Abstract base class for disturbance memory modules.

    Defines the interface for memory tracking and avoidance behavior.
    """

    def __init__(self, params: 'SimulationParameters'):
        """Initialize memory module."""
        self.params = params

    @abstractmethod
    def record_disturbance(
        self,
        state: DisturbanceMemoryState,
        context: DisturbanceMemoryContext,
        mask: np.ndarray,
    ) -> None:
        """Record disturbance events in memory."""
        pass

    @abstractmethod
    def decay_memory(
        self,
        state: DisturbanceMemoryState,
        mask: np.ndarray,
        ticks_elapsed: int = 1,
    ) -> None:
        """Apply memory decay."""
        pass

    @abstractmethod
    def compute_avoidance(
        self,
        state: DisturbanceMemoryState,
        agent_x: np.ndarray,
        agent_y: np.ndarray,
        mask: np.ndarray,
    ) -> AvoidanceResult:
        """Compute avoidance vectors based on memory."""
        pass

    @abstractmethod
    def get_mode(self) -> MemoryMode:
        """Return the memory mode."""
        pass

    def get_statistics(
        self,
        state: DisturbanceMemoryState,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        """Get memory statistics for reporting."""
        active = mask
        if not np.any(active):
            return {}

        return {
            'mean_exposure': float(np.mean(state.total_disturbance_exposure[active])),
            'mean_event_count': float(np.mean(state.disturbance_event_count[active])),
            'mean_avoidance_strength': float(np.mean(state.avoidance_strength[active])),
            'agents_with_memory': int(np.sum(state.disturbance_event_count[active] > 0)),
        }


class DEPONSMemoryModule(DisturbanceMemoryModule):
    """
    DEPONS memory module - no persistent disturbance memory.

    In DEPONS mode, deterrence is stateless and agents don't remember
    past disturbances. This module provides a no-op implementation.
    """

    def record_disturbance(
        self,
        state: DisturbanceMemoryState,
        context: DisturbanceMemoryContext,
        mask: np.ndarray,
    ) -> None:
        """No-op: DEPONS doesn't track disturbance memory."""
        pass

    def decay_memory(
        self,
        state: DisturbanceMemoryState,
        mask: np.ndarray,
        ticks_elapsed: int = 1,
    ) -> None:
        """No-op: DEPONS doesn't have memory decay."""
        pass

    def compute_avoidance(
        self,
        state: DisturbanceMemoryState,
        agent_x: np.ndarray,
        agent_y: np.ndarray,
        mask: np.ndarray,
    ) -> AvoidanceResult:
        """Return zero avoidance (DEPONS uses immediate deterrence only)."""
        count = len(agent_x)
        return AvoidanceResult(
            avoidance_dx=np.zeros(count, dtype=np.float32),
            avoidance_dy=np.zeros(count, dtype=np.float32),
            avoidance_strength=np.zeros(count, dtype=np.float32),
            cells_avoided=np.zeros(count, dtype=np.int32),
        )

    def get_mode(self) -> MemoryMode:
        return MemoryMode.DEPONS


class JASMINEMemoryModule(DisturbanceMemoryModule):
    """
    JASMINE memory module - full disturbance memory with learned avoidance.

    Implements:
    - Spatial memory grid for disturbance zones
    - Configurable memory decay (exponential)
    - Avoidance behavior based on remembered disturbances
    - Habituation (reduced response to repeated exposure)
    """

    # Memory grid parameters (same as PSM for consistency)
    MEMORY_CELL_SIZE = 5  # Grid cells per memory cell (2km for 400m cells)

    # Decay parameters
    DEFAULT_DECAY_RATE = 0.001     # Per-tick decay (slow decay over ~1000 ticks)
    DEFAULT_DECAY_HALF_LIFE = 720  # ~15 days (720 ticks) for 50% decay

    # Avoidance parameters
    AVOIDANCE_RADIUS = 20          # Memory cells to consider for avoidance
    AVOIDANCE_THRESHOLD = 0.1     # Minimum memory strength to trigger avoidance
    MAX_AVOIDANCE_STRENGTH = 0.8  # Maximum avoidance influence on movement

    # Habituation parameters
    HABITUATION_RATE = 0.05       # Rate of habituation per exposure
    MIN_RESPONSE = 0.2            # Minimum response after habituation

    def __init__(self, params: 'SimulationParameters'):
        super().__init__(params)

        # Extract parameters
        self.decay_rate = getattr(params, 'memory_decay_rate', self.DEFAULT_DECAY_RATE)
        self.avoidance_radius = getattr(params, 'avoidance_radius', self.AVOIDANCE_RADIUS)
        self.habituation_enabled = getattr(params, 'habituation_enabled', True)

        # Grid dimensions (will be set on first use)
        self._cells_per_row = 0
        self._cells_per_col = 0

    def _position_to_cell(self, x: float, y: float) -> int:
        """Convert position to memory cell index."""
        mem_x = int(x) // self.MEMORY_CELL_SIZE
        mem_y = int(y) // self.MEMORY_CELL_SIZE
        mem_x = max(0, min(mem_x, self._cells_per_row - 1))
        mem_y = max(0, min(mem_y, self._cells_per_col - 1))
        return mem_y * self._cells_per_row + mem_x

    def _cell_to_position(self, cell_id: int) -> Tuple[float, float]:
        """Convert memory cell to center position."""
        mem_x = cell_id % self._cells_per_row
        mem_y = cell_id // self._cells_per_row
        x = (mem_x + 0.5) * self.MEMORY_CELL_SIZE
        y = (mem_y + 0.5) * self.MEMORY_CELL_SIZE
        return x, y

    def _ensure_grid_dims(self, max_x: float, max_y: float) -> None:
        """Ensure grid dimensions are set."""
        if self._cells_per_row == 0:
            self._cells_per_row = int(max_x / self.MEMORY_CELL_SIZE) + 1
            self._cells_per_col = int(max_y / self.MEMORY_CELL_SIZE) + 1

    def record_disturbance(
        self,
        state: DisturbanceMemoryState,
        context: DisturbanceMemoryContext,
        mask: np.ndarray,
    ) -> None:
        """
        Record disturbance events in spatial memory.

        Each agent maintains a sparse grid of remembered disturbance intensities.
        """
        # Ensure grid dimensions
        if np.any(mask):
            max_x = np.max(context.agent_x[mask]) + 100
            max_y = np.max(context.agent_y[mask]) + 100
            self._ensure_grid_dims(max_x, max_y)

        # Find disturbed agents
        disturbed = mask & context.is_disturbed
        disturbed_indices = np.where(disturbed)[0]

        for idx in disturbed_indices:
            # Get disturbance source location
            dist_x = context.disturbance_x[idx]
            dist_y = context.disturbance_y[idx]
            intensity = context.disturbance_intensity[idx]

            # Convert to memory cell
            cell_id = self._position_to_cell(dist_x, dist_y)

            # Update memory grid (max of current and new)
            current = state.memory_grids[idx].get(cell_id, 0.0)
            state.memory_grids[idx][cell_id] = max(current, intensity)

            # Update statistics
            state.total_disturbance_exposure[idx] += intensity
            state.disturbance_event_count[idx] += 1
            state.last_disturbance_tick[idx] = context.current_tick

    def decay_memory(
        self,
        state: DisturbanceMemoryState,
        mask: np.ndarray,
        ticks_elapsed: int = 1,
    ) -> None:
        """
        Apply exponential decay to disturbance memories.

        Memory strength decreases over time, allowing agents to
        eventually return to areas where disturbances have stopped.
        
        Optimized: batch-processes all agents' grids by collecting
        entries into flat arrays, applying vectorized decay, and
        writing back.
        """
        decay_factor = (1.0 - self.decay_rate) ** ticks_elapsed
        threshold = 0.01  # Remove very weak memories

        active_indices = np.where(mask)[0]

        # --- Batch approach: collect all (agent, cell, strength) triples ---
        # For small agent counts, the per-dict loop is fine.
        # For large counts, we pre-allocate and vectorize the decay.
        total_entries = sum(len(state.memory_grids[idx]) for idx in active_indices)
        
        if total_entries == 0:
            state.avoidance_strength[mask] *= decay_factor
            return
        
        if total_entries < 500:
            # Small scale: simple loop is faster than array overhead
            for idx in active_indices:
                grid = state.memory_grids[idx]
                if not grid:
                    continue
                cells_to_remove = []
                for cell_id, strength in grid.items():
                    new_strength = strength * decay_factor
                    if new_strength < threshold:
                        cells_to_remove.append(cell_id)
                    else:
                        grid[cell_id] = new_strength
                for cell_id in cells_to_remove:
                    del grid[cell_id]
        else:
            # Large scale: vectorized decay
            strengths = np.empty(total_entries, dtype=np.float64)
            agent_ids = np.empty(total_entries, dtype=np.int32)
            cell_ids = np.empty(total_entries, dtype=np.int32)
            
            pos = 0
            for idx in active_indices:
                grid = state.memory_grids[idx]
                n = len(grid)
                if n == 0:
                    continue
                keys = list(grid.keys())
                vals = list(grid.values())
                agent_ids[pos:pos+n] = idx
                cell_ids[pos:pos+n] = keys
                strengths[pos:pos+n] = vals
                pos += n
            
            # Vectorized decay
            strengths[:pos] *= decay_factor
            
            # Rebuild grids, filtering below threshold
            survive = strengths[:pos] >= threshold
            
            # Clear all active grids and rebuild
            for idx in active_indices:
                state.memory_grids[idx].clear()
            
            surviving_agents = agent_ids[:pos][survive]
            surviving_cells = cell_ids[:pos][survive]
            surviving_strengths = strengths[:pos][survive]
            
            for i in range(len(surviving_agents)):
                state.memory_grids[surviving_agents[i]][surviving_cells[i]] = surviving_strengths[i]

        # Decay avoidance strength
        state.avoidance_strength[mask] *= decay_factor

    def compute_avoidance(
        self,
        state: DisturbanceMemoryState,
        agent_x: np.ndarray,
        agent_y: np.ndarray,
        mask: np.ndarray,
    ) -> AvoidanceResult:
        """
        Compute avoidance vectors based on remembered disturbances.

        Agents are biased away from areas where they remember
        experiencing disturbances.
        
        Optimized: per-agent cell processing is vectorized with numpy
        array operations on (cell_ids, strengths) instead of Python loops.
        """
        count = len(agent_x)

        avoidance_dx = np.zeros(count, dtype=np.float32)
        avoidance_dy = np.zeros(count, dtype=np.float32)
        avoidance_strength = np.zeros(count, dtype=np.float32)
        cells_avoided = np.zeros(count, dtype=np.int32)

        # Ensure grid dimensions
        if np.any(mask):
            max_x = np.max(agent_x[mask]) + 100
            max_y = np.max(agent_y[mask]) + 100
            self._ensure_grid_dims(max_x, max_y)

        active_indices = np.where(mask)[0]
        max_dist = self.avoidance_radius * self.MEMORY_CELL_SIZE
        cell_size = self.MEMORY_CELL_SIZE

        for idx in active_indices:
            grid = state.memory_grids[idx]
            if not grid:
                continue

            # Vectorized: extract all cell ids and strengths at once
            cell_ids_arr = np.fromiter(grid.keys(), dtype=np.int32, count=len(grid))
            strengths_arr = np.fromiter(grid.values(), dtype=np.float64, count=len(grid))

            # Filter by threshold
            above_thresh = strengths_arr >= self.AVOIDANCE_THRESHOLD
            if not np.any(above_thresh):
                continue

            cell_ids_arr = cell_ids_arr[above_thresh]
            strengths_arr = strengths_arr[above_thresh]

            # Vectorized cell_to_position
            mem_x = cell_ids_arr % self._cells_per_row
            mem_y = cell_ids_arr // self._cells_per_row
            cx = (mem_x + 0.5) * cell_size
            cy = (mem_y + 0.5) * cell_size

            # Vectorized distance from agent to each cell
            ax = agent_x[idx]
            ay = agent_y[idx]
            dx = cx - ax
            dy = cy - ay
            dist = np.sqrt(dx * dx + dy * dy)

            # Filter by distance
            in_range = (dist <= max_dist) & (dist > 0.1)
            if not np.any(in_range):
                continue

            dx = dx[in_range]
            dy = dy[in_range]
            dist = dist[in_range]
            s = strengths_arr[in_range]

            # Vectorized weight computation
            weights = s / (dist + 1.0)
            inv_dist = 1.0 / dist

            # Direction AWAY from disturbance (negative)
            total_weight = np.sum(weights)
            sum_dx = -np.sum(weights * dx * inv_dist)
            sum_dy = -np.sum(weights * dy * inv_dist)

            # Normalize and scale
            avoidance_dx[idx] = sum_dx / total_weight
            avoidance_dy[idx] = sum_dy / total_weight
            avoidance_strength[idx] = min(float(total_weight), self.MAX_AVOIDANCE_STRENGTH)
            cells_avoided[idx] = int(np.sum(in_range))

        # Update state with computed avoidance
        state.avoidance_strength[mask] = avoidance_strength[mask]

        return AvoidanceResult(
            avoidance_dx=avoidance_dx,
            avoidance_dy=avoidance_dy,
            avoidance_strength=avoidance_strength,
            cells_avoided=cells_avoided,
        )

    def get_mode(self) -> MemoryMode:
        return MemoryMode.JASMINE

    def get_avoidance_map(
        self,
        state: DisturbanceMemoryState,
        agent_idx: int,
    ) -> Dict[Tuple[float, float], float]:
        """
        Get remembered disturbance locations for visualization.

        Returns dict of {(x, y): strength} for the specified agent.
        """
        result = {}
        grid = state.memory_grids[agent_idx]

        for cell_id, strength in grid.items():
            x, y = self._cell_to_position(cell_id)
            result[(x, y)] = strength

        return result


def create_memory_module(
    params: 'SimulationParameters',
    mode: MemoryMode = MemoryMode.DEPONS,
) -> DisturbanceMemoryModule:
    """
    Factory function to create appropriate memory module.

    Args:
        params: Simulation parameters
        mode: Memory mode (DEPONS or JASMINE)

    Returns:
        Configured DisturbanceMemoryModule instance
    """
    if mode == MemoryMode.DEPONS:
        return DEPONSMemoryModule(params)
    elif mode == MemoryMode.JASMINE:
        return JASMINEMemoryModule(params)
    else:
        # Default to JASMINE for hybrid/unknown
        return JASMINEMemoryModule(params)
