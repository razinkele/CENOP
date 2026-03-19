"""
Vectorized Porpoise Population Manager.

This module implements a Structure-of-Arrays (SoA) approach to managing
the porpoise population efficiently using NumPy.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict

from cenop.parameters.simulation_params import SimulationParameters
from cenop.landscape.cell_data import CellData
from cenop.parameters.demography import AGE_DISTRIBUTION_FREQUENCY
from cenop.behavior.psm import PersistentSpatialMemory
from cenop.behavior.sound import calculate_received_level, response_probability_from_rl

import logging
import os

# Import scipy.spatial at module level for performance
try:
    from scipy.spatial import cKDTree as _cKDTree
    _HAS_SCIPY = True
except ImportError:
    _cKDTree = None
    _HAS_SCIPY = False

# Import numba helpers at module level
try:
    from cenop.optimizations.numba_helpers import accumulate_social_totals as _accumulate_social_totals
    from cenop.optimizations.numba_helpers import weighted_direction_sum as _weighted_direction_sum
    _HAS_NUMBA_HELPERS = True
except ImportError:
    _accumulate_social_totals = None
    _weighted_direction_sum = None
    _HAS_NUMBA_HELPERS = False

try:
    from cenop.optimizations.kernels import reflect_boundaries_kernel as _reflect_kernel
    from cenop.optimizations.kernels import crw_angle_step_kernel as _crw_kernel
    from cenop.optimizations.kernels import seed_numba_rng as _seed_numba_rng
    from cenop.optimizations.kernels import turn_position_kernel as _turn_kernel
    from cenop.optimizations.kernels import social_accumulate_kernel as _social_kernel
    _HAS_KERNELS = True
except ImportError:
    _HAS_KERNELS = False

try:
    from cenop.optimizations.kernels import land_avoidance_kernel as _land_avoidance_kernel
    _HAS_LAND_KERNEL = True
except ImportError:
    _land_avoidance_kernel = None
    _HAS_LAND_KERNEL = False

logger = logging.getLogger('cenop.agents.population')


class PorpoisePopulation:
    """
    Manages the entire population of porpoises using vectorized numpy arrays.
    Replaces the list of individual Porpoise objects for performance.
    """
    
    def __init__(self, count: int, params: SimulationParameters, landscape: Optional[CellData] = None,
                 movement_module=None, behavior_fsm=None, energy_module=None, memory_module=None):
        self.params = params
        self.landscape = landscape
        self.count = count # Initial count capacity

        # JASMINE integration modules (Phase 2-5)
        self._movement_module = movement_module
        self._behavior_fsm = behavior_fsm
        self._energy_module = energy_module
        self._memory_module = memory_module
        self._behavior_state = None
        
        # === Arrays (Structure of Arrays) ===
        # Use a dictionary of arrays or direct attributes? Direct attributes are faster.
        
        # Identity
        self.ids = np.arange(count, dtype=np.int32)
        self.active_mask = np.ones(count, dtype=bool) # True if alive/active slot
        
        # Position
        self.x = np.zeros(count, dtype=np.float32)
        self.y = np.zeros(count, dtype=np.float32)
        self.heading = np.zeros(count, dtype=np.float32)
        
        # Movement State
        self.prev_log_mov = np.full(count, 0.8, dtype=np.float64)
        self.prev_angle = np.full(count, 10.0, dtype=np.float64)
        
        # Demography
        self.is_female = np.zeros(count, dtype=bool)
        self.age = np.zeros(count, dtype=np.float32)
        
        # Energy
        self.energy = np.random.normal(
            params.energy_init_mean, params.energy_init_sd, count
        ).clip(0, 20).astype(np.float32)
        
        # Reproduction
        self.mating_day = np.full(count, -99, dtype=np.int16)
        self.days_since_mating = np.full(count, -99, dtype=np.int16)
        self.days_since_birth = np.full(count, -99, dtype=np.int16)
        self.with_calf = np.zeros(count, dtype=bool)
        # Pregnancy FSM (DEPONS 3.2: 0=immature, 1=pregnant, 2=ready-to-mate)
        self.pregnancy_status = np.zeros(count, dtype=np.int8)
        
        # Deterrence status
        self.deter_strength = np.zeros(count, dtype=np.float32)
        # Tracks any porpoise deterred at least once during the reporting period
        self._was_deterred = np.zeros(count, dtype=bool)

        # Reference memory circular buffers (DEPONS 3.2 — 120 entries per agent)
        _REF_MEM_SIZE = params.ref_mem_size if hasattr(params, 'ref_mem_size') else 120
        self._stored_util = np.zeros((count, _REF_MEM_SIZE), dtype=np.float32)
        self._pos_history_x = np.zeros((count, _REF_MEM_SIZE), dtype=np.float32)
        self._pos_history_y = np.zeros((count, _REF_MEM_SIZE), dtype=np.float32)
        self._mem_ptr = np.zeros(count, dtype=np.int32)   # Current write index
        self._mem_count = np.zeros(count, dtype=np.int32)  # Entries stored
        self._ve_total = np.zeros(count, dtype=np.float32)  # Expected food value
        self._vt_x = np.zeros(count, dtype=np.float32)  # Attraction vector x
        self._vt_y = np.zeros(count, dtype=np.float32)  # Attraction vector y

        # === PSM and Dispersal State (Phase 2) ===
        # Energy history for dispersal trigger (5 days = 5*48 ticks)
        self._energy_history = np.zeros((count, 10), dtype=np.float32)  # Last 10 daily averages (need 8 for energy-based stop)
        self._energy_ticks_today = np.zeros(count, dtype=np.float32)   # Energy sum for current day
        self._tick_counter = 0  # Track ticks for daily updates
        self._last_energy_update_tick = -1  # last global tick when energy was accumulated

        # Per-step energy metrics (exposed for dashboard)
        self.avg_food_gained = 0.0   # Average food gained per active agent (last step)
        self.avg_energy_cost = 0.0   # Average energy cost per active agent (last step)
        
        # Dispersal state
        self.is_dispersing = np.zeros(count, dtype=bool)
        self.days_declining_energy = np.zeros(count, dtype=np.int16)
        self.dispersal_target_x = np.zeros(count, dtype=np.float32)
        self.dispersal_target_y = np.zeros(count, dtype=np.float32)
        self.dispersal_target_distance = np.zeros(count, dtype=np.float32)
        self.dispersal_distance_traveled = np.zeros(count, dtype=np.float32)
        self.dispersal_start_x = np.zeros(count, dtype=np.float32)
        self.dispersal_start_y = np.zeros(count, dtype=np.float32)
        self._prev_step_heading = np.zeros(count, dtype=np.float32)

        # PSM instances - one per porpoise (list for object storage)
        world_w = self.params.world_width
        world_h = self.params.world_height
        if landscape is not None:
            world_w = landscape.width
            world_h = landscape.height
        
        # Store basic PSM config per agent (preferred distance)
        # We can still use the class for helper methods or just store distances array
        # For full optimization, we replace list of objects with arrays
        self._psm_instances: List[PersistentSpatialMemory] = [
             PersistentSpatialMemory(world_w, world_h) for _ in range(count)
        ]
        
        # Vectorized PSM Storage (Optimized)
        # Shape: (count, grid_h, grid_w, 2) where last dim is [ticks, food]
        # Grid size is roughly width/5
        self.psm_cell_size = 5
        self.psm_cols = world_w // self.psm_cell_size
        self.psm_rows = world_h // self.psm_cell_size
        self.psm_buffer = np.zeros((count, self.psm_rows, self.psm_cols, 2), dtype=np.float32)
        
        # Initialize
        self._initialize_population()

        # Initialize JASMINE module states
        self._energy_state = None
        self._memory_state = None
        self._movement_state = None
        self._avoidance_result = None

        if self._behavior_fsm is not None:
            from cenop.behavior.states import BehaviorStateVector
            self._behavior_state = BehaviorStateVector.create(count)

        if self._energy_module is not None:
            from cenop.physiology.energy_budget import EnergyState
            self._energy_state = EnergyState.create(count, initial_energy=10.0)
            # Share energy array: eliminates 3 full-array sync copies per tick
            self._energy_state.energy = self.energy

        if self._memory_module is not None:
            from cenop.behavior.disturbance_memory import DisturbanceMemoryState
            self._memory_state = DisturbanceMemoryState.create(count)

        if self._movement_module is not None:
            from cenop.movement.base import MovementState
            self._movement_state = MovementState.create(count)

        # Random generator for reproducibility (per-instance)
        # Use SimulationParameters.random_seed when available
        seed = getattr(self.params, 'random_seed', None)
        self.rng = np.random.default_rng(seed)

        # Instrumentation controls: set via params.debug_instrumentation or env var CENOP_INSTRUMENT
        self._debug_instrumentation = bool(getattr(self.params, 'debug_instrumentation', False) or os.getenv('CENOP_INSTRUMENT', '0').lower() in ('1','true','yes'))
        self._instrument_events: list = []
        # Global tick counter for instrumentation logs (incremented each step)
        self._global_tick: int = 0

        # Cache for neighbor topology used by social communication
        # Stores: {'idx_i': idx_i, 'idx_j': idx_j, 'ncols': ncols, 'active_len': len(active_idx)}
        self._social_cache: dict | None = None
        # Counter (ticks) until next recompute; 0 forces recompute now
        self._neighbor_recompute_counter: int = 0
        # Current recompute interval (may adapt over time)
        self._current_recompute_interval: int = max(1, int(getattr(self.params, 'communication_recompute_interval', 4)))
        
        # Cache communication parameters to avoid repeated getattr calls
        self._comm_enabled: bool = bool(getattr(self.params, 'communication_enabled', False))
        self._comm_range_km: float = float(getattr(self.params, 'communication_range_km', 10.0))
        self._comm_cells: int = max(1, int(np.ceil((self._comm_range_km * 1000.0) / 400.0)))
        self._comm_source_level: float = float(getattr(self.params, 'communication_source_level', 160.0))
        self._comm_threshold: float = float(getattr(self.params, 'communication_threshold', 120.0))
        self._comm_slope: float = float(getattr(self.params, 'communication_response_slope', 0.2))
        self._social_weight: float = float(getattr(self.params, 'social_weight', 0.3))
        # Previous positions for displacement calculation (in cell units)
        self._prev_x = self.x.copy()
        self._prev_y = self.y.copy()
        # EMA of mean displacement (meters per tick)
        self._disp_ema_m: float = 0.0

        # === Pre-allocated arrays for step() to avoid GC pressure ===
        # These are reused each tick instead of creating new arrays
        self._rand_angle = np.zeros(count, dtype=np.float64)
        self._rand_len = np.zeros(count, dtype=np.float64)
        self._pres_angle = np.zeros(count, dtype=np.float64)
        self._log_mov = np.zeros(count, dtype=np.float64)
        self._step_dist = np.zeros(count, dtype=np.float32)
        self._rads = np.zeros(count, dtype=np.float32)
        self._dx = np.zeros(count, dtype=np.float32)
        self._dy = np.zeros(count, dtype=np.float32)
        self._new_x = np.zeros(count, dtype=np.float32)
        self._new_y = np.zeros(count, dtype=np.float32)
        self._new_xi = np.zeros(count, dtype=np.int32)
        self._new_yi = np.zeros(count, dtype=np.int32)
        self._depths = np.zeros(count, dtype=np.float64)
        self._salinity_vals = np.zeros(count, dtype=np.float64)  # For CRW environmental modulation
        self._env_mod_angle = np.zeros(count, dtype=np.float32)  # b1*depth + b2*salinity + b3
        self._scaling_factor = np.zeros(count, dtype=np.float32)

        # === Pre-allocated reference memory workspace ===
        _REF_MEM_SIZE_INIT = params.ref_mem_size if hasattr(params, 'ref_mem_size') else 120
        from cenop.behavior.ref_mem import RefMemWorkspace
        self._ref_mem_workspace = RefMemWorkspace.create(count, _REF_MEM_SIZE_INIT)

        # === Pre-allocated buffers to eliminate per-tick .copy() calls ===
        self._pre_move_x = np.zeros(count, dtype=np.float32)
        self._pre_move_y = np.zeros(count, dtype=np.float32)
        self._orig_dx = np.zeros(count, dtype=np.float32)
        self._orig_dy = np.zeros(count, dtype=np.float32)
        self._pre_heading = np.zeros(count, dtype=np.float32)
        self._positions = np.zeros((count, 2), dtype=np.float32)  # Reusable (N,2) buffer

        # === Pre-allocated float64 buffers for Numba turn_position_kernel ===
        self._f64_x = np.zeros(count, dtype=np.float64)
        self._f64_y = np.zeros(count, dtype=np.float64)
        self._f64_heading = np.zeros(count, dtype=np.float64)
        self._f64_step = np.zeros(count, dtype=np.float64)
        self._f64_out_x = np.zeros(count, dtype=np.float64)
        self._f64_out_y = np.zeros(count, dtype=np.float64)
        self._f64_out_heading = np.zeros(count, dtype=np.float64)
        self._int32_out_xi = np.zeros(count, dtype=np.int32)
        self._int32_out_yi = np.zeros(count, dtype=np.int32)
        # All-true mask for turn_position fallback path
        self._all_mask = np.ones(count, dtype=bool)

        # === Pre-allocated buffers for land avoidance kernel ===
        self._la_f64_x = np.empty(count, dtype=np.float64)
        self._la_f64_y = np.empty(count, dtype=np.float64)
        self._la_f64_heading = np.empty(count, dtype=np.float64)
        self._la_f64_step = np.empty(count, dtype=np.float64)
        self._la_out_x = np.empty(count, dtype=np.float64)
        self._la_out_y = np.empty(count, dtype=np.float64)
        self._la_out_heading = np.empty(count, dtype=np.float64)
        self._la_resolved = np.empty(count, dtype=np.bool_)

        # === Pre-allocated cell index buffers (D1: compute once per tick) ===
        self._cell_xi = np.zeros(count, dtype=np.int32)
        self._cell_yi = np.zeros(count, dtype=np.int32)

        # === Social kernel pre-allocated buffers (D2) ===
        self._social_ux = np.zeros(count, dtype=np.float64)
        self._social_uy = np.zeros(count, dtype=np.float64)
        self._social_sw = np.zeros(count, dtype=np.float64)
        self._social_out_dx = np.zeros(count, dtype=np.float32)
        self._social_out_dy = np.zeros(count, dtype=np.float32)
        # Pair-sized buffers (lazily grown)
        self._social_f64_dx = np.empty(0, dtype=np.float64)
        self._social_f64_dy = np.empty(0, dtype=np.float64)
        self._social_f64_dist = np.empty(0, dtype=np.float64)
        self._social_f64_pi = np.empty(0, dtype=np.float64)
        self._social_f64_pj = np.empty(0, dtype=np.float64)
        self._social_buf_size = 0

        # === Pre-allocated energy/context buffers ===
        self._water_temp = np.full(count, 10.0, dtype=np.float32)
        self._food_quality = np.ones(count, dtype=np.float32)
        self._behavioral_state_buf = np.full(
            count, 0, dtype=np.int32
        )  # 0 = FORAGING
        self._speed_ms = np.zeros(count, dtype=np.float32)

        # === Cached mortality constants (avoid per-tick getattr) ===
        self._m_mort_prob_const = getattr(params, 'm_mort_prob_const', 1.0)
        self._x_survival_const = getattr(params, 'x_survival_const', 0.4)

        # === Pre-allocated arrays for land avoidance loop ===
        self._on_land = np.zeros(count, dtype=bool)
        self._still_blocked = np.zeros(count, dtype=bool)
        self._right_x = np.zeros(count, dtype=np.float32)
        self._right_y = np.zeros(count, dtype=np.float32)
        self._left_x = np.zeros(count, dtype=np.float32)
        self._left_y = np.zeros(count, dtype=np.float32)
        self._right_xi = np.zeros(count, dtype=np.int32)
        self._right_yi = np.zeros(count, dtype=np.int32)
        self._left_xi = np.zeros(count, dtype=np.int32)
        self._left_yi = np.zeros(count, dtype=np.int32)
        self._right_depths = np.zeros(count, dtype=np.float32)
        self._left_depths = np.zeros(count, dtype=np.float32)

        # --- Glue-tax optimization: skip land avoidance on all-water landscapes ---
        self._skip_land_avoidance = False
        if self.landscape is None:
            self._skip_land_avoidance = True
        elif getattr(self.landscape, "landscape_name", "") == "Homogeneous":
            self._skip_land_avoidance = True
        elif (
            hasattr(self.landscape, "_depth")
            and self.landscape._depth is not None
        ):
            depth = self.landscape._depth
            has_land = np.any(np.isnan(depth))
            if not has_land:
                min_depth = self.params.min_depth if self.params else 1.0
                self._skip_land_avoidance = bool(
                    np.all(depth >= min_depth)
                )

        # Compute initial cell indices from position arrays
        self._recompute_cell_indices()

    def _recompute_cell_indices(self) -> None:
        """Recompute clamped int32 cell indices from current x/y positions."""
        if self.landscape is not None:
            w = self.landscape.width
            h = self.landscape.height
        else:
            w = self.params.world_width
            h = self.params.world_height
        np.copyto(self._cell_xi, np.clip(self.x.astype(np.int32), 0, w - 1))
        np.copyto(self._cell_yi, np.clip(self.y.astype(np.int32), 0, h - 1))

    @property
    def population_size(self) -> int:
        """Current number of living porpoises."""
        return np.sum(self.active_mask)
        
    def _initialize_population(self):
        """Vectorized initialization logic with land avoidance."""
        # Random positions - must place in water (depth > 0)
        world_w = self.params.world_width
        world_h = self.params.world_height
        
        if self.landscape is None:
            # No landscape - use simple random positions
            self.x = np.random.uniform(0, world_w, self.count).astype(np.float32)
            self.y = np.random.uniform(0, world_h, self.count).astype(np.float32)
        else:
            # Use landscape - place only in water cells (depth >= min_depth)
            lw = self.landscape.width
            lh = self.landscape.height
            min_depth = self.params.min_depth if self.params else 1.0
            
            if hasattr(self.landscape, '_depth') and self.landscape._depth is not None:
                # Find all valid water cells (depth >= min_depth AND not NaN)
                # NaN values indicate land (-9999 NODATA converted to NaN during loading)
                valid_mask = (self.landscape._depth >= min_depth) & ~np.isnan(self.landscape._depth)
                valid_y, valid_x = np.where(valid_mask)
                
                if len(valid_x) > 0:
                    # Randomly select from valid positions
                    indices = np.random.choice(len(valid_x), self.count, replace=True)
                    self.x = valid_x[indices].astype(np.float32) + np.random.uniform(0, 1, self.count).astype(np.float32)
                    self.y = valid_y[indices].astype(np.float32) + np.random.uniform(0, 1, self.count).astype(np.float32)
                else:
                    # Fallback - no valid water cells (shouldn't happen)
                    self.x = np.random.uniform(0, lw, self.count).astype(np.float32)
                    self.y = np.random.uniform(0, lh, self.count).astype(np.float32)
            else:
                # No depth data - use full area
                self.x = np.random.uniform(0, lw, self.count).astype(np.float32)
                self.y = np.random.uniform(0, lh, self.count).astype(np.float32)
            
        self.heading = np.random.uniform(0, 360, self.count).astype(np.float32)
        
        # Sex ratio 50%
        self.is_female = np.random.choice([True, False], self.count)
        
        # Ages from distribution
        self.age = np.random.choice(
            AGE_DISTRIBUTION_FREQUENCY, 
            size=self.count
        ).astype(np.float32)

        # --- Social communication implementation (vectorized neighborhood search) ---
        def _ensure_social_buffers(self, n_pairs: int) -> None:
            """Grow pair-sized social buffers if needed (never shrink)."""
            if self._social_buf_size >= n_pairs:
                return
            self._social_f64_dx = np.empty(n_pairs, dtype=np.float64)
            self._social_f64_dy = np.empty(n_pairs, dtype=np.float64)
            self._social_f64_dist = np.empty(n_pairs, dtype=np.float64)
            self._social_f64_pi = np.empty(n_pairs, dtype=np.float64)
            self._social_f64_pj = np.empty(n_pairs, dtype=np.float64)
            self._social_buf_size = n_pairs

        self._ensure_social_buffers = _ensure_social_buffers.__get__(self, self.__class__)

        def _compute_social_vectors(self, mask: np.ndarray, ambient_rl: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute social attraction vectors for active agents.

            Implementation uses a cKDTree-based neighbor lookup (fast C implementation) when
            SciPy is available, otherwise falls back to the previous binning approach.
            Detection is still probabilistic and masked by ambient noise (SNR).
            """
            self._social_out_dx.fill(0.0)
            self._social_out_dy.fill(0.0)
            social_dx = self._social_out_dx
            social_dy = self._social_out_dy

            if not self._comm_enabled:
                return social_dx, social_dy

            # Use cached communication parameters
            comm_cells = self._comm_cells
            source_level = self._comm_source_level
            threshold = self._comm_threshold
            slope = self._comm_slope
            social_weight = self._social_weight

            active_idx = np.where(mask)[0]
            if len(active_idx) == 0:
                return social_dx, social_dy

            if _HAS_SCIPY:
                # Build KD-tree in cell units (consistent with existing code)
                positions = np.column_stack((self.x, self.y))
                radius = float(comm_cells)

                # Determine whether to rebuild neighbor topology this tick
                interval = self._current_recompute_interval
                rebuild = False
                if self._social_cache is None:
                    rebuild = True
                elif self._neighbor_recompute_counter <= 0:
                    rebuild = True
                elif len(active_idx) != self._social_cache.get('active_len', -1):
                    # If active set size changed (births/deaths), rebuild
                    rebuild = True

                if rebuild:
                    pos_active = positions[active_idx]
                    kd_active = _cKDTree(pos_active)

                    try:
                        # Use query_pairs with output_type='ndarray' for maximum speed
                        # Returns (N, 2) array of indices into pos_active
                        pairs = kd_active.query_pairs(radius, output_type='ndarray')
                    except (TypeError, ValueError) as e:
                        # Fallback for older scipy versions that don't support output_type
                        logger.debug("query_pairs fallback: %s", e)
                        pairs = np.array([], dtype=np.int32).reshape(0, 2)
                        
                        try:
                            # Fallback to query_ball_tree only if query_pairs failed
                            neigh_lists = kd_active.query_ball_tree(kd_active, r=radius)
                             # Build canonical pairs using pre-allocation
                            # First pass: count pairs where j > i
                            total_pairs = 0
                            for i_local, neigh in enumerate(neigh_lists):
                                for j_local in neigh:
                                    if j_local > i_local:
                                        total_pairs += 1
                            
                            if total_pairs > 0:
                                rows_fb = np.empty(total_pairs, dtype=np.int32)
                                cols_fb = np.empty(total_pairs, dtype=np.int32)
                                pair_idx = 0
                                for i_local, neigh in enumerate(neigh_lists):
                                    for j_local in neigh:
                                        if j_local > i_local:
                                            rows_fb[pair_idx] = i_local
                                            cols_fb[pair_idx] = j_local
                                            pair_idx += 1
                                pairs = np.column_stack((rows_fb, cols_fb))
                        except (TypeError, ValueError) as e:
                            logger.debug("query_ball_tree fallback: %s", e)
                            pairs = np.empty((0, 2), dtype=np.int32)

                    if pairs.shape[0] == 0:
                        # Reset cache counter to avoid repeated work
                        self._neighbor_recompute_counter = interval
                        self._social_cache = {'idx_i': np.array([], dtype=np.int64), 'idx_j': np.array([], dtype=np.int64), 'ncols': 0, 'active_len': len(active_idx)}
                        return social_dx, social_dy

                    # Extract rows and cols directly
                    rows = pairs[:, 0]
                    cols = pairs[:, 1]

                    # Map to global indices and cache
                    # Use int64 for indices to be compatible with Numba helper default expectations if any
                    idx_i = active_idx[rows].astype(np.int64)
                    idx_j = active_idx[cols].astype(np.int64)
                    ncols = len(idx_i)

                    self._social_cache = {
                        'idx_i': idx_i,
                        'idx_j': idx_j,
                        'ncols': ncols,
                        'active_len': len(active_idx)
                    }

                    # Reset counter
                    self._neighbor_recompute_counter = interval
                else:
                    # Reuse cached topology
                    idx_i = self._social_cache['idx_i']
                    idx_j = self._social_cache['idx_j']
                    ncols = self._social_cache['ncols']

                if ncols == 0:
                    # Nothing to do
                    self._neighbor_recompute_counter -= 1
                    return social_dx, social_dy

                # Coordinates - use float32 to save memory bandwidth (precision is sufficient for agents)
                # Avoid copy if possible, but indexing creates copy anyway
                xi = self.x[idx_i] # already float32
                yi = self.y[idx_i]
                xj = self.x[idx_j]
                yj = self.y[idx_j]

                # Displacements and distances recomputed each tick (topology reused)
                dx_ij = xj - xi
                dy_ij = yj - yi
                dist = np.hypot(dx_ij, dy_ij) + 1e-6
                dist_m = dist * np.float32(400.0)

                # Received level (same for both directions since distance symmetric)
                rl_pairs = calculate_received_level(source_level, dist_m, self.params.alpha_hat, self.params.beta_hat)

                # Probabilities: listener i hearing caller j
                if ambient_rl is not None:
                    ambient_i = np.asarray(ambient_rl[idx_i], dtype=np.float32)
                    snr_i = rl_pairs - ambient_i
                    p_i = response_probability_from_rl(snr_i, threshold, slope)
                    ambient_j = np.asarray(ambient_rl[idx_j], dtype=np.float32)
                    snr_j = rl_pairs - ambient_j
                    p_j = response_probability_from_rl(snr_j, threshold, slope)
                else:
                    p_i = response_probability_from_rl(rl_pairs, threshold, slope)
                    p_j = response_probability_from_rl(rl_pairs, threshold, slope)

                # Accumulate per-agent social vectors (unit-vector + weighting + accumulation)
                self._social_ux.fill(0.0)
                self._social_uy.fill(0.0)
                self._social_sw.fill(0.0)
                ux_total = self._social_ux
                uy_total = self._social_uy
                sw_total = self._social_sw

                if _HAS_KERNELS:
                    # Fused kernel: unit-vector, weighting, and accumulation in one pass
                    self._ensure_social_buffers(ncols)
                    self._social_f64_dx[:ncols] = dx_ij
                    self._social_f64_dy[:ncols] = dy_ij
                    self._social_f64_dist[:ncols] = dist
                    self._social_f64_pi[:ncols] = p_i
                    self._social_f64_pj[:ncols] = p_j
                    _social_kernel(
                        idx_i, idx_j,
                        self._social_f64_dx[:ncols],
                        self._social_f64_dy[:ncols],
                        self._social_f64_dist[:ncols],
                        self._social_f64_pi[:ncols],
                        self._social_f64_pj[:ncols],
                        ux_total, uy_total, sw_total,
                    )
                elif _HAS_NUMBA_HELPERS and _accumulate_social_totals is not None:
                    ux_ij = dx_ij / dist
                    uy_ij = dy_ij / dist
                    ux_contrib_i = ux_ij * p_i
                    uy_contrib_i = uy_ij * p_i
                    ux_contrib_j = -ux_ij * p_j
                    uy_contrib_j = -uy_ij * p_j
                    try:
                        _accumulate_social_totals(
                            np.int64(self.count), idx_i, idx_j,
                            ux_contrib_i, uy_contrib_i, ux_contrib_j, uy_contrib_j, p_i, p_j,
                            ux_total, uy_total, sw_total
                        )
                    except (TypeError, ValueError) as e:
                        logger.debug("Numba social accumulator fallback: %s", e)
                        ux_total = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([ux_contrib_i, ux_contrib_j]), minlength=self.count)
                        uy_total = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([uy_contrib_i, uy_contrib_j]), minlength=self.count)
                        sw_total = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([p_i, p_j]), minlength=self.count)
                else:
                    ux_ij = dx_ij / dist
                    uy_ij = dy_ij / dist
                    ux_contrib_i = ux_ij * p_i
                    uy_contrib_i = uy_ij * p_i
                    ux_contrib_j = -ux_ij * p_j
                    uy_contrib_j = -uy_ij * p_j
                    ux_total = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([ux_contrib_i, ux_contrib_j]), minlength=self.count)
                    uy_total = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([uy_contrib_i, uy_contrib_j]), minlength=self.count)
                    sw_total = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([p_i, p_j]), minlength=self.count)

                # Compute unit direction and scale for those agents that had any contribution
                has_signal = sw_total > 0
                norm = np.hypot(ux_total, uy_total)
                nonzero = (norm > 0) & has_signal

                unit_x = np.zeros(self.count, dtype=np.float64)
                unit_y = np.zeros(self.count, dtype=np.float64)
                unit_x[nonzero] = ux_total[nonzero] / norm[nonzero]
                unit_y[nonzero] = uy_total[nonzero] / norm[nonzero]

                # Step distances for active agents
                step_dist = (10.0 ** self.prev_log_mov) / 4.0

                # Apply social weight and step length into pre-allocated output
                self._social_out_dx[:] = (
                    unit_x * social_weight * step_dist
                ).astype(np.float32)
                self._social_out_dy[:] = (
                    unit_y * social_weight * step_dist
                ).astype(np.float32)
                social_dx = self._social_out_dx
                social_dy = self._social_out_dy

                # Zero out inactive slots
                social_dx[~mask] = 0.0
                social_dy[~mask] = 0.0

                # Decrement recompute counter (if positive) so we eventually rebuild
                self._neighbor_recompute_counter = max(0, self._neighbor_recompute_counter - 1)

                # Update EMA of displacement based on movement this tick (meters per tick)
                # Note: actual displacement update happens in step() after positions update; here we
                # rely on that value to adjust the recompute interval on next call.
                return social_dx, social_dy

            # Fallback: previous binning approach (keeps behavior stable when SciPy not available)
            bin_size = max(1, comm_cells)
            bins = {}
            xs = (self.x[active_idx].astype(int))
            ys = (self.y[active_idx].astype(int))
            bx = (xs // bin_size).astype(int)
            by = (ys // bin_size).astype(int)
            for idx, bxi, byi in zip(active_idx, bx, by):
                bins.setdefault((bxi, byi), []).append(int(idx))

            search_range = 1
            for idx in active_idx:
                x_i = float(self.x[idx])
                y_i = float(self.y[idx])
                bxi = int(x_i) // bin_size
                byi = int(y_i) // bin_size

                # Gather candidates from neighboring bins
                cand = []
                for nx in range(bxi - search_range, bxi + search_range + 1):
                    for ny in range(byi - search_range, byi + search_range + 1):
                        if (nx, ny) in bins:
                            cand.extend(bins[(nx, ny)])

                if not cand:
                    continue

                # Exclude self
                cand = [c for c in cand if c != idx]
                if not cand:
                    continue

                cand = np.array(cand, dtype=int)
                dxs = self.x[cand].astype(np.float32) - x_i
                dys = self.y[cand].astype(np.float32) - y_i
                dist_cells = np.sqrt(dxs * dxs + dys * dys)

                within = dist_cells <= comm_cells
                if not np.any(within):
                    continue

                dxs = dxs[within]
                dys = dys[within]
                dist_cells_in = dist_cells[within]
                dist_m = dist_cells_in * 400.0

                # Received levels from callers at listener
                rl = calculate_received_level(source_level, dist_m, self.params.alpha_hat, self.params.beta_hat)

                # Mask by ambient noise if provided (SNR)
                if ambient_rl is not None and idx < len(ambient_rl):
                    ambient_listener = float(ambient_rl[idx])
                    snr = rl - ambient_listener
                    p = response_probability_from_rl(snr, threshold, slope)
                else:
                    p = response_probability_from_rl(rl, threshold, slope)

                weights = np.array(p, dtype=np.float64)
                sum_w = np.sum(weights)
                if sum_w <= 0:
                    continue

                # Weighted direction (use numba helper if available)
                if _HAS_NUMBA_HELPERS and _weighted_direction_sum is not None:
                    ux, uy, sw = _weighted_direction_sum(dxs.astype(np.float64), dys.astype(np.float64), weights)
                else:
                    # Fallback: pure numpy weighted sum
                    dist_safe = np.maximum(np.hypot(dxs, dys), 1e-6)
                    ux = np.sum((dxs / dist_safe) * weights)
                    uy = np.sum((dys / dist_safe) * weights)
                    sw = np.sum(weights)
                if sw <= 0:
                    continue
                norm = np.hypot(ux, uy)
                if norm <= 0:
                    continue
                unit_x = ux / norm
                unit_y = uy / norm

                step_dist_i = (10.0 ** self.prev_log_mov[idx]) / 4.0
                social_dx[idx] = unit_x * social_weight * step_dist_i
                social_dy[idx] = unit_y * social_weight * step_dist_i

            return social_dx, social_dy

        # Bind to instance
        self._compute_social_vectors = _compute_social_vectors.__get__(self, self.__class__)

        # Initialize day-of-year counter for reproduction timing
        self._day_of_year = 0

        # Mating day (females only, N(225, 20))
        mating_days = np.random.normal(225, 20, self.count).astype(np.int16)
        # Apply only to females, others stay -99
        self.mating_day = np.where(self.is_female, mating_days, -99)

        # Initialize pregnancy state (Java Porpoise.java:165-178)
        maturity_age = self.params.maturity_age
        mature_females = self.is_female & (self.age >= maturity_age) & self.active_mask

        # All mature females start as ready-to-mate (status=2)
        self.pregnancy_status[mature_females] = 2

        # Some conceive with probability conceive_prob → pregnant (status=1)
        conceive_roll = np.random.random(self.count)
        conceives = mature_females & (conceive_roll < self.params.conceive_prob)
        self.pregnancy_status[conceives] = 1
        self.days_since_mating[conceives] = 0

        # Pregnant females get random days_since_mating representing progress into gestation
        n_conceives = int(np.sum(conceives))
        if n_conceives > 0:
            initial_dsm = (360 - np.round(np.random.normal(225, 20, n_conceives))).astype(np.int16)
            initial_dsm = np.clip(initial_dsm, 0, 360)
            self.days_since_mating[conceives] = initial_dsm

        # Failed conceive → set back to immature (status=0)
        failed = mature_females & ~conceives
        self.pregnancy_status[failed] = 0

    # === Step Sub-Methods (P1.5 Refactoring) ===

    def _update_movement(
        self,
        mask: np.ndarray,
        deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]],
        ambient_rl: Optional[np.ndarray],
    ) -> None:
        """
        Update movement calculations for active agents.

        Handles:
        - Correlated random walk (CRW) turning angles with environmental modulation
        - Dispersal heading override for dispersing porpoises
        - Step length calculation with environmental modulation
        - Deterrence vector application
        - Social cohesion vectors
        """
        if self._movement_module is not None:
            self._update_movement_jasmine(mask, deterrence_vectors, ambient_rl)
            return
        # === Get environmental variables from landscape ===
        # DEPONS CRW uses depth and salinity to modulate movement
        if self.landscape is not None:
            # Build positions array for vectorized lookup (reuse pre-allocated buffer)
            self._positions[:, 0] = self.x
            self._positions[:, 1] = self.y
            np.copyto(
                self._depths,
                self.landscape.get_depths_vectorized(
                    self._positions, xi=self._cell_xi, yi=self._cell_yi
                ),
            )
            np.copyto(
                self._salinity_vals,
                self.landscape.get_salinities_vectorized(
                    self._positions, xi=self._cell_xi, yi=self._cell_yi
                ),
            )
            # Kattegat salinity override (Java Porpoise.java:339-345)
            landscape_name = getattr(self.landscape, 'landscape_name', '')
            if landscape_name == 'Kattegat':
                self._salinity_vals[:] = 34.069105813295
        else:
            # Default values when no landscape (homogeneous case)
            self._depths.fill(30.0)  # Default depth
            self._salinity_vals.fill(30.0)  # Default salinity

        # === Calculate Turning Angle + Step Length (Full DEPONS CRW) ===
        if _HAS_KERNELS:
            # Numba kernel: computes pres_angle and log_mov with rejection sampling
            # Seed Numba's internal RNG from NumPy's RNG for reproducibility
            _seed_numba_rng(np.random.randint(0, 2**31))
            np.copyto(self._rand_angle, np.random.normal(self.params.r2_mean, self.params.r2_sd, self.count))
            np.copyto(self._rand_len, np.random.normal(self.params.r1_mean, self.params.r1_sd, self.count))
            _crw_kernel(
                self.prev_angle, self.prev_log_mov,
                self._depths, self._salinity_vals,
                self._rand_angle, self._rand_len, mask,
                self._pres_angle, self._log_mov,
                self.params.corr_angle_base, self.params.corr_angle_bathy,
                self.params.corr_angle_salinity, self.params.corr_angle_base_sd,
                self.params.corr_logmov_length, self.params.corr_logmov_bathy,
                self.params.corr_logmov_salinity, self.params.max_mov,
                self.params.r2_mean, self.params.r2_sd,
                self.params.r1_mean, self.params.r1_sd,
            )
        else:
            # --- Turning Angle (NumPy fallback) ---
            # DEPONS formula: angleTmp = b0 * prevAngle + N(0,4)
            #                 presAngle = angleTmp * (b1*depth + b2*salinity + b3)
            np.copyto(self._rand_angle, np.random.normal(self.params.r2_mean, self.params.r2_sd, self.count))

            # angleTmp = b0 * prevAngle + R2
            np.multiply(self.params.corr_angle_base, self.prev_angle, out=self._pres_angle)
            self._pres_angle += self._rand_angle

            # Environmental modulation: (b1*depth + b2*salinity + b3)
            np.multiply(self.params.corr_angle_bathy, self._depths, out=self._env_mod_angle)
            self._env_mod_angle += self.params.corr_angle_salinity * self._salinity_vals
            self._env_mod_angle += self.params.corr_angle_base_sd

            # presAngle = angleTmp * env_modulation
            self._pres_angle *= self._env_mod_angle

            # Rejection sampling for turning angle (Java Porpoise.java:332-360)
            violations = np.abs(self._pres_angle) > 180
            retry = 0
            while np.any(violations & mask) and retry < 200:
                idx = np.where(violations & mask)[0]
                new_rand = np.random.normal(self.params.r2_mean, self.params.r2_sd, len(idx))
                angle_tmp = self.params.corr_angle_base * self.prev_angle[idx] + new_rand
                self._pres_angle[idx] = angle_tmp * (
                    self.params.corr_angle_bathy * self._depths[idx]
                    + self.params.corr_angle_salinity * self._salinity_vals[idx]
                    + self.params.corr_angle_base_sd
                )
                violations = np.abs(self._pres_angle) > 180
                retry += 1
            # Emergency fallback: clamp to +/-90 (Java Porpoise.java:354)
            if np.any(violations & mask):
                self._pres_angle[violations & mask] = np.sign(self._pres_angle[violations & mask]) * 90

            # Second angle loop: distance-dependent modulation (Java Porpoise.java:374-393)
            max_mov_value = np.power(10.0, self.params.max_mov)
            prev_mov = np.power(10.0, self.prev_log_mov)
            needs_modulation = mask & (prev_mov <= max_mov_value)
            if np.any(needs_modulation):
                retry = 0
                violations2 = np.ones(self.count, dtype=bool)
                while np.any(violations2 & needs_modulation) and retry < 200:
                    idx = np.where(violations2 & needs_modulation)[0]
                    rnd = np.random.uniform(0, 20, len(idx))
                    new_angle = (np.abs(self._pres_angle[idx]) + rnd
                                 - rnd * prev_mov[idx] / max_mov_value)
                    self._pres_angle[idx] = np.sign(self._pres_angle[idx]) * new_angle
                    violations2 = np.abs(self._pres_angle) >= 180
                    retry += 1
                # Fallback: random(0,20) + 90 (Java Porpoise.java:389)
                if np.any(violations2 & needs_modulation):
                    fb_idx = np.where(violations2 & needs_modulation)[0]
                    self._pres_angle[fb_idx] = np.sign(self._pres_angle[fb_idx]) * (
                        np.random.uniform(0, 20, len(fb_idx)) + 90
                    )

            # --- Step Length (NumPy fallback) ---
            # DEPONS formula: log10_mov = a0 * prev_log_mov + a1*depth + a2*salinity + R1
            np.copyto(self._rand_len, np.random.normal(self.params.r1_mean, self.params.r1_sd, self.count))
            np.multiply(self.params.corr_logmov_length, self.prev_log_mov, out=self._log_mov)
            self._log_mov += self.params.corr_logmov_bathy * self._depths
            self._log_mov += self.params.corr_logmov_salinity * self._salinity_vals
            self._log_mov += self._rand_len

            # Rejection sampling for step length (Java Porpoise.java:367-391)
            violations = self._log_mov > self.params.max_mov
            retry = 0
            while np.any(violations & mask) and retry < 200:
                idx = np.where(violations & mask)[0]
                new_rand = np.random.normal(self.params.r1_mean, self.params.r1_sd, len(idx))
                self._log_mov[idx] = (
                    self.params.corr_logmov_length * self.prev_log_mov[idx]
                    + self.params.corr_logmov_bathy * self._depths[idx]
                    + self.params.corr_logmov_salinity * self._salinity_vals[idx]
                    + new_rand
                )
                violations = self._log_mov > self.params.max_mov
                retry += 1
            # Emergency fallback: clamp to maxMov (Java Porpoise.java:387)
            if np.any(violations & mask):
                self._log_mov[violations & mask] = self.params.max_mov

            self.prev_log_mov[mask] = self._log_mov[mask]

        # Capture pre-movement heading for prev_angle computation (Task 6)
        np.copyto(self._pre_heading, self.heading)

        self.heading[mask] += self._pres_angle[mask]
        self.heading[mask] %= 360.0

        # Apply dispersal heading override for dispersing porpoises
        self._apply_dispersal_heading(mask)

        # Update reference memory (stores food, computes veTotal and vt)
        self._update_reference_memory(mask)

        # Save dispersal heading before CRW composition overwrites it
        _disp_mask = mask & self.is_dispersing
        _saved_disp_heading = self.heading[_disp_mask].copy() if np.any(_disp_mask) else None

        # Compute CRW unit direction vector from heading
        np.radians(self.heading, out=self._rads)
        np.sin(self._rads, out=self._dx)
        np.cos(self._rads, out=self._dy)

        # Heading composition (Java Porpoise.java:556-566)
        # crwContrib = inertiaConst + presMov * veTotal
        # Compute 10^log_mov once and reuse for both crwContrib and step distance
        np.power(10.0, self._log_mov, out=self._step_dist)
        crw_contrib = self.params.inertia_const + self._step_dist * self._ve_total

        # totalD = (dx,dy) * crwContrib + vt + deterVt
        total_dx = self._dx * crw_contrib + self._vt_x
        total_dy = self._dy * crw_contrib + self._vt_y

        # Apply deterrence vectors
        if deterrence_vectors is not None:
            d_dx, d_dy = deterrence_vectors
            self.deter_strength[mask] = np.abs(d_dx[mask]) + np.abs(d_dy[mask])
            self._was_deterred |= (self.deter_strength > 0) & mask
            total_dx[mask] += d_dx[mask]
            total_dy[mask] += d_dy[mask]
        else:
            self.deter_strength[mask] = 0.0

        # Social communication & cohesion
        if getattr(self.params, 'communication_enabled', False):
            soc_dx, soc_dy = self._compute_social_vectors(mask, ambient_rl)
            total_dx[mask] += soc_dx[mask]
            total_dy[mask] += soc_dy[mask]

        # facePoint: new heading from composite vector (Java Porpoise.java:567)
        new_heading = np.degrees(np.arctan2(total_dx, total_dy)) % 360
        self.heading[mask] = new_heading[mask]

        # Restore dispersal heading — dispersing agents skip CRW composition
        if _saved_disp_heading is not None:
            self.heading[_disp_mask] = _saved_disp_heading

        # Store total turn for next step (Java Porpoise.java:583)
        total_turn = (self.heading - self._pre_heading + 180) % 360 - 180
        self.prev_angle[mask] = total_turn[mask]

        # Step distance: presMov / 4.0 (Java Porpoise.java:589)
        # _step_dist already holds 10^log_mov from heading composition above
        self._step_dist /= 4.0

        # Override step distance for dispersing agents (Java AbstractPSMDispersal.java:210)
        dispersing = mask & self.is_dispersing
        if np.any(dispersing):
            disp_step = getattr(self.params, 'mean_disp_dist', 2.0) / 0.4
            self._step_dist[dispersing] = disp_step
            self.dispersal_distance_traveled[dispersing] += disp_step

        # Final dx/dy for actual movement from composite heading
        np.radians(self.heading, out=self._rads)
        np.sin(self._rads, out=self._dx)
        self._dx *= self._step_dist
        np.cos(self._rads, out=self._dy)
        self._dy *= self._step_dist

    def _apply_dispersal_heading(self, mask: np.ndarray) -> None:
        """Apply PSM-Type2 dispersal heading using the dispersal module formula.

        Delegates to the same formula as PSMType2Dispersal.get_dispersal_move():
        - angleDelta = U(-psm_angle, +psm_angle) * SSLogis(distLogX)
        - distLogX = 3 * distPercent - 1.5
        - newHeading = previousStepHeading + angleDelta
        """
        from cenop.behavior.dispersal import sslogis

        dispersing = mask & self.is_dispersing
        if not np.any(dispersing):
            return

        n = int(np.sum(dispersing))
        psm_angle = getattr(self.params, 'psm_type2_random_angle', 20.0)
        psm_log = getattr(self.params, 'psm_log', 0.6)

        # Random angle delta: U(-psm_angle, +psm_angle)
        delta = np.random.uniform(-psm_angle, psm_angle, n)

        # Distance-based logistic scaling (DispersalPSMType2.java:77)
        target_dist = self.dispersal_target_distance[dispersing]
        dx_disp = self.x[dispersing] - self.dispersal_start_x[dispersing]
        dy_disp = self.y[dispersing] - self.dispersal_start_y[dispersing]
        dist_traveled = np.sqrt(dx_disp**2 + dy_disp**2)

        with np.errstate(divide='ignore', invalid='ignore'):
            dist_percent = np.where(target_dist > 0, dist_traveled / target_dist, 0.0)
        dist_percent = np.nan_to_num(dist_percent, nan=0.0, posinf=1.0, neginf=0.0)
        dist_percent = np.clip(dist_percent, 0.0, 10.0)

        dist_log_x = 3 * dist_percent - 1.5

        # SSLogis: phi1 / (1 + exp((phi2 - x) / phi3))
        # Vectorized version of sslogis(dist_log_x, 1.0, 0.0, psm_log)
        dist_log_x = np.clip(dist_log_x, -100, 100)
        logistic = 1.0 / (1.0 + np.exp((0.0 - dist_log_x) / psm_log))

        # Scale angle by logistic
        delta = delta * logistic

        # New heading from previous step heading (not CRW heading)
        self.heading[dispersing] = self._prev_step_heading[dispersing] + delta
        self.heading[dispersing] %= 360.0

        # Update prev_step_heading for next tick
        self._prev_step_heading[dispersing] = self.heading[dispersing]

    def _update_movement_jasmine(
        self,
        mask: np.ndarray,
        deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]],
        ambient_rl: Optional[np.ndarray],
    ) -> None:
        """
        JASMINE movement path: delegates to movement module.

        Syncs population arrays with MovementState, calls the module,
        and writes results back to population arrays.
        """
        from cenop.movement.base import EnvironmentContext

        # Build environment context
        if self.landscape is not None:
            env = EnvironmentContext.from_landscape(self.landscape, self.x, self.y)
        else:
            env = EnvironmentContext.create_homogeneous(self.count)

        # Sync population → MovementState
        state = self._movement_state
        state.heading[:] = self.heading
        state.prev_log_mov[:] = self.prev_log_mov
        state.prev_angle[:] = self.prev_angle
        state.is_dispersing[:] = self.is_dispersing

        # Compute dispersal headings for dispersing agents
        dispersing = mask & self.is_dispersing
        if np.any(dispersing):
            disp_dx = self.dispersal_target_x - self.x
            disp_dy = self.dispersal_target_y - self.y
            state.dispersal_heading[dispersing] = np.degrees(
                np.arctan2(disp_dx[dispersing], disp_dy[dispersing])
            ) % 360.0

        # Extract deterrence components
        det_dx = None
        det_dy = None
        if deterrence_vectors is not None:
            det_dx, det_dy = deterrence_vectors

        # Add memory-based avoidance to deterrence
        if self._avoidance_result is not None:
            av = self._avoidance_result
            if det_dx is not None:
                det_dx = det_dx + av.avoidance_dx * av.avoidance_strength
                det_dy = det_dy + av.avoidance_dy * av.avoidance_strength
            else:
                strength = av.avoidance_strength
                if np.any(strength > 0):
                    det_dx = av.avoidance_dx * strength
                    det_dy = av.avoidance_dy * strength

        # Call movement module
        result = self._movement_module.compute_step(
            self.x, self.y, state, env, mask, det_dx, det_dy
        )

        # Write results back to population arrays
        self._dx[mask] = result.dx[mask]
        self._dy[mask] = result.dy[mask]
        self.heading[mask] = result.new_heading[mask]
        self._step_dist[mask] = result.step_distance[mask]
        self.prev_angle[mask] = result.turning_angle[mask]

        # Update prev_log_mov from step distance
        safe_dist = np.maximum(result.step_distance * 4.0, 1e-6)
        self.prev_log_mov[mask] = np.log10(safe_dist[mask])

        # Deterrence tracking
        if deterrence_vectors is not None:
            d_dx, d_dy = deterrence_vectors
            self.deter_strength[mask] = np.abs(d_dx[mask]) + np.abs(d_dy[mask])
            self._was_deterred |= (self.deter_strength > 0) & mask
        else:
            self.deter_strength[mask] = 0.0

        # Social communication & cohesion (same as DEPONS path)
        if getattr(self.params, 'communication_enabled', False):
            soc_dx, soc_dy = self._compute_social_vectors(mask, ambient_rl)
            self._dx[mask] += soc_dx[mask]
            self._dy[mask] += soc_dy[mask]

    def _handle_land_avoidance(self, mask: np.ndarray) -> None:
        """
        Handle land avoidance using DEPONS pattern.

        Checks if proposed positions are on land and tries turning
        40°, 70°, 120° in both directions to find water.
        
        Boundary handling uses DEPONS-style reflection (BouncyBorders):
        when an agent would move past an edge, the overshot component
        is negated (reflected) and the heading is recalculated.
        """
        # Calculate proposed new positions
        np.add(self.x, self._dx, out=self._new_x)
        np.add(self.y, self._dy, out=self._new_y)

        # DEPONS-style bouncy borders: reflect instead of clamp
        world_w = self.landscape.width if self.landscape else self.params.world_width
        world_h = self.landscape.height if self.landscape else self.params.world_height
        # Save original dx/dy to detect which agents got reflected
        # (_reflect_boundaries flips dx/dy signs for reflected agents)
        np.copyto(self._orig_dx, self._dx)
        np.copyto(self._orig_dy, self._dy)

        self._reflect_boundaries(self._new_x, self._new_y, self._dx, self._dy,
                                 world_w, world_h, mask)

        # Recalculate heading ONLY for agents whose displacement was reflected
        # (DEPONS Porpoise.forward(): setHeading + setPrevAngle(0) after bounce)
        reflected = mask & ((self._dx != self._orig_dx) | (self._dy != self._orig_dy))
        if np.any(reflected):
            self.heading[reflected] = np.degrees(
                np.arctan2(self._dx[reflected], self._dy[reflected])
            ) % 360.0

        # Early exit: skip depth-check loop on all-water landscapes
        if self._skip_land_avoidance:
            return

        if not self.landscape:
            return

        # Check depth at new positions
        np.copyto(self._new_xi, self._new_x.astype(np.int32))
        np.copyto(self._new_yi, self._new_y.astype(np.int32))
        np.clip(self._new_xi, 0, world_w - 1, out=self._new_xi)
        np.clip(self._new_yi, 0, world_h - 1, out=self._new_yi)

        if hasattr(self.landscape, '_depth') and self.landscape._depth is not None:
            np.copyto(self._depths, self.landscape._depth[self._new_yi, self._new_xi])
        else:
            self._depths.fill(20.0)  # Default to water

        # Identify agents on land
        min_depth = self.params.min_depth if self.params else 1.0
        np.copyto(self._on_land, ((self._depths < min_depth) | np.isnan(self._depths)) & mask)

        if not np.any(self._on_land):
            return

        # Try turning to avoid land (DEPONS pattern with random jitter)
        if _HAS_LAND_KERNEL:
            # Fused kernel path: one Numba call for all 3 angles x 2 directions
            blocked_idx = np.where(self._on_land)[0]
            n_blocked = len(blocked_idx)
            if n_blocked > 0:
                bx = self._la_f64_x[:n_blocked]
                by = self._la_f64_y[:n_blocked]
                bh = self._la_f64_heading[:n_blocked]
                bs = self._la_f64_step[:n_blocked]
                bx[:] = self.x[blocked_idx]  # float32->float64 implicit upcast
                by[:] = self.y[blocked_idx]
                bh[:] = self.heading[blocked_idx]
                bs[:] = self._step_dist[blocked_idx]

                base_angles = np.array([40.0, 70.0, 120.0], dtype=np.float64)
                jitter = np.random.uniform(0, 10, 3).astype(np.float64)

                out_x = self._la_out_x[:n_blocked]
                out_y = self._la_out_y[:n_blocked]
                out_h = self._la_out_heading[:n_blocked]
                resolved = self._la_resolved[:n_blocked]

                _land_avoidance_kernel(
                    bx, by, bh, bs,
                    self.landscape._depth, min_depth,
                    base_angles, jitter,
                    out_x, out_y, out_h, resolved,
                )

                # Apply results for resolved agents
                resolved_global = blocked_idx[resolved]
                if len(resolved_global) > 0:
                    self._new_x[resolved_global] = out_x[resolved]
                    self._new_y[resolved_global] = out_y[resolved]
                    self.heading[resolved_global] = out_h[resolved]
                    self._on_land[resolved_global] = False
        else:
            # Fallback: original 6-call loop (preserved exactly)
            for base_angle in [40, 70, 120]:
                turn_angle = base_angle + np.random.uniform(0, 10)
                np.copyto(self._still_blocked, self._on_land)

                # Compute positions for both turn directions
                right_heading = self._compute_turn_position(
                    turn_angle, world_w, world_h,
                    self._right_x, self._right_y,
                    self._right_xi, self._right_yi,
                    self._right_depths,
                    blocked_mask=self._still_blocked,
                )
                left_heading = self._compute_turn_position(
                    -turn_angle, world_w, world_h,
                    self._left_x, self._left_y,
                    self._left_xi, self._left_yi,
                    self._left_depths,
                    blocked_mask=self._still_blocked,
                )

                # Pick deeper direction if valid water
                right_ok = (
                    (self._right_depths >= min_depth)
                    & ~np.isnan(self._right_depths)
                ) & self._still_blocked
                left_ok = (
                    (self._left_depths >= min_depth)
                    & ~np.isnan(self._left_depths)
                ) & self._still_blocked
                both_ok = right_ok & left_ok

                # If both OK, pick deeper
                use_right = both_ok & (
                    self._right_depths >= self._left_depths
                )
                use_left = both_ok & (
                    self._left_depths > self._right_depths
                )

                # If only one OK
                use_right = use_right | (right_ok & ~left_ok)
                use_left = use_left | (left_ok & ~right_ok)

                # Update positions for those who found water
                self._new_x[use_right] = self._right_x[use_right]
                self._new_y[use_right] = self._right_y[use_right]
                self.heading[use_right] = right_heading[use_right]

                self._new_x[use_left] = self._left_x[use_left]
                self._new_y[use_left] = self._left_y[use_left]
                self.heading[use_left] = left_heading[use_left]

                # Mark as no longer blocked
                self._on_land[use_right | use_left] = False

        # Backtrack fallback (Java Porpoise.java:505-533) — vectorized
        still_blocked = np.where(self._on_land)[0]
        if len(still_blocked) > 0 and hasattr(self.landscape, '_depth') and self.landscape._depth is not None:
            depth_grid = self.landscape._depth
            mem_size = self._stored_util.shape[1]
            max_hist = 20
            for h in range(max_hist):
                remaining = np.where(self._on_land)[0]
                if len(remaining) == 0:
                    break
                valid = remaining[self._mem_count[remaining] > h]
                if len(valid) == 0:
                    continue
                buf_idx = (self._mem_ptr[valid] - 1 - h) % mem_size
                px = self._pos_history_x[valid, buf_idx]
                py = self._pos_history_y[valid, buf_idx]
                xi = np.clip(px.astype(np.int32), 0, self.landscape.width - 1)
                yi = np.clip(py.astype(np.int32), 0, self.landscape.height - 1)
                depths_at = depth_grid[yi, xi]
                found = depths_at > 0
                found_idx = valid[found]
                self._new_x[found_idx] = px[found]
                self._new_y[found_idx] = py[found]
                self._on_land[found_idx] = False

        # Deepest-neighbor fallback (Java Porpoise.java:962-976) — vectorized
        still_blocked2 = np.where(self._on_land)[0]
        if len(still_blocked2) > 0 and hasattr(self.landscape, '_depth') and self.landscape._depth is not None:
            depth_grid = self.landscape._depth
            best_depth = np.full(len(still_blocked2), -9999.0, dtype=np.float32)
            best_x = self.x[still_blocked2].copy()
            best_y = self.y[still_blocked2].copy()
            for dx_off in [-1, 0, 1]:
                for dy_off in [-1, 0, 1]:
                    nx = self.x[still_blocked2] + dx_off
                    ny = self.y[still_blocked2] + dy_off
                    xi = np.clip(nx.astype(np.int32), 0, self.landscape.width - 1)
                    yi = np.clip(ny.astype(np.int32), 0, self.landscape.height - 1)
                    d = depth_grid[yi, xi]
                    better = d > best_depth
                    best_depth[better] = d[better]
                    best_x[better] = nx[better]
                    best_y[better] = ny[better]
            self._new_x[still_blocked2] = best_x
            self._new_y[still_blocked2] = best_y
            self._on_land[still_blocked2] = best_depth <= 0

        # For any still blocked, stay in place and turn around
        self._new_x[self._on_land] = self.x[self._on_land]
        self._new_y[self._on_land] = self.y[self._on_land]
        self.heading[self._on_land] = (self.heading[self._on_land] + 180) % 360

    @staticmethod
    def _reflect_boundaries(
        new_x: np.ndarray, new_y: np.ndarray,
        dx: np.ndarray, dy: np.ndarray,
        world_w: int, world_h: int,
        mask: np.ndarray,
    ) -> None:
        """
        DEPONS-style bouncy borders.

        When a position overshoots an edge the component is reflected
        back into the domain and the displacement sign is flipped so
        that heading recalculation (done in the caller where needed)
        points inward.

        Uses Numba kernel when available, pure NumPy fallback otherwise.
        """
        if _HAS_KERNELS:
            _reflect_kernel(new_x, new_y, dx, dy, world_w, world_h, mask)
            return

        max_x = world_w - 1.0
        max_y = world_h - 1.0

        # --- X reflection ---
        under_x = mask & (new_x < 0)
        over_x  = mask & (new_x > max_x)
        if np.any(under_x):
            new_x[under_x] = -new_x[under_x]
            dx[under_x]    = -dx[under_x]
        if np.any(over_x):
            new_x[over_x] = 2.0 * max_x - new_x[over_x]
            dx[over_x]    = -dx[over_x]
        np.clip(new_x, 0, max_x, out=new_x)

        # --- Y reflection ---
        under_y = mask & (new_y < 0)
        over_y  = mask & (new_y > max_y)
        if np.any(under_y):
            new_y[under_y] = -new_y[under_y]
            dy[under_y]    = -dy[under_y]
        if np.any(over_y):
            new_y[over_y] = 2.0 * max_y - new_y[over_y]
            dy[over_y]    = -dy[over_y]
        np.clip(new_y, 0, max_y, out=new_y)

    def _apply_positions(self, mask: np.ndarray) -> None:
        """Apply final positions and update adaptive neighbor recompute."""
        # Save pre-move positions for post-move depth check (reuse pre-allocated buffers)
        np.copyto(self._pre_move_x, self.x)
        np.copyto(self._pre_move_y, self.y)

        self.x[mask] = self._new_x[mask]
        self.y[mask] = self._new_y[mask]

        # Recompute cached cell indices for fresh positions
        self._recompute_cell_indices()

        # Post-move depth check (Java Porpoise.java:639-660)
        if self.landscape is not None:
            self._positions[:, 0] = self.x
            self._positions[:, 1] = self.y
            post_depths = self.landscape.get_depths_vectorized(
                self._positions, xi=self._cell_xi, yi=self._cell_yi
            )
            on_land = mask & (post_depths <= 0)
            if np.any(on_land):
                self.x[on_land] = self._pre_move_x[on_land]
                self.y[on_land] = self._pre_move_y[on_land]
                # Re-recompute after land rollback
                self._recompute_cell_indices()

        # Adaptive neighbor recompute based on displacement
        try:
            if getattr(self.params, 'communication_recompute_adaptive', False):
                dx_m = (self.x - self._prev_x) * 400.0  # meters
                dy_m = (self.y - self._prev_y) * 400.0
                disp = np.hypot(dx_m, dy_m)
                if np.any(mask):
                    mean_disp = float(np.mean(disp[mask]))
                else:
                    mean_disp = 0.0

                alpha = float(getattr(self.params, 'communication_recompute_ema_alpha', 0.3))
                self._disp_ema_m = alpha * mean_disp + (1.0 - alpha) * self._disp_ema_m
                self._update_neighbor_recompute_interval(self._disp_ema_m)
        except (AttributeError, ValueError) as e:
            logger.debug("Adaptive recompute interval error: %s", e)

        # Save positions for next tick
        self._prev_x[mask] = self.x[mask]
        self._prev_y[mask] = self.y[mask]

    def _update_behavior_fsm(self, mask: np.ndarray) -> None:
        """Update behavioral FSM states (JASMINE)."""
        from cenop.behavior.states import BehaviorContext

        # Compute time since last disturbance
        time_since = np.full(self.count, 9999, dtype=np.int32)
        if self._memory_state is not None:
            valid = self._memory_state.last_disturbance_tick >= 0
            time_since[valid] = self._global_tick - self._memory_state.last_disturbance_tick[valid]

        # Count PSM memory cells per agent
        memory_cells = np.zeros(self.count, dtype=np.int32)
        active_idx = np.where(mask)[0]
        if len(active_idx) > 0:
            memory_cells[active_idx] = np.count_nonzero(
                self.psm_buffer[active_idx, :, :, 0], axis=(1, 2)
            ).astype(np.int32)

        context = BehaviorContext(
            deterrence_magnitude=self.deter_strength,
            time_since_disturbance=time_since,
            current_energy=self.energy / 20.0,
            energy_declining_days=self.days_declining_energy.astype(np.int32),
            current_speed=self._step_dist,
            memory_cell_count=memory_cells,
            is_dispersing=self.is_dispersing,
            dispersal_complete=(
                self.dispersal_distance_traveled >= self.dispersal_target_distance
            ) & self.is_dispersing,
        )

        self._behavior_fsm.update_states(self._behavior_state, context, mask)

    def _update_energy_dynamics(self, mask: np.ndarray) -> None:
        """
        Update energy dynamics (DEPONS Pattern).

        Handles food consumption, BMR, swimming cost, and PSM updates.
        """
        if self._energy_module is not None:
            self._update_energy_jasmine(mask)
            return
        # Food consumption - hungry porpoises eat more
        fract_to_eat = np.clip((20.0 - self.energy) / 10.0, 0.0, 0.99)

        if self.landscape is not None and hasattr(self.landscape, 'eat_food'):
            food_gained = self._eat_food_vectorized(mask, fract_to_eat)
        else:
            food_gained = fract_to_eat * np.random.uniform(0.1, 0.5, self.count)

        self.energy[mask] += food_gained[mask]

        # Energy consumption (BMR + Swimming)
        current_month = self._get_current_month()
        scaling_factor = self._get_energy_scaling(current_month, mask)

        bmr_cost = 0.001 * scaling_factor * self.params.e_use_per_30_min
        # E_USE_PER_KM = 0 in DEPONS, so swimming_cost is always zero
        total_cost = bmr_cost
        self.energy[mask] -= total_cost[mask]

        # Expose per-step averages for dashboard
        n_active = int(np.sum(mask))
        if n_active > 0:
            self.avg_food_gained = float(np.mean(food_gained[mask]))
            self.avg_energy_cost = float(np.mean(total_cost[mask]))
        else:
            self.avg_food_gained = 0.0
            self.avg_energy_cost = 0.0

        # Update PSM and energy history
        self._update_psm(mask, food_gained)
        self._update_energy_history(mask)
        self._update_dispersal(mask)

        # Clamp energy
        np.clip(self.energy, 0, 20.0, out=self.energy)

    def _update_disturbance_memory(self, mask: np.ndarray) -> None:
        """Update disturbance memory (JASMINE)."""
        from cenop.behavior.disturbance_memory import DisturbanceMemoryContext

        is_disturbed = self.deter_strength > 0.01

        # Approximate disturbance source position from deterrence vector direction
        # Disturbance is "behind" the deterrence push — opposite direction
        det_mag = self.deter_strength + 1e-6
        disturbance_x = self.x.copy()
        disturbance_y = self.y.copy()

        # If we have deterrence dx/dy stored from movement, use them
        # Otherwise approximate from deter_strength alone (use agent position)
        disturbed_mask = is_disturbed & mask
        if np.any(disturbed_mask):
            disturbance_x[disturbed_mask] = self.x[disturbed_mask] - self._dx[disturbed_mask] * 5.0
            disturbance_y[disturbed_mask] = self.y[disturbed_mask] - self._dy[disturbed_mask] * 5.0

        context = DisturbanceMemoryContext(
            is_disturbed=is_disturbed,
            disturbance_intensity=self.deter_strength,
            disturbance_x=disturbance_x,
            disturbance_y=disturbance_y,
            agent_x=self.x,
            agent_y=self.y,
            current_tick=self._global_tick,
        )

        self._memory_module.record_disturbance(self._memory_state, context, mask)
        self._memory_module.decay_memory(self._memory_state, mask)

        # Store avoidance result for use in next tick's movement
        self._avoidance_result = self._memory_module.compute_avoidance(
            self._memory_state, self.x, self.y, mask
        )

    def _update_energy_jasmine(self, mask: np.ndarray) -> None:
        """JASMINE energy path: delegates to energy module (legacy combined path, kept for reference).

        NOTE: The simulation step() now uses the split path (_apply_food_intake_jasmine /
        _apply_bmr_cost_jasmine) to match Java's starvation-check ordering:
          food intake → starvation check → BMR cost
        This combined method is preserved for test compatibility and is no longer called by step().
        """
        from cenop.physiology.energy_budget import EnergyContext
        from cenop.behavior.states import BehaviorState

        # No sync needed — shared view, same array

        # Food availability
        fract_to_eat = np.clip((20.0 - self.energy) / 10.0, 0.0, 0.99)
        if self.landscape is not None and hasattr(self.landscape, 'eat_food'):
            food_available = self._eat_food_vectorized(mask, fract_to_eat)
        else:
            food_available = fract_to_eat * np.random.uniform(0.1, 0.5, self.count)

        # Behavioral state
        if self._behavior_state is not None:
            np.copyto(self._behavioral_state_buf, self._behavior_state.state)
            behavioral_state = self._behavioral_state_buf
        else:
            self._behavioral_state_buf.fill(BehaviorState.FORAGING.value)
            behavioral_state = self._behavioral_state_buf

        current_month = self._get_current_month()

        # Water temperature
        self._water_temp.fill(10.0)
        water_temp = self._water_temp
        if self.landscape is not None and hasattr(self.landscape, 'get_temperature'):
            self._positions[:, 0] = self.x
            self._positions[:, 1] = self.y
            water_temp = self.landscape.get_temperature(self._positions)

        # Convert step_dist (cells/tick) to speed (m/s): cells * 400m/cell / 1800s/tick
        np.multiply(self._step_dist, 400.0 / 1800.0, out=self._speed_ms)
        speed_ms = self._speed_ms

        context = EnergyContext(
            food_available=food_available,
            food_quality=self._food_quality,
            current_speed=speed_ms,
            behavioral_state=behavioral_state,
            water_temperature=water_temp,
            current_month=current_month,
            is_disturbed=self.deter_strength > 0,
            deterrence_magnitude=self.deter_strength,
            is_lactating=self.with_calf,
            is_pregnant=(self.days_since_mating > 0) & (self.days_since_mating < 300),
        )

        result = self._energy_module.compute_energy_update(self._energy_state, context, mask)
        self._energy_module.apply_result(self._energy_state, result, mask)

        # No sync needed — shared view, same array

        # Update distance traveled for energy module tracking
        self._energy_state.distance_traveled[mask] = self._step_dist[mask] * 400.0  # cells → meters

        # Expose per-step averages for dashboard
        n_active = int(np.sum(mask))
        if n_active > 0:
            self.avg_food_gained = float(np.mean(result.energy_intake[mask]))
            self.avg_energy_cost = float(np.mean(result.total_cost[mask]))
        else:
            self.avg_food_gained = 0.0
            self.avg_energy_cost = 0.0

        # PSM, energy history, and dispersal still needed
        self._update_psm(mask, food_available)
        self._update_energy_history(mask)
        self._update_dispersal(mask)

        # Clamp energy
        np.clip(self.energy, 0, 20.0, out=self.energy)

    def _build_energy_context(self, mask: np.ndarray) -> tuple:
        """Build EnergyContext and food_available for JASMINE path.

        Returns (context, food_available) tuple.
        """
        from cenop.physiology.energy_budget import EnergyContext
        from cenop.behavior.states import BehaviorState

        # No sync needed — shared view, same array

        # Food availability
        fract_to_eat = np.clip((20.0 - self.energy) / 10.0, 0.0, 0.99)
        if self.landscape is not None and hasattr(self.landscape, 'eat_food'):
            food_available = self._eat_food_vectorized(mask, fract_to_eat)
        else:
            food_available = fract_to_eat * np.random.uniform(0.1, 0.5, self.count)

        # Behavioral state
        if self._behavior_state is not None:
            np.copyto(self._behavioral_state_buf, self._behavior_state.state)
            behavioral_state = self._behavioral_state_buf
        else:
            self._behavioral_state_buf.fill(BehaviorState.FORAGING.value)
            behavioral_state = self._behavioral_state_buf

        current_month = self._get_current_month()

        # Water temperature
        self._water_temp.fill(10.0)
        water_temp = self._water_temp
        if self.landscape is not None and hasattr(self.landscape, 'get_temperature'):
            self._positions[:, 0] = self.x
            self._positions[:, 1] = self.y
            water_temp = self.landscape.get_temperature(self._positions)

        # Convert step_dist (cells/tick) to speed (m/s): cells * 400m/cell / 1800s/tick
        np.multiply(self._step_dist, 400.0 / 1800.0, out=self._speed_ms)
        speed_ms = self._speed_ms

        context = EnergyContext(
            food_available=food_available,
            food_quality=self._food_quality,
            current_speed=speed_ms,
            behavioral_state=behavioral_state,
            water_temperature=water_temp,
            current_month=current_month,
            is_disturbed=self.deter_strength > 0,
            deterrence_magnitude=self.deter_strength,
            is_lactating=self.with_calf,
            is_pregnant=(self.days_since_mating > 0) & (self.days_since_mating < 300),
        )
        return context, food_available

    def _apply_food_intake_jasmine(self, mask: np.ndarray) -> None:
        """Phase 1 of split JASMINE energy update: food intake only.

        Builds the EnergyContext (including eating food from landscape), computes
        food intake via compute_food_intake(), applies it to energy, and stores
        the context and food_available for use by _apply_bmr_cost_jasmine().

        After this method the population energy reflects post-food, pre-BMR values —
        matching Java's ordering so that starvation is checked at the right moment.
        """
        context, food_available = self._build_energy_context(mask)

        # Compute food intake only
        intake = self._energy_module.compute_food_intake(self._energy_state, context, mask)

        # Apply food intake (shared view — no sync needed)
        self._energy_state.energy[mask] += intake[mask]
        # Clamp deferred to end of _apply_bmr_cost_jasmine

        # Store for BMR phase
        self._pending_energy_context = context
        self._pending_food_available = food_available
        self._pending_food_intake = intake

        # Dashboard metric: food gained
        n_active = int(np.sum(mask))
        if n_active > 0:
            self.avg_food_gained = float(np.mean(intake[mask]))
        else:
            self.avg_food_gained = 0.0

    def _apply_bmr_cost_jasmine(self, mask: np.ndarray) -> None:
        """Phase 2 of split JASMINE energy update: BMR + activity costs.

        Uses the context built by _apply_food_intake_jasmine(). Deducts BMR/activity
        costs, syncs energy back to population, updates PSM, energy history, dispersal.
        Must be called after _check_mortality() so that only surviving agents pay BMR.
        """
        context = getattr(self, '_pending_energy_context', None)
        food_available = getattr(self, '_pending_food_available', None)
        if context is None or food_available is None:
            # Fallback: no pending context (shouldn't happen in normal flow)
            return

        # No sync needed — shared view, same array

        # Compute BMR + activity cost
        cost = self._energy_module.compute_bmr_cost(self._energy_state, context, mask)

        # Apply cost
        self._energy_state.energy[mask] -= cost[mask]
        # Clamp deferred to final clamp below

        # Also track disturbance costs in energy_state
        if hasattr(self._energy_state, 'disturbance_energy_cost'):
            disturbance_cost = np.where(
                context.is_disturbed[mask],
                0.002 * context.deterrence_magnitude[mask],
                0.0
            ).astype(np.float32)
            self._energy_state.disturbance_energy_cost[mask] += disturbance_cost

        # No sync needed — shared view, same array

        # Update distance traveled for energy module tracking
        self._energy_state.distance_traveled[mask] = self._step_dist[mask] * 400.0  # cells → meters

        # Dashboard metric: energy cost
        n_active = int(np.sum(mask))
        if n_active > 0:
            self.avg_energy_cost = float(np.mean(cost[mask]))
        else:
            self.avg_energy_cost = 0.0

        # PSM, energy history, and dispersal
        self._update_psm(mask, food_available)
        self._update_energy_history(mask)
        self._update_dispersal(mask)

        # Clamp energy
        np.clip(self.energy, 0, 20.0, out=self.energy)

        # Clear pending state
        self._pending_energy_context = None
        self._pending_food_available = None
        self._pending_food_intake = None

    def _apply_food_intake(self, mask: np.ndarray) -> None:
        """Dispatch food intake phase (DEPONS or JASMINE)."""
        if self._energy_module is not None:
            self._apply_food_intake_jasmine(mask)
        else:
            # DEPONS inline path: food + PSM, leave BMR for _apply_bmr_cost
            fract_to_eat = np.clip((20.0 - self.energy) / 10.0, 0.0, 0.99)
            if self.landscape is not None and hasattr(self.landscape, 'eat_food'):
                food_gained = self._eat_food_vectorized(mask, fract_to_eat)
            else:
                food_gained = fract_to_eat * np.random.uniform(0.1, 0.5, self.count)
            self.energy[mask] += food_gained[mask]
            np.clip(self.energy, 0, 20.0, out=self.energy)
            # Store for BMR phase
            self._pending_food_available = food_gained
            n_active = int(np.sum(mask))
            if n_active > 0:
                self.avg_food_gained = float(np.mean(food_gained[mask]))
            else:
                self.avg_food_gained = 0.0

    def _apply_bmr_cost(self, mask: np.ndarray) -> None:
        """Dispatch BMR cost phase (DEPONS or JASMINE)."""
        if self._energy_module is not None:
            self._apply_bmr_cost_jasmine(mask)
        else:
            # DEPONS inline path: BMR + swimming cost
            current_month = self._get_current_month()
            scaling_factor = self._get_energy_scaling(current_month, mask)
            bmr_cost = 0.001 * scaling_factor * self.params.e_use_per_30_min
            swimming_cost = (10.0 ** self.prev_log_mov) * 0.001 * scaling_factor * 0.0  # E_USE_PER_KM = 0
            total_cost = bmr_cost + swimming_cost
            self.energy[mask] -= total_cost[mask]
            n_active = int(np.sum(mask))
            food_gained = getattr(self, '_pending_food_available', np.zeros(self.count))
            if n_active > 0:
                self.avg_energy_cost = float(np.mean(total_cost[mask]))
            else:
                self.avg_energy_cost = 0.0
            # PSM, energy history, and dispersal
            self._update_psm(mask, food_gained)
            self._update_energy_history(mask)
            self._update_dispersal(mask)
            # Clamp energy
            np.clip(self.energy, 0, 20.0, out=self.energy)
            self._pending_food_available = None

    def _check_mortality(self, mask: np.ndarray, active_before: int) -> None:
        """
        Check and apply mortality (DEPONS Pattern).

        Handles starvation, max-age death, and bycatch (DEPONS 3.2).
        Uses parameters from SimulationParameters for all mortality constants.
        """
        # Energy-based starvation mortality (parameterized)
        m_mort_prob_const = self._m_mort_prob_const
        x_survival_const = self._x_survival_const

        yearly_surv_prob = np.where(
            self.energy > 0,
            1.0 - (m_mort_prob_const * np.exp(-self.energy * x_survival_const)),
            0.0
        )
        # Convert yearly survival to per-tick survival: P_tick = P_year^(1/(360*48))
        # Using np.power instead of exp(log(x)/n) to avoid unnecessary intermediate arrays
        _RECIP_TICKS_PER_YEAR = 1.0 / (360 * 48)  # 360 days/year * 48 ticks/day, consistent with DEPONS
        step_surv_prob = np.where(
            self.energy > 0,
            np.power(np.maximum(yearly_surv_prob, 1e-10), _RECIP_TICKS_PER_YEAR),
            0.0
        )

        starvation_check = np.random.random(self.count)
        starving = (starvation_check > step_surv_prob) & mask

        # Two-step starvation logic (Java Porpoise.java:766-776):
        # if (!this.withLactCalf || this.energyLevel <= 0) { die(); }
        # if (this.withLactCalf) { this.withLactCalf = false; }
        was_with_calf = starving & self.with_calf
        self.with_calf[was_with_calf] = False  # abandon calf first
        # Die if: not lactating, or energy <= 0
        starved = starving & ((self.energy <= 0) | ~was_with_calf)

        # Bycatch + max-age: daily schedule (Java Porpoise.java:1137-1153)
        # Only check on day boundaries (tick % 48 == 0)
        bycatch = np.zeros(self.count, dtype=bool)
        old_age = np.zeros(self.count, dtype=bool)
        if self._global_tick % 48 == 0:
            bycatch_annual = getattr(self.params, 'bycatch_prob', 0.0)
            if bycatch_annual > 0:
                # Java: dailySurvivalProb = exp(log(1 - bycatchProb) / 360)
                daily_surv = np.exp(np.log(1 - bycatch_annual) / 360)
                bycatch = (self.rng.random(self.count) > daily_surv) & mask

            # Max-age also daily (Java: same updMortality method)
            max_age = getattr(self.params, 'max_age', 30.0)
            old_age = mask & (self.age > max_age)

        # Apply deaths
        all_deaths = starved | old_age | bycatch
        if np.any(all_deaths):
            death_count = int(np.sum(all_deaths))
            starved_count = int(np.sum(starved))
            old_age_count = int(np.sum(old_age))
            bycatch_count = int(np.sum(bycatch))
            self.active_mask[all_deaths] = False
            if self._debug_instrumentation or death_count > 0:
                active_after = int(np.sum(self.active_mask))
                logger.debug(
                    "[INSTR] tick=%d deaths=%d starved=%d old_age=%d bycatch=%d active_before=%d active_after=%d",
                    self._global_tick, death_count, starved_count, old_age_count, bycatch_count,
                    active_before, active_after
                )

    def _update_aging(self, mask: np.ndarray) -> None:
        """Update aging for active agents (continuous small increments)."""
        self.age[mask] += 1.0 / 360.0 / 48.0  # Age in years per tick (360 days/year, consistent with DEPONS)

    def _update_reference_memory(self, mask: np.ndarray) -> None:
        """Update reference memory: record food and position, compute veTotal and vt.

        Called every tick before movement heading computation.
        Java ref: FastRefMemTurn.java:53-64 (store), Porpoise.java:688-705 (veTotal)
        """
        from cenop.behavior.ref_mem import (
            get_ref_mem_strength_table, get_work_mem_strength_table,
            compute_ve_total, compute_attraction_vector,
        )

        if self.landscape is None:
            return

        active = np.where(mask)[0]
        if len(active) == 0:
            return

        mem_size = self._stored_util.shape[1]

        # IMPORTANT: Java computes vt and veTotal BEFORE storing current position
        # (Porpoise.java:264-277 — refMemTurn + getExpFoodVal before posList.add)

        # 1. Compute veTotal FIRST (uses existing buffer, before new entry)
        work_table = get_work_mem_strength_table(self.params.r_s, mem_size)
        self._ve_total = compute_ve_total(
            self._stored_util, self._mem_ptr, self._mem_count, work_table, mask,
            workspace=self._ref_mem_workspace,
        )

        # 2. Compute attraction vector vt FIRST (uses existing buffer)
        ref_table = get_ref_mem_strength_table(self.params.r_r, mem_size)
        world_w = self.landscape.width if self.landscape else 0
        world_h = self.landscape.height if self.landscape else 0
        new_vt_x, new_vt_y = compute_attraction_vector(
            self._stored_util, self._pos_history_x, self._pos_history_y,
            self._mem_ptr, self._mem_count,
            self.x, self.y, ref_table, mask, world_w, world_h,
            workspace=self._ref_mem_workspace,
        )
        # Java: if refMemTurn returns null, keep previous vt (Porpoise.java:266-267)
        has_history = self._mem_count >= 2
        update = mask & has_history
        self._vt_x[update] = new_vt_x[update]
        self._vt_y[update] = new_vt_y[update]

        # 3. NOW store current food and position in circular buffer (vectorized)
        n_active = len(active)
        pos_buf = np.empty((n_active, 2), dtype=np.float32)
        pos_buf[:, 0] = self.x[active]
        pos_buf[:, 1] = self.y[active]
        food_levels = self.landscape.get_food_levels_vectorized(pos_buf)

        ptrs = self._mem_ptr[active]
        self._stored_util[active, ptrs] = food_levels
        self._pos_history_x[active, ptrs] = self.x[active]
        self._pos_history_y[active, ptrs] = self.y[active]
        self._mem_ptr[active] = (ptrs + 1) % mem_size
        self._mem_count[active] = np.minimum(self._mem_count[active] + 1, mem_size)

    def _handle_reproduction(self, mask: np.ndarray) -> None:
        """Handle reproduction — delegates to pregnancy FSM on day boundaries.

        Called every tick from step(), but only runs FSM on day boundary (tick%48==0).
        Java ref: Porpoise.java:1124-1128 — if (updMortality()) { updPregnancyStatus(); }
        """
        # Update day-of-year counter every tick
        self._day_of_year = (self._day_of_year + 1) % (360 * 48)

        # Only run pregnancy FSM once per day
        if self._global_tick % 48 != 0:
            return

        self._update_pregnancy_status(mask)

    def rerandomize_mating_days(self) -> None:
        """Re-draw mating days for all active females. Called yearly.

        Java ref: YearlyTask.java:99 — p.setRandomMatingDay() for each porpoise
        Porpoise.java:1237-1241 — matingDay = round(N(tmating_mean, tmating_sd))
        """
        females = self.is_female & self.active_mask
        n_females = int(np.sum(females))
        if n_females > 0:
            new_days = np.round(np.random.normal(
                self.params.mating_day_mean, self.params.mating_day_sd, n_females
            )).astype(np.int16)
            self.mating_day[females] = new_days

    def _update_pregnancy_status(self, mask: np.ndarray) -> None:
        """Update pregnancy FSM — called once per day (Java Porpoise.java:1155-1231)."""
        female_mask = mask & self.is_female

        # 0 → 2: Immature to ready-to-mate
        immature = female_mask & (self.pregnancy_status == 0) & (self.age >= self.params.maturity_age)
        self.pregnancy_status[immature] = 2

        # 2 → 1: Ready to pregnant on mating day
        current_day = self._day_of_year // 48
        ready = female_mask & (self.pregnancy_status == 2) & (self.mating_day == current_day)
        if np.any(ready):
            conceive_roll = np.random.random(self.count)
            conceives = ready & (conceive_roll < self.params.conceive_prob)
            self.pregnancy_status[conceives] = 1
            self.days_since_mating[conceives] = 0

        # 1 → 2 + birth: Pregnant gives birth at gestation_time
        giving_birth = female_mask & (self.pregnancy_status == 1) & \
                       (self.days_since_mating == self.params.gestation_time)
        if np.any(giving_birth):
            self.pregnancy_status[giving_birth] = 2
            self.with_calf[giving_birth] = True
            self.days_since_mating[giving_birth] = -99
            self.days_since_birth[giving_birth] = 0

        # Weaning: at nursing_time, create female calf, end lactation
        weaning = female_mask & self.with_calf & \
                  (self.days_since_birth == self.params.nursing_time)
        if np.any(weaning):
            calf_roll = np.random.random(self.count)
            creates_calf = weaning & (calf_roll > 0.5)

            n_calves = int(np.sum(creates_calf))
            if n_calves > 0:
                inactive_slots = np.where(~self.active_mask)[0]
                slots_to_use = min(n_calves, len(inactive_slots))
                if slots_to_use > 0:
                    new_slots = inactive_slots[:slots_to_use]
                    mother_indices = np.where(creates_calf)[0][:slots_to_use]

                    self.active_mask[new_slots] = True
                    self.x[new_slots] = self.x[mother_indices]
                    self.y[new_slots] = self.y[mother_indices]
                    self.heading[new_slots] = self.heading[mother_indices]
                    self.age[new_slots] = 0.0
                    self.is_female[new_slots] = True
                    self.energy[new_slots] = np.random.normal(
                        self.params.energy_init_mean, self.params.energy_init_sd, slots_to_use
                    ).clip(0, 20).astype(np.float32)
                    self.pregnancy_status[new_slots] = 0
                    self.with_calf[new_slots] = False
                    self.days_since_mating[new_slots] = -99
                    self.days_since_birth[new_slots] = -99
                    self.mating_day[new_slots] = -99

            self.with_calf[weaning] = False
            self.days_since_birth[weaning] = -99

        # Increment counters
        pregnant = female_mask & (self.pregnancy_status == 1)
        self.days_since_mating[pregnant] += 1

        lactating = female_mask & self.with_calf
        self.days_since_birth[lactating] += 1

    def step(self, deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None, ambient_rl: Optional[np.ndarray] = None):
        """
        Main simulation step for the entire population.

        Orchestrates all sub-steps in sequence matching Java DEPONS 3.2 ordering:
        1. Movement calculations (CRW, deterrence, social)
        2. Land avoidance (DEPONS pattern)
        3. Position updates
        4a. Food intake (energy gain from foraging)
        4b. Starvation check on post-food, pre-BMR energy (Java ordering)
        4c. BMR cost (PSM, energy history, dispersal — after mortality so dead agents excluded)
        5. Disturbance memory update (JASMINE)
        6. Aging
        7. Reproduction

        The food → starvation → BMR ordering matches Java Porpoise.java where starvation
        is evaluated after food gain but before metabolic costs are subtracted. This means
        a starving animal that just ate sees higher energy at the starvation check, improving
        survival compared to the previous Python ordering (food + BMR → starvation).

        Args:
            deterrence_vectors: Tuple of (dx_array, dy_array) for deterrence
            ambient_rl: Ambient received level for social communication
        """
        mask = self.active_mask
        if not np.any(mask):
            return

        active_before = int(np.sum(self.active_mask))
        self._global_tick += 1

        # Ensure cached cell indices are consistent with current landscape
        # (handles landscape reassignment after __init__)
        if self._global_tick == 1:
            self._recompute_cell_indices()

        # 1. Movement calculations
        self._update_movement(mask, deterrence_vectors, ambient_rl)

        # 2. Land avoidance
        self._handle_land_avoidance(mask)

        # 3. Apply positions
        self._apply_positions(mask)

        # 3.5 Behavioral FSM update (JASMINE)
        if self._behavior_fsm is not None:
            self._update_behavior_fsm(mask)

        # 4a. Food intake (post-food energy is used for starvation check below)
        self._apply_food_intake(mask)

        # 4b. Starvation check on post-food, pre-BMR energy (Java ordering)
        self._check_mortality(mask, active_before)

        # 4c. BMR cost — use updated active_mask so dead agents are excluded
        self._apply_bmr_cost(self.active_mask)

        # 4.5 Disturbance memory update (JASMINE)
        if self._memory_module is not None:
            self._update_disturbance_memory(self.active_mask)

        # 5. Aging
        self._update_aging(self.active_mask)

        # 6. Reproduction
        self._handle_reproduction(self.active_mask)

    def to_dataframe(self) -> pd.DataFrame:
        """Export active agents to DataFrame for UI helpers."""
        mask = self.active_mask
        n_active = np.sum(mask)

        # Behavioral state export (if FSM present)
        if self._behavior_state is not None:
            behavioral_state = self._behavior_state.state.astype(np.int32)
        else:
            behavioral_state = np.ones(self.count, dtype=np.int32)

        # Disturbance flag (boolean): whether deter strength exceeds a small threshold
        is_disturbed = self.deter_strength > 0.1

        # Get depth at current position for debugging land-avoidance
        if self.landscape is not None and hasattr(self.landscape, '_depth'):
            xi = self.x.astype(np.int32)
            yi = self.y.astype(np.int32)
            world_w = self.landscape.width
            world_h = self.landscape.height
            xi = np.clip(xi, 0, world_w - 1)
            yi = np.clip(yi, 0, world_h - 1)
            depths = self.landscape._depth[yi, xi]
        else:
            depths = np.full(self.count, 20.0, dtype=np.float32)

        return pd.DataFrame({
            'id': self.ids[mask],
            'x': self.x[mask],
            'y': self.y[mask],
            'age': self.age[mask],
            'is_female': self.is_female[mask],
            'energy': self.energy[mask],
            'heading': self.heading[mask],
            'is_disturbed': is_disturbed[mask],
            'behavioral_state': behavioral_state[mask],
            'depth': depths[mask],  # Debug: depth at current position
            'alive': np.ones(n_active, dtype=bool)
        })

    # === PSM and Dispersal Methods (Phase 2) ===
    
    def _update_psm(self, mask: np.ndarray, food_gained: np.ndarray) -> None:
        """
        Update Persistent Spatial Memory (Vectorized).

        Only records ticks where food was actually consumed
        (Java PersistentSpatialMemory.java:119).
        """
        # Gate: only count agents that actually gained food
        food_positive = food_gained > 0
        active_idx = np.where(mask & food_positive)[0]
        if len(active_idx) == 0:
            return
            
        # Convert positions to PSM grid coordinates
        psm_x = (self.x[active_idx] // self.psm_cell_size).astype(np.int32)
        psm_y = (self.y[active_idx] // self.psm_cell_size).astype(np.int32)

        # Clip to bounds
        np.clip(psm_x, 0, self.psm_cols - 1, out=psm_x)
        np.clip(psm_y, 0, self.psm_rows - 1, out=psm_y)

        # Use efficient accumulator (Numba-accelerated when available)
        from cenop.optimizations import accumulate_psm_updates  # noqa: E402 — kept deferred to avoid circular import at module level

        idx_arr = active_idx  # int64 from np.where; both Numba and fallback accept it
        ys_arr = psm_y  # already int32
        xs_arr = psm_x  # already int32
        food_arr = food_gained[active_idx].astype(np.float32)

        try:
            accumulate_psm_updates(self.psm_buffer, idx_arr, ys_arr, xs_arr, food_arr)
        except (TypeError, ValueError) as e:
            # Fallback to np.add.at if Numba accelerator unavailable
            logger.debug("PSM accumulator fallback: %s", e)
            np.add.at(self.psm_buffer[:, :, :, 0], (active_idx, psm_y, psm_x), 1.0)
            np.add.at(self.psm_buffer[:, :, :, 1], (active_idx, psm_y, psm_x), food_gained[active_idx])

        # NOTE: Per-agent PSM instances (_psm_instances) are kept only for
        # preferred_distance access. Memory data is stored in psm_buffer only.
        # The individual update() calls were removed for performance optimization.

        # NOTE: Energy history accumulation is handled by _update_energy_history()
        # which is called separately from step(). Do NOT duplicate here to avoid
        # double-counting energy (Critical Bug Fix - Jan 2026).
            
    def _check_dispersal_trigger(self, mask: np.ndarray) -> None:
        """
        Check if dispersal should trigger based on energy decline.
        
        DEPONS Pattern:
        - If energy has declined for t_disp consecutive days (default 5)
        - And porpoise has sufficient memory (50+ cells visited)
        - Then trigger dispersal to remembered high-food area
        """
        t_disp = getattr(self.params, 't_disp', 5)  # Days before dispersal triggers
        min_memory_cells = 50  # Minimum PSM cells for dispersal
        
        # Vectorized check for declining energy
        # history shape: (count, t_disp)
        max_hist = self._energy_history.shape[1]
        # Clamp t_disp to available history length
        t_disp = max(1, min(t_disp, max_hist))

        # If t_disp == 1, declining means today's < yesterday doesn't apply; skip
        if t_disp <= 1:
            return

        # Check all consecutive pairs: history[:, i] < history[:, i+1]
        is_declining = np.ones(self.count, dtype=bool)
        for i in range(t_disp - 1):
            is_declining &= (self._energy_history[:, i] < self._energy_history[:, i + 1])

        # Add to mask
        candidates = mask & is_declining & (~self.is_dispersing)
        candidate_indices = np.where(candidates)[0]

        if len(candidate_indices) > 0:
            # Vectorized memory count check: count cells with ticks > 0
            visited_counts = np.count_nonzero(
                self.psm_buffer[candidate_indices, :, :, 0], axis=(1, 2)
            )
            qualified = candidate_indices[visited_counts >= min_memory_cells]
            if len(qualified) > 0:
                # Batch-initialize dispersal state
                self.is_dispersing[qualified] = True
                self.dispersal_start_x[qualified] = self.x[qualified]
                self.dispersal_start_y[qualified] = self.y[qualified]
                self.dispersal_distance_traveled[qualified] = 0.0
                self._prev_step_heading[qualified] = self.heading[qualified]
                # Per-agent PSM target selection (cannot vectorize)
                for idx in qualified:
                    self._select_dispersal_target(idx)

    def _update_neighbor_recompute_interval(self, mean_disp_m: float) -> None:
        """Update the current recompute interval based on mean displacement EMA.

        Rules (simple heuristic):
        - If mean_disp_m < 0.5 * threshold -> double the interval (up to max)
        - If mean_disp_m > 1.5 * threshold -> set to min_interval
        - Otherwise leave unchanged
        """
        if not getattr(self.params, 'communication_recompute_adaptive', False):
            return

        min_i = int(getattr(self.params, 'communication_recompute_min_interval', 1))
        max_i = int(getattr(self.params, 'communication_recompute_max_interval', 16))
        threshold = float(getattr(self.params, 'communication_recompute_disp_threshold_m', 50.0))

        # Defensive clamp
        min_i = max(1, min_i)
        max_i = max(min_i, max_i)

        cur = int(self._current_recompute_interval)
        new = cur

        if mean_disp_m < 0.5 * threshold:
            new = min(max_i, cur * 2)
        elif mean_disp_m > 1.5 * threshold:
            new = min_i

        # Always clamp final value to valid range (defensive bounds check)
        new = max(min_i, min(max_i, new))

        if new != cur:
            self._current_recompute_interval = int(new)
            # Reset counter to new interval so change takes effect
            self._neighbor_recompute_counter = self._current_recompute_interval
        
    def _update_energy_history(self, mask: np.ndarray) -> None:
        """
        Accumulate per-tick energy into daily totals and update 5-day history when a day completes.
        Safe to call multiple times per tick: each tick is only recorded once using _last_energy_update_tick.
        """
        # Prevent double-update within the same tick
        if getattr(self, '_last_energy_update_tick', -1) == self._global_tick:
            return

        # Accumulate energy for current day
        self._energy_ticks_today[mask] += self.energy[mask]
        self._tick_counter += 1

        # At end of day (48 ticks), update history
        if self._tick_counter >= 48:
            self._tick_counter = 0

            # Calculate daily average
            daily_avg = self._energy_ticks_today / 48.0

            # Shift history and add new day (newest at index 0)
            self._energy_history[:, 1:] = self._energy_history[:, :-1]
            self._energy_history[:, 0] = daily_avg

            # Reset daily accumulator
            self._energy_ticks_today[:] = 0.0

            # Check for declining energy trend (t_disp consecutive days)
            self._check_dispersal_trigger(mask)

        # Record this tick as updated
        self._last_energy_update_tick = self._global_tick
                    
    def _start_dispersal(self, idx: int) -> None:
        """
        Start dispersal behavior for a single porpoise.

        Uses PSM to find target cell at approximately preferred distance.
        Initializes dispersal state then selects a target.
        """
        self.is_dispersing[idx] = True
        self.dispersal_start_x[idx] = self.x[idx]
        self.dispersal_start_y[idx] = self.y[idx]
        self.dispersal_distance_traveled[idx] = 0.0
        self._prev_step_heading[idx] = self.heading[idx]

        self._select_dispersal_target(idx)

    def _select_dispersal_target(self, idx: int) -> None:
        """
        Select a dispersal target for a single porpoise using PSM data.

        Finds the highest-food cell at approximately the preferred distance.
        Falls back to a random target if no suitable cell is found.
        """
        # Use vectorized PSM buffer scan
        mem_slice = self.psm_buffer[idx]  # (rows, cols, 2)
        ticks = mem_slice[:, :, 0]
        food = mem_slice[:, :, 1]

        # Get visited cells
        visited_y, visited_x = np.nonzero(ticks)

        if len(visited_x) == 0:
            self._set_random_dispersal_target(idx)
            return

        # Calculate expectations efficiently
        # food / ticks where ticks > 0
        visited_ticks = ticks[visited_y, visited_x]
        visited_food = food[visited_y, visited_x]
        expectations = visited_food / visited_ticks

        max_exp = np.max(expectations)
        if max_exp <= 0:
            self._set_random_dispersal_target(idx)
            return

        # Get preferred distance (stored in object list or default)
        pref_dist_km = self._psm_instances[idx].preferred_distance
        pref_dist_cells = pref_dist_km * 1000 / 400.0

        # Get world coordinates of visited cells (center of PSM cell)
        # psm_cell_size in world units = 5 * 400 = 2000m = 5 cells
        world_x = visited_x * self.psm_cell_size + (self.psm_cell_size / 2)
        world_y = visited_y * self.psm_cell_size + (self.psm_cell_size / 2)

        # Calculate distances to current position
        dx = world_x - self.x[idx]
        dy = world_y - self.y[idx]
        dists = np.sqrt(dx * dx + dy * dy)

        # Filter for tolerance (5km approx 12.5 cells)
        tolerance_cells = 12.5
        valid_mask = np.abs(dists - pref_dist_cells) < tolerance_cells

        if np.any(valid_mask):
            # Pick highest value among valid distance cells
            valid_expectations = expectations[valid_mask]

            # Find best
            best_local_idx = np.argmax(valid_expectations)

            # Map back to original indices
            valid_indices_in_visited = np.where(valid_mask)[0]
            best_index = valid_indices_in_visited[best_local_idx]

            target_x = world_x[best_index]
            target_y = world_y[best_index]
            target_dist = dists[best_index]

            self.dispersal_target_x[idx] = target_x
            self.dispersal_target_y[idx] = target_y
            self.dispersal_target_distance[idx] = target_dist
        else:
            self._set_random_dispersal_target(idx)

        # Set heading toward target
        dx = self.dispersal_target_x[idx] - self.x[idx]
        dy = self.dispersal_target_y[idx] - self.y[idx]
        self.heading[idx] = np.degrees(np.arctan2(dx, dy)) % 360.0

    def _set_random_dispersal_target(self, idx: int) -> None:
        """Set a random dispersal target at preferred distance.
        
        Uses reflection to keep targets inside the map instead of
        clamping (which biased targets toward edges/corners).
        """
        pref_dist_km = self._psm_instances[idx].preferred_distance
        angle_rad = np.random.uniform(0, 2 * np.pi)
        dist_cells = pref_dist_km * 1000 / 400.0
        
        tx = self.x[idx] + np.sin(angle_rad) * dist_cells
        ty = self.y[idx] + np.cos(angle_rad) * dist_cells
        
        # Reflect into world (DEPONS bouncy-border style)
        w = self.landscape.width if self.landscape else self.params.world_width
        h = self.landscape.height if self.landscape else self.params.world_height
        max_x = float(w - 1)
        max_y = float(h - 1)
        
        # Reflect X
        if tx < 0:
            tx = -tx
        elif tx > max_x:
            tx = 2.0 * max_x - tx
        tx = float(np.clip(tx, 0, max_x))
        
        # Reflect Y
        if ty < 0:
            ty = -ty
        elif ty > max_y:
            ty = 2.0 * max_y - ty
        ty = float(np.clip(ty, 0, max_y))
        
        self.dispersal_target_x[idx] = tx
        self.dispersal_target_y[idx] = ty
        self.dispersal_target_distance[idx] = dist_cells

        
    def _update_dispersal(self, mask: np.ndarray) -> None:
        """Update dispersal progress for dispersing porpoises."""
        dispersing = mask & self.is_dispersing
        if not np.any(dispersing):
            return

        # Pre-compute distances once (shared across all checks)
        dx = self.x - self.dispersal_start_x
        dy = self.y - self.dispersal_start_y
        distances = np.sqrt(dx * dx + dy * dy)

        # --- Deterrence deactivates dispersal (Java Porpoise.java:1277-1278) ---
        deterred = dispersing & (self.deter_strength > 0)
        if np.any(deterred):
            self.is_dispersing[deterred] = False
            self.dispersal_distance_traveled[deterred] = 0.0
            self.days_declining_energy[deterred] = 0
            dispersing = mask & self.is_dispersing

        if not np.any(dispersing):
            return

        # --- Energy-based stop (Java Porpoise.java:1105-1118) ---
        # Check at day boundary (every 48 ticks)
        if self._tick_counter == 0 and self._global_tick > 0:
            today = self._energy_history[dispersing, 0]
            past_min = np.min(self._energy_history[dispersing, 1:8], axis=1)
            recovering = today > past_min
            if np.any(recovering):
                disp_indices = np.where(dispersing)[0]
                stop_indices = disp_indices[recovering]
                self.is_dispersing[stop_indices] = False
                self.dispersal_distance_traveled[stop_indices] = 0.0
                self.days_declining_energy[stop_indices] = 0
                dispersing = mask & self.is_dispersing

        if not np.any(dispersing):
            return

        # --- Distance completion check (reuse pre-computed distances) ---
        completed = dispersing & (distances >= 0.95 * self.dispersal_target_distance)
        if np.any(completed):
            self.is_dispersing[completed] = False
            self.dispersal_distance_traveled[completed] = 0.0
            self.days_declining_energy[completed] = 0
            
    def get_psm(self, idx: int) -> PersistentSpatialMemory:
        """Get PSM instance for a specific porpoise."""
        return self._psm_instances[idx]
        
    def get_dispersal_stats(self) -> Dict[str, Any]:
        """Get statistics about dispersal behavior."""
        active = self.active_mask
        # Calculate avg visited cells from buffer
        # This is expensive for all agents, allow sampling or simplified metric
        avg_cells = 0.0
        if np.any(active):
             # Just sample first 10 for performance in UI? Or calc all?
             # Vectorized count:
             counts = np.count_nonzero(self.psm_buffer[active, :, :, 0], axis=(1,2))
             avg_cells = float(np.mean(counts))
             
        return {
            'dispersing_count': int(np.sum(self.is_dispersing & active)),
            'total_active': int(np.sum(active)),
            'avg_psm_cells': avg_cells,
            'max_declining_days': 0  # Simplified out of model array for now
        }

    # === Land Avoidance Helper Methods ===

    def _compute_turn_position(
        self,
        turn_delta: float,
        world_w: int,
        world_h: int,
        out_x: np.ndarray,
        out_y: np.ndarray,
        out_xi: np.ndarray,
        out_yi: np.ndarray,
        out_depths: np.ndarray,
        blocked_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute position and depth after turning by turn_delta degrees.

        Uses pre-allocated output arrays to avoid memory allocation.
        Returns the heading array for the turn direction.

        Args:
            turn_delta: Degrees to turn (positive = right, negative = left)
            world_w: World width in cells
            world_h: World height in cells
            out_x, out_y: Output arrays for new positions (float32)
            out_xi, out_yi: Output arrays for cell indices (int32)
            out_depths: Output array for depths at new positions (float32)
            blocked_mask: Optional bool mask — when provided and sparse,
                only blocked agents are processed through the kernel.

        Returns:
            Heading array after turning
        """
        if _HAS_KERNELS:
            if blocked_mask is not None and np.sum(blocked_mask) < len(self.x) // 2:
                # Optimized path: only compute for blocked agents
                idx = np.where(blocked_mask)[0]
                n_blocked = len(idx)
                if n_blocked == 0:
                    out_depths.fill(20.0)
                    return self.heading.astype(np.float64)
                # Use slices of the pre-allocated f64 buffers
                bx = self._f64_x[:n_blocked]
                by = self._f64_y[:n_blocked]
                bh = self._f64_heading[:n_blocked]
                bs = self._f64_step[:n_blocked]
                box = self._f64_out_x[:n_blocked]
                boy = self._f64_out_y[:n_blocked]
                boh = self._f64_out_heading[:n_blocked]
                boxi = self._int32_out_xi[:n_blocked]
                boyi = self._int32_out_yi[:n_blocked]
                bx[:] = self.x[idx]
                by[:] = self.y[idx]
                bh[:] = self.heading[idx]
                bs[:] = self._step_dist[idx]
                _turn_kernel(
                    bx, by, bh, bs,
                    float(turn_delta), world_w, world_h,
                    box, boy, boh, boxi, boyi,
                )
                # Scatter results back into full-size output arrays
                out_x[idx] = box.astype(np.float32)
                out_y[idx] = boy.astype(np.float32)
                out_xi[idx] = boxi
                out_yi[idx] = boyi
                if (
                    hasattr(self.landscape, '_depth')
                    and self.landscape._depth is not None
                ):
                    out_depths[idx] = self.landscape._depth[
                        out_yi[idx], out_xi[idx]
                    ]
                else:
                    out_depths[idx] = 20.0
                # Build heading result — copy current heading, update blocked
                heading_out = self.heading.astype(np.float64)
                heading_out[idx] = boh
                return heading_out
            else:
                # Full path (original code) — all agents
                np.copyto(self._f64_x, self.x)
                np.copyto(self._f64_y, self.y)
                np.copyto(self._f64_heading, self.heading)
                np.copyto(self._f64_step, self._step_dist)
                _turn_kernel(
                    self._f64_x, self._f64_y,
                    self._f64_heading, self._f64_step,
                    float(turn_delta), world_w, world_h,
                    self._f64_out_x, self._f64_out_y,
                    self._f64_out_heading,
                    self._int32_out_xi, self._int32_out_yi,
                )
                np.copyto(out_x, self._f64_out_x)
                np.copyto(out_y, self._f64_out_y)
                # Cell indices already clamped by the kernel
                np.copyto(out_xi, self._int32_out_xi)
                np.copyto(out_yi, self._int32_out_yi)
                # Depth lookup stays in Python
                if (
                    hasattr(self.landscape, '_depth')
                    and self.landscape._depth is not None
                ):
                    np.copyto(
                        out_depths,
                        self.landscape._depth[out_yi, out_xi],
                    )
                else:
                    out_depths.fill(20.0)
                return self._f64_out_heading

        heading = (self.heading + turn_delta) % 360
        rads = np.radians(heading)

        # Compute dx, dy using pre-allocated _dx, _dy arrays
        np.multiply(np.sin(rads), self._step_dist, out=self._dx)
        np.multiply(np.cos(rads), self._step_dist, out=self._dy)

        # Proposed position
        np.add(self.x, self._dx, out=out_x)
        np.add(self.y, self._dy, out=out_y)

        # DEPONS-style reflection at boundaries (reuse pre-allocated all-true mask)
        self._reflect_boundaries(out_x, out_y, self._dx, self._dy,
                                 world_w, world_h, self._all_mask)

        # Get cell indices
        np.copyto(out_xi, out_x.astype(np.int32))
        np.copyto(out_yi, out_y.astype(np.int32))
        np.clip(out_xi, 0, world_w - 1, out=out_xi)
        np.clip(out_yi, 0, world_h - 1, out=out_yi)

        # Get depths at new positions
        if hasattr(self.landscape, '_depth') and self.landscape._depth is not None:
            np.copyto(out_depths, self.landscape._depth[out_yi, out_xi])
        else:
            out_depths.fill(20.0)  # Default to water

        return heading

    # === Phase 3: Enhanced Energetics Methods ===

    def _eat_food_vectorized(self, mask: np.ndarray, fract_to_eat: np.ndarray) -> np.ndarray:
        """
        Eat food from landscape cells (Vectorized).
        
        Uses CellData.eat_food_vectorized for high performance block update.
        """
        food_eaten = np.zeros(self.count, dtype=np.float32)
        
        if self.landscape is None:
            return food_eaten
            
        # Only active agents eat
        active_idx = np.where(mask)[0]
        if len(active_idx) == 0:
             return food_eaten
        
        # Delegate to landscape vectorized method
        consumed = self.landscape.eat_food_vectorized(
            self.x[active_idx],
            self.y[active_idx],
            fract_to_eat[active_idx],
            xi=self._cell_xi[active_idx],
            yi=self._cell_yi[active_idx],
            energy=self.energy[active_idx],
        )
        
        food_eaten[active_idx] = consumed
        return food_eaten

        
    def _get_current_month(self) -> int:
        """
        Get current month of simulation (1-12).
        
        Based on tick counter (48 ticks/day, ~30 days/month).
        """
        if not hasattr(self, '_day_of_year'):
            return 1
            
        day = self._day_of_year // 48
        # Approximate month (30 days each)
        month = (day // 30) % 12 + 1
        return month
        
    def _get_energy_scaling(self, month: int, mask: np.ndarray) -> np.ndarray:
        """
        Calculate energy scaling factor based on season and lactation.
        
        DEPONS Pattern:
        - Nov-Mar (cold): 1.0 (baseline)
        - Apr, Oct: 1.15 (transition)
        - May-Sep (warm): 1.3 (e_warm)
        - Lactating females: multiply by 1.4 (e_lact)
        
        Args:
            month: Current month (1-12)
            mask: Active porpoise mask
            
        Returns:
            Scaling factor array for each porpoise
        """
        scaling = np.ones(self.count, dtype=np.float32)
        
        # Seasonal scaling
        if month == 4 or month == 10:
            # April and October - transition months
            scaling[:] = 1.15
        elif 5 <= month <= 9:
            # May through September - warm months
            scaling[:] = self.params.e_warm
        # Nov-Mar stays at 1.0 (cold months, lower metabolism)
        
        # Lactation scaling (40% increase)
        lactating = self.with_calf & mask
        scaling[lactating] *= self.params.e_lact
        
        return scaling
        
    def get_energy_stats(self) -> Dict[str, Any]:
        """Get statistics about population energy levels."""
        active = self.active_mask
        if not np.any(active):
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'hungry': 0, 'starving': 0}
            
        active_energy = self.energy[active]
        return {
            'mean': float(np.mean(active_energy)),
            'std': float(np.std(active_energy)),
            'min': float(np.min(active_energy)),
            'max': float(np.max(active_energy)),
            'hungry': int(np.sum(active_energy < 10)),  # Below neutral
            'starving': int(np.sum(active_energy < 5))  # Critical
        }
