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

logger = logging.getLogger('cenop.agents.population')


class PorpoisePopulation:
    """
    Manages the entire population of porpoises using vectorized numpy arrays.
    Replaces the list of individual Porpoise objects for performance.
    """
    
    def __init__(self, count: int, params: SimulationParameters, landscape: Optional[CellData] = None):
        self.params = params
        self.landscape = landscape
        self.count = count # Initial count capacity
        
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
        self.prev_log_mov = np.full(count, 0.8, dtype=np.float32)
        self.prev_angle = np.full(count, 10.0, dtype=np.float32)
        
        # Demography
        self.is_female = np.zeros(count, dtype=bool)
        self.age = np.zeros(count, dtype=np.float32)
        
        # Energy
        self.energy = np.full(count, 10.0, dtype=np.float32)
        
        # Reproduction
        self.mating_day = np.full(count, -99, dtype=np.int16)
        self.days_since_mating = np.full(count, -99, dtype=np.int16)
        self.days_since_birth = np.full(count, -99, dtype=np.int16)
        self.with_calf = np.zeros(count, dtype=bool)
        
        # Deterrence status
        self.deter_strength = np.zeros(count, dtype=np.float32)
        
        # === PSM and Dispersal State (Phase 2) ===
        # Energy history for dispersal trigger (5 days = 5*48 ticks)
        self._energy_history = np.zeros((count, 5), dtype=np.float32)  # Last 5 daily averages
        self._energy_ticks_today = np.zeros(count, dtype=np.float32)   # Energy sum for current day
        self._tick_counter = 0  # Track ticks for daily updates
        self._last_energy_update_tick = -1  # last global tick when energy was accumulated
        
        # Dispersal state
        self.is_dispersing = np.zeros(count, dtype=bool)
        self.days_declining_energy = np.zeros(count, dtype=np.int16)
        self.dispersal_target_x = np.zeros(count, dtype=np.float32)
        self.dispersal_target_y = np.zeros(count, dtype=np.float32)
        self.dispersal_target_distance = np.zeros(count, dtype=np.float32)
        self.dispersal_distance_traveled = np.zeros(count, dtype=np.float32)
        self.dispersal_start_x = np.zeros(count, dtype=np.float32)
        self.dispersal_start_y = np.zeros(count, dtype=np.float32)
        
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
        self._rand_angle = np.zeros(count, dtype=np.float32)
        self._rand_len = np.zeros(count, dtype=np.float32)
        self._pres_angle = np.zeros(count, dtype=np.float32)
        self._log_mov = np.zeros(count, dtype=np.float32)
        self._step_dist = np.zeros(count, dtype=np.float32)
        self._rads = np.zeros(count, dtype=np.float32)
        self._dx = np.zeros(count, dtype=np.float32)
        self._dy = np.zeros(count, dtype=np.float32)
        self._new_x = np.zeros(count, dtype=np.float32)
        self._new_y = np.zeros(count, dtype=np.float32)
        self._new_xi = np.zeros(count, dtype=np.int32)
        self._new_yi = np.zeros(count, dtype=np.int32)
        self._depths = np.zeros(count, dtype=np.float32)
        self._salinity_vals = np.zeros(count, dtype=np.float32)  # For CRW environmental modulation
        self._env_mod_angle = np.zeros(count, dtype=np.float32)  # b1*depth + b2*salinity + b3
        self._scaling_factor = np.zeros(count, dtype=np.float32)

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
        def _compute_social_vectors(self, mask: np.ndarray, ambient_rl: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute social attraction vectors for active agents.

            Implementation uses a cKDTree-based neighbor lookup (fast C implementation) when
            SciPy is available, otherwise falls back to the previous binning approach.
            Detection is still probabilistic and masked by ambient noise (SNR).
            """
            social_dx = np.zeros(self.count, dtype=np.float32)
            social_dy = np.zeros(self.count, dtype=np.float32)

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
                    except Exception:
                        # Fallback for older scipy versions or errors
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
                        except Exception:
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

                # Unit vectors for i's listener (pointing towards callers)
                ux_ij = dx_ij / dist
                uy_ij = dy_ij / dist

                # For j's listener, vectors are reversed
                # Just use multipliers later instead of allocating new arrays?
                # No, we need explicit vectors for accumulator.
                ux_ji = -ux_ij
                uy_ji = -uy_ij

                # Weighted contributions
                ux_contrib_i = ux_ij * p_i
                uy_contrib_i = uy_ij * p_i
                ux_contrib_j = ux_ji * p_j
                uy_contrib_j = uy_ji * p_j

                # Accumulate per-agent contributions using a Numba-accelerated accumulator (fallback to bincount)
                # Use float64 accumulators for precision, but inputs can be float32
                ux_total = np.zeros(self.count, dtype=np.float64)
                uy_total = np.zeros(self.count, dtype=np.float64)
                sw_total = np.zeros(self.count, dtype=np.float64)
                
                if _HAS_NUMBA_HELPERS and _accumulate_social_totals is not None:
                    # Numba will compile a specialization for the input types (float32)
                    # We pass the arrays directly without casting to float64
                    try:
                        _accumulate_social_totals(
                            np.int64(self.count), idx_i, idx_j,
                            ux_contrib_i, uy_contrib_i, ux_contrib_j, uy_contrib_j, p_i, p_j,
                            ux_total, uy_total, sw_total
                        )
                    except Exception:
                        # Fallback if numba call fails
                        ux_total = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([ux_contrib_i, ux_contrib_j]), minlength=self.count)
                        uy_total = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([uy_contrib_i, uy_contrib_j]), minlength=self.count)
                        sw_total = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([p_i, p_j]), minlength=self.count)
                else:
                    # Fallback to bincount if no numba
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

                # Apply social weight and step length
                social_dx = unit_x.astype(np.float32) * social_weight * step_dist.astype(np.float32)
                social_dy = unit_y.astype(np.float32) * social_weight * step_dist.astype(np.float32)

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
        # === Get environmental variables from landscape ===
        # DEPONS CRW uses depth and salinity to modulate movement
        if self.landscape is not None:
            # Build positions array for vectorized lookup
            positions = np.column_stack((self.x, self.y))
            np.copyto(self._depths, self.landscape.get_depths_vectorized(positions))
            np.copyto(self._salinity_vals, self.landscape.get_salinities_vectorized(positions))
        else:
            # Default values when no landscape (homogeneous case)
            self._depths.fill(30.0)  # Default depth
            self._salinity_vals.fill(30.0)  # Default salinity

        # === Calculate Turning Angle (Full DEPONS CRW) ===
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

        # Wrap angles > 180 or < -180
        np.clip(self._pres_angle, -180, 180, out=self._pres_angle)

        self.heading[mask] += self._pres_angle[mask]
        self.heading[mask] %= 360.0

        # Apply dispersal heading override for dispersing porpoises
        self._apply_dispersal_heading(mask)

        self.prev_angle[mask] = self._pres_angle[mask]

        # === Calculate Step Length (Full DEPONS CRW) ===
        # DEPONS formula: log10_mov = a0 * prev_log_mov + a1*depth + a2*salinity + R1
        np.copyto(self._rand_len, np.random.normal(self.params.r1_mean, self.params.r1_sd, self.count))
        np.multiply(self.params.corr_logmov_length, self.prev_log_mov, out=self._log_mov)
        self._log_mov += self.params.corr_logmov_bathy * self._depths
        self._log_mov += self.params.corr_logmov_salinity * self._salinity_vals
        self._log_mov += self._rand_len

        # Clip max speed
        np.minimum(self._log_mov, self.params.max_mov, out=self._log_mov)
        self.prev_log_mov[mask] = self._log_mov[mask]

        # Convert to distance (10^log_mov) / 4.0 (for 400m cell adjustment)
        np.power(10.0, self._log_mov, out=self._step_dist)
        self._step_dist /= 4.0

        # Determine new positions
        np.radians(self.heading, out=self._rads)
        np.sin(self._rads, out=self._dx)
        self._dx *= self._step_dist
        np.cos(self._rads, out=self._dy)
        self._dy *= self._step_dist

        # Apply deterrence if exists
        if deterrence_vectors is not None:
            d_dx, d_dy = deterrence_vectors
            self.deter_strength[mask] = np.abs(d_dx[mask]) + np.abs(d_dy[mask])
            self._dx[mask] += d_dx[mask]
            self._dy[mask] += d_dy[mask]
        else:
            self.deter_strength[mask] = 0.0

        # Social communication & cohesion
        if getattr(self.params, 'communication_enabled', False):
            soc_dx, soc_dy = self._compute_social_vectors(mask, ambient_rl)
            self._dx[mask] += soc_dx[mask]
            self._dy[mask] += soc_dy[mask]

    def _apply_dispersal_heading(self, mask: np.ndarray) -> None:
        """Apply reduced turning for dispersing porpoises (PSM-Type2)."""
        dispersing = mask & self.is_dispersing
        if not np.any(dispersing):
            return

        # Calculate distance progress for logistic dampening
        dx_disp = self.x - self.dispersal_start_x
        dy_disp = self.y - self.dispersal_start_y
        dist_traveled = np.sqrt(dx_disp**2 + dy_disp**2)

        # Safe division: avoid divide by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            dist_perc = np.where(
                self.dispersal_target_distance > 0,
                dist_traveled / self.dispersal_target_distance,
                0.0
            )
        dist_perc = np.nan_to_num(dist_perc, nan=0.0, posinf=1.0, neginf=0.0)
        dist_perc = np.clip(dist_perc, 0.0, 2.0)

        # Logistic dampening: as progress -> 1, turning -> 0
        dist_log_x = 3 * dist_perc - 1.5
        dist_log_x = np.clip(dist_log_x, -100, 100)  # Prevent overflow
        log_mult = 1.0 / (1.0 + np.exp(0.6 * dist_log_x))

        # Reduce turning angle for dispersing porpoises
        dampened_angle = self._pres_angle * log_mult * 0.3  # 70% reduction base
        self.heading[dispersing] -= self._pres_angle[dispersing]  # Remove normal turn
        self.heading[dispersing] += dampened_angle[dispersing]  # Add dampened
        self.heading[dispersing] %= 360.0

    def _handle_land_avoidance(self, mask: np.ndarray) -> None:
        """
        Handle land avoidance using DEPONS pattern.

        Checks if proposed positions are on land and tries turning
        40°, 70°, 120° in both directions to find water.
        """
        # Calculate proposed new positions
        np.add(self.x, self._dx, out=self._new_x)
        np.add(self.y, self._dy, out=self._new_y)

        # Clamp to bounds
        world_w = self.landscape.width if self.landscape else self.params.world_width
        world_h = self.landscape.height if self.landscape else self.params.world_height
        np.clip(self._new_x, 0, world_w - 1, out=self._new_x)
        np.clip(self._new_y, 0, world_h - 1, out=self._new_y)

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

        # Try turning to avoid land (DEPONS pattern)
        for turn_angle in [40, 70, 120]:
            np.copyto(self._still_blocked, self._on_land)

            # Compute positions for both turn directions
            right_heading = self._compute_turn_position(
                turn_angle, world_w, world_h,
                self._right_x, self._right_y,
                self._right_xi, self._right_yi,
                self._right_depths
            )
            left_heading = self._compute_turn_position(
                -turn_angle, world_w, world_h,
                self._left_x, self._left_y,
                self._left_xi, self._left_yi,
                self._left_depths
            )

            # Pick deeper direction if valid water
            right_ok = ((self._right_depths >= min_depth) & ~np.isnan(self._right_depths)) & self._still_blocked
            left_ok = ((self._left_depths >= min_depth) & ~np.isnan(self._left_depths)) & self._still_blocked
            both_ok = right_ok & left_ok

            # If both OK, pick deeper
            use_right = both_ok & (self._right_depths >= self._left_depths)
            use_left = both_ok & (self._left_depths > self._right_depths)

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

        # For any still blocked, stay in place and turn around
        self._new_x[self._on_land] = self.x[self._on_land]
        self._new_y[self._on_land] = self.y[self._on_land]
        self.heading[self._on_land] = (self.heading[self._on_land] + 180) % 360

    def _apply_positions(self, mask: np.ndarray) -> None:
        """Apply final positions and update adaptive neighbor recompute."""
        self.x[mask] = self._new_x[mask]
        self.y[mask] = self._new_y[mask]

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
        except Exception:
            pass

        # Save positions for next tick
        self._prev_x[mask] = self.x[mask]
        self._prev_y[mask] = self.y[mask]

    def _update_energy_dynamics(self, mask: np.ndarray) -> None:
        """
        Update energy dynamics (DEPONS Pattern).

        Handles food consumption, BMR, swimming cost, and PSM updates.
        """
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
        swimming_cost = (10.0 ** self.prev_log_mov) * 0.001 * scaling_factor * 0.0  # E_USE_PER_KM = 0

        total_cost = bmr_cost + swimming_cost
        self.energy[mask] -= total_cost[mask]

        # Update PSM and energy history
        self._update_psm(mask, food_gained)
        self._update_energy_history(mask)
        self._update_dispersal(mask)

        # Clamp energy
        np.clip(self.energy, 0, 20.0, out=self.energy)

    def _check_mortality(self, mask: np.ndarray, active_before: int) -> None:
        """
        Check and apply mortality (DEPONS Pattern).

        Handles starvation, age-based natural mortality, and bycatch.
        Uses parameters from SimulationParameters for all mortality constants.
        """
        # Energy-based starvation mortality (parameterized)
        m_mort_prob_const = getattr(self.params, 'm_mort_prob_const', 0.5)
        x_survival_const = getattr(self.params, 'x_survival_const', 0.15)

        yearly_surv_prob = np.where(
            self.energy > 0,
            1.0 - (m_mort_prob_const * np.exp(-self.energy * x_survival_const)),
            0.0
        )
        step_surv_prob = np.where(
            self.energy > 0,
            np.exp(np.log(np.maximum(yearly_surv_prob, 1e-10)) / (360 * 48)),
            0.0
        )

        starvation_check = np.random.random(self.count)
        starving = (starvation_check > step_surv_prob) & mask

        # Lactating mothers abandon calf before dying
        abandon_calf = starving & self.with_calf
        self.with_calf[abandon_calf] = False
        starved = starving & ~abandon_calf

        # Age-dependent natural mortality (parameterized)
        annual_juvenile_mortality = getattr(self.params, 'mortality_juvenile', 0.15)  # age < 1
        annual_adult_mortality = getattr(self.params, 'mortality_adult', 0.05)        # 1 <= age <= 20
        annual_elderly_mortality = getattr(self.params, 'mortality_elderly', 0.15)    # age > 20

        per_tick_juvenile = annual_juvenile_mortality / 365.0 / 48.0
        per_tick_adult = annual_adult_mortality / 365.0 / 48.0
        per_tick_elderly = annual_elderly_mortality / 365.0 / 48.0

        daily_mortality_prob = np.where(
            self.age < 1, per_tick_juvenile,
            np.where(self.age > 20, per_tick_elderly, per_tick_adult)
        )
        natural_death = (np.random.random(self.count) < daily_mortality_prob) & mask

        # Bycatch mortality (already parameterized)
        bycatch_prob = getattr(self.params, 'bycatch_prob', 0.0) / 365.0 / 48.0
        bycatch = (np.random.random(self.count) < bycatch_prob) & mask

        # Apply deaths
        all_deaths = starved | natural_death | bycatch
        if np.any(all_deaths):
            death_count = int(np.sum(all_deaths))
            starved_count = int(np.sum(starved))
            natural_count = int(np.sum(natural_death))
            bycatch_count = int(np.sum(bycatch))
            self.active_mask[all_deaths] = False
            if self._debug_instrumentation or death_count > 0:
                active_after = int(np.sum(self.active_mask))
                logger.debug(
                    "[INSTR] tick=%d deaths=%d starved=%d natural=%d bycatch=%d active_before=%d active_after=%d",
                    self._global_tick, death_count, starved_count, natural_count, bycatch_count,
                    active_before, active_after
                )

    def _update_aging(self, mask: np.ndarray) -> None:
        """Update aging for active agents (continuous small increments)."""
        self.age[mask] += 1.0 / 365.0 / 48.0  # Age in years per tick

    def _handle_reproduction(self, mask: np.ndarray) -> None:
        """
        Handle reproduction during breeding season.

        Mature females (age 4-20) can give birth during days 195-255.
        """
        # Update day of year
        if hasattr(self, '_day_of_year'):
            self._day_of_year = (self._day_of_year + 1) % (365 * 48)
        else:
            self._day_of_year = 0

        current_day = self._day_of_year // 48

        # Breeding season: days 195-255
        if not (195 <= current_day <= 255):
            return

        # Eligible females: maturity_age to max_breeding_age, not already with calf
        maturity_age = getattr(self.params, 'maturity_age', 3.44)
        max_breeding_age = getattr(self.params, 'max_breeding_age', 20.0)
        eligible = mask & self.is_female & (self.age >= maturity_age) & (self.age <= max_breeding_age) & ~self.with_calf

        # Per-tick birth probability (~60% over 60-day season)
        birth_prob = 0.0003
        giving_birth = (np.random.random(self.count) < birth_prob) & eligible

        if not np.any(giving_birth):
            return

        # Find inactive slots for new calves
        inactive_slots = np.where(~self.active_mask)[0]
        birth_count = int(np.sum(giving_birth))
        mother_indices = np.where(giving_birth)[0]
        inactive_before = int(len(inactive_slots))

        slots_to_use = min(birth_count, len(inactive_slots))
        if slots_to_use > 0:
            new_slots = inactive_slots[:slots_to_use]
            mothers = mother_indices[:slots_to_use]

            # Activate new calves
            self.active_mask[new_slots] = True
            self.x[new_slots] = self.x[mothers]
            self.y[new_slots] = self.y[mothers]
            self.heading[new_slots] = self.heading[mothers]
            self.age[new_slots] = 0.0
            self.is_female[new_slots] = np.random.choice([True, False], size=int(slots_to_use))
            self.energy[new_slots] = 10.0
            self.with_calf[mothers] = True
            self.with_calf[new_slots] = False

            if self._debug_instrumentation or birth_count > 0:
                created = int(slots_to_use)
                logger.debug(
                    "[INSTR] tick=%d births_attempted=%d births_created=%d inactive_slots_before=%d",
                    self._global_tick, birth_count, created, inactive_before
                )

    def step(self, deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None, ambient_rl: Optional[np.ndarray] = None):
        """
        Main simulation step for the entire population.

        Orchestrates all sub-steps in sequence:
        1. Movement calculations (CRW, deterrence, social)
        2. Land avoidance (DEPONS pattern)
        3. Position updates
        4. Energy dynamics (food, BMR, PSM)
        5. Mortality (starvation, age-based, bycatch)
        6. Aging
        7. Reproduction

        Args:
            deterrence_vectors: Tuple of (dx_array, dy_array) for deterrence
            ambient_rl: Ambient received level for social communication
        """
        mask = self.active_mask
        if not np.any(mask):
            return

        active_before = int(np.sum(self.active_mask))
        self._global_tick += 1

        # 1. Movement calculations
        self._update_movement(mask, deterrence_vectors, ambient_rl)

        # 2. Land avoidance
        self._handle_land_avoidance(mask)

        # 3. Apply positions
        self._apply_positions(mask)

        # 4. Energy dynamics
        self._update_energy_dynamics(mask)

        # 5. Mortality
        self._check_mortality(mask, active_before)

        # 6. Aging
        self._update_aging(mask)

        # 7. Reproduction
        self._handle_reproduction(mask)

    def to_dataframe(self) -> pd.DataFrame:
        """Export active agents to DataFrame for UI helpers."""
        mask = self.active_mask
        n_active = np.sum(mask)
        return pd.DataFrame({
            'id': self.ids[mask],
            'x': self.x[mask],
            'y': self.y[mask],
            'age': self.age[mask],
            'is_female': self.is_female[mask],
            'energy': self.energy[mask],
            'alive': np.ones(n_active, dtype=bool)
        })

    # === PSM and Dispersal Methods (Phase 2) ===
    
    def _update_psm(self, mask: np.ndarray, food_gained: np.ndarray) -> None:
        """
        Update Persistent Spatial Memory (Vectorized).
        
        Records food obtained at current location for dispersal targeting directly into
        the vectorized PSM buffer.
        
        Args:
            mask: Active porpoise mask
            food_gained: Array of food gained this tick
        """
        active_idx = np.where(mask)[0]
        if len(active_idx) == 0:
            return
            
        # Convert positions to PSM grid coordinates
        psm_x = (self.x[active_idx] // self.psm_cell_size).astype(int)
        psm_y = (self.y[active_idx] // self.psm_cell_size).astype(int)
        
        # Clip to bounds
        np.clip(psm_x, 0, self.psm_cols - 1, out=psm_x)
        np.clip(psm_y, 0, self.psm_rows - 1, out=psm_y)
        
        # Use efficient accumulator (Numba-accelerated when available)
        from cenop.optimizations import accumulate_psm_updates

        idx_arr = active_idx.astype(np.int32)
        ys_arr = psm_y.astype(np.int32)
        xs_arr = psm_x.astype(np.int32)
        food_arr = food_gained[active_idx].astype(np.float32)

        try:
            accumulate_psm_updates(self.psm_buffer, idx_arr, ys_arr, xs_arr, food_arr)
        except Exception:
            # Fallback to np.add.at for safety
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

        for idx in candidate_indices:
            # Check memory count from buffer
            # Count cells with ticks > 0
            visited_count = int(np.count_nonzero(self.psm_buffer[idx, :, :, 0]))
            if visited_count >= min_memory_cells:
                self._start_dispersal(idx)

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
        """
        self.is_dispersing[idx] = True
        self.dispersal_start_x[idx] = self.x[idx]
        self.dispersal_start_y[idx] = self.y[idx]
        self.dispersal_distance_traveled[idx] = 0.0
        
        # Use vectorized PSM buffer scan
        mem_slice = self.psm_buffer[idx] # (rows, cols, 2)
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
        dists = np.sqrt(dx*dx + dy*dy)
        
        # Filter for tolerance (5km approx 12.5 cells)
        tolerance_cells = 12.5
        valid_mask = np.abs(dists - pref_dist_cells) < tolerance_cells
        
        if np.any(valid_mask):
            # Pick highest value among valid distance cells
            # Filter arrays
            valid_expectations = expectations[valid_mask]
            
            # Find best
            best_local_idx = np.argmax(valid_expectations)
            
            # Map back to original indices
            # valid_mask is a boolean mask into visited_x/y arrays
            # We need the index in the filtered array -> corresponding index in visited arrays
            
            # Indices of valid cells in the 'visited' arrays
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
        """Set a random dispersal target at preferred distance."""
        pref_dist_km = self._psm_instances[idx].preferred_distance
        angle_rad = np.random.uniform(0, 2 * np.pi)
        dist_cells = pref_dist_km * 1000 / 400.0
        
        tx = self.x[idx] + np.sin(angle_rad) * dist_cells
        ty = self.y[idx] + np.cos(angle_rad) * dist_cells
        
        # Clamp to world
        w = self.landscape.width if self.landscape else self.params.world_width
        h = self.landscape.height if self.landscape else self.params.world_height
        
        self.dispersal_target_x[idx] = np.clip(tx, 0, w - 1)
        self.dispersal_target_y[idx] = np.clip(ty, 0, h - 1)
        self.dispersal_target_distance[idx] = dist_cells

        
    def _update_dispersal(self, mask: np.ndarray) -> None:
        """
        Update dispersal progress for dispersing porpoises.
        
        Check if target distance reached and end dispersal if so.
        """
        dispersing = mask & self.is_dispersing
        if not np.any(dispersing):
            return
            
        # Calculate distance from start
        dx = self.x - self.dispersal_start_x
        dy = self.y - self.dispersal_start_y
        distances = np.sqrt(dx**2 + dy**2)
        
        # Check for completion (95% of target distance - PSM-Type2 rule)
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

        Returns:
            Heading array after turning
        """
        heading = (self.heading + turn_delta) % 360
        rads = np.radians(heading)

        # Compute dx, dy using pre-allocated _dx, _dy arrays
        np.multiply(np.sin(rads), self._step_dist, out=self._dx)
        np.multiply(np.cos(rads), self._step_dist, out=self._dy)

        # Clip to world bounds
        np.clip(self.x + self._dx, 0, world_w - 1, out=out_x)
        np.clip(self.y + self._dy, 0, world_h - 1, out=out_y)

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
             fract_to_eat[active_idx]
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
