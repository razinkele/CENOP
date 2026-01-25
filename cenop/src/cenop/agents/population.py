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
        # Guard to prevent double-updating energy within the same tick
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
        self._current_recompute_interval: int = max(1, int(getattr(self.params, 'communication_recompute_interval', 1)))
        # Previous positions for displacement calculation (in cell units)
        self._prev_x = self.x.copy()
        self._prev_y = self.y.copy()
        # EMA of mean displacement (meters per tick)
        self._disp_ema_m: float = 0.0
        
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

            if not getattr(self.params, 'communication_enabled', False):
                return social_dx, social_dy

            # Communication parameters
            comm_range_km = getattr(self.params, 'communication_range_km', 10.0)
            comm_cells = max(1, int(np.ceil((comm_range_km * 1000.0) / 400.0)))
            source_level = getattr(self.params, 'communication_source_level', 160.0)
            threshold = getattr(self.params, 'communication_threshold', 120.0)
            slope = getattr(self.params, 'communication_response_slope', 0.2)
            social_weight = getattr(self.params, 'social_weight', 0.3)

            active_idx = np.where(mask)[0]
            if len(active_idx) == 0:
                return social_dx, social_dy

            # Try to use scipy.spatial.cKDTree for fast neighbor queries
            cKDTree: Any = None
            try:
                from scipy.spatial import cKDTree  # type: ignore
                use_kdtree = True
            except Exception:
                use_kdtree = False

            if use_kdtree:
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
                    kd_active = cKDTree(pos_active)

                    # Use query_ball_tree to get neighbor lists (no large sparse matrix allocation)
                    try:
                        neigh_lists = kd_active.query_ball_tree(kd_active, r=radius)
                    except Exception:
                        # Fallback to empty
                        neigh_lists = []

                    if not neigh_lists:
                        # Reset cache counter to avoid repeated work
                        self._neighbor_recompute_counter = interval
                        self._social_cache = {'idx_i': np.array([], dtype=int), 'idx_j': np.array([], dtype=int), 'ncols': 0, 'active_len': len(active_idx)}
                        return social_dx, social_dy

                    # Build canonical pairs (local indices) rows < cols to avoid duplicates
                    rows_local = []
                    cols_local = []
                    for i_local, neigh in enumerate(neigh_lists):
                        if not neigh:
                            continue
                        for j_local in neigh:
                            if j_local <= i_local:
                                continue
                            rows_local.append(i_local)
                            cols_local.append(j_local)

                    if len(rows_local) == 0:
                        self._neighbor_recompute_counter = interval
                        self._social_cache = {'idx_i': np.array([], dtype=int), 'idx_j': np.array([], dtype=int), 'ncols': 0, 'active_len': len(active_idx)}
                        return social_dx, social_dy

                    rows = np.array(rows_local, dtype=int)
                    cols = np.array(cols_local, dtype=int)

                    # Map to global indices and cache
                    idx_i = active_idx[rows].astype(int)
                    idx_j = active_idx[cols].astype(int)
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

                # Coordinates
                xi = self.x[idx_i].astype(np.float64)
                yi = self.y[idx_i].astype(np.float64)
                xj = self.x[idx_j].astype(np.float64)
                yj = self.y[idx_j].astype(np.float64)

                # Displacements and distances recomputed each tick (topology reused)
                dx_ij = xj - xi
                dy_ij = yj - yi
                dist = np.hypot(dx_ij, dy_ij) + 1e-6
                dist_m = dist * 400.0

                # Received level (same for both directions since distance symmetric)
                rl_pairs = calculate_received_level(source_level, dist_m, self.params.alpha_hat, self.params.beta_hat)

                # Probabilities: listener i hearing caller j
                if ambient_rl is not None:
                    ambient_i = np.asarray(ambient_rl[idx_i], dtype=np.float64)
                    snr_i = rl_pairs - ambient_i
                    p_i = response_probability_from_rl(snr_i, threshold, slope)
                    ambient_j = np.asarray(ambient_rl[idx_j], dtype=np.float64)
                    snr_j = rl_pairs - ambient_j
                    p_j = response_probability_from_rl(snr_j, threshold, slope)
                else:
                    p_i = response_probability_from_rl(rl_pairs, threshold, slope)
                    p_j = response_probability_from_rl(rl_pairs, threshold, slope)

                # Unit vectors for i's listener (pointing towards callers)
                ux_ij = dx_ij / dist
                uy_ij = dy_ij / dist

                # For j's listener, vectors are reversed
                ux_ji = -ux_ij
                uy_ji = -uy_ij

                # Weighted contributions
                ux_contrib_i = ux_ij * p_i
                uy_contrib_i = uy_ij * p_i
                ux_contrib_j = ux_ji * p_j
                uy_contrib_j = uy_ji * p_j

                # Accumulate per-agent contributions using a Numba-accelerated accumulator (fallback to bincount)
                ux_total = np.zeros(self.count, dtype=np.float64)
                uy_total = np.zeros(self.count, dtype=np.float64)
                sw_total = np.zeros(self.count, dtype=np.float64)
                try:
                    from cenop.optimizations.numba_helpers import accumulate_social_totals

                    # Ensure dtypes are compatible for the compiled function
                    idx_i_arr = np.asarray(idx_i, dtype=np.int64)
                    idx_j_arr = np.asarray(idx_j, dtype=np.int64)

                    ux_ci = np.asarray(ux_contrib_i, dtype=np.float64)
                    uy_ci = np.asarray(uy_contrib_i, dtype=np.float64)
                    ux_cj = np.asarray(ux_contrib_j, dtype=np.float64)
                    uy_cj = np.asarray(uy_contrib_j, dtype=np.float64)
                    p_i_arr = np.asarray(p_i, dtype=np.float64)
                    p_j_arr = np.asarray(p_j, dtype=np.float64)

                    accumulate_social_totals(
                        np.int64(self.count), idx_i_arr, idx_j_arr,
                        ux_ci, uy_ci, ux_cj, uy_cj, p_i_arr, p_j_arr,
                        ux_total, uy_total, sw_total
                    )

                except Exception:
                    # Fallback to bincount if no numba or accumulator fails
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
