"""DEPONS 3.2 Reference Memory — decay tables and vectorized computation.

Ports Java RefMem.java and FastRefMemTurn.java to vectorized NumPy.
"""

import numpy as np
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class RefMemWorkspace:
    """Pre-allocated workspace for reference memory computations.

    Create once in population.__init__ and pass to compute functions
    to avoid per-tick allocations.
    """

    # Shape: (n_agents, mem_size) - shared by both compute functions
    active_indices: np.ndarray  # int16
    ordered_food: np.ndarray  # float32
    valid_mask: np.ndarray  # bool
    weighted: np.ndarray  # float32
    # Additional arrays for compute_attraction_vector
    ordered_util: np.ndarray  # float32
    ordered_px: np.ndarray  # float32
    ordered_py: np.ndarray  # float32
    attr_x: np.ndarray  # float32
    attr_y: np.ndarray  # float32
    dist: np.ndarray  # float32
    unit_x: np.ndarray  # float32
    unit_y: np.ndarray  # float32
    factor: np.ndarray  # float32

    @classmethod
    def create(cls, n_agents: int, mem_size: int) -> "RefMemWorkspace":
        return cls(
            active_indices=np.zeros((n_agents, mem_size), dtype=np.int16),
            ordered_food=np.zeros((n_agents, mem_size), dtype=np.float32),
            valid_mask=np.zeros((n_agents, mem_size), dtype=bool),
            weighted=np.zeros((n_agents, mem_size), dtype=np.float32),
            ordered_util=np.zeros((n_agents, mem_size), dtype=np.float32),
            ordered_px=np.zeros((n_agents, mem_size), dtype=np.float32),
            ordered_py=np.zeros((n_agents, mem_size), dtype=np.float32),
            attr_x=np.zeros((n_agents, mem_size), dtype=np.float32),
            attr_y=np.zeros((n_agents, mem_size), dtype=np.float32),
            dist=np.zeros((n_agents, mem_size), dtype=np.float32),
            unit_x=np.zeros((n_agents, mem_size), dtype=np.float32),
            unit_y=np.zeros((n_agents, mem_size), dtype=np.float32),
            factor=np.zeros((n_agents, mem_size), dtype=np.float32),
        )


def _compute_decay_table(first_value: float, rate: float, size: int) -> np.ndarray:
    """Compute logistic decay table: s[i+1] = s[i] - rate * s[i] * (1 - s[i]).

    Java ref: RefMem.java:100-108 (calcArray)
    """
    table = np.zeros(size, dtype=np.float64)
    table[0] = first_value
    for i in range(1, size):
        prev = table[i - 1]
        table[i] = prev - rate * prev * (1.0 - prev)
    # Round to 4 decimals to match Java BigDecimal rounding
    result = np.round(table, 4)
    result.flags.writeable = False  # Prevent mutation of cached array
    return result


@lru_cache(maxsize=4)
def get_ref_mem_strength_table(r_r: float = 0.03, size: int = 120) -> np.ndarray:
    """Get reference memory strength table (rR decay).

    Controls how strongly past locations attract the porpoise.
    Slow decay = long memory.
    """
    return _compute_decay_table(0.999, r_r, size)


@lru_cache(maxsize=4)
def get_work_mem_strength_table(r_s: float = 0.03, size: int = 120) -> np.ndarray:
    """Get working memory strength table (rS decay).

    Controls veTotal — the expected food value weighting.
    Faster decay = more recent experience weighted.
    """
    return _compute_decay_table(0.999, r_s, size)


def _build_ordered_indices(
    mem_ptr: np.ndarray,
    mem_count: np.ndarray,
    mem_size: int,
    mask: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Build (count, mem_size) array of circular buffer indices ordered most-recent-first.

    For each agent, indices[agent, 0] = most recent entry, indices[agent, 1] = second most recent,
    etc. Entries beyond mem_count are set to -1 (masked out).

    If *out* is provided it must have shape >= (len(mask), mem_size) and will be written into
    (avoiding a fresh allocation).
    """
    count = len(mask)
    # Offsets 0..mem_size-1
    offsets = np.arange(mem_size, dtype=np.int16)
    # For each agent: (ptr - 1 - offset) % mem_size
    computed = (mem_ptr[:, np.newaxis] - 1 - offsets[np.newaxis, :]) % mem_size
    if out is not None:
        out[:count, :] = computed
        return out[:count, :]
    return computed


def compute_ve_total(
    stored_util: np.ndarray,
    mem_ptr: np.ndarray,
    mem_count: np.ndarray,
    work_mem_table: np.ndarray,
    mask: np.ndarray,
    workspace: RefMemWorkspace | None = None,
) -> np.ndarray:
    """Compute veTotal (expected food value) for each agent — fully vectorized.

    veTotal = sum(workMemStrength[i] * storedUtilList[i]) for i in 0..N-2

    Java ref: Porpoise.java:688-705 (getExpFoodVal)

    If *workspace* is provided, intermediate arrays are taken from pre-allocated buffers
    instead of allocating fresh memory each call.
    """
    count = len(mask)
    ve_total = np.zeros(count, dtype=np.float32)

    if not np.any(mask):
        return ve_total

    mem_size = stored_util.shape[1]
    active = np.where(mask)[0]
    if len(active) == 0:
        return ve_total

    n_active = len(active)

    # Try Numba fused kernel first
    try:
        from cenop.optimizations.kernels import compute_ve_total_kernel
        out = np.zeros(n_active, dtype=np.float64)
        compute_ve_total_kernel(
            stored_util, mem_ptr,
            mem_count,
            work_mem_table if work_mem_table.dtype == np.float64 else work_mem_table.astype(np.float64),
            active.astype(np.int64), out,
        )
        ve_total[active] = out.astype(np.float32)
        return ve_total
    except ImportError:
        pass

    # Build ordered indices: (count, mem_size) — most recent first
    indices = _build_ordered_indices(mem_ptr, mem_count, mem_size, mask)

    # Gather food values in recency order for active agents
    # Advanced indexing always creates a copy; workspace saves on downstream buffers
    active_indices = indices[active]  # (n_active, mem_size)
    row_idx = active[:, np.newaxis]  # (n_active, 1) for broadcasting
    ordered_food = stored_util[row_idx, active_indices]  # (n_active, mem_size)

    # Mask out entries beyond each agent's count (Java skips oldest, so use n-1)
    n_valid = np.minimum(mem_count[active].astype(np.int32), mem_size) - 1  # n-1 entries
    n_valid = np.maximum(n_valid, 0)
    entry_idx = np.arange(mem_size)[np.newaxis, :]  # (1, mem_size)

    if workspace is not None:
        valid_mask = workspace.valid_mask[:n_active, :]
        np.less(entry_idx, n_valid[:, np.newaxis], out=valid_mask)
        weighted = workspace.weighted[:n_active, :]
        weights = work_mem_table[:mem_size].astype(np.float32)
        np.multiply(ordered_food, weights[np.newaxis, :], out=weighted)
        weighted *= valid_mask
    else:
        valid_mask = entry_idx < n_valid[:, np.newaxis]  # (n_active, mem_size)
        weights = work_mem_table[:mem_size].astype(np.float32)
        weighted = ordered_food * weights[np.newaxis, :] * valid_mask

    ve_total[active] = weighted.sum(axis=1).astype(np.float32)

    return ve_total


def compute_attraction_vector(
    stored_util: np.ndarray,
    pos_history_x: np.ndarray,
    pos_history_y: np.ndarray,
    mem_ptr: np.ndarray,
    mem_count: np.ndarray,
    current_x: np.ndarray,
    current_y: np.ndarray,
    ref_mem_table: np.ndarray,
    mask: np.ndarray,
    world_width: int = 0,
    world_height: int = 0,
    workspace: RefMemWorkspace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute attraction vector vt for each agent — fully vectorized.

    For each past position with nonzero food:
      weight = storedUtil[i] * refMemStrength[i] / distance
      vt += weight * unit_direction_to_past_position

    Java ref: FastRefMemTurn.java:44-125

    If *workspace* is provided, intermediate arrays are taken from pre-allocated buffers
    instead of allocating fresh memory each call.

    Returns: (vt_x, vt_y) arrays of shape (count,)
    """
    count = len(mask)
    vt_x = np.zeros(count, dtype=np.float32)
    vt_y = np.zeros(count, dtype=np.float32)

    if not np.any(mask):
        return vt_x, vt_y

    mem_size = stored_util.shape[1]
    active = np.where(mask)[0]
    if len(active) == 0:
        return vt_x, vt_y

    n_active = len(active)

    # Try Numba fused kernel first
    try:
        from cenop.optimizations.kernels import compute_attraction_kernel
        out_x = np.zeros(n_active, dtype=np.float64)
        out_y = np.zeros(n_active, dtype=np.float64)
        compute_attraction_kernel(
            stored_util, pos_history_x, pos_history_y,
            mem_ptr, mem_count,
            current_x, current_y,
            ref_mem_table if ref_mem_table.dtype == np.float64 else ref_mem_table.astype(np.float64),
            active.astype(np.int64), out_x, out_y,
        )
        vt_x[active] = out_x.astype(np.float32)
        vt_y[active] = out_y.astype(np.float32)
        return vt_x, vt_y
    except ImportError:
        pass

    # Build ordered indices
    indices = _build_ordered_indices(mem_ptr, mem_count, mem_size, mask)
    active_indices = indices[active]  # (n_active, mem_size)
    row_idx = active[:, np.newaxis]  # (n_active, 1)

    # Gather values in recency order (advanced indexing always copies)
    ordered_util = stored_util[row_idx, active_indices]  # (n_active, mem_size)
    ordered_px = pos_history_x[row_idx, active_indices]  # (n_active, mem_size)
    ordered_py = pos_history_y[row_idx, active_indices]  # (n_active, mem_size)

    # Direction vectors from current position to past positions
    cx = current_x[active][:, np.newaxis]  # (n_active, 1)
    cy = current_y[active][:, np.newaxis]  # (n_active, 1)

    if workspace is not None:
        # Use pre-allocated buffers sliced to (n_active, mem_size)
        attr_x = workspace.attr_x[:n_active, :]
        attr_y = workspace.attr_y[:n_active, :]
        np.subtract(ordered_px, cx, out=attr_x)
        np.subtract(ordered_py, cy, out=attr_y)

        # World wrapping (in-place via boolean indexing)
        if world_width > 0:
            half_w = world_width / 2
            too_pos = attr_x > half_w
            too_neg = attr_x < -half_w
            attr_x[too_pos] -= world_width
            attr_x[too_neg] += world_width
        if world_height > 0:
            half_h = world_height / 2
            too_pos = attr_y > half_h
            too_neg = attr_y < -half_h
            attr_y[too_pos] -= world_height
            attr_y[too_neg] += world_height

        # Distance and unit vectors
        dist = workspace.dist[:n_active, :]
        np.multiply(attr_x, attr_x, out=dist)
        dist += attr_y * attr_y
        np.sqrt(dist, out=dist)

        unit_x = workspace.unit_x[:n_active, :]
        unit_y = workspace.unit_y[:n_active, :]
        # safe_dist: avoid division by zero
        safe_dist = np.where(dist < 1e-20, 1.0, dist)
        np.divide(attr_x, safe_dist, out=unit_x)
        np.divide(attr_y, safe_dist, out=unit_y)

        # Weight = util * refMemStrength[i] / distance
        ref_weights = ref_mem_table[:mem_size].astype(np.float32)
        factor = workspace.factor[:n_active, :]
        factor[:] = np.where(
            dist < 1e-20,
            9999.0 * ordered_util,
            ordered_util * ref_weights[np.newaxis, :] / safe_dist,
        )

        # Validity mask (reuse valid_mask from workspace)
        valid = workspace.valid_mask[:n_active, :]
        n_valid = np.minimum(mem_count[active].astype(np.int32), mem_size)
        entry_idx = np.arange(mem_size)[np.newaxis, :]
        np.bitwise_and(entry_idx >= 1, entry_idx < n_valid[:, np.newaxis], out=valid)
        valid &= ordered_util != 0
        valid &= dist > 0

        factor *= valid

        vt_x[active] = (factor * unit_x).sum(axis=1).astype(np.float32)
        vt_y[active] = (factor * unit_y).sum(axis=1).astype(np.float32)
    else:
        attr_x = ordered_px - cx  # (n_active, mem_size)
        attr_y = ordered_py - cy

        # World wrapping
        if world_width > 0:
            half_w = world_width / 2
            attr_x = np.where(attr_x > half_w, attr_x - world_width, attr_x)
            attr_x = np.where(attr_x < -half_w, attr_x + world_width, attr_x)
        if world_height > 0:
            half_h = world_height / 2
            attr_y = np.where(attr_y > half_h, attr_y - world_height, attr_y)
            attr_y = np.where(attr_y < -half_h, attr_y + world_height, attr_y)

        # Distance and unit vectors
        dist = np.sqrt(attr_x * attr_x + attr_y * attr_y)
        safe_dist = np.where(dist < 1e-20, 1.0, dist)
        unit_x = attr_x / safe_dist
        unit_y = attr_y / safe_dist

        # Weight = util * refMemStrength[i] / distance
        ref_weights = ref_mem_table[:mem_size].astype(np.float32)
        factor = np.where(
            dist < 1e-20,
            9999.0 * ordered_util,
            ordered_util * ref_weights[np.newaxis, :] / safe_dist,
        )

        # Mask: skip index 0 (current position), skip entries beyond count, skip zero util
        n_valid = np.minimum(mem_count[active].astype(np.int32), mem_size)
        entry_idx = np.arange(mem_size)[np.newaxis, :]
        valid = (
            (entry_idx >= 1) & (entry_idx < n_valid[:, np.newaxis]) & (ordered_util != 0)
        )
        # Also mask zero-distance entries (same position)
        valid &= dist > 0

        factor *= valid

        vt_x[active] = (factor * unit_x).sum(axis=1).astype(np.float32)
        vt_y[active] = (factor * unit_y).sum(axis=1).astype(np.float32)

    return vt_x, vt_y
