"""DEPONS 3.2 Reference Memory — decay tables and vectorized computation.

Ports Java RefMem.java and FastRefMemTurn.java to vectorized NumPy.
"""

import numpy as np
from functools import lru_cache


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


def _build_ordered_indices(mem_ptr: np.ndarray, mem_count: np.ndarray,
                           mem_size: int, mask: np.ndarray) -> np.ndarray:
    """Build (count, mem_size) array of circular buffer indices ordered most-recent-first.

    For each agent, indices[agent, 0] = most recent entry, indices[agent, 1] = second most recent, etc.
    Entries beyond mem_count are set to -1 (masked out).
    """
    count = len(mask)
    # Offsets 0..mem_size-1
    offsets = np.arange(mem_size, dtype=np.int16)
    # For each agent: (ptr - 1 - offset) % mem_size
    indices = (mem_ptr[:, np.newaxis] - 1 - offsets[np.newaxis, :]) % mem_size
    return indices


def compute_ve_total(
    stored_util: np.ndarray,
    mem_ptr: np.ndarray,
    mem_count: np.ndarray,
    work_mem_table: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute veTotal (expected food value) for each agent — fully vectorized.

    veTotal = sum(workMemStrength[i] * storedUtilList[i]) for i in 0..N-2

    Java ref: Porpoise.java:688-705 (getExpFoodVal)
    """
    count = len(mask)
    ve_total = np.zeros(count, dtype=np.float32)

    if not np.any(mask):
        return ve_total

    mem_size = stored_util.shape[1]
    active = np.where(mask)[0]
    if len(active) == 0:
        return ve_total

    # Build ordered indices: (count, mem_size) — most recent first
    indices = _build_ordered_indices(mem_ptr, mem_count, mem_size, mask)

    # Gather food values in recency order for active agents
    # Use advanced indexing: stored_util[active, indices[active]]
    active_indices = indices[active]  # (n_active, mem_size)
    row_idx = active[:, np.newaxis]  # (n_active, 1) for broadcasting
    ordered_food = stored_util[row_idx, active_indices]  # (n_active, mem_size)

    # Mask out entries beyond each agent's count (Java skips oldest, so use n-1)
    n_valid = np.minimum(mem_count[active].astype(np.int32), mem_size) - 1  # n-1 entries
    n_valid = np.maximum(n_valid, 0)
    entry_idx = np.arange(mem_size)[np.newaxis, :]  # (1, mem_size)
    valid_mask = entry_idx < n_valid[:, np.newaxis]  # (n_active, mem_size)

    # Weight by work_mem_table and sum
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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute attraction vector vt for each agent — fully vectorized.

    For each past position with nonzero food:
      weight = storedUtil[i] * refMemStrength[i] / distance
      vt += weight * unit_direction_to_past_position

    Java ref: FastRefMemTurn.java:44-125

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

    # Build ordered indices
    indices = _build_ordered_indices(mem_ptr, mem_count, mem_size, mask)
    active_indices = indices[active]  # (n_active, mem_size)
    row_idx = active[:, np.newaxis]  # (n_active, 1)

    # Gather values in recency order
    ordered_util = stored_util[row_idx, active_indices]    # (n_active, mem_size)
    ordered_px = pos_history_x[row_idx, active_indices]    # (n_active, mem_size)
    ordered_py = pos_history_y[row_idx, active_indices]    # (n_active, mem_size)

    # Direction vectors from current position to past positions
    cx = current_x[active][:, np.newaxis]  # (n_active, 1)
    cy = current_y[active][:, np.newaxis]  # (n_active, 1)
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
    valid = (entry_idx >= 1) & (entry_idx < n_valid[:, np.newaxis]) & (ordered_util != 0)
    # Also mask zero-distance entries (same position)
    valid &= (dist > 0)

    factor *= valid

    vt_x[active] = (factor * unit_x).sum(axis=1).astype(np.float32)
    vt_y[active] = (factor * unit_y).sum(axis=1).astype(np.float32)

    return vt_x, vt_y
