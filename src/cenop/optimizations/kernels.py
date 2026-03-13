"""
Numba-accelerated simulation kernels.

Each kernel is a standalone @njit function that receives and returns flat NumPy
arrays.  No class references or Python objects.  Wrapper methods in population.py
unpack SoA arrays, call the kernel, and write results back.

Existing Numba functions in numba_helpers.py and __init__.py are NOT moved here;
this module holds NEW kernels extracted from population.py hot paths.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args):
        return range(*args)


@njit(cache=True)
def seed_numba_rng(seed):
    """Seed Numba's internal RNG for reproducibility."""
    np.random.seed(seed)


@njit(cache=True, parallel=True)
def reflect_boundaries_kernel(
    new_x: np.ndarray,
    new_y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    world_w: int,
    world_h: int,
    mask: np.ndarray,
) -> None:
    """
    DEPONS-style bouncy borders — Numba kernel.

    Reflects positions that overshoot domain edges back into bounds
    and flips the displacement sign so heading recalculation points inward.
    Modifies arrays in-place.
    """
    max_x = float(world_w - 1)
    max_y = float(world_h - 1)
    n = new_x.shape[0]

    for i in prange(n):
        if mask[i]:
            if new_x[i] < 0.0:
                new_x[i] = -new_x[i]
                dx[i] = -dx[i]
            elif new_x[i] > max_x:
                new_x[i] = 2.0 * max_x - new_x[i]
                dx[i] = -dx[i]
        # Safety clamp applied to all (matches NumPy np.clip behaviour)
        if new_x[i] < 0.0:
            new_x[i] = 0.0
        elif new_x[i] > max_x:
            new_x[i] = max_x

    for i in prange(n):
        if mask[i]:
            if new_y[i] < 0.0:
                new_y[i] = -new_y[i]
                dy[i] = -dy[i]
            elif new_y[i] > max_y:
                new_y[i] = 2.0 * max_y - new_y[i]
                dy[i] = -dy[i]
        # Safety clamp applied to all (matches NumPy np.clip behaviour)
        if new_y[i] < 0.0:
            new_y[i] = 0.0
        elif new_y[i] > max_y:
            new_y[i] = max_y


@njit(cache=True)
def crw_angle_step_kernel(
    prev_angle,       # float64[n] — previous turning angle (read)
    prev_log_mov,     # float64[n] — previous log step length (read+write)
    depths,           # float64[n] — depth at each agent's position
    salinity,         # float64[n] — salinity at each agent's position
    rand_angle,       # float64[n] — pre-generated N(r2_mean, r2_sd)
    rand_len,         # float64[n] — pre-generated N(r1_mean, r1_sd)
    mask,             # bool[n]
    out_pres_angle,   # float64[n] — output: turning angle
    out_log_mov,      # float64[n] — output: log step length
    # CRW parameters
    corr_angle_base, corr_angle_bathy, corr_angle_salinity, corr_angle_base_sd,
    corr_logmov_length, corr_logmov_bathy, corr_logmov_salinity, max_mov,
    r2_mean, r2_sd, r1_mean, r1_sd,
):
    """
    CRW angle + step length kernel with rejection sampling.

    Implements DEPONS 3.2 CRW (Java Porpoise.java:332-393):
    1. Turning angle with rejection sampling (max 200 retries)
    2. Distance-dependent angle modulation with rejection loop
    3. Step length with rejection sampling (max 200 retries)

    Outputs pres_angle and log_mov. Does NOT compute dx/dy (those
    come later after heading composition with vt, deterrence, social).
    """
    n = prev_angle.shape[0]
    max_retries = 200

    for i in range(n):
        if not mask[i]:
            out_pres_angle[i] = 0.0
            out_log_mov[i] = prev_log_mov[i]
            continue

        # === Angle calculation ===
        env_mod = (corr_angle_bathy * depths[i]
                   + corr_angle_salinity * salinity[i]
                   + corr_angle_base_sd)
        angle_tmp = corr_angle_base * prev_angle[i] + rand_angle[i]
        pres_angle = angle_tmp * env_mod

        # Rejection sampling for angle (Java Porpoise.java:332-360)
        retry = 0
        while abs(pres_angle) > 180.0 and retry < max_retries:
            new_rand = np.random.normal(r2_mean, r2_sd)
            angle_tmp = corr_angle_base * prev_angle[i] + new_rand
            pres_angle = angle_tmp * env_mod
            retry += 1

        # Emergency fallback: +/-90 (Java Porpoise.java:354)
        if abs(pres_angle) > 180.0:
            if pres_angle > 0:
                pres_angle = 90.0
            else:
                pres_angle = -90.0

        # Distance-dependent modulation (Java Porpoise.java:374-393)
        max_mov_value = 10.0 ** max_mov
        prev_mov_value = 10.0 ** prev_log_mov[i]

        if prev_mov_value <= max_mov_value:
            retry = 0
            while abs(pres_angle) >= 180.0 and retry < max_retries:
                rnd = np.random.uniform(0.0, 20.0)
                new_angle = abs(pres_angle) + rnd - rnd * prev_mov_value / max_mov_value
                if pres_angle >= 0:
                    pres_angle = new_angle
                else:
                    pres_angle = -new_angle
                retry += 1

            if abs(pres_angle) >= 180.0:
                rnd = np.random.uniform(0.0, 20.0)
                if pres_angle >= 0:
                    pres_angle = rnd + 90.0
                else:
                    pres_angle = -(rnd + 90.0)

        out_pres_angle[i] = pres_angle

        # === Step length calculation ===
        log_mov = (corr_logmov_length * prev_log_mov[i]
                   + corr_logmov_bathy * depths[i]
                   + corr_logmov_salinity * salinity[i]
                   + rand_len[i])

        retry = 0
        while log_mov > max_mov and retry < max_retries:
            new_rand = np.random.normal(r1_mean, r1_sd)
            log_mov = (corr_logmov_length * prev_log_mov[i]
                       + corr_logmov_bathy * depths[i]
                       + corr_logmov_salinity * salinity[i]
                       + new_rand)
            retry += 1

        if log_mov > max_mov:
            log_mov = max_mov

        out_log_mov[i] = log_mov
        prev_log_mov[i] = log_mov


@njit(cache=True, parallel=True)
def turn_position_kernel(
    x, y, heading, step_dist,
    turn_delta,
    world_w, world_h,
    out_x, out_y, out_heading,
):
    """
    Compute new positions after turning by turn_delta degrees.

    For each agent: turn heading, compute displacement, add to position,
    reflect at boundaries.  Writes results to out_x, out_y, out_heading.
    """
    max_x = float(world_w - 1)
    max_y = float(world_h - 1)
    n = x.shape[0]

    for i in prange(n):
        h = (heading[i] + turn_delta) % 360.0
        out_heading[i] = h

        rads = h * np.pi / 180.0
        dx_i = np.sin(rads) * step_dist[i]
        dy_i = np.cos(rads) * step_dist[i]

        nx = x[i] + dx_i
        ny = y[i] + dy_i

        # Reflect X
        if nx < 0.0:
            nx = -nx
        elif nx > max_x:
            nx = 2.0 * max_x - nx
        if nx < 0.0:
            nx = 0.0
        elif nx > max_x:
            nx = max_x

        # Reflect Y
        if ny < 0.0:
            ny = -ny
        elif ny > max_y:
            ny = 2.0 * max_y - ny
        if ny < 0.0:
            ny = 0.0
        elif ny > max_y:
            ny = max_y

        out_x[i] = nx
        out_y[i] = ny


@njit(cache=True)
def eat_food_kernel(
    food_grid,   # 2D float32 array (rows, cols) — modified in-place
    x_indices,   # 1D int32 — column indices per agent
    y_indices,   # 1D int32 — row indices per agent
    fraction,    # 1D float32 — fraction to eat per agent
    food_eaten,  # 1D float32 — output: actual food eaten per agent
    min_food,    # float — minimum food floor (ADD_ARTIFICIAL_FOOD)
    demand_grid, # 2D float32 array (rows, cols) — pre-allocated buffer
):
    """
    Eat food from grid cells — two-pass proportional-sharing kernel.

    Preserves the same semantics as the original np.add.at implementation:
    agents in the same cell get proportional shares of available food,
    independent of processing order. This avoids the full-grid allocation
    that was the #1 bottleneck.

    Pass 1: Compute per-agent demand and accumulate per-cell total demand.
    Pass 2: For each agent, if cell demand <= supply, eat full amount;
            otherwise, eat proportional share of supply (matching DEPONS
            Java semantics: eat first, floor after).

    demand_grid must be pre-allocated with shape matching food_grid;
    it is zeroed at the start of each call and used as scratch space.

    Modifies food_grid and food_eaten in-place.
    """
    rows = food_grid.shape[0]
    cols = food_grid.shape[1]
    n = x_indices.shape[0]

    # --- Pass 1: accumulate per-cell demand ---
    demand_grid[:, :] = 0.0
    agent_demand = np.empty(n, dtype=np.float32)

    for i in range(n):
        row = y_indices[i]
        col = x_indices[i]
        demand = food_grid[row, col] * fraction[i]
        agent_demand[i] = demand
        demand_grid[row, col] += demand

    # --- Pass 2: distribute proportionally ---
    for i in range(n):
        row = y_indices[i]
        col = x_indices[i]
        current = food_grid[row, col]
        total_demand = demand_grid[row, col]

        if total_demand <= 0.0:
            food_eaten[i] = 0.0
            continue

        # Match DEPONS Java: agents eat from full cell food, floor
        # is enforced AFTER consumption (not subtracted beforehand).
        available = current
        if available < 0.0:
            available = 0.0

        if total_demand <= available:
            # No competition: everyone gets what they asked for
            food_eaten[i] = agent_demand[i]
        else:
            # Competition: proportional share of available food
            share = agent_demand[i] / total_demand
            food_eaten[i] = available * share

    # --- Update grid: subtract actual total eaten per cell ---
    for i in range(n):
        row = y_indices[i]
        col = x_indices[i]
        food_grid[row, col] -= food_eaten[i]

    # --- Enforce minimum food floor (only touched cells) ---
    for i in range(n):
        row = y_indices[i]
        col = x_indices[i]
        if food_grid[row, col] < min_food:
            food_grid[row, col] = min_food


@njit(cache=True, parallel=True)
def depons_bmr_cost_kernel(
    speed,              # 1D float32 — current speed in m/s
    scaling,            # 1D float32 — seasonal scaling factor (pre-computed)
    is_lactating,       # 1D bool
    is_disturbed,       # 1D bool
    deter_magnitude,    # 1D float32
    mask,               # 1D bool
    out_total_cost,     # 1D float32 — output
    e_use_per_30_min,   # float — BMR parameter
    e_lact,             # float — lactation multiplier
):
    """
    DEPONS BMR + activity + disturbance cost kernel.

    Matches DEPONSEnergyModule.compute_bmr_cost() exactly:
    - BMR: 0.001 * scaling * e_use_per_30_min (* e_lact if lactating)
    - Activity: speed * 0.0001 * scaling
    - Disturbance: 0.002 * deter_magnitude * scaling (if disturbed)
    """
    n = speed.shape[0]
    for i in prange(n):
        if not mask[i]:
            out_total_cost[i] = 0.0
            continue

        bmr = 0.001 * scaling[i] * e_use_per_30_min
        if is_lactating[i]:
            bmr *= e_lact

        activity = speed[i] * 0.0001 * scaling[i]

        disturbance = 0.0
        if is_disturbed[i]:
            disturbance = 0.002 * deter_magnitude[i] * scaling[i]

        out_total_cost[i] = bmr + activity + disturbance


@njit(cache=True)
def social_accumulate_kernel(
    idx_i, idx_j, dx_ij, dy_ij, dist, p_i, p_j,
    ux_total, uy_total, sw_total,
):
    """
    Fused social vector accumulation: unit-vector + weighting + accumulation.

    Replaces separate NumPy unit-vector computation, weighting, and the
    accumulate_social_totals helper in a single pass over pairs.

    For each pair (i, j):
    - i gets unit vector towards j, weighted by p_i
    - j gets unit vector towards i (reversed), weighted by p_j

    dist should already include eps (caller adds 1e-6).
    """
    n = idx_i.shape[0]
    for k in range(n):
        i = idx_i[k]
        j = idx_j[k]
        d = dist[k]

        ux = dx_ij[k] / d
        uy = dy_ij[k] / d

        # i hears j (towards j)
        ux_total[i] += ux * p_i[k]
        uy_total[i] += uy * p_i[k]
        sw_total[i] += p_i[k]

        # j hears i (towards i = reverse direction)
        ux_total[j] -= ux * p_j[k]
        uy_total[j] -= uy * p_j[k]
        sw_total[j] += p_j[k]


def warmup_kernels():
    """Pre-compile all kernels with small dummy data to avoid first-call latency."""
    if not NUMBA_AVAILABLE:
        return False
    x = np.array([1.0, -1.0, 25.0], dtype=np.float64)
    y = np.array([1.0, -1.0, 25.0], dtype=np.float64)
    dx = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    dy = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    mask = np.array([True, True, True])
    reflect_boundaries_kernel(x, y, dx, dy, 20, 20, mask)
    # Warmup CRW angle+step kernel
    n = 3
    pa = np.zeros(n, dtype=np.float64)
    plm = np.ones(n, dtype=np.float64)
    dep = np.full(n, 30.0, dtype=np.float64)
    sal = np.full(n, 30.0, dtype=np.float64)
    ra = np.zeros(n, dtype=np.float64)
    rl = np.zeros(n, dtype=np.float64)
    m = np.ones(n, dtype=np.bool_)
    opa = np.zeros(n, dtype=np.float64)
    olm = np.zeros(n, dtype=np.float64)
    crw_angle_step_kernel(pa, plm, dep, sal, ra, rl, m, opa, olm,
                          0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 3.0,
                          0.0, 4.0, 0.0, 1.0)
    # Warmup turn_position kernel
    ox = np.zeros(3, dtype=np.float64)
    oy = np.zeros(3, dtype=np.float64)
    oh = np.zeros(3, dtype=np.float64)
    sd = np.ones(3, dtype=np.float64)
    hd = np.zeros(3, dtype=np.float64)
    turn_position_kernel(x, y, hd, sd, 90.0, 20, 20, ox, oy, oh)
    # Warmup eat_food kernel
    fg = np.ones((2, 2), dtype=np.float32) * 100.0
    xi = np.array([0, 1], dtype=np.int32)
    yi = np.array([0, 0], dtype=np.int32)
    fr = np.array([0.1, 0.2], dtype=np.float32)
    fe = np.zeros(2, dtype=np.float32)
    dg = np.zeros((2, 2), dtype=np.float32)
    eat_food_kernel(fg, xi, yi, fr, fe, 0.01, dg)
    # Warmup BMR cost kernel
    spd = np.ones(2, dtype=np.float32)
    scl = np.ones(2, dtype=np.float32)
    lac = np.array([False, True])
    dis = np.array([False, True])
    dmg = np.array([0.0, 0.5], dtype=np.float32)
    msk = np.array([True, True])
    cost = np.zeros(2, dtype=np.float32)
    depons_bmr_cost_kernel(spd, scl, lac, dis, dmg, msk, cost, 4.5, 1.4)
    # Warmup social accumulate kernel
    si = np.array([0, 1], dtype=np.int64)
    sj = np.array([1, 0], dtype=np.int64)
    sdx = np.array([1.0, -1.0], dtype=np.float64)
    sdy = np.array([0.0, 0.0], dtype=np.float64)
    sdi = np.array([1.0, 1.0], dtype=np.float64)
    spi = np.array([0.5, 0.5], dtype=np.float64)
    spj = np.array([0.5, 0.5], dtype=np.float64)
    sux = np.zeros(2, dtype=np.float64)
    suy = np.zeros(2, dtype=np.float64)
    ssw = np.zeros(2, dtype=np.float64)
    social_accumulate_kernel(si, sj, sdx, sdy, sdi, spi, spj, sux, suy, ssw)
    return True
