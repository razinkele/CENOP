"""
Numba-accelerated simulation kernels.

Each kernel is a standalone @njit function that receives and returns flat NumPy
arrays.  No class references or Python objects.  Wrapper methods in population.py
unpack SoA arrays, call the kernel, and write results back.

Existing Numba functions in numba_helpers.py and __init__.py are NOT moved here;
this module holds NEW kernels extracted from population.py hot paths.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    import logging as _logging

    _logging.getLogger("cenop.optimizations.kernels").warning(
        "Numba not available — kernels will run as interpreted Python (~100x slower)"
    )

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


@njit(cache=True)
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

    for i in range(n):
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

    for i in range(n):
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
    prev_angle,  # float64[n] — previous turning angle (read)
    prev_log_mov,  # float64[n] — previous log step length (read+write)
    depths,  # float64[n] — depth at each agent's position
    salinity,  # float64[n] — salinity at each agent's position
    rand_angle,  # float64[n] — pre-generated N(r2_mean, r2_sd)
    rand_len,  # float64[n] — pre-generated N(r1_mean, r1_sd)
    mask,  # bool[n]
    out_pres_angle,  # float64[n] — output: turning angle
    out_log_mov,  # float64[n] — output: log step length
    # CRW parameters
    corr_angle_base,
    corr_angle_bathy,
    corr_angle_salinity,
    corr_angle_base_sd,
    corr_logmov_length,
    corr_logmov_bathy,
    corr_logmov_salinity,
    max_mov,
    r2_mean,
    r2_sd,
    r1_mean,
    r1_sd,
    m_param,
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
        env_mod = (
            corr_angle_bathy * depths[i] + corr_angle_salinity * salinity[i] + corr_angle_base_sd
        )
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

        # Distance-dependent modulation (Java Porpoise.java:367-397)
        prev_mov_value = 10.0 ** prev_log_mov[i]

        if prev_mov_value <= m_param:
            sign = 1.0 if pres_angle >= 0 else -1.0
            pres_angle = abs(pres_angle)  # Java line 367: abs once before loop
            retry = 0
            while pres_angle >= 180.0 and retry < max_retries:
                rnd = np.random.normal(0.0, 1.0)  # Java: nextCrwAngleWithM() = N(0,1)
                pres_angle = pres_angle + rnd - rnd * prev_mov_value / m_param
                retry += 1

            if pres_angle >= 180.0:
                rnd = np.random.uniform(0.0, 20.0)  # Emergency fallback (Java line 386)
                pres_angle = rnd + 90.0

            pres_angle = pres_angle * sign  # Restore sign (Java line 397)

        out_pres_angle[i] = pres_angle

        # === Step length calculation ===
        log_mov = (
            corr_logmov_length * prev_log_mov[i]
            + corr_logmov_bathy * depths[i]
            + corr_logmov_salinity * salinity[i]
            + rand_len[i]
        )

        retry = 0
        while log_mov > max_mov and retry < max_retries:
            new_rand = np.random.normal(r1_mean, r1_sd)
            log_mov = (
                corr_logmov_length * prev_log_mov[i]
                + corr_logmov_bathy * depths[i]
                + corr_logmov_salinity * salinity[i]
                + new_rand
            )
            retry += 1

        if log_mov > max_mov:
            log_mov = max_mov

        out_log_mov[i] = log_mov
        prev_log_mov[i] = log_mov


@njit(cache=True)
def turn_position_kernel(
    x,
    y,
    heading,
    step_dist,
    turn_delta,
    world_w,
    world_h,
    out_x,
    out_y,
    out_heading,
    out_xi,
    out_yi,
):
    """
    Compute new positions after turning by turn_delta degrees.

    For each agent: turn heading, compute displacement, add to position,
    reflect at boundaries.  Writes results to out_x, out_y, out_heading.
    Also outputs clamped int32 cell indices in out_xi, out_yi to avoid
    post-kernel astype(np.int32) + np.clip calls.
    """
    max_x = float(world_w - 1)
    max_y = float(world_h - 1)
    n = x.shape[0]

    for i in range(n):
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
        out_xi[i] = min(world_w - 1, max(0, int(nx)))
        out_yi[i] = min(world_h - 1, max(0, int(ny)))


@njit(cache=True)
def eat_food_kernel(
    food_grid,  # 2D float32 array (rows, cols) — modified in-place
    x_indices,  # 1D int32 — column indices per agent
    y_indices,  # 1D int32 — row indices per agent
    fraction,  # 1D float32 — fraction to eat per agent
    food_eaten,  # 1D float32 — output: actual food eaten per agent
    min_food,  # float — minimum food floor (ADD_ARTIFICIAL_FOOD)
    demand_grid,  # 2D float32 array (rows, cols) — pre-allocated buffer
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


@njit(cache=True)
def eat_food_kernel_v2(
    food_grid,  # 2D float32 array (rows, cols) — modified in-place
    x_indices,  # 1D int32 — column indices per agent
    y_indices,  # 1D int32 — row indices per agent
    energy,  # 1D float32 — energy level per agent
    food_eaten,  # 1D float32 — output: actual food eaten per agent
    min_food,  # float — minimum food floor (ADD_ARTIFICIAL_FOOD)
    demand_grid,  # 2D float32 array (rows, cols) — pre-allocated buffer
):
    """Two-pass proportional food with inline fraction computation.

    Same as eat_food_kernel but computes frac = clip((20-energy)/10, 0, 0.99)
    internally. demand = food_at_cell * frac (NOT raw frac).
    """
    n = x_indices.shape[0]
    agent_demand = np.empty(n, dtype=np.float32)

    # Pass 1: accumulate per-cell demand
    demand_grid[:, :] = 0.0
    for i in range(n):
        frac = min(max((20.0 - energy[i]) / 10.0, 0.0), 0.99)
        row = y_indices[i]
        col = x_indices[i]
        demand = food_grid[row, col] * frac
        agent_demand[i] = demand
        demand_grid[row, col] += demand

    # Pass 2: distribute proportionally
    for i in range(n):
        row = y_indices[i]
        col = x_indices[i]
        total_demand = demand_grid[row, col]
        if total_demand <= 0.0:
            food_eaten[i] = 0.0
            continue
        available = max(food_grid[row, col], 0.0)
        if total_demand <= available:
            food_eaten[i] = agent_demand[i]
        else:
            share = agent_demand[i] / total_demand
            food_eaten[i] = available * share

    # Pass 3: update grid and enforce floor
    for i in range(n):
        row = y_indices[i]
        col = x_indices[i]
        food_grid[row, col] -= food_eaten[i]
        if food_grid[row, col] < min_food:
            food_grid[row, col] = min_food


@njit(cache=True)
def depons_bmr_cost_kernel(
    speed,  # 1D float32 — current speed in m/s
    scaling,  # 1D float32 — seasonal scaling factor (pre-computed)
    is_lactating,  # 1D bool
    is_disturbed,  # 1D bool
    deter_magnitude,  # 1D float32
    mask,  # 1D bool
    out_total_cost,  # 1D float32 — output
    e_use_per_30_min,  # float — BMR parameter
    e_lact,  # float — lactation multiplier
    e_use_per_km,  # float — swimming activity coefficient (0.0 in DEPONS)
    disturbance_coeff,  # float — disturbance energy coefficient (0.0 in DEPONS)
):
    """
    DEPONS BMR + activity + disturbance cost kernel.

    Matches DEPONSEnergyModule.compute_bmr_cost() exactly:
    - BMR: 0.001 * scaling * e_use_per_30_min (* e_lact if lactating)
    - Activity: speed * e_use_per_km * scaling
    - Disturbance: disturbance_coeff * deter_magnitude * scaling (if disturbed)
    """
    n = speed.shape[0]
    for i in range(n):
        if not mask[i]:
            out_total_cost[i] = 0.0
            continue

        bmr = 0.001 * scaling[i] * e_use_per_30_min
        if is_lactating[i]:
            bmr *= e_lact

        activity = speed[i] * e_use_per_km * scaling[i]

        disturbance = 0.0
        if is_disturbed[i]:
            disturbance = disturbance_coeff * deter_magnitude[i] * scaling[i]

        out_total_cost[i] = bmr + activity + disturbance


@njit(cache=True)
def social_accumulate_kernel(
    idx_i,
    idx_j,
    dx_ij,
    dy_ij,
    dist,
    p_i,
    p_j,
    ux_total,
    uy_total,
    sw_total,
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


@njit(cache=True, parallel=True)
def regrow_food_kernel(food, k_vals, rate, n_iter):
    """Apply logistic food regrowth for n_iter iterations (prange-parallel).

    Modifies food array in-place.
    Each cell: F = F + rate * F * (1 - F/K), repeated n_iter times.
    """
    for i in prange(len(food)):
        k = k_vals[i]
        if k <= 0.0:
            continue
        f = food[i]
        for _ in range(n_iter):
            f = f + rate * f * (1.0 - f / k)
        food[i] = f


@njit(cache=True)
def compute_ve_total_kernel(
    stored_util,
    mem_ptr,
    mem_count,
    work_mem_table,
    active_indices,
    out_ve_total,
):
    """Fused veTotal: gather + weight + sum in one pass per agent.

    Args:
        stored_util: (n_agents, mem_size) float32
        mem_ptr: (n_agents,) int32 — circular buffer write pointer
        mem_count: (n_agents,) int32 — entries stored
        work_mem_table: (mem_size,) float64 — decay weights
        active_indices: (n_active,) int64 — indices of active agents
        out_ve_total: (n_active,) float64 — output
    """
    mem_size = stored_util.shape[1]
    for ai in range(len(active_indices)):
        agent = active_indices[ai]
        n = min(int(mem_count[agent]), mem_size) - 1  # use n-1 entries
        if n <= 0:
            out_ve_total[ai] = 0.0
            continue
        ptr = int(mem_ptr[agent])
        total = 0.0
        for i in range(n):
            buf_idx = (ptr - 1 - i) % mem_size
            total += work_mem_table[i] * stored_util[agent, buf_idx]
        out_ve_total[ai] = total


@njit(cache=True)
def compute_attraction_kernel(
    stored_util,
    pos_history_x,
    pos_history_y,
    mem_ptr,
    mem_count,
    current_x,
    current_y,
    ref_mem_table,
    active_indices,
    out_vt_x,
    out_vt_y,
):
    """Fused attraction vector: gather + direction + weight + sum per agent.

    Args:
        stored_util: (n_agents, mem_size) float32
        pos_history_x, pos_history_y: (n_agents, mem_size) float32
        mem_ptr: (n_agents,) int32
        mem_count: (n_agents,) int32
        current_x, current_y: (n_agents,) float32 — current positions
        ref_mem_table: (mem_size,) float64 — decay weights
        active_indices: (n_active,) int64 — indices of active agents
        out_vt_x, out_vt_y: (n_active,) float64 — output
    """
    mem_size = stored_util.shape[1]
    for ai in range(len(active_indices)):
        agent = active_indices[ai]
        n = min(int(mem_count[agent]), mem_size)
        cx = current_x[agent]
        cy = current_y[agent]
        ptr = int(mem_ptr[agent])
        vx = 0.0
        vy = 0.0
        # Skip index 0 (most recent = current position)
        for i in range(1, n):
            buf_idx = (ptr - 1 - i) % mem_size
            util = stored_util[agent, buf_idx]
            if util == 0.0:
                continue
            px = pos_history_x[agent, buf_idx]
            py = pos_history_y[agent, buf_idx]
            dx = px - cx
            dy = py - cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 1e-20:
                # direction undefined at zero distance, skip
                continue
            else:
                weight = util * ref_mem_table[i] / dist
            vx += weight * dx / dist
            vy += weight * dy / dist
        out_vt_x[ai] = vx
        out_vt_y[ai] = vy


@njit(cache=True)
def land_avoidance_kernel(
    x,
    y,
    heading,
    step_dist,
    depth_grid,
    min_depth,
    base_angles,
    jitter,
    out_x,
    out_y,
    out_heading,
    resolved,
):
    """Try 3 angles x 2 directions per blocked agent in one pass.

    For each angle, try right then left; pick deeper if both valid;
    break on first successful angle. Matches DEPONS Java and current
    Python _handle_land_avoidance pattern.

    Args:
        x, y, heading, step_dist: blocked agent arrays (float64)
        depth_grid: float32[rows, cols] landscape depth
        min_depth: minimum water depth threshold
        base_angles: float64[3] = [40, 70, 120] degree turn angles
        jitter: float64[3] = pre-drawn uniform(0, 10) per angle
        out_x, out_y, out_heading: output position/heading (float64)
        resolved: output bool array — True if found water
    """
    max_x = float(depth_grid.shape[1] - 1)
    max_y = float(depth_grid.shape[0] - 1)

    for i in range(len(x)):
        resolved[i] = False
        for a_idx in range(3):
            if resolved[i]:
                break
            angle = base_angles[a_idx] + jitter[a_idx]

            # Right turn
            rh = (heading[i] + angle) % 360.0
            rr = rh * 3.141592653589793 / 180.0
            rx = x[i] + np.sin(rr) * step_dist[i]
            ry = y[i] + np.cos(rr) * step_dist[i]
            # Reflect boundaries
            if rx < 0.0:
                rx = -rx
            elif rx > max_x:
                rx = 2.0 * max_x - rx
            if rx < 0.0:
                rx = 0.0
            elif rx > max_x:
                rx = max_x
            if ry < 0.0:
                ry = -ry
            elif ry > max_y:
                ry = 2.0 * max_y - ry
            if ry < 0.0:
                ry = 0.0
            elif ry > max_y:
                ry = max_y
            rd = depth_grid[int(ry), int(rx)]

            # Left turn
            lh = (heading[i] - angle + 360.0) % 360.0
            lr = lh * 3.141592653589793 / 180.0
            lx = x[i] + np.sin(lr) * step_dist[i]
            ly = y[i] + np.cos(lr) * step_dist[i]
            # Reflect boundaries
            if lx < 0.0:
                lx = -lx
            elif lx > max_x:
                lx = 2.0 * max_x - lx
            if lx < 0.0:
                lx = 0.0
            elif lx > max_x:
                lx = max_x
            if ly < 0.0:
                ly = -ly
            elif ly > max_y:
                ly = 2.0 * max_y - ly
            if ly < 0.0:
                ly = 0.0
            elif ly > max_y:
                ly = max_y
            ld = depth_grid[int(ly), int(lx)]

            # Pick best
            rok = rd >= min_depth
            lok = ld >= min_depth
            if rok and lok:
                if rd >= ld:
                    out_x[i] = rx
                    out_y[i] = ry
                    out_heading[i] = rh
                else:
                    out_x[i] = lx
                    out_y[i] = ly
                    out_heading[i] = lh
                resolved[i] = True
            elif rok:
                out_x[i] = rx
                out_y[i] = ry
                out_heading[i] = rh
                resolved[i] = True
            elif lok:
                out_x[i] = lx
                out_y[i] = ly
                out_heading[i] = lh
                resolved[i] = True


@njit(cache=True)
def heading_position_reflect_kernel(
    heading,
    pres_angle,
    log_mov,
    ve_total,
    vt_x,
    vt_y,
    deter_dx,
    deter_dy,
    x,
    y,
    mask,
    is_dispersing,
    inertia_const,
    disp_step,
    world_w,
    world_h,
    out_heading,
    out_dx,
    out_dy,
    out_step_dist,
):
    """Fused: heading composition + step distance + dx/dy.

    Replaces separate heading composition, step distance, and final dx/dy
    phases with a single pass per agent.  Position update and boundary
    reflection are left to _handle_land_avoidance.

    Args:
        heading: float32[N] current heading (degrees, already has CRW turn applied)
        pres_angle: float64[N] CRW turning angle (unused here — already in heading)
        log_mov: float64[N] current tick's log10(movement) from CRW kernel
        ve_total: float32[N] expected food value from RefMem
        vt_x, vt_y: float32[N] attraction vectors from RefMem
        deter_dx, deter_dy: float[N] deterrence + social displacement
        x, y: float32[N] current positions (unused — dx/dy only)
        mask: bool[N] active agents
        is_dispersing: bool[N] dispersing agents skip heading composition
        inertia_const: float scalar
        disp_step: float scalar — fixed step for dispersing agents
        world_w, world_h: int — grid dimensions (unused, kept for API compat)
        out_heading: float32[N] output heading (may alias heading)
        out_dx, out_dy: float32[N] output displacement
        out_step_dist: float32[N] output step distance
    """
    DEG2RAD = 0.017453292519943295
    RAD2DEG = 57.29577951308232
    for i in range(len(heading)):
        if not mask[i]:
            continue

        pres_mov = 10.0 ** log_mov[i]

        if is_dispersing[i]:
            # Dispersing: keep current heading, use fixed step
            new_h = heading[i]
            step = disp_step
        else:
            # Heading composition (Java Porpoise.java:556-567)
            h = heading[i]
            rad = h * DEG2RAD
            dx_crw = math.sin(rad)
            dy_crw = math.cos(rad)
            crw_c = inertia_const + pres_mov * ve_total[i]
            total_dx = dx_crw * crw_c + vt_x[i] + deter_dx[i]
            total_dy = dy_crw * crw_c + vt_y[i] + deter_dy[i]

            # facePoint: new heading from composite vector
            new_h = math.atan2(total_dx, total_dy) * RAD2DEG
            if new_h < 0:
                new_h += 360.0

            # Step distance = 10^log_mov / 4.0 (Java Porpoise.java:589)
            step = pres_mov / 4.0

        out_heading[i] = new_h
        out_step_dist[i] = step

        # Final dx/dy from composite heading
        rad2 = new_h * DEG2RAD
        out_dx[i] = math.sin(rad2) * step
        out_dy[i] = math.cos(rad2) * step


@njit(cache=True)
def social_sound_kernel(
    xi,
    yi,
    xj,
    yj,
    idx_i,
    idx_j,
    ambient_i,
    ambient_j,
    source_level,
    alpha_hat,
    beta_hat,
    threshold,
    slope,
    cell_size,
    n_agents,
    out_ux,
    out_uy,
    out_sw,
):
    """Fused social sound: distance + TL + RL + probability + accumulation.

    Uses sequential iteration (no prange) because multiple pairs may share
    agent indices, causing race conditions on accumulation.

    Handles ambient noise: p_i and p_j computed independently using
    per-agent ambient levels (SNR = RL - ambient).

    Args:
        xi, yi: float64[n_pairs] -- agent i positions (grid cells)
        xj, yj: float64[n_pairs] -- agent j positions (grid cells)
        idx_i, idx_j: int64[n_pairs] -- global agent indices
        ambient_i, ambient_j: float64[n_pairs] -- ambient RL at each agent (0 if none)
        source_level, alpha_hat, beta_hat: sound propagation params
        threshold, slope: response probability params
        cell_size: grid cell size in meters (400.0)
        n_agents: total agent count
        out_ux, out_uy, out_sw: float64[n_agents] -- accumulated (must be zeroed before call)
    """
    n_pairs = len(xi)
    for p in range(n_pairs):
        ddx = xj[p] - xi[p]
        ddy = yj[p] - yi[p]
        dist_cells = math.sqrt(ddx * ddx + ddy * ddy) + 1.0e-6
        dist_m = dist_cells * cell_size

        ux = ddx / dist_cells
        uy = ddy / dist_cells

        dist_safe = max(dist_m, 1.0)
        tl = beta_hat * math.log10(dist_safe) + alpha_hat * dist_safe
        rl = source_level - tl

        snr_i = rl - ambient_i[p]
        linear_i = slope * (snr_i - threshold)
        linear_i = max(-500.0, min(500.0, linear_i))
        p_i = 1.0 / (1.0 + math.exp(-linear_i))
        p_i = max(0.0, min(1.0, p_i))

        snr_j = rl - ambient_j[p]
        linear_j = slope * (snr_j - threshold)
        linear_j = max(-500.0, min(500.0, linear_j))
        p_j = 1.0 / (1.0 + math.exp(-linear_j))
        p_j = max(0.0, min(1.0, p_j))

        ii = idx_i[p]
        jj = idx_j[p]

        out_ux[ii] += ux * p_i
        out_uy[ii] += uy * p_i
        out_sw[ii] += p_i
        out_ux[jj] += -ux * p_j
        out_uy[jj] += -uy * p_j
        out_sw[jj] += p_j


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
    crw_angle_step_kernel(
        pa,
        plm,
        dep,
        sal,
        ra,
        rl,
        m,
        opa,
        olm,
        0.0,
        0.0,
        0.0,
        1.0,
        0.5,
        0.0,
        0.0,
        3.0,
        0.0,
        4.0,
        0.0,
        1.0,
        0.00001,
    )
    # Warmup turn_position kernel
    ox = np.zeros(3, dtype=np.float64)
    oy = np.zeros(3, dtype=np.float64)
    oh = np.zeros(3, dtype=np.float64)
    oxi = np.zeros(3, dtype=np.int32)
    oyi = np.zeros(3, dtype=np.int32)
    sd = np.ones(3, dtype=np.float64)
    hd = np.zeros(3, dtype=np.float64)
    turn_position_kernel(x, y, hd, sd, 90.0, 20, 20, ox, oy, oh, oxi, oyi)
    # Warmup eat_food kernel
    fg = np.ones((2, 2), dtype=np.float32) * 100.0
    xi = np.array([0, 1], dtype=np.int32)
    yi = np.array([0, 0], dtype=np.int32)
    fr = np.array([0.1, 0.2], dtype=np.float32)
    fe = np.zeros(2, dtype=np.float32)
    dg = np.zeros((2, 2), dtype=np.float32)
    eat_food_kernel(fg, xi, yi, fr, fe, 0.01, dg)
    # Warmup eat_food_kernel_v2
    fg2 = np.ones((2, 2), dtype=np.float32) * 100.0
    xi2 = np.array([0, 1], dtype=np.int32)
    yi2 = np.array([0, 0], dtype=np.int32)
    en2 = np.array([10.0, 15.0], dtype=np.float32)
    fe2 = np.zeros(2, dtype=np.float32)
    dg2 = np.zeros((2, 2), dtype=np.float32)
    eat_food_kernel_v2(fg2, xi2, yi2, en2, fe2, 0.01, dg2)
    # Warmup BMR cost kernel
    spd = np.ones(2, dtype=np.float32)
    scl = np.ones(2, dtype=np.float32)
    lac = np.array([False, True])
    dis = np.array([False, True])
    dmg = np.array([0.0, 0.5], dtype=np.float32)
    msk = np.array([True, True])
    cost = np.zeros(2, dtype=np.float32)
    depons_bmr_cost_kernel(spd, scl, lac, dis, dmg, msk, cost, 4.5, 1.4, 0.0001, 0.002)
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
    # Warmup social_sound_kernel
    ss_xi = np.array([1.0, 2.0], dtype=np.float64)
    ss_yi = np.array([1.0, 2.0], dtype=np.float64)
    ss_xj = np.array([3.0, 4.0], dtype=np.float64)
    ss_yj = np.array([3.0, 4.0], dtype=np.float64)
    ss_ii = np.array([0, 1], dtype=np.int64)
    ss_jj = np.array([1, 0], dtype=np.int64)
    ss_ai = np.zeros(2, dtype=np.float64)
    ss_aj = np.zeros(2, dtype=np.float64)
    ss_oux = np.zeros(2, dtype=np.float64)
    ss_ouy = np.zeros(2, dtype=np.float64)
    ss_osw = np.zeros(2, dtype=np.float64)
    social_sound_kernel(
        ss_xi,
        ss_yi,
        ss_xj,
        ss_yj,
        ss_ii,
        ss_jj,
        ss_ai,
        ss_aj,
        150.0,
        0.01,
        15.0,
        100.0,
        0.1,
        400.0,
        2,
        ss_oux,
        ss_ouy,
        ss_osw,
    )
    # Warmup regrow_food kernel
    rf = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rk = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    regrow_food_kernel(rf, rk, 0.1, 2)
    # Reference memory kernels
    _su = np.zeros((2, 4), dtype=np.float32)
    _mp = np.zeros(2, dtype=np.int32)
    _mc = np.array([2, 2], dtype=np.int32)
    _wt = np.ones(4, dtype=np.float64)
    _ai = np.array([0, 1], dtype=np.int64)
    _ov = np.zeros(2, dtype=np.float64)
    compute_ve_total_kernel(_su, _mp, _mc, _wt, _ai, _ov)
    _cx = np.zeros(2, dtype=np.float32)
    _cy = np.zeros(2, dtype=np.float32)
    _phx = np.zeros((2, 4), dtype=np.float32)
    _phy = np.zeros((2, 4), dtype=np.float32)
    _ovx = np.zeros(2, dtype=np.float64)
    _ovy = np.zeros(2, dtype=np.float64)
    compute_attraction_kernel(_su, _phx, _phy, _mp, _mc, _cx, _cy, _wt, _ai, _ovx, _ovy)
    # Warmup land_avoidance kernel
    la_x = np.array([5.0], dtype=np.float64)
    la_y = np.array([5.0], dtype=np.float64)
    la_h = np.array([0.0], dtype=np.float64)
    la_s = np.array([2.0], dtype=np.float64)
    la_dg = np.full((10, 10), 20.0, dtype=np.float32)
    la_ba = np.array([40.0, 70.0, 120.0], dtype=np.float64)
    la_j = np.zeros(3, dtype=np.float64)
    la_ox = np.zeros(1, dtype=np.float64)
    la_oy = np.zeros(1, dtype=np.float64)
    la_oh = np.zeros(1, dtype=np.float64)
    la_r = np.zeros(1, dtype=np.bool_)
    land_avoidance_kernel(
        la_x,
        la_y,
        la_h,
        la_s,
        la_dg,
        1.0,
        la_ba,
        la_j,
        la_ox,
        la_oy,
        la_oh,
        la_r,
    )
    # Warmup heading_position_reflect kernel
    hp_n = 3
    hp_h = np.array([0.0, 90.0, 180.0], dtype=np.float32)
    hp_pa = np.zeros(hp_n, dtype=np.float64)
    hp_lm = np.ones(hp_n, dtype=np.float64)
    hp_ve = np.zeros(hp_n, dtype=np.float32)
    hp_vtx = np.zeros(hp_n, dtype=np.float32)
    hp_vty = np.zeros(hp_n, dtype=np.float32)
    hp_ddx = np.zeros(hp_n, dtype=np.float64)
    hp_ddy = np.zeros(hp_n, dtype=np.float64)
    hp_x = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    hp_y = np.array([5.0, 10.0, 15.0], dtype=np.float32)
    hp_mask = np.ones(hp_n, dtype=np.bool_)
    hp_disp = np.zeros(hp_n, dtype=np.bool_)
    hp_oh = np.zeros(hp_n, dtype=np.float32)
    hp_odx = np.zeros(hp_n, dtype=np.float32)
    hp_ody = np.zeros(hp_n, dtype=np.float32)
    hp_osd = np.zeros(hp_n, dtype=np.float32)
    heading_position_reflect_kernel(
        hp_h,
        hp_pa,
        hp_lm,
        hp_ve,
        hp_vtx,
        hp_vty,
        hp_ddx,
        hp_ddy,
        hp_x,
        hp_y,
        hp_mask,
        hp_disp,
        1.0,
        4.0,
        20,
        20,
        hp_oh,
        hp_odx,
        hp_ody,
        hp_osd,
    )
    return True
