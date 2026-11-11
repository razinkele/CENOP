"""
JAX-accelerated simulation kernels.

Each kernel is a pure function operating on JAX arrays. No side effects,
no mutation — returns new arrays. Composed into jax_tick_movement and
jax_tick_energy in tick_jax.py.

Requires: jax, jaxlib
Fallback: if JAX unavailable, these functions are never imported.
"""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.lax as lax
import jax.numpy as jnp


def jax_crw_kernel(
    prev_angle,
    prev_log_mov,
    depths,
    salinity,
    mask,
    key,
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
    """CRW angle + step length — fixed-iteration batch rejection.

    Draws K=4 candidate samples per agent, picks the first valid one.
    No while_loop — pure vectorized ops, fully GPU-friendly.
    Rejection rates are typically <5%, so K=4 provides >99.99% coverage
    (P(all 4 fail) ≈ 6e-6 per agent per tick).

    Parameters
    ----------
    prev_angle : float64[n] — previous turning angle
    prev_log_mov : float64[n] — previous log step length
    depths : float64[n] — depth at each agent's position
    salinity : float64[n] — salinity at each agent's position
    mask : bool[n] — active agent mask
    key : JAX PRNGKey
    corr_angle_base .. r1_sd : scalar CRW parameters (12 total)

    Returns
    -------
    out_angle : float64[n] — turning angle per agent
    out_log_mov : float64[n] — log step length per agent
    """
    K = 4  # candidate samples per agent (was 32; <5% rejection rate → K=4 safe)
    n = prev_angle.shape[0]
    env_mod = corr_angle_bathy * depths + corr_angle_salinity * salinity + corr_angle_base_sd

    # === Loop 1: Angle — draw K candidates, pick first with |angle| <= 180 ===
    key, k1 = jax.random.split(key)
    rand_angles = jax.random.normal(k1, (K, n)) * r2_sd + r2_mean  # (K, n)
    angle_tmps = corr_angle_base * prev_angle[None, :] + rand_angles  # (K, n)
    pres_angles = angle_tmps * env_mod[None, :]  # (K, n)

    valid1 = jnp.abs(pres_angles) <= 180.0  # (K, n)
    # First valid index per agent (K if none valid)
    first_valid1 = jnp.argmax(valid1, axis=0)  # (n,)
    any_valid1 = jnp.any(valid1, axis=0)  # (n,)
    pres_angle = pres_angles[first_valid1, jnp.arange(n)]
    # Fallback: clamp to ±90
    pres_angle = jnp.where(any_valid1, pres_angle, jnp.sign(pres_angles[0]) * 90.0)

    # === Loop 2: Distance-dependent modulation ===
    prev_mov_value = 10.0**prev_log_mov
    needs_mod = prev_mov_value <= m_param

    key, k2 = jax.random.split(key)
    rnds = jax.random.normal(k2, (K, n))  # N(0,1), not uniform
    abs_angle = jnp.abs(pres_angle)
    # Each candidate: |angle| + rnd - rnd * prev_mov/m_param
    new_angle_mags = abs_angle[None, :] + rnds - rnds * (prev_mov_value / m_param)[None, :]
    dist_angles = jnp.sign(pres_angle)[None, :] * new_angle_mags  # (K, n)

    valid2 = jnp.abs(dist_angles) < 180.0  # (K, n)
    first_valid2 = jnp.argmax(valid2, axis=0)
    any_valid2 = jnp.any(valid2, axis=0)
    dist_result = dist_angles[first_valid2, jnp.arange(n)]

    # Fallback: sign * (random + 90)
    key, kfb = jax.random.split(key)
    fb_rnd = jax.random.uniform(kfb, (n,)) * 20.0
    dist_result = jnp.where(any_valid2, dist_result, jnp.sign(pres_angle) * (fb_rnd + 90.0))

    # Only apply distance mod where needed AND angle was >= 180
    need_dist = needs_mod & (jnp.abs(pres_angle) >= 180.0)
    pres_angle = jnp.where(need_dist, dist_result, pres_angle)

    # === Loop 3: Step length — draw K candidates, pick first <= max_mov ===
    key, k3 = jax.random.split(key)
    rand_lens = jax.random.normal(k3, (K, n)) * r1_sd + r1_mean  # (K, n)
    log_movs = (
        corr_logmov_length * prev_log_mov[None, :]
        + corr_logmov_bathy * depths[None, :]
        + corr_logmov_salinity * salinity[None, :]
        + rand_lens
    )  # (K, n)

    valid3 = log_movs <= max_mov  # (K, n)
    first_valid3 = jnp.argmax(valid3, axis=0)
    any_valid3 = jnp.any(valid3, axis=0)
    log_mov = log_movs[first_valid3, jnp.arange(n)]
    log_mov = jnp.where(any_valid3, log_mov, max_mov)

    # Apply mask: inactive agents get 0 angle, keep prev_log_mov
    pres_angle = jnp.where(mask, pres_angle, 0.0)
    log_mov = jnp.where(mask, log_mov, prev_log_mov)

    return pres_angle, log_mov


# ---------------------------------------------------------------------------
# Reference Memory Kernels
# ---------------------------------------------------------------------------


def jax_compute_ve_total(
    stored_util: jnp.ndarray,
    mem_ptr: jnp.ndarray,
    mem_count: jnp.ndarray,
    work_mem_table: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Compute veTotal (expected food value) for each agent.

    Vectorised JAX equivalent of ``compute_ve_total`` in ref_mem.py.
    No loops — pure array indexing + weighted sum.

    Parameters
    ----------
    stored_util : float32[n, mem_size] — circular buffer of stored utilities
    mem_ptr : int32[n] — write pointer (next slot to write)
    mem_count : int32[n] — number of entries written so far
    work_mem_table : float64[mem_size] — decay weights (workMemStrength)
    mask : bool[n] — active agent mask

    Returns
    -------
    ve_total : float32[n] — expected food value per agent
    """
    n, mem_size = stored_util.shape

    # Build ordered indices: most-recent-first circular buffer unwrap
    offsets = jnp.arange(mem_size, dtype=jnp.int32)  # [0, 1, ..., mem_size-1]
    # (n, mem_size): for each agent, ordered[i, j] = (ptr_i - 1 - j) % mem_size
    ordered = (mem_ptr[:, None] - 1 - offsets[None, :]) % mem_size

    # Gather food values in recency order
    row_idx = jnp.arange(n)[:, None]  # (n, 1)
    ordered_food = stored_util[row_idx, ordered]  # (n, mem_size)

    # Valid entries: Java uses n-1 entries (skips oldest)
    n_valid = jnp.maximum(jnp.minimum(mem_count, mem_size) - 1, 0)  # (n,)
    entry_idx = jnp.arange(mem_size)[None, :]  # (1, mem_size)
    valid_mask = entry_idx < n_valid[:, None]  # (n, mem_size)

    # Weighted sum
    weights = work_mem_table[:mem_size].astype(jnp.float32)  # (mem_size,)
    weighted = ordered_food * weights[None, :] * valid_mask

    ve_total = weighted.sum(axis=1).astype(jnp.float32)

    # Zero out inactive agents
    ve_total = jnp.where(mask, ve_total, 0.0)

    return ve_total


def jax_compute_attraction(
    stored_util: jnp.ndarray,
    pos_hist_x: jnp.ndarray,
    pos_hist_y: jnp.ndarray,
    mem_ptr: jnp.ndarray,
    mem_count: jnp.ndarray,
    current_x: jnp.ndarray,
    current_y: jnp.ndarray,
    work_mem_table: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute attraction vector (vt_x, vt_y) from reference memory.

    Vectorised JAX equivalent of ``compute_attraction_vector`` in ref_mem.py.
    No world wrapping (matching the typical simulation call with world_width=0).

    Parameters
    ----------
    stored_util : float32[n, mem_size] — circular buffer of stored utilities
    pos_hist_x : float32[n, mem_size] — x positions circular buffer
    pos_hist_y : float32[n, mem_size] — y positions circular buffer
    mem_ptr : int32[n] — write pointer
    mem_count : int32[n] — number of entries written
    current_x : float32[n] — current x position
    current_y : float32[n] — current y position
    work_mem_table : float64[mem_size] — refMemStrength decay weights
    mask : bool[n] — active agent mask

    Returns
    -------
    (vt_x, vt_y) : float32[n] each — attraction vector per agent
    """
    n, mem_size = stored_util.shape

    # Build ordered indices (most-recent-first)
    offsets = jnp.arange(mem_size, dtype=jnp.int32)
    ordered = (mem_ptr[:, None] - 1 - offsets[None, :]) % mem_size

    row_idx = jnp.arange(n)[:, None]

    # Gather in recency order
    ordered_util = stored_util[row_idx, ordered]
    ordered_px = pos_hist_x[row_idx, ordered]
    ordered_py = pos_hist_y[row_idx, ordered]

    # Direction vectors: past_pos - current_pos
    dx = ordered_px - current_x[:, None]
    dy = ordered_py - current_y[:, None]

    # Distance
    dist = jnp.sqrt(dx * dx + dy * dy)
    safe_dist = jnp.where(dist < 1e-20, 1.0, dist)

    # Unit direction
    unit_x = dx / safe_dist
    unit_y = dy / safe_dist

    # Weight = util * refMemStrength / distance
    ref_weights = work_mem_table[:mem_size].astype(jnp.float32)
    factor = jnp.where(
        dist < 1e-20,
        9999.0 * ordered_util,
        ordered_util * ref_weights[None, :] / safe_dist,
    )

    # Validity mask: skip index 0 (current position), skip beyond count, skip zero util
    n_valid = jnp.minimum(mem_count, mem_size)  # (n,)
    entry_idx = jnp.arange(mem_size)[None, :]
    valid = (entry_idx >= 1) & (entry_idx < n_valid[:, None]) & (ordered_util != 0) & (dist > 0)

    factor = factor * valid

    vt_x = (factor * unit_x).sum(axis=1).astype(jnp.float32)
    vt_y = (factor * unit_y).sum(axis=1).astype(jnp.float32)

    # Zero out inactive agents
    vt_x = jnp.where(mask, vt_x, 0.0)
    vt_y = jnp.where(mask, vt_y, 0.0)

    return vt_x, vt_y


# ---------------------------------------------------------------------------
# Dispersal step (DEPONS AbstractPSMDispersal.disperse())
# ---------------------------------------------------------------------------


def jax_disperse_step(
    x,
    y,
    heading,
    prev_step_heading,
    is_dispersing,
    dispersal_distance_traveled,
    dispersal_target_distance,
    mask,
    psm_angle,
    psm_log,
    mean_disp_dist,
    depth_grid,
    min_depth_dispersal,
    world_w,
    world_h,
    key,
    enough_water_ahead=None,
):
    """PSM-Type2 dispersal step for ``mask & is_dispersing``; mirrors
    PorpoisePopulation._disperse_step, which runs before the CRW step.

    In DEPONS order:
    1. completion: traveled >= 0.95 * target ends dispersal (no turn);
    2. PSM turn: prev_step_heading + U(-psm_angle, psm_angle) * SSLogis(3p - 1.5), with
       path length p = traveled / (0.95 * target) (0 where target <= 0); prev_step_heading
       takes the turned heading, even when a stop follows;
    2b. stop where ``enough_water_ahead`` (the land check's stored result, frozen while
       dispersing: DEPONS disperse()'s first-call filter, D9) is False; None means all True;
    3. look-ahead point one dispersal step (mean_disp_dist / 0.4 cells) along it;
    4. stop if that point is off the grid (DEPONS getPointAtHeadingAndDistNoBorder);
    5. stop unless the depth there exceeds min_depth_dispersal (NaN stops);
    6. a stopped animal ends dispersal and keeps its pre-turn heading.

    The caller ends dispersal for completed and stopped animals (traveled reset to 0
    here) and must also reset days_declining_energy for them, as _end_dispersal does.

    Returns
    -------
    (heading, prev_step_heading, is_dispersing, dispersal_distance_traveled)
    """
    d = mask & is_dispersing

    # 1. Completion on path length (float32 compare, as NumPy)
    done = d & (dispersal_distance_traveled >= 0.95 * dispersal_target_distance)
    d = d & ~done

    # 2. PSM turn (DispersalPSMType2.java:76, 91)
    has_target = dispersal_target_distance > 0
    safe_target = jnp.where(has_target, dispersal_target_distance, 1.0)
    dist_percent = jnp.where(has_target, dispersal_distance_traveled / (0.95 * safe_target), 0.0)
    dist_percent = jnp.nan_to_num(dist_percent, nan=0.0, posinf=1.0, neginf=0.0)
    dist_percent = jnp.clip(dist_percent, 0.0, 10.0)
    dist_log_x = jnp.clip(3 * dist_percent - 1.5, -100, 100)
    logistic = 1.0 / (1.0 + jnp.exp((0.0 - dist_log_x) / psm_log))
    delta = jax.random.uniform(
        key, shape=heading.shape, dtype=jnp.float64, minval=-psm_angle, maxval=psm_angle
    )
    # Store in heading's dtype, then wrap, as NumPy assigns into the float32 heading array
    turned = (prev_step_heading + delta * logistic).astype(heading.dtype) % 360.0
    new_prev_step_heading = jnp.where(d, turned, prev_step_heading).astype(prev_step_heading.dtype)

    # 3. Look-ahead point from the stored (rounded) heading, in float64
    step = mean_disp_dist / 0.4
    rad = jnp.radians(turned.astype(jnp.float64))
    px = x + step * jnp.sin(rad)
    py = y + step * jnp.cos(rad)

    # 2b. Not-enough-water-ahead stop, then 4. off-grid stop
    no_water = jnp.zeros_like(d) if enough_water_ahead is None else ~enough_water_ahead
    stop = no_water | (px < 0) | (py < 0) | (px >= world_w - 0.5) | (py >= world_h - 0.5)

    # 5. Depth stop; off-grid points are already stopped, clipping keeps the lookup in bounds
    xi = jnp.clip(px.astype(jnp.int32), 0, world_w - 1)
    yi = jnp.clip(py.astype(jnp.int32), 0, world_h - 1)
    stop = d & (stop | ~(depth_grid[yi, xi] > min_depth_dispersal))

    # 6. End dispersal for completed and stopped animals; stopped keep the pre-turn heading
    ended = done | stop
    new_is_dispersing = jnp.where(ended, False, is_dispersing)
    new_traveled = jnp.where(ended, 0.0, dispersal_distance_traveled).astype(
        dispersal_distance_traveled.dtype
    )
    new_heading = jnp.where(d & ~stop, turned, heading)

    return new_heading, new_prev_step_heading, new_is_dispersing, new_traveled


# ---------------------------------------------------------------------------
# Heading Composition + Position Update + Boundary Reflection (J3)
# ---------------------------------------------------------------------------


def jax_heading_composition(
    heading,
    pres_angle,
    log_mov,
    ve_total,
    vt_x,
    vt_y,
    deter_dx,
    deter_dy,
    social_dx,
    social_dy,
    mask,
    is_dispersing,
    prev_angle,
    inertia_const,
    mean_disp_dist,
    crw_heading=None,
):
    """Heading composition: CRW + ref_mem + deterrence + social -> final heading/dx/dy.

    Combines all movement influences into a final heading and displacement vector for
    CRW movers (``mask & ~is_dispersing``). Dispersers (after jax_disperse_step) skip
    the CRW turn and the composition: they keep the PSM heading, take the fixed
    dispersal step, and keep prev_angle (DEPONS skips stdMove() while dispersing).

    Parameters
    ----------
    heading : float64[n] — current heading (degrees), PSM heading for dispersers
    pres_angle : float64[n] — CRW turning angle from jax_crw_kernel
    log_mov : float64[n] — CRW log step length from jax_crw_kernel
    ve_total : float32[n] — expected food value from jax_compute_ve_total
    vt_x, vt_y : float32[n] — attraction vectors from jax_compute_attraction
    deter_dx, deter_dy : float64[n] — deterrence displacement vectors
    social_dx, social_dy : float64[n] — social cohesion vectors
    mask : bool[n] — active agent mask
    is_dispersing : bool[n] — which agents are in dispersal mode
    prev_angle : float64[n] — previous total turn; updated for CRW movers only
    inertia_const : float scalar — CRW inertia constant
    mean_disp_dist : float scalar — mean dispersal distance per step
    crw_heading : float64[n] or None — CRW movers' heading with the CRW turn (and the D15
        land check) already applied; ``pres_angle`` is then not added again. prev_angle is
        still measured from ``heading``.

    Returns
    -------
    (new_heading, new_prev_angle, step_dist, dx, dy, deter_strength,
     dispersal_distance_delta) — tuple of float64[n] arrays
    """
    inertia_const = jnp.float64(inertia_const)
    mean_disp_dist = jnp.float64(mean_disp_dist)
    disp_mask = mask & is_dispersing
    crw_mask = mask & ~is_dispersing

    # 1. Apply CRW turning angle to heading (CRW movers only), unless the caller already did
    if crw_heading is None:
        new_heading = jnp.where(crw_mask, (heading + pres_angle) % 360.0, heading)
    else:
        new_heading = jnp.where(crw_mask, crw_heading, heading)

    # 2. CRW unit direction from heading
    rads = jnp.radians(new_heading)
    dx_unit = jnp.sin(rads)
    dy_unit = jnp.cos(rads)

    # 3. Heading composition
    step_pow = jnp.power(10.0, log_mov)
    crw_contrib = inertia_const + step_pow * ve_total

    total_dx = dx_unit * crw_contrib + vt_x + deter_dx + social_dx
    total_dy = dy_unit * crw_contrib + vt_y + deter_dy + social_dy

    # Deterrence strength (for tracking)
    deter_strength = jnp.hypot(deter_dx, deter_dy)
    deter_strength = jnp.where(mask, deter_strength, 0.0)

    # 4. New heading from composite vector; dispersers keep their PSM heading
    composed_heading = jnp.degrees(jnp.arctan2(total_dx, total_dy)) % 360.0
    final_heading = jnp.where(crw_mask, composed_heading, new_heading)

    # 5. Prev angle: total turn relative to the original heading (CRW movers only)
    total_turn = (final_heading - heading + 180.0) % 360.0 - 180.0
    new_prev_angle = jnp.where(crw_mask, total_turn, prev_angle)

    # 6. Step distance
    step_dist = step_pow / 4.0
    disp_step = mean_disp_dist / 0.4
    step_dist = jnp.where(disp_mask, disp_step, step_dist)

    # Dispersal distance traveled delta (only for dispersing agents)
    dispersal_distance_delta = jnp.where(disp_mask, disp_step, 0.0)

    # 7. Final dx/dy from final heading and step distance
    final_rads = jnp.radians(final_heading)
    dx = jnp.sin(final_rads) * step_dist
    dy = jnp.cos(final_rads) * step_dist

    # Zero out inactive agents
    dx = jnp.where(mask, dx, 0.0)
    dy = jnp.where(mask, dy, 0.0)
    step_dist = jnp.where(mask, step_dist, 0.0)

    return (
        final_heading,
        new_prev_angle,
        step_dist,
        dx,
        dy,
        deter_strength,
        dispersal_distance_delta,
    )


def jax_reflect_boundaries(new_x, new_y, dx, dy, world_w, world_h):
    """DEPONS-style bouncy borders — reflect positions that overshoot.

    When a position overshoots an edge, the overshot component is
    reflected back into the domain and the displacement sign is flipped.

    Parameters
    ----------
    new_x, new_y : float64[n] — proposed positions (x + dx, y + dy)
    dx, dy : float64[n] — displacement vectors
    world_w, world_h : int — world dimensions in cells

    Returns
    -------
    (reflected_x, reflected_y, reflected_dx, reflected_dy)
    """
    max_x = jnp.float64(world_w - 1)
    max_y = jnp.float64(world_h - 1)

    # X reflection
    under_x = new_x < 0.0
    over_x = new_x > max_x
    rx = jnp.where(under_x, -new_x, new_x)
    rx = jnp.where(over_x, 2.0 * max_x - new_x, rx)
    rx = jnp.clip(rx, 0.0, max_x)
    rdx = jnp.where(under_x | over_x, -dx, dx)

    # Y reflection
    under_y = new_y < 0.0
    over_y = new_y > max_y
    ry = jnp.where(under_y, -new_y, new_y)
    ry = jnp.where(over_y, 2.0 * max_y - new_y, ry)
    ry = jnp.clip(ry, 0.0, max_y)
    rdy = jnp.where(under_y | over_y, -dy, dy)

    return rx, ry, rdx, rdy


def jax_land_avoidance(x, y, heading, step_dist, depth_grid, min_depth, on_land, key):
    """Land avoidance: try 3 angles x 2 directions.

    For each blocked agent (on land), tries 3 base angles (40, 70, 120) with
    jitter, testing both right and left turns. Picks the deeper valid direction.
    Unresolved agents stay in place and turn 180 degrees.

    Uses ``lax.fori_loop`` with a ``resolved`` carry flag for early-exit semantics.

    Parameters
    ----------
    x, y : float32[n] — current positions (pre-move, used as turn origin)
    heading : float32[n] — current heading (degrees)
    step_dist : float32[n] — step distance per agent
    depth_grid : float32[rows, cols] — landscape depth grid
    min_depth : float — minimum water depth threshold
    on_land : bool[n] — True for agents needing land avoidance
    key : JAX PRNG key (for jitter)

    Returns
    -------
    (new_x, new_y, new_heading, resolved, key) — updated positions for
    resolved agents; unresolved agents stay in place with heading + 180.
    """
    base_angles = jnp.array([40.0, 70.0, 120.0])
    max_x = jnp.float32(depth_grid.shape[1] - 1)
    max_y = jnp.float32(depth_grid.shape[0] - 1)

    # Draw 3 jitter values up front
    key, k_jitter = jax.random.split(key)
    jitter = jax.random.uniform(k_jitter, (3,)) * 10.0

    def loop_body(angle_idx, carry):
        out_x, out_y, out_heading, resolved = carry
        angle = base_angles[angle_idx] + jitter[angle_idx]
        should_try = on_land & ~resolved

        # Right turn
        rh = (heading + angle) % 360.0
        rr = rh * (jnp.pi / 180.0)
        rx = jnp.clip(x + jnp.sin(rr) * step_dist, 0.0, max_x)
        ry = jnp.clip(y + jnp.cos(rr) * step_dist, 0.0, max_y)
        rd = depth_grid[
            jnp.clip(ry.astype(jnp.int32), 0, depth_grid.shape[0] - 1),
            jnp.clip(rx.astype(jnp.int32), 0, depth_grid.shape[1] - 1),
        ]

        # Left turn
        lh = (heading - angle + 360.0) % 360.0
        lr = lh * (jnp.pi / 180.0)
        lx = jnp.clip(x + jnp.sin(lr) * step_dist, 0.0, max_x)
        ly = jnp.clip(y + jnp.cos(lr) * step_dist, 0.0, max_y)
        ld = depth_grid[
            jnp.clip(ly.astype(jnp.int32), 0, depth_grid.shape[0] - 1),
            jnp.clip(lx.astype(jnp.int32), 0, depth_grid.shape[1] - 1),
        ]

        rok = rd >= min_depth
        lok = ld >= min_depth

        # Pick deeper when both valid
        use_right = (rok & lok & (rd >= ld)) | (rok & ~lok)
        use_left = (rok & lok & (ld > rd)) | (lok & ~rok)

        found = (use_right | use_left) & should_try

        new_ox = jnp.where(found & use_right, rx, jnp.where(found & use_left, lx, out_x))
        new_oy = jnp.where(found & use_right, ry, jnp.where(found & use_left, ly, out_y))
        new_oh = jnp.where(found & use_right, rh, jnp.where(found & use_left, lh, out_heading))
        new_resolved = resolved | found

        # Preserve dtypes to match carry input types
        return (
            new_ox.astype(jnp.float32),
            new_oy.astype(jnp.float32),
            new_oh.astype(jnp.float32),
            new_resolved,
        )

    init = (x, y, heading, jnp.zeros_like(on_land))
    out_x, out_y, out_heading, resolved = lax.fori_loop(0, 3, loop_body, init)

    # Unresolved: stay in place, turn 180 degrees
    unresolved = on_land & ~resolved
    out_heading = jnp.where(unresolved, (heading + 180.0) % 360.0, out_heading)

    return out_x, out_y, out_heading, resolved, key


def jax_update_positions(
    x, y, dx, dy, heading, world_w, world_h, mask, depth_grid=None, start_x=None, start_y=None
):
    """Compute new positions, reflect at boundaries, recalculate heading for reflected.

    Combines position addition with boundary reflection and heading correction. With
    ``depth_grid`` (the D15 land-check path) it also applies DEPONS beh 6 (stdMove 639-661),
    as PorpoisePopulation._apply_positions does: an active agent whose new cell has depth <= 0
    returns to its start-of-tick position (``start_x/_y``; DEPONS posList.get(0), even when
    beh 4 moved the step base ``x/y``), facing it from the attempted position. Facing one's
    own position is a no-op; NaN depth is not <= 0. The cell is read from the float32 position
    _step_jax stores, clipped as _recompute_cell_indices does.

    Parameters
    ----------
    x, y : float32[n] — current positions
    dx, dy : float64[n] — displacement vectors
    heading : float64[n] — current heading (degrees)
    world_w, world_h : int — world dimensions in cells
    mask : bool[n] — active agent mask
    depth_grid : float32[rows, cols] or None — None keeps the plain update (no beh 6)
    start_x, start_y : float32[n] — start-of-tick positions (with ``depth_grid``)

    Returns
    -------
    (new_x, new_y, new_heading) — reflected positions and corrected heading
    """
    # Proposed positions
    new_x = x + dx
    new_y = y + dy

    # Reflect at boundaries
    ref_x, ref_y, ref_dx, ref_dy = jax_reflect_boundaries(new_x, new_y, dx, dy, world_w, world_h)

    # Detect reflected agents (displacement sign changed)
    reflected = mask & ((ref_dx != dx) | (ref_dy != dy))

    # Recalculate heading for reflected agents
    reflected_heading = jnp.degrees(jnp.arctan2(ref_dx, ref_dy)) % 360.0
    new_heading = jnp.where(reflected, reflected_heading, heading)

    # Inactive agents keep original positions
    final_x = jnp.where(mask, ref_x, x)
    final_y = jnp.where(mask, ref_y, y)
    final_heading = jnp.where(mask, new_heading, heading)

    if depth_grid is not None:
        # beh 6: roll back from depth <= 0 to the start-of-tick position, facing it
        tried_x = final_x.astype(jnp.float32)
        tried_y = final_y.astype(jnp.float32)
        xi = jnp.clip(tried_x.astype(jnp.int32), 0, world_w - 1)
        yi = jnp.clip(tried_y.astype(jnp.int32), 0, world_h - 1)
        back = mask & (depth_grid[yi, xi] <= 0)
        sx = start_x.astype(jnp.float64)
        sy = start_y.astype(jnp.float64)
        fdx = sx - tried_x.astype(jnp.float64)
        fdy = sy - tried_y.astype(jnp.float64)
        face = back & ((fdx != 0.0) | (fdy != 0.0))
        face_heading = jnp.degrees(jnp.arctan2(fdx, fdy)) % 360.0
        final_heading = jnp.where(face, face_heading, final_heading)
        final_x = jnp.where(back, sx, final_x)
        final_y = jnp.where(back, sy, final_y)

    return final_x, final_y, final_heading


# ---------------------------------------------------------------------------
# DEPONS land avoidance before heading composition (D15)
# ---------------------------------------------------------------------------

_LC_NO_DEPTH = -9999.0
_LC_DEG = math.pi / 180.0
# Moore neighbours, in the order beh 5 scans them (first maximum wins), as in land_check.py
_LC_NB_DX = (-1, 0, 1, -1, 1, -1, 0, 1)
_LC_NB_DY = (-1, -1, -1, 0, 0, 1, 1, 1)


def _lc_depth_at(depth_grid, x, y):
    """Depth (float64) of the cell holding (x, y); -9999 off the grid or for NaN."""
    rows, cols = depth_grid.shape
    inside = (x >= 0.0) & (y >= 0.0) & (x < cols) & (y < rows)
    xi = jnp.clip(jnp.floor(jnp.where(inside, x, 0.0)), 0, cols - 1).astype(jnp.int32)
    yi = jnp.clip(jnp.floor(jnp.where(inside, y, 0.0)), 0, rows - 1).astype(jnp.int32)
    v = depth_grid[yi, xi].astype(jnp.float64)
    return jnp.where(inside & ~jnp.isnan(v), v, _LC_NO_DEPTH)


def _lc_reflect(v, m):
    """Bounce a coordinate into [0, m) (Repast BouncyBorders), as land_check._reflect:
    v < 0 -> -v, v > m -> 2m - v, v == m -> the largest float below m. NaN passes through."""
    v = jnp.where(v < 0.0, -v, jnp.where(v > m, 2.0 * m - v, v))
    return jnp.where(v == m, math.nextafter(m, 0.0), v)


def _lc_look_depth(depth_grid, x, y):
    """Depth at a look-ahead point, reflected into the grid first."""
    rows, cols = depth_grid.shape
    return _lc_depth_at(depth_grid, _lc_reflect(x, float(cols)), _lc_reflect(y, float(rows)))


def _lc_path_ok(depth_grid, px, py, h, pm, checking, n_samples_max):
    """DEPONS checkDepth for the ``checking`` agents (other entries are meaningless): the point
    presMov ahead has depth >= 0 and every 0.1-cell sample k = 1 .. ceil(presMov / 0.1)
    depth > 0. The sample loop stops once every checking agent has failed or run out of
    samples, and never passes ``n_samples_max``."""
    rad = h * _LC_DEG
    s = jnp.sin(rad)
    c = jnp.cos(rad)
    ok = _lc_look_depth(depth_grid, px + s * pm, py + c * pm) >= 0.0
    vx = 0.1 * s
    vy = 0.1 * c
    n = jnp.ceil(pm / 0.1)
    k_max = jnp.minimum(jnp.max(jnp.where(checking, n, 0.0)), float(n_samples_max))

    def cond(carry):
        k, ok = carry
        return (k <= k_max) & jnp.any(checking & ok & (k <= n))

    def body(carry):
        k, ok = carry
        d = _lc_look_depth(depth_grid, px + vx * k, py + vy * k)
        return k + 1.0, ok & ~(checking & (k <= n) & ~(d > 0.0))

    _, ok = lax.while_loop(cond, body, (jnp.float64(1.0), ok))
    return ok


def jax_depons_land_check(
    x,
    y,
    heading,
    log_mov,
    mask,
    depth_grid,
    wmin,
    pos_hist_x,
    pos_hist_y,
    mem_ptr,
    mem_count,
    key,
    n_samples_max,
):
    """DEPONS 3.2 land avoidance (stdMove 467-533) for ``mask`` agents; the JAX twin of
    cenop.movement.land_check.depons_land_check, with the same semantics.

    Up to 100 rounds (``lax.while_loop``, ending early once no agent needs avoidance): the
    path check (checkDepth) looks presMov = 10**log_mov cells ahead. An agent failing it draws
    one U(0, 10) jitter r (one draw per agent per round, from a key split per round) and tries
    +-(40 + r), +-(70 + r), +-(120 + r) at presMov ahead; a point is good if depth >= wmin; it
    turns toward the good side (the deeper one if both, right on ties; right = +). If no tier
    is good (beh 4) it walks back from the newest remembered position (ring index
    ``(mem_ptr - 1) % mem_size``) through at most min(20, mem_count) entries, facing and
    stepping onto each (a no-op for its own position), until the path check passes; 20 steps
    end its rounds, a shorter history that runs out does not. Then a final path check gives
    ``enough``; where it fails (beh 5) the agent faces the centre of the deepest of its 8
    neighbours (first maximum in a fixed order) and log_mov becomes 0.

    Look-ahead points are reflected into [0, W) x [0, H) before the floor lookup; off-grid
    beh-5 neighbours and NaN cells read -9999. ``n_samples_max`` (static under jit) bounds the
    path samples: ceil(10**max_mov / 0.1).

    Returns
    -------
    (x, y, heading, log_mov, enough) — inputs' dtypes; only masked agents change; heading in
    [0, 360) where turned; ``enough`` is the final path check, True outside ``mask``.
    """
    n = x.shape[0]
    mem_size = pos_hist_x.shape[1]
    rows = jnp.arange(n)
    f64 = jnp.float64
    pm = jnp.power(10.0, log_mov.astype(f64))
    n_back = jnp.minimum(20, mem_count.astype(jnp.int32))
    no = jnp.zeros(n, dtype=bool)

    def walk_back(args):
        px, py, h, moved, turned, walk = args

        def cond(c):
            return (c[0] < 20) & jnp.any(c[6])

        def body(c):
            j, px, py, h, moved, turned, walking, passed = c
            b = jnp.mod(mem_ptr.astype(jnp.int32) - 1 - j, mem_size)
            bx = pos_hist_x[rows, b].astype(f64)
            by = pos_hist_y[rows, b].astype(f64)
            face = walking & ((bx != px) | (by != py))
            h = jnp.where(face, jnp.arctan2(bx - px, by - py) / _LC_DEG, h)
            px = jnp.where(face, bx, px)
            py = jnp.where(face, by, py)
            ok = _lc_path_ok(depth_grid, px, py, h, pm, walking, n_samples_max)
            passed = passed | (walking & ok)
            walking = walking & ~ok & (j + 1 < n_back)
            return j + 1, px, py, h, moved | face, turned | face, walking, passed

        init = (jnp.int32(0), px, py, h, moved, turned, walk & (n_back > 0), no)
        _, px, py, h, moved, turned, _, passed = lax.while_loop(cond, body, init)
        return px, py, h, moved, turned, passed

    def no_walk(args):
        px, py, h, moved, turned, _ = args
        return px, py, h, moved, turned, no

    def round_cond(s):
        return (s[0] < 100) & jnp.any(s[5])

    def round_body(s):
        rnd, key, px, py, h, need, moved, turned, known_ok = s
        ok = _lc_path_ok(depth_grid, px, py, h, pm, need, n_samples_max)
        known_ok = known_ok | (need & ok)
        need = need & ~ok

        # avoidLand: one jitter per agent per round, shared by the three tiers
        key, sub = jax.random.split(key)
        r = jax.random.uniform(sub, (n,), dtype=f64, minval=0.0, maxval=10.0)
        found = no
        turn = jnp.zeros(n, dtype=f64)
        for base in (40.0, 70.0, 120.0):
            a = base + r
            rr = (h + a) * _LC_DEG
            rl = (h - a) * _LC_DEG
            dr = _lc_look_depth(depth_grid, px + jnp.sin(rr) * pm, py + jnp.cos(rr) * pm)
            dl = _lc_look_depth(depth_grid, px + jnp.sin(rl) * pm, py + jnp.cos(rl) * pm)
            good_r = dr >= wmin
            good_l = dl >= wmin
            t = jnp.where(good_r & good_l, jnp.where(dr >= dl, a, -a), jnp.where(good_r, a, -a))
            take = ~found & (good_r | good_l)
            turn = jnp.where(take, t, turn)
            found = found | take
        tier = need & found
        h = jnp.where(tier, h + turn, h)
        turned = turned | tier

        # beh 4 for the agents no tier helped
        walk = need & ~found
        px, py, h, moved, turned, passed = lax.cond(
            jnp.any(walk), walk_back, no_walk, (px, py, h, moved, turned, walk)
        )
        known_ok = known_ok | passed
        need = need & ~(walk & (passed | (n_back == 20)))
        return rnd + 1, key, px, py, h, need, moved, turned, known_ok

    init = (
        jnp.int32(0),
        key,
        x.astype(f64),
        y.astype(f64),
        heading.astype(f64),
        mask,
        no,
        no,
        no,
    )
    _, _, px, py, h, _, moved, turned, known_ok = lax.while_loop(round_cond, round_body, init)

    # Final path check; an agent that left the rounds on a pass is unchanged since then
    recheck = mask & ~known_ok
    enough = jnp.where(
        recheck, _lc_path_ok(depth_grid, px, py, h, pm, recheck, n_samples_max), True
    )

    # beh 5: face the centre of the deepest Moore neighbour, presMov = 1
    beh5 = mask & ~enough
    cx = jnp.floor(px)
    cy = jnp.floor(py)
    nb = jnp.stack(
        [
            _lc_depth_at(depth_grid, cx + ndx + 0.5, cy + ndy + 0.5)
            for ndx, ndy in zip(_LC_NB_DX, _LC_NB_DY)
        ]
    )
    best = jnp.argmax(nb, axis=0)
    tx = cx + jnp.asarray(_LC_NB_DX, dtype=f64)[best] + 0.5
    ty = cy + jnp.asarray(_LC_NB_DY, dtype=f64)[best] + 0.5
    h = jnp.where(beh5, jnp.arctan2(tx - px, ty - py) / _LC_DEG, h)
    turned = turned | beh5

    h_out = jnp.mod(h, 360.0).astype(heading.dtype)
    h_out = jnp.where(h_out >= 360.0, jnp.zeros_like(h_out), h_out)  # rounding just below 360
    return (
        jnp.where(moved, px.astype(x.dtype), x),
        jnp.where(moved, py.astype(y.dtype), y),
        jnp.where(turned, h_out, heading),
        jnp.where(beh5, jnp.zeros_like(log_mov), log_mov),
        enough,
    )


# ---------------------------------------------------------------------------
# Food Intake + Energy + Mortality Kernels (J5)
# ---------------------------------------------------------------------------


def jax_eat_food(food_grid, xi, yi, energy, min_food, active_mask=None):
    """Two-pass proportional food sharing.

    Pass 1: fraction = clip((20 - energy) / 10, 0, 0.99)
            demand = food_at_cell * fraction
            accumulate total demand per cell (scatter-add)
    Pass 2: if total_demand <= available: eat full demand
            else: eat proportional share (demand / total_demand * available)

    The min_food floor is applied only to cells that had food and were visited by a live
    agent (DEPONS CellData.eatFood acts only if food > 0). Empty cells stay at 0.

    Parameters
    ----------
    food_grid : float32[rows, cols]
    xi, yi : int32[n] — cell indices (x=col, y=row)
    energy : float32[n] — current energy
    min_food : float — minimum food floor (0.01)
    active_mask : bool[n] or None — live agents; inactive slots neither eat nor floor
        (None = all active)

    Returns
    -------
    (food_eaten, new_food_grid) — float32[n], float32[rows, cols]
    """
    if active_mask is None:
        active_mask = jnp.ones(energy.shape, dtype=bool)
    frac = jnp.clip((20.0 - energy) / 10.0, 0.0, 0.99)
    cell_food = food_grid[yi, xi]
    agent_demand = jnp.where(active_mask, cell_food * frac, 0.0)

    # Scatter-add demands per cell
    demand_grid = jnp.zeros_like(food_grid)
    demand_grid = demand_grid.at[yi, xi].add(agent_demand)
    total_demand = demand_grid[yi, xi]

    # Proportional sharing
    available = jnp.maximum(cell_food, 0.0)
    share = jnp.where(total_demand > 0, agent_demand / total_demand, 0.0)
    eaten_comp = available * share
    eaten_noncomp = agent_demand
    eaten = jnp.where(total_demand <= available, eaten_noncomp, eaten_comp)
    eaten = jnp.where(active_mask, eaten, 0.0)

    # Update food grid; floor only visited cells that had food (no phantom food)
    eaten_total_grid = jnp.zeros_like(food_grid)
    eaten_total_grid = eaten_total_grid.at[yi, xi].add(eaten)
    visits = (
        jnp.zeros(food_grid.shape, dtype=jnp.int32).at[yi, xi].add(active_mask.astype(jnp.int32))
    )
    eat_here = (visits > 0) & (food_grid > 0.0)
    new_food = jnp.where(eat_here, jnp.maximum(food_grid - eaten_total_grid, min_food), food_grid)

    return eaten, new_food


def jax_bmr_cost(
    energy,
    active_mask,
    speed,
    is_lactating,
    is_disturbed,
    deter_magnitude,
    scaling,
    e_use_per_30_min,
    e_lact,
):
    """BMR + activity + disturbance cost.

    Matches DEPONSEnergyModule / depons_bmr_cost_kernel:
    - BMR: 0.001 * scaling * e_use_per_30_min (* e_lact if lactating)
    - Activity: speed * 0.0001 * scaling
    - Disturbance: 0.002 * deter_magnitude * scaling (if disturbed)

    Parameters
    ----------
    energy : float32[n] — current energy
    active_mask : bool[n] — active agent mask
    speed : float32[n] — current speed (m/s)
    is_lactating : bool[n] — lactating agents
    is_disturbed : bool[n] — disturbed agents
    deter_magnitude : float32[n] — deterrence magnitude
    scaling : float32 — seasonal scaling factor (scalar)
    e_use_per_30_min : float — BMR parameter
    e_lact : float — lactation multiplier

    Returns
    -------
    (new_energy, total_cost) — float32[n] each
    """
    bmr = 0.001 * scaling * e_use_per_30_min
    bmr = jnp.where(is_lactating, bmr * e_lact, bmr)

    activity = speed * 0.0001 * scaling

    disturbance = jnp.where(is_disturbed, 0.002 * deter_magnitude * scaling, 0.0)

    total_cost = bmr + activity + disturbance
    total_cost = jnp.where(active_mask, total_cost, 0.0)

    new_energy = jnp.where(active_mask, energy - total_cost, energy)
    return new_energy, total_cost


def jax_mortality(
    energy,
    active_mask,
    with_calf,
    age,
    key,
    m_mort_prob_const,
    x_survival_const,
    is_day_boundary,
    bycatch_prob,
    max_age,
):
    """Mortality check: starvation + bycatch + max-age.

    Starvation (every tick):
        yearly_surv = 1 - m_mort_prob_const * exp(-energy * x_survival_const)
        step_surv = yearly_surv ^ (1 / (360*48))
        dies = random > step_surv
        Two-step: if with_calf, abandon calf first; die only if not lactating
                  or energy<=0

    Bycatch + max-age (day boundary only):
        daily_surv = exp(log(1 - bycatch_prob) / 360)
        bycatch = random > daily_surv
        old_age = age > max_age

    Parameters
    ----------
    energy : float32[n]
    active_mask : bool[n]
    with_calf : bool[n]
    age : float32[n] — agent age (years)
    key : JAX PRNGKey
    m_mort_prob_const : float
    x_survival_const : float
    is_day_boundary : jnp.bool_ — dynamic boolean, use lax.cond
    bycatch_prob : float — annual bycatch probability
    max_age : float — maximum age in years

    Returns
    -------
    (new_active_mask, new_with_calf, key)
    """
    n = energy.shape[0]

    # --- Starvation (every tick) ---
    yearly_surv = jnp.where(
        energy > 0,
        1.0 - m_mort_prob_const * jnp.exp(-energy * x_survival_const),
        0.0,
    )
    recip_ticks = 1.0 / (360.0 * 48.0)
    step_surv = jnp.where(
        energy > 0,
        jnp.power(jnp.maximum(yearly_surv, 1e-10), recip_ticks),
        0.0,
    )

    key, subkey = jax.random.split(key)
    rand_starv = jax.random.uniform(subkey, (n,))
    starving = (rand_starv > step_surv) & active_mask

    # Two-step calf logic:
    # 1. If starving and with_calf -> abandon calf first
    was_with_calf = starving & with_calf
    new_with_calf = jnp.where(was_with_calf, False, with_calf)

    # 2. Die if: starving and (energy <= 0 OR was not with calf)
    starved = starving & ((energy <= 0) | ~was_with_calf)

    # --- Bycatch + max-age (day boundary only) ---
    def day_boundary_deaths(_):
        daily_surv = jnp.exp(jnp.log(jnp.maximum(1.0 - bycatch_prob, 1e-30)) / 360.0)
        k, sk = jax.random.split(key)
        rand_bc = jax.random.uniform(sk, (n,))
        bc = (rand_bc > daily_surv) & active_mask & (bycatch_prob > 0)
        oa = active_mask & (age > max_age)
        return bc | oa

    def no_day_boundary_deaths(_):
        return jnp.zeros(n, dtype=bool)

    day_deaths = lax.cond(is_day_boundary, day_boundary_deaths, no_day_boundary_deaths, None)

    # Apply deaths
    all_deaths = starved | day_deaths
    new_active_mask = jnp.where(all_deaths, False, active_mask)

    return new_active_mask, new_with_calf, key


def jax_energy_history_update(
    energy, active_mask, energy_ticks_today, energy_history, tick_counter, is_day_boundary
):
    """Accumulate energy; at day boundary shift history.

    Every tick: energy_ticks_today[mask] += energy[mask]; tick_counter += 1
    At day boundary (tick_counter >= 48):
        daily_avg = round(energy_ticks_today / 48, 3)  (DEPONS 3-d.p. daily mean)
        history[:, 1:] = history[:, :-1]  (shift right)
        history[:, 0] = daily_avg
        energy_ticks_today = 0; tick_counter = 0

    Uses lax.cond for the day-boundary branch.

    Parameters
    ----------
    energy : float32[n]
    active_mask : bool[n]
    energy_ticks_today : float32[n] — accumulator for current day
    energy_history : float32[n, hist_len] — daily energy history
    tick_counter : int32 — ticks accumulated this day
    is_day_boundary : jnp.bool_ — dynamic boolean

    Returns
    -------
    (energy_ticks_today, energy_history, tick_counter)
    """
    # Accumulate energy for current day
    energy_ticks_today = jnp.where(active_mask, energy_ticks_today + energy, energy_ticks_today)
    tick_counter = tick_counter + 1

    def do_shift(args):
        ett, eh, _tc = args
        # NumPy: np.round(f32 / 48.0, 3) in float32, i.e. a correctly rounded f32 quotient,
        # then f32 multiply by 1000, rint, and a correctly rounded f32 divide by 1000. XLA
        # turns float32 division by a constant into a reciprocal multiply (5.0010004, not
        # 5.001), so both divisions are done in float64 and rounded once to float32, which
        # gives the correctly rounded float32 result. The multiply and rint stay in float32.
        q = (ett.astype(jnp.float64) / 48.0).astype(jnp.float32)
        r = jnp.rint(q * jnp.float32(1000.0))
        daily_avg = (r.astype(jnp.float64) / 1000.0).astype(eh.dtype)
        # Shift right: history[:, 1:] = history[:, :-1]
        new_hist = jnp.concatenate([daily_avg[:, None], eh[:, :-1]], axis=1).astype(eh.dtype)
        new_ett = jnp.zeros_like(ett)
        new_tc = jnp.int32(0)
        return new_ett, new_hist, new_tc

    def no_shift(args):
        ett, eh, tc = args
        return ett, eh, tc

    energy_ticks_today, energy_history, tick_counter = lax.cond(
        is_day_boundary,
        do_shift,
        no_shift,
        (energy_ticks_today, energy_history, tick_counter),
    )

    return energy_ticks_today, energy_history, tick_counter


def jax_dispersal_update(
    is_dispersing,
    dispersal_distance_traveled,
    days_declining_energy,
    turbine_deter_strength,
    energy_history,
    active_mask,
    is_day_boundary,
):
    """Update dispersal: deterrence cancel, energy stop.

    Two checks in order:
    1. Deterrence cancels dispersal (any tick)
    2. Energy recovery stops dispersal (day boundary only)

    Path-length completion runs at the start of the next movement tick, in
    jax_disperse_step (DEPONS shouldStopDispersing inside disperse()).

    Parameters
    ----------
    is_dispersing : bool[n]
    dispersal_distance_traveled : float32[n]
    days_declining_energy : int32[n]
    turbine_deter_strength : float32[n]
    energy_history : float32[n, hist_len]
    active_mask : bool[n]
    is_day_boundary : jnp.bool_

    Returns
    -------
    (is_dispersing, dispersal_distance_traveled, days_declining_energy)
    """
    dispersing = active_mask & is_dispersing

    # 1. Deterrence cancels dispersal
    deterred = dispersing & (turbine_deter_strength > 0)
    is_dispersing = jnp.where(deterred, False, is_dispersing)
    dispersal_distance_traveled = jnp.where(deterred, 0.0, dispersal_distance_traveled)
    days_declining_energy = jnp.where(deterred, 0, days_declining_energy)

    dispersing = active_mask & is_dispersing

    # 2. Energy-based stop (day boundary only)
    def energy_stop(args):
        disp, ddt, dde, eh = args
        today = eh[:, 0]
        past_min = jnp.min(eh[:, 1:8], axis=1)
        recovering = (today > past_min) & disp
        new_is_dispersing = jnp.where(recovering, False, is_dispersing)
        new_ddt = jnp.where(recovering, 0.0, ddt)
        new_dde = jnp.where(recovering, 0, dde)
        return new_is_dispersing, new_ddt, new_dde

    def no_energy_stop(args):
        _disp, ddt, dde, _eh = args
        return is_dispersing, ddt, dde

    is_dispersing, dispersal_distance_traveled, days_declining_energy = lax.cond(
        is_day_boundary,
        energy_stop,
        no_energy_stop,
        (dispersing, dispersal_distance_traveled, days_declining_energy, energy_history),
    )

    return is_dispersing, dispersal_distance_traveled, days_declining_energy
