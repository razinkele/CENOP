"""
JAX-accelerated simulation kernels.

Each kernel is a pure function operating on JAX arrays. No side effects,
no mutation — returns new arrays. Composed into jax_tick_movement and
jax_tick_energy in tick_jax.py.

Requires: jax, jaxlib
Fallback: if JAX unavailable, these functions are never imported.
"""

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
    dispersal_target_x,
    dispersal_target_y,
    dispersal_target_distance,
    prev_step_heading,
    x,
    y,
    inertia_const,
    mean_disp_dist,
    dispersal_start_x,
    dispersal_start_y,
    psm_angle,
    psm_log,
    disp_key,
):
    """Heading composition: CRW + ref_mem + deterrence + social -> final heading/dx/dy.

    Combines all movement influences into a final heading and displacement vector.
    Dispersing agents get a PSM-Type2 heading override: a logistic-scaled
    uniform-random turn from previousStepHeading (NOT steered toward the target),
    matching the NumPy reference _apply_dispersal_heading / DispersalPSMType2.java.

    Parameters
    ----------
    heading : float64[n] — current heading (degrees)
    pres_angle : float64[n] — CRW turning angle from jax_crw_kernel
    log_mov : float64[n] — CRW log step length from jax_crw_kernel
    ve_total : float32[n] — expected food value from jax_compute_ve_total
    vt_x, vt_y : float32[n] — attraction vectors from jax_compute_attraction
    deter_dx, deter_dy : float64[n] — deterrence displacement vectors
    social_dx, social_dy : float64[n] — social cohesion vectors
    mask : bool[n] — active agent mask
    is_dispersing : bool[n] — which agents are in dispersal mode
    dispersal_target_x, dispersal_target_y : float64[n] — dispersal target positions
    dispersal_target_distance : float64[n] — total dispersal distance (cells)
    prev_step_heading : float64[n] — previous step heading for dispersing agents
    x, y : float32[n] — current positions (for dispersal distance calc)
    inertia_const : float scalar — CRW inertia constant
    mean_disp_dist : float scalar — mean dispersal distance per step
    dispersal_start_x, dispersal_start_y : float64[n] — dispersal start positions
        (distance TRAVELLED from here drives the logistic scaling)
    psm_angle : float scalar — half-range of the uniform random turn (degrees)
    psm_log : float scalar — SSLogis phi3 (logistic scale) for the turn magnitude
    disp_key : jax PRNGKey — RNG key for the uniform random dispersal turn

    Returns
    -------
    (new_heading, new_prev_angle, step_dist, dx, dy,
     new_prev_step_heading, deter_strength,
     dispersal_distance_delta) — tuple of float64[n] arrays
    """
    inertia_const = jnp.float64(inertia_const)
    mean_disp_dist = jnp.float64(mean_disp_dist)

    # 1. Apply CRW turning angle to heading
    new_heading = (heading + pres_angle) % 360.0

    # 2. Dispersal heading override (PSM-Type2 random walk)
    # Matches NumPy reference _apply_dispersal_heading (DispersalPSMType2.java):
    #   distPercent = distance TRAVELLED from dispersal_start / dispersal_target_distance
    #   distLogX    = 3 * distPercent - 1.5
    #   logistic    = SSLogis(distLogX, phi1=1, phi2=0, phi3=psm_log)
    #   angleDelta  = U(-psm_angle, +psm_angle) * logistic
    #   newHeading  = previousStepHeading + angleDelta
    dx_disp = x - dispersal_start_x
    dy_disp = y - dispersal_start_y
    dist_traveled = jnp.sqrt(dx_disp**2 + dy_disp**2)

    dist_percent = jnp.where(
        dispersal_target_distance > 0,
        dist_traveled / dispersal_target_distance,
        0.0,
    )
    dist_percent = jnp.clip(dist_percent, 0.0, 10.0)

    dist_log_x = jnp.clip(3.0 * dist_percent - 1.5, -100.0, 100.0)
    logistic = 1.0 / (1.0 + jnp.exp((0.0 - dist_log_x) / psm_log))

    rand_delta = jax.random.uniform(
        disp_key,
        shape=prev_step_heading.shape,
        dtype=prev_step_heading.dtype,
        minval=-psm_angle,
        maxval=psm_angle,
    )
    dispersal_heading = (prev_step_heading + rand_delta * logistic) % 360.0

    # Apply dispersal override for dispersing agents
    disp_mask = mask & is_dispersing
    new_heading = jnp.where(disp_mask, dispersal_heading, new_heading)

    # Save dispersal heading before CRW composition
    saved_disp_heading = new_heading  # will select back for dispersing agents

    # 3. CRW unit direction from heading
    rads = jnp.radians(new_heading)
    dx_unit = jnp.sin(rads)
    dy_unit = jnp.cos(rads)

    # 4. Heading composition
    step_pow = jnp.power(10.0, log_mov)
    crw_contrib = inertia_const + step_pow * ve_total

    total_dx = dx_unit * crw_contrib + vt_x + deter_dx + social_dx
    total_dy = dy_unit * crw_contrib + vt_y + deter_dy + social_dy

    # Deterrence strength (for tracking)
    deter_strength = jnp.hypot(deter_dx, deter_dy)
    deter_strength = jnp.where(mask, deter_strength, 0.0)

    # 5. New heading from composite vector
    composed_heading = jnp.degrees(jnp.arctan2(total_dx, total_dy)) % 360.0
    final_heading = jnp.where(mask, composed_heading, new_heading)

    # 6. Restore dispersal heading for dispersing agents (they skip CRW composition)
    final_heading = jnp.where(disp_mask, saved_disp_heading, final_heading)

    # 7. Prev angle: total turn relative to original heading
    original_heading = heading  # heading before pres_angle was applied
    new_prev_angle = (final_heading - original_heading + 180.0) % 360.0 - 180.0

    # 8. Step distance
    step_dist = step_pow / 4.0
    disp_step = mean_disp_dist / 0.4
    step_dist = jnp.where(disp_mask, disp_step, step_dist)

    # Dispersal distance traveled delta (only for dispersing agents)
    dispersal_distance_delta = jnp.where(disp_mask, disp_step, 0.0)

    # 9. Final dx/dy from final heading and step distance
    final_rads = jnp.radians(final_heading)
    dx = jnp.sin(final_rads) * step_dist
    dy = jnp.cos(final_rads) * step_dist

    # Zero out inactive agents
    dx = jnp.where(mask, dx, 0.0)
    dy = jnp.where(mask, dy, 0.0)
    step_dist = jnp.where(mask, step_dist, 0.0)
    new_prev_angle = jnp.where(mask, new_prev_angle, 0.0)

    # Update prev_step_heading for dispersing agents
    new_prev_step_heading = jnp.where(disp_mask, final_heading, prev_step_heading)

    return (
        final_heading,
        new_prev_angle,
        step_dist,
        dx,
        dy,
        new_prev_step_heading,
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


def jax_update_positions(x, y, dx, dy, heading, world_w, world_h, mask):
    """Compute new positions, reflect at boundaries, recalculate heading for reflected.

    Combines position addition with boundary reflection and heading correction.

    Parameters
    ----------
    x, y : float32[n] — current positions
    dx, dy : float64[n] — displacement vectors
    heading : float64[n] — current heading (degrees)
    world_w, world_h : int — world dimensions in cells
    mask : bool[n] — active agent mask

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

    return final_x, final_y, final_heading


# ---------------------------------------------------------------------------
# Food Intake + Energy + Mortality Kernels (J5)
# ---------------------------------------------------------------------------


def jax_eat_food(food_grid, xi, yi, energy, min_food):
    """Two-pass proportional food sharing.

    Pass 1: fraction = clip((20 - energy) / 10, 0, 0.99)
            demand = food_at_cell * fraction
            accumulate total demand per cell (scatter-add)
    Pass 2: if total_demand <= available: eat full demand
            else: eat proportional share (demand / total_demand * available)

    Parameters
    ----------
    food_grid : float32[rows, cols]
    xi, yi : int32[n] — cell indices (x=col, y=row)
    energy : float32[n] — current energy
    min_food : float — minimum food floor (0.01)

    Returns
    -------
    (food_eaten, new_food_grid) — float32[n], float32[rows, cols]
    """
    frac = jnp.clip((20.0 - energy) / 10.0, 0.0, 0.99)
    cell_food = food_grid[yi, xi]
    agent_demand = cell_food * frac

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

    # Update food grid
    eaten_total_grid = jnp.zeros_like(food_grid)
    eaten_total_grid = eaten_total_grid.at[yi, xi].add(eaten)
    new_food = jnp.maximum(food_grid - eaten_total_grid, min_food)

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
        daily_avg = energy_ticks_today / 48
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
        daily_avg = (ett / 48.0).astype(eh.dtype)
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
    dispersal_start_x,
    dispersal_start_y,
    dispersal_target_distance,
    dispersal_distance_traveled,
    days_declining_energy,
    x,
    y,
    turbine_deter_strength,
    energy_history,
    active_mask,
    is_day_boundary,
):
    """Update dispersal: deterrence cancel, energy stop, distance check.

    Three checks in order:
    1. Deterrence cancels dispersal (any tick)
    2. Energy recovery stops dispersal (day boundary only)
    3. Distance completion check (any tick)

    Parameters
    ----------
    is_dispersing : bool[n]
    dispersal_start_x, dispersal_start_y : float32[n]
    dispersal_target_distance : float32[n]
    dispersal_distance_traveled : float32[n]
    days_declining_energy : int32[n]
    x, y : float32[n] — current position
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
        new_disp = jnp.where(recovering, False, disp)  # stop is_dispersing for disp mask
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

    dispersing = active_mask & is_dispersing

    # 3. Distance completion check
    dx = x - dispersal_start_x
    dy = y - dispersal_start_y
    distances = jnp.sqrt(dx * dx + dy * dy)
    completed = dispersing & (distances >= 0.95 * dispersal_target_distance)
    is_dispersing = jnp.where(completed, False, is_dispersing)
    dispersal_distance_traveled = jnp.where(completed, 0.0, dispersal_distance_traveled)
    days_declining_energy = jnp.where(completed, 0, days_declining_energy)

    return is_dispersing, dispersal_distance_traveled, days_declining_energy
