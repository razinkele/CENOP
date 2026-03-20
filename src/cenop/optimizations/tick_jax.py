"""
Composed JAX tick functions for the full simulation step.

Assembles individual kernels from jax_kernels.py into two JIT-compiled
functions: jax_tick_movement (phases 1-3) and jax_tick_energy (phases 4-8).

Both functions are pure — all state is passed in as flat arrays, all
parameters as scalars, and new arrays are returned. No side effects.
"""

from functools import partial

import jax
import jax.numpy as jnp

from cenop.optimizations.jax_kernels import (
    jax_crw_kernel,
    jax_compute_ve_total,
    jax_compute_attraction,
    jax_heading_composition,
    jax_update_positions,
    jax_land_avoidance,
    jax_eat_food,
    jax_bmr_cost,
    jax_mortality,
    jax_energy_history_update,
    jax_dispersal_update,
)


def is_jax_available() -> bool:
    """Return True if JAX is installed and has a working backend."""
    try:
        x = jnp.ones(1)
        _ = float(x[0])
        return True
    except Exception:
        return False


@jax.jit
def jax_tick_movement(
    # Agent state arrays
    x,
    y,
    heading,
    prev_angle,
    prev_log_mov,
    active_mask,
    # Reference memory state
    stored_util,
    pos_hist_x,
    pos_hist_y,
    mem_ptr,
    mem_count,
    work_mem_table,
    # Deterrence / social vectors (pre-computed in Python)
    deter_dx,
    deter_dy,
    social_dx,
    social_dy,
    # Dispersal state
    is_dispersing,
    dispersal_target_x,
    dispersal_target_y,
    dispersal_target_distance,
    dispersal_distance_traveled,
    prev_step_heading,
    # Landscape grids
    depth_grid,
    salinity_grid,
    # Scalar month index (0-11) for salinity lookup
    month_idx,
    # Scalar parameters
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
    inertia_const,
    mean_disp_dist,
    min_depth,
    world_w,
    world_h,
    # RNG key
    key,
):
    """Full movement tick: CRW + ref mem + heading + position + land avoidance.

    Returns
    -------
    Tuple of:
        new_x, new_y, new_heading, new_prev_angle, new_prev_log_mov,
        step_dist, deter_strength, dispersal_distance_delta,
        new_prev_step_heading, key
    """
    # Compute per-agent depth and salinity from grids
    xi = jnp.clip(x.astype(jnp.int32), 0, world_w - 1)
    yi = jnp.clip(y.astype(jnp.int32), 0, world_h - 1)
    depths = depth_grid[yi, xi].astype(jnp.float64)
    salinity = salinity_grid[month_idx, yi, xi].astype(jnp.float64)

    # --- Phase 1: CRW angle + step ---
    key, crw_key = jax.random.split(key)
    pres_angle, log_mov = jax_crw_kernel(
        prev_angle,
        prev_log_mov,
        depths,
        salinity,
        active_mask,
        crw_key,
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
    )

    # --- Phase 2: Reference memory ---
    ve_total = jax_compute_ve_total(
        stored_util, mem_ptr, mem_count, work_mem_table, active_mask
    )
    vt_x, vt_y = jax_compute_attraction(
        stored_util,
        pos_hist_x,
        pos_hist_y,
        mem_ptr,
        mem_count,
        x.astype(jnp.float32),
        y.astype(jnp.float32),
        work_mem_table,
        active_mask,
    )
    # Only use attraction if agent has >= 2 memory entries
    has_history = mem_count >= 2
    vt_x = jnp.where(has_history, vt_x, 0.0)
    vt_y = jnp.where(has_history, vt_y, 0.0)

    # --- Phase 3: Heading composition + position update ---
    (
        new_heading,
        new_prev_angle,
        step_dist,
        dx,
        dy,
        new_prev_step_heading,
        deter_strength,
        dispersal_distance_delta,
    ) = jax_heading_composition(
        heading.astype(jnp.float64),
        pres_angle,
        log_mov,
        ve_total,
        vt_x,
        vt_y,
        deter_dx,
        deter_dy,
        social_dx,
        social_dy,
        active_mask,
        is_dispersing,
        dispersal_target_x.astype(jnp.float64),
        dispersal_target_y.astype(jnp.float64),
        dispersal_target_distance.astype(jnp.float64),
        prev_step_heading.astype(jnp.float64),
        x.astype(jnp.float64),
        y.astype(jnp.float64),
        inertia_const,
        mean_disp_dist,
    )

    # Position update with boundary reflection
    new_x, new_y, new_heading = jax_update_positions(
        x.astype(jnp.float64),
        y.astype(jnp.float64),
        dx,
        dy,
        new_heading,
        world_w,
        world_h,
        active_mask,
    )

    # Land avoidance
    new_x_f32 = new_x.astype(jnp.float32)
    new_y_f32 = new_y.astype(jnp.float32)
    new_heading_f32 = new_heading.astype(jnp.float32)
    step_dist_f32 = step_dist.astype(jnp.float32)

    # Check which agents are on land
    xi_check = jnp.clip(new_x_f32.astype(jnp.int32), 0, world_w - 1)
    yi_check = jnp.clip(new_y_f32.astype(jnp.int32), 0, world_h - 1)
    depth_at_new = depth_grid[yi_check, xi_check]
    on_land = active_mask & (depth_at_new < min_depth)

    key, la_key = jax.random.split(key)
    la_x, la_y, la_heading, la_resolved, _ = jax_land_avoidance(
        x.astype(jnp.float32),
        y.astype(jnp.float32),
        new_heading_f32,
        step_dist_f32,
        depth_grid,
        min_depth,
        on_land,
        la_key,
    )

    # Apply land avoidance results
    new_x_f32 = jnp.where(la_resolved, la_x, new_x_f32)
    new_y_f32 = jnp.where(la_resolved, la_y, new_y_f32)
    new_heading_f32 = jnp.where(la_resolved, la_heading, new_heading_f32)

    # Unresolved agents: stay in place, turn 180
    unresolved = on_land & ~la_resolved
    new_x_f32 = jnp.where(unresolved, x, new_x_f32)
    new_y_f32 = jnp.where(unresolved, y, new_y_f32)
    new_heading_f32 = jnp.where(
        unresolved, (new_heading_f32 + 180.0) % 360.0, new_heading_f32
    )

    # Update prev_log_mov for active agents
    new_prev_log_mov = jnp.where(active_mask, log_mov, prev_log_mov)

    return (
        new_x_f32,
        new_y_f32,
        new_heading_f32,
        new_prev_angle.astype(jnp.float64),
        new_prev_log_mov,
        step_dist_f32,
        deter_strength.astype(jnp.float32),
        dispersal_distance_delta.astype(jnp.float32),
        new_prev_step_heading.astype(jnp.float32),
        key,
    )


@jax.jit
def jax_tick_energy(
    # Agent state
    energy,
    active_mask,
    x,
    y,
    speed,
    is_lactating,
    is_disturbed,
    deter_magnitude,
    with_calf,
    age,
    # Food grid
    food_grid,
    xi,
    yi,
    # Energy history
    energy_ticks_today,
    energy_history,
    tick_counter,
    # Dispersal state
    is_dispersing,
    dispersal_start_x,
    dispersal_start_y,
    dispersal_target_distance,
    dispersal_distance_traveled,
    days_declining_energy,
    deter_strength,
    # Scalar parameters
    scaling,
    e_use_per_30_min,
    e_lact,
    min_food,
    m_mort_prob_const,
    x_survival_const,
    bycatch_prob,
    max_age,
    is_day_boundary,
    # RNG key
    key,
):
    """Full energy tick: food + mortality + BMR + history + dispersal.

    Returns
    -------
    Tuple of:
        new_energy, new_active_mask, new_with_calf, new_food_grid,
        food_eaten, total_cost,
        new_energy_ticks_today, new_energy_history, new_tick_counter,
        new_is_dispersing, new_dispersal_distance_traveled,
        new_days_declining_energy, key
    """
    # --- Phase 4: Food intake ---
    food_eaten, new_food_grid = jax_eat_food(
        food_grid, xi, yi, energy, min_food
    )
    # Only active agents eat
    food_eaten = jnp.where(active_mask, food_eaten, 0.0)
    energy_after_food = jnp.where(active_mask, energy + food_eaten, energy)

    # --- Phase 5: Mortality check (on post-food, pre-BMR energy) ---
    key, mort_key = jax.random.split(key)
    new_active_mask, new_with_calf, _ = jax_mortality(
        energy_after_food,
        active_mask,
        with_calf,
        age,
        mort_key,
        m_mort_prob_const,
        x_survival_const,
        is_day_boundary,
        bycatch_prob,
        max_age,
    )

    # --- Phase 6: BMR cost (only surviving agents) ---
    energy_after_bmr, total_cost = jax_bmr_cost(
        energy_after_food,
        new_active_mask,
        speed,
        is_lactating,
        is_disturbed,
        deter_magnitude,
        scaling,
        e_use_per_30_min,
        e_lact,
    )

    # --- Phase 7: Energy clamp ---
    new_energy = jnp.clip(energy_after_bmr, 0.0, 20.0)

    # --- Phase 8: Energy history update ---
    new_energy_ticks_today, new_energy_history, new_tick_counter = (
        jax_energy_history_update(
            new_energy,
            new_active_mask,
            energy_ticks_today,
            energy_history,
            tick_counter,
            is_day_boundary,
        )
    )

    # --- Phase 9: Dispersal update ---
    new_is_dispersing, new_dispersal_distance_traveled, new_days_declining_energy = (
        jax_dispersal_update(
            is_dispersing,
            dispersal_start_x,
            dispersal_start_y,
            dispersal_target_distance,
            dispersal_distance_traveled,
            days_declining_energy,
            x,
            y,
            deter_strength,
            new_energy_history,
            new_active_mask,
            is_day_boundary,
        )
    )

    return (
        new_energy,
        new_active_mask,
        new_with_calf,
        new_food_grid,
        food_eaten,
        total_cost,
        new_energy_ticks_today,
        new_energy_history,
        new_tick_counter,
        new_is_dispersing,
        new_dispersal_distance_traveled,
        new_days_declining_energy,
        key,
    )
