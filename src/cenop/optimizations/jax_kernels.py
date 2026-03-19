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

import jax.numpy as jnp
import jax.lax as lax


def _crw_single_agent(carry, _x):
    """Process one agent's CRW angle + step length via scan.

    carry: (idx, prev_angle, prev_log_mov, depths, salinity, mask, key,
            corr_angle_base, corr_angle_bathy, corr_angle_salinity,
            corr_angle_base_sd, corr_logmov_length, corr_logmov_bathy,
            corr_logmov_salinity, max_mov, r2_mean, r2_sd, r1_mean, r1_sd)

    Returns (carry_next, (out_angle, out_log_mov)) per agent.
    """
    (
        idx,
        prev_angle_arr,
        prev_log_mov_arr,
        depths_arr,
        salinity_arr,
        mask_arr,
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
    ) = carry

    active = mask_arr[idx]
    pa = prev_angle_arr[idx]
    plm = prev_log_mov_arr[idx]
    d = depths_arr[idx]
    s = salinity_arr[idx]

    key, subkey = jax.random.split(key)

    # Compute active path
    out_a, out_l, key = _crw_active(
        pa,
        plm,
        d,
        s,
        subkey,
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

    # Masked: angle=0, log_mov=prev
    out_angle = jnp.where(active, out_a, 0.0)
    out_log_mov = jnp.where(active, out_l, plm)

    new_carry = (
        idx + 1,
        prev_angle_arr,
        prev_log_mov_arr,
        depths_arr,
        salinity_arr,
        mask_arr,
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
    )
    return new_carry, (out_angle, out_log_mov)


def _crw_active(
    prev_angle,
    prev_log_mov,
    depth,
    salinity,
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
):
    """Compute CRW angle + step for a single active agent."""
    max_retries = 200

    # === 1. Angle rejection sampling ===
    env_mod = corr_angle_bathy * depth + corr_angle_salinity * salinity + corr_angle_base_sd

    key, subkey = jax.random.split(key)
    rand_angle = jax.random.normal(subkey, dtype=jnp.float64) * r2_sd + r2_mean
    angle_tmp = corr_angle_base * prev_angle + rand_angle
    pres_angle = angle_tmp * env_mod

    # while_loop for angle rejection: retry while |angle| > 180
    def angle_cond(state):
        angle, retry, _key = state
        return jnp.logical_and(jnp.abs(angle) > 180.0, retry < max_retries)

    def angle_body(state):
        angle, retry, k = state
        k, sk = jax.random.split(k)
        new_rand = jax.random.normal(sk, dtype=jnp.float64) * r2_sd + r2_mean
        new_tmp = corr_angle_base * prev_angle + new_rand
        new_angle = new_tmp * env_mod
        return (new_angle, retry + 1, k)

    pres_angle, _retry, key = lax.while_loop(
        angle_cond, angle_body, (pres_angle, jnp.int32(0), key)
    )

    # Emergency fallback: +/-90
    pres_angle = jnp.where(
        jnp.abs(pres_angle) > 180.0,
        jnp.where(pres_angle > 0.0, 90.0, -90.0),
        pres_angle,
    )

    # === 2. Distance-dependent modulation ===
    max_mov_value = 10.0**max_mov
    prev_mov_value = 10.0**prev_log_mov

    do_dist_mod = prev_mov_value <= max_mov_value

    def dist_mod_cond(state):
        angle, retry, _key = state
        return jnp.logical_and(jnp.abs(angle) >= 180.0, retry < max_retries)

    def dist_mod_body(state):
        angle, retry, k = state
        k, sk = jax.random.split(k)
        rnd = jax.random.uniform(sk, dtype=jnp.float64, minval=0.0, maxval=20.0)
        new_abs = jnp.abs(angle) + rnd - rnd * prev_mov_value / max_mov_value
        new_angle = jnp.where(angle >= 0.0, new_abs, -new_abs)
        return (new_angle, retry + 1, k)

    # Run distance-dependent loop
    angle_after_dist, _retry2, key_after_dist = lax.while_loop(
        dist_mod_cond, dist_mod_body, (pres_angle, jnp.int32(0), key)
    )

    # Distance-dependent fallback
    key_after_dist, fb_key = jax.random.split(key_after_dist)
    rnd_fb = jax.random.uniform(fb_key, dtype=jnp.float64, minval=0.0, maxval=20.0)
    angle_fallback = jnp.where(
        jnp.abs(angle_after_dist) >= 180.0,
        jnp.where(angle_after_dist >= 0.0, rnd_fb + 90.0, -(rnd_fb + 90.0)),
        angle_after_dist,
    )

    # Only apply dist mod if condition met
    pres_angle = jnp.where(do_dist_mod, angle_fallback, pres_angle)
    key = jnp.where(do_dist_mod, key_after_dist, key)

    # === 3. Step length rejection sampling ===
    key, subkey = jax.random.split(key)
    rand_len = jax.random.normal(subkey, dtype=jnp.float64) * r1_sd + r1_mean
    log_mov = (
        corr_logmov_length * prev_log_mov
        + corr_logmov_bathy * depth
        + corr_logmov_salinity * salinity
        + rand_len
    )

    def step_cond(state):
        lm, retry, _key = state
        return jnp.logical_and(lm > max_mov, retry < max_retries)

    def step_body(state):
        _lm, retry, k = state
        k, sk = jax.random.split(k)
        new_rand = jax.random.normal(sk, dtype=jnp.float64) * r1_sd + r1_mean
        new_lm = (
            corr_logmov_length * prev_log_mov
            + corr_logmov_bathy * depth
            + corr_logmov_salinity * salinity
            + new_rand
        )
        return (new_lm, retry + 1, k)

    log_mov, _retry3, key = lax.while_loop(step_cond, step_body, (log_mov, jnp.int32(0), key))

    # Fallback: clamp to max_mov
    log_mov = jnp.where(log_mov > max_mov, max_mov, log_mov)

    return pres_angle, log_mov, key


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
):
    """
    JAX CRW angle + step length kernel with rejection sampling.

    Implements DEPONS 3.2 CRW algorithm using lax.while_loop for
    rejection sampling. All state in carry — no side effects,
    fully JIT-compilable.

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
    n = prev_angle.shape[0]

    init_carry = (
        jnp.int32(0),
        prev_angle,
        prev_log_mov,
        depths,
        salinity,
        mask,
        key,
        jnp.float64(corr_angle_base),
        jnp.float64(corr_angle_bathy),
        jnp.float64(corr_angle_salinity),
        jnp.float64(corr_angle_base_sd),
        jnp.float64(corr_logmov_length),
        jnp.float64(corr_logmov_bathy),
        jnp.float64(corr_logmov_salinity),
        jnp.float64(max_mov),
        jnp.float64(r2_mean),
        jnp.float64(r2_sd),
        jnp.float64(r1_mean),
        jnp.float64(r1_sd),
    )

    _final_carry, (out_angles, out_log_movs) = lax.scan(
        _crw_single_agent, init_carry, None, length=n
    )

    return out_angles, out_log_movs
