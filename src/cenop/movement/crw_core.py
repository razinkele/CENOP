"""Shared DEPONS CRW turning-angle + step-length generation and heading composition.

Extracted verbatim from PorpoisePopulation._update_movement (NumPy fallback branch) so
the inline path and the injected movement module produce identical RNG draws and results.
Java ref: Porpoise.java move() (332-397 rejection loops, 556-589 composition).
"""

from __future__ import annotations

import numpy as np


def generate_crw_angle_step(
    rng,
    prev_angle,
    prev_log_mov,
    depths,
    salinity,
    mask,
    params,
    pres_angle,
    log_mov,
    env_mod_angle,
    rand_angle,
    rand_len,
):
    """Fill pres_angle (f64) and log_mov (f64). Does NOT update prev_log_mov."""
    count = pres_angle.shape[0]

    # --- Turning angle: angleTmp = b0*prevAngle + R2; presAngle = angleTmp*(b1*d+b2*s+b3)
    np.copyto(rand_angle, rng.normal(params.r2_mean, params.r2_sd, count))
    np.multiply(params.corr_angle_base, prev_angle, out=pres_angle)
    pres_angle += rand_angle
    np.multiply(params.corr_angle_bathy, depths, out=env_mod_angle)
    env_mod_angle += params.corr_angle_salinity * salinity
    env_mod_angle += params.corr_angle_base_sd
    pres_angle *= env_mod_angle

    # Reject-and-redraw (Porpoise.java:332-360)
    violations = np.abs(pres_angle) > 180
    retry = 0
    while (violations & mask).any() and retry < 200:
        idx = np.where(violations & mask)[0]
        new_rand = rng.normal(params.r2_mean, params.r2_sd, len(idx))
        angle_tmp = params.corr_angle_base * prev_angle[idx] + new_rand
        pres_angle[idx] = angle_tmp * (
            params.corr_angle_bathy * depths[idx]
            + params.corr_angle_salinity * salinity[idx]
            + params.corr_angle_base_sd
        )
        violations = np.abs(pres_angle) > 180
        retry += 1
    if (violations & mask).any():
        pres_angle[violations & mask] = np.sign(pres_angle[violations & mask]) * 90

    # Distance-dependent second angle loop (Porpoise.java:367-397)
    prev_mov = np.power(10.0, prev_log_mov)
    needs_modulation = mask & (prev_mov <= params.m)
    if needs_modulation.any():
        mod_idx = np.where(needs_modulation)[0]
        signs = np.sign(pres_angle[mod_idx])
        pres_angle[mod_idx] = np.abs(pres_angle[mod_idx])
        retry = 0
        violations2 = pres_angle[mod_idx] >= 180.0
        while violations2.any() and retry < 200:
            v_idx = mod_idx[violations2]
            rnd = rng.normal(0, 1, len(v_idx))
            pres_angle[v_idx] += rnd - rnd * prev_mov[v_idx] / params.m
            violations2 = pres_angle[mod_idx] >= 180.0
            retry += 1
        still_bad = pres_angle[mod_idx] >= 180.0
        if still_bad.any():
            fb_idx = mod_idx[still_bad]
            pres_angle[fb_idx] = rng.uniform(0, 20, len(fb_idx)) + 90
        pres_angle[mod_idx] *= signs

    # --- Step length: log10mov = a0*prev + a1*d + a2*s + R1 (Porpoise.java:367-391)
    np.copyto(rand_len, rng.normal(params.r1_mean, params.r1_sd, count))
    np.multiply(params.corr_logmov_length, prev_log_mov, out=log_mov)
    log_mov += params.corr_logmov_bathy * depths
    log_mov += params.corr_logmov_salinity * salinity
    log_mov += rand_len

    violations = log_mov > params.max_mov
    retry = 0
    while (violations & mask).any() and retry < 200:
        idx = np.where(violations & mask)[0]
        new_rand = rng.normal(params.r1_mean, params.r1_sd, len(idx))
        log_mov[idx] = (
            params.corr_logmov_length * prev_log_mov[idx]
            + params.corr_logmov_bathy * depths[idx]
            + params.corr_logmov_salinity * salinity[idx]
            + new_rand
        )
        violations = log_mov > params.max_mov
        retry += 1
    if (violations & mask).any():
        log_mov[violations & mask] = params.max_mov


def compose_movement(
    heading,
    pres_angle,
    log_mov,
    ve_total,
    vt_x,
    vt_y,
    d_dx,
    d_dy,
    is_dispersing,
    mask,
    inertia_const,
    disp_step,
    rads,
    dx,
    dy,
    step_dist,
):
    """DEPONS heading composition + displacement. Mutates heading, dx, dy, step_dist.

    Assumes heading already has pres_angle added and any dispersal override applied.
    `pres_angle` is accepted for call-site symmetry but is not read here.
    """
    _disp_mask = mask & is_dispersing
    _saved_disp_heading = heading[_disp_mask].copy() if _disp_mask.any() else None

    np.radians(heading, out=rads)
    np.sin(rads, out=dx)
    np.cos(rads, out=dy)

    np.power(10.0, log_mov, out=step_dist)
    crw_contrib = inertia_const + step_dist * ve_total

    total_dx = dx * crw_contrib + vt_x + d_dx
    total_dy = dy * crw_contrib + vt_y + d_dy

    new_heading = np.degrees(np.arctan2(total_dx, total_dy)) % 360
    heading[mask] = new_heading[mask]

    if _saved_disp_heading is not None:
        heading[_disp_mask] = _saved_disp_heading

    step_dist /= 4.0

    dispersing = mask & is_dispersing
    if dispersing.any():
        step_dist[dispersing] = disp_step

    np.radians(heading, out=rads)
    np.sin(rads, out=dx)
    dx *= step_dist
    np.cos(rads, out=dy)
    dy *= step_dist
