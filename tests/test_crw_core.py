"""Unit tests for the shared DEPONS CRW core (generation + composition)."""

import numpy as np

from cenop.parameters import SimulationParameters


def test_generate_crw_angle_step_matches_formula_no_rejection():
    """With defaults + depth/salinity=30 the angle/step never violate bounds, so the
    function must draw exactly rand_angle(count) then rand_len(count) and apply the
    plain DEPONS formulas (incl. the float32 env_mod accumulation quirk).

    Note: params.m defaults to 1e-05, and prev_mov = 10**0.8 ~= 6.31 > m, so the
    distance-dependent second angle loop does NOT fire (no extra RNG draws) — the
    reconstruction below is exact."""
    from cenop.movement.crw_core import generate_crw_angle_step

    params = SimulationParameters(porpoise_count=8)
    count = 8
    prev_angle = np.full(count, 10.0)
    prev_log_mov = np.full(count, 0.8)
    depths = np.full(count, 30.0)
    salinity = np.full(count, 30.0)
    mask = np.ones(count, dtype=bool)

    pres = np.zeros(count)
    logm = np.zeros(count)
    envm = np.zeros(count, dtype=np.float32)
    ra = np.zeros(count)
    rl = np.zeros(count)
    generate_crw_angle_step(
        np.random.default_rng(2024),
        prev_angle,
        prev_log_mov,
        depths,
        salinity,
        mask,
        params,
        pres,
        logm,
        envm,
        ra,
        rl,
    )

    rng2 = np.random.default_rng(2024)
    exp_ra = rng2.normal(params.r2_mean, params.r2_sd, count)
    em = np.zeros(count, dtype=np.float32)
    np.multiply(params.corr_angle_bathy, depths, out=em)
    em += params.corr_angle_salinity * salinity
    em += params.corr_angle_base_sd
    exp_pres = (params.corr_angle_base * prev_angle + exp_ra) * em
    assert np.all(np.abs(exp_pres) <= 180)  # regime has no rejection
    exp_rl = rng2.normal(params.r1_mean, params.r1_sd, count)
    exp_logm = (
        params.corr_logmov_length * prev_log_mov
        + params.corr_logmov_bathy * depths
        + params.corr_logmov_salinity * salinity
        + exp_rl
    )

    np.testing.assert_allclose(pres, exp_pres, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(logm, exp_logm, rtol=1e-5, atol=1e-5)


def test_compose_movement_simple_no_attraction():
    """ve_total=vt=deterrence=0 -> heading direction is preserved and the step magnitude
    equals 10**log_mov / 4 (DEPONS 400 m cell conversion)."""
    from cenop.movement.crw_core import compose_movement

    count = 5
    heading = np.array([0.0, 90.0, 180.0, 270.0, 45.0], dtype=np.float32)
    pres = np.zeros(count)
    logm = np.full(count, 0.9)
    ve = np.zeros(count, np.float32)
    vtx = np.zeros(count, np.float32)
    vty = np.zeros(count, np.float32)
    ddx = np.zeros(count)
    ddy = np.zeros(count)
    disp = np.zeros(count, bool)
    mask = np.ones(count, bool)
    rads = np.zeros(count, np.float32)
    dx = np.zeros(count, np.float32)
    dy = np.zeros(count, np.float32)
    step = np.zeros(count, np.float32)

    compose_movement(
        heading, pres, logm, ve, vtx, vty, ddx, ddy, disp, mask, 0.001, 5.0, rads, dx, dy, step
    )

    exp_step = (10.0**0.9) / 4.0
    np.testing.assert_allclose(step, exp_step, rtol=1e-4)
    np.testing.assert_allclose(dx, np.sin(np.radians(heading)) * exp_step, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(dy, np.cos(np.radians(heading)) * exp_step, rtol=1e-3, atol=1e-4)
