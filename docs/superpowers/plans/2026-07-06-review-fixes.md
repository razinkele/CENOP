# CENOP Review-Fixes Implementation Plan (2026-07-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 17 default-path defects surfaced by the 2026-07-06 deep review (restoring the shipped web app + committed baselines to DEPONS-3.2 validity and seed-reproducibility) and repair the JAX + Cython backends to parity. 29 tasks across 6 phases.

**Architecture:** Test-driven, one finding-cluster per task group. Each task writes a failing test that captures the defect (preferring a differential/parity test against the validated `movement_module=None` reference path), makes it pass with a minimal real change, and commits. Fixes are ordered so the CRITICAL shipped-path items land first, then HIGH reproducibility/correctness, then MEDIUM/LOW, then the two opt-in backends.

**Tech Stack:** Python 3 (micromamba env `shiny`), NumPy, Numba (`@njit`), JAX, Cython, Shiny for Python, pytest. Source of truth for parity: DEPONS 3.2 `parameters.xml` + `Porpoise.java` (vendored at `DEPONS-3.2/`).

**Source review report:** `docs/superpowers/specs/2026-07-06-deep-codebase-review.md` (the `#N` finding numbers referenced below map to that report).

## Global Constraints

- Run everything from `/home/razinka/cenjas/CENOP/`. Use `python3` (not `python`) via `micromamba run -n shiny ...`.
- Tests: `micromamba run -n shiny python3 -m pytest tests/ -q` (fast suite; slow multi-year tests auto-deselected). Slow tier: add `-m slow` (MANUAL — no CI). A single slow test needs `-m "slow or not slow"`.
- CENOP has its OWN nested git repo — commit from within `CENOP/`. **Setup step (do this ONCE before Task 1):** create and check out a feature branch off `CENOP-JASMINE` — `git -C /home/razinka/cenjas/CENOP switch -c review-fixes-2026-07-06 CENOP-JASMINE`. **Every task's commit lands on `review-fixes-2026-07-06`, NOT on `CENOP-JASMINE`** (never commit straight to `CENOP-JASMINE`/`main`/`master`; force-push to those is hook-blocked). Where a task's commit step says "branch CENOP-JASMINE", read it as this feature branch.
- Every git commit message must end with the trailer line `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` (some task examples below show it explicitly; apply it to all).
- Use `git -C /home/razinka/cenjas/CENOP <cmd>` for all git operations — never `cd <path> && git ...` (compound `cd && git` is hook-blocked in this repo).
- Line length 100; black + ruff auto-applied via hooks. Do NOT add type annotations or docstrings to unchanged code.
- Parity target is DEPONS **3.2** (`parameters.xml` runtime values), NOT stale Java field initializers.
- **Baseline regeneration:** the behaviour-changing fixes (turbine deterrence units, deterministic turbine deterrence, energy terms, double food regrow, movement parity, PSM distance) invalidate the committed Kattegat baselines. After such a fix lands, regenerate and re-run the slow validation tier (`-m slow`) before release — **once at the end of each phase, not per-task** (`--turbines` is value-taking and `--ships` is a `store_true` flag, and `--out` must be given explicitly or it defaults to the UNDISTURBED dir and overwrites it):
  - undisturbed: `micromamba run -n shiny python3 scripts/run_kattegat_reference.py --years 5 --seed 42 --out output/kattegat_ref`
  - ships: `micromamba run -n shiny python3 scripts/run_kattegat_reference.py --ships --years 2 --seed 42 --out output/kattegat_ref_ships`
  - turbines: `micromamba run -n shiny python3 scripts/run_kattegat_reference.py --turbines Kattegat-test --years 2 --seed 42 --out output/kattegat_ref_turbines`

## Decisions required before implementing (2 tasks)

Two Phase-3 tasks fix behaviour that diverges from DEPONS but **may be intended JASMINE research extensions**. They are authored to restore DEPONS purity by default while gating the behaviour behind a flag so JASMINE can opt in — but **confirm intent before implementing**:
- **Non-DEPONS energy terms** (finding #10, Task 13): swimming (`E_USE_PER_KM`) + disturbance energy costs in `DEPONSEnergyModule`. Bug (remove) or JASMINE extension (gate)?
- **Probabilistic turbine deterrence** (finding #12, Task 14): logistic strength scaling on the turbine path. Bug (make deterministic for DEPONS) or JASMINE extension (gate)?

Each of those tasks opens with a `> DECISION REQUIRED:` callout. (Task 2's group also carries a DECISION REQUIRED callout because bringing the web movement module to parity changes production movement output.)

## Out of scope (deferred to a later pass)

Per the agreed scope (default-path findings + JAX/Cython parity), these six verified findings are **deliberately excluded** — all are real but gated behind a non-default mode or an opt-in API, so they do not affect shipped behaviour:
- **#9** JASMINE energy mode crashes on tick 1 (`energy_budget.py:490`) — non-default energy mode. Fix if/when JASMINE energy mode ships.
- **#17** JASMINE movement freezes step length (`jasmine_physics.py:326`) — non-default movement mode.
- **#20** `batch_runner._run_parallel` fallback `NameError` (`batch_runner.py:257`) — opt-in parallel batch API (`parallel` defaults False; no in-repo caller).
- **#23** `batch_runner._run_sequential` `param_str` unbound (`batch_runner.py:224`) — needs `progress=False` + a `progress_callback`, which never co-occur on exercised paths.
- **#24** `batch_runner` seeds shorter than replicates -> `IndexError` (`batch_runner.py:212`) — opt-in public API robustness.
- **#27** FSM `recovery_ticks` gate inert unless JASMINE memory active (`hybrid_fsm.py:177`) — only bites under a JASMINE-energy + DEPONS-memory combination.

---


## Phase 1 — CRITICAL: restore shipped-path model validity

The two defects that make the interactive web app run a different model than the one validated. Do these first; task ordering matters — the turbine-units fix feeds the deterrence vector the movement-parity task consumes.


### Task 1: Fix vectorized turbine deterrence vector to use grid displacement (not metres)

**Files:**
- Modify: `src/cenop/agents/turbine.py` (lines 481-484, inside `TurbineManager.calculate_aggregate_deterrence_vectorized`)
- Test: `tests/test_deterrence.py` (replace weak method at lines 50-75; add two parity/magnitude tests in class `TestTurbineDeterrenceVector`)

**Interfaces:**
- Consumes: `SimulationParameters.deter_coeff` (float, 0.012), `cell_size` (float, 400.0), the in-function locals `dx_m`/`dy_m` (metre displacement, `np.ndarray`), `full_mask` (`np.ndarray[bool]`), `s` (`np.ndarray`, per-porpoise strength). Scalar oracle `TurbineManager.calculate_aggregate_deterrence(porpoise_x, porpoise_y, params, cell_size) -> (max_strength, total_dx, total_dy)` (grid-unit vector via `sound.calculate_deterrence_vector`).
- Produces: corrected `TurbineManager.calculate_aggregate_deterrence_vectorized(porpoise_x, porpoise_y, params, cell_size) -> (total_dx, total_dy)` now emitting GRID-unit vectors. No signature change; return shape/dtype unchanged.

- [ ] **Step 1: Write the failing parity + magnitude tests.** Replace the existing weak `test_no_normalization_vectorized_turbine` method (tests/test_deterrence.py lines 50-75) with two tests: (a) a parity test asserting the vectorized turbine vector equals the scalar DEPONS oracle for several porpoises, and (b) an explicit grid-unit magnitude test. Both must fail on today's metre-based code. NOTE: the parity test MUST lower `deter_threshold` so the test porpoises are actually in-range — at the default 152 dB (vs impact 200 dB) a porpoise deters only within ~4 cells, so the 5-10 cell porpoises would all yield zero strength, degenerating the parity assertions into a vacuous `0 == 0` that passes on the buggy AND fixed code. A non-vacuity guard is added to make that failure mode impossible to reintroduce.

```python
    def test_vectorized_turbine_matches_scalar_oracle(self):
        """Vectorized turbine deterrence vector must equal the scalar DEPONS path.

        Regression for the metre-vs-grid displacement bug: the vectorized path
        built the vector from METRE displacement (dx_m = grid_disp * cell_size),
        making the emitted vector ~cell_size (400x) too large versus the scalar
        calculate_deterrence_vector path, which uses GRID units
        (Porpoise.java:1290-1292). dist_m (metres) stays correct for TL/range.
        """
        from cenop.agents.turbine import Turbine, TurbineManager, TurbinePhase
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        # Scalar oracle applies no probabilistic scaling -> disable for parity.
        params.deter_probabilistic = False
        # Lower the threshold so the assorted porpoises below are actually
        # in-range. At the default 152 dB (vs impact 200 dB) deterrence only
        # reaches ~4 cells, so the 5-10 cell porpoises here would ALL yield
        # zero strength -> the parity comparison would degenerate to a vacuous
        # 0 == 0 that passes on the buggy code too. At threshold 100 dB every
        # porpoise deters (strength ~45.9-50.9), so the assertions are live.
        params.deter_threshold = 100.0

        t = Turbine(id=0, x=50.0, y=50.0, impact=200.0, phase=TurbinePhase.CONSTRUCTION)
        t._is_active = True
        mgr = TurbineManager([t])
        mgr.phase = TurbinePhase.CONSTRUCTION

        # Assorted grid displacements, incl. a 3-4-5 diagonal (porpoise index 2).
        px = np.array([55.0, 50.0, 53.0, 60.0, 45.0])
        py = np.array([50.0, 56.0, 54.0, 50.0, 42.0])

        vec_dx, vec_dy = mgr.calculate_aggregate_deterrence_vectorized(
            px, py, params, cell_size=400.0
        )

        saw_nonzero = False
        for i in range(len(px)):
            _, exp_dx, exp_dy = mgr.calculate_aggregate_deterrence(
                float(px[i]), float(py[i]), params, cell_size=400.0
            )
            if exp_dx != 0.0 or exp_dy != 0.0:
                saw_nonzero = True
            assert vec_dx[i] == pytest.approx(exp_dx, rel=1e-9, abs=1e-12), (
                f"porpoise {i}: vectorized dx {vec_dx[i]} != scalar oracle {exp_dx}"
            )
            assert vec_dy[i] == pytest.approx(exp_dy, rel=1e-9, abs=1e-12), (
                f"porpoise {i}: vectorized dy {vec_dy[i]} != scalar oracle {exp_dy}"
            )
        assert saw_nonzero, (
            "parity test is vacuous: no porpoise deterred (all-zero comparison) — "
            "raise impact / lower threshold / move porpoises in-range"
        )

    def test_vectorized_turbine_vector_is_grid_units(self):
        """The vector magnitude must be built from GRID displacement, not metres.

        Porpoise 5 cells east of the turbine: dx must be strength*5*coeff
        (grid), NOT strength*2000*coeff (metres). The buggy code emitted the
        latter (~400x too large).
        """
        from cenop.agents.turbine import Turbine, TurbineManager, TurbinePhase
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        params.deter_probabilistic = False
        params.deter_coeff = 1.0
        params.deter_threshold = 0.0

        t = Turbine(id=0, x=50.0, y=50.0, impact=200.0, phase=TurbinePhase.CONSTRUCTION)
        t._is_active = True
        mgr = TurbineManager([t])
        mgr.phase = TurbinePhase.CONSTRUCTION

        px = np.array([55.0])  # 5 grid cells east
        py = np.array([50.0])
        vec_dx, vec_dy = mgr.calculate_aggregate_deterrence_vectorized(
            px, py, params, cell_size=400.0
        )

        # Strength = RL - threshold; RL = 200 - (beta*log10(2000) + alpha*2000)
        dist_m = 5.0 * 400.0
        tl = params.beta_hat * np.log10(dist_m) + params.alpha_hat * dist_m
        strength = (200.0 - tl) - params.deter_threshold
        expected_dx = strength * 5.0 * params.deter_coeff  # GRID units
        assert vec_dx[0] == pytest.approx(expected_dx, rel=1e-9), (
            f"grid-unit dx expected {expected_dx}, got {vec_dx[0]}"
        )
        # Guard against the metres value (which is cell_size=400x larger).
        assert vec_dx[0] < expected_dx * 2.0, "vector still in metres (400x too large)?"
        assert vec_dy[0] == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run the new tests — expect FAIL.** Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_deterrence.py::TestTurbineDeterrenceVector -q`. Expected: both new tests FAIL. Concrete reason: today's `calculate_aggregate_deterrence_vectorized` computes `vec_x = dx_m[full_mask] * s * deter_coeff` where `dx_m` is in metres (`grid_disp * cell_size`). In the parity test (threshold lowered to 100 so porpoise 0 deters with strength ~50.9) the vectorized dx for porpoise 0 is `strength*2000*coeff` while the scalar oracle returns `strength*5*coeff` — a factor of `cell_size`=400 mismatch, so `pytest.approx(exp_dx, rel=1e-9)` fails. `test_vectorized_turbine_vector_is_grid_units` fails its `vec_dx[0] < expected_dx*2` guard (buggy vec_x ≈ 301737 vs guard ≈ 1508). The two pre-existing scalar tests `test_no_normalization_scalar`/`test_no_normalization_diagonal` and the ship test `test_ship_vectorized_uses_unit_vector_times_magnitude` still pass (unchanged code paths).

- [ ] **Step 3: Apply the minimal fix in `src/cenop/agents/turbine.py`.** Divide the metre displacement by `cell_size` to recover grid displacement before scaling by strength and `deter_coeff`. Replace the block at lines 481-484:

```python
            s = s_final[full_mask]
            # DEPONS 3.2: raw displacement, NOT unit vector (Porpoise.java:1290-1292)
            vec_x = dx_m[full_mask] * s * deter_coeff
            vec_y = dy_m[full_mask] * s * deter_coeff
```

with:

```python
            s = s_final[full_mask]
            # DEPONS 3.2 (Porpoise.java:1290-1292): raw GRID displacement (cell
            # units), NOT metres. dx_m/dy_m are metres (needed above for TL/range),
            # so divide by cell_size to recover grid displacement — matching the
            # scalar calculate_deterrence_vector path (grid units, no *cell_size).
            grid_dx = dx_m[full_mask] / cell_size
            grid_dy = dy_m[full_mask] / cell_size
            vec_x = grid_dx * s * deter_coeff
            vec_y = grid_dy * s * deter_coeff
```

- [ ] **Step 4: Re-run tests — expect PASS.** Commands: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_deterrence.py tests/test_depons_deterrence.py -q`. Expected: all pass, including the two new tests (parity now compares live non-zero vectors; grid-units guard holds). The ship parity test `test_ship_vectorized_uses_unit_vector_times_magnitude` is unaffected (ships use a separate unit-vector×magnitude path). Then run the broader deterrence/movement surface to catch regressions: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/ -q -k "deter or turbine or movement or dispersal"`. Expected: all pass (0 failures).

- [ ] **Step 5: Commit (from within the nested CENOP repo).** Commands: `git -C /home/razinka/cenjas/CENOP add src/cenop/agents/turbine.py tests/test_deterrence.py` then `git -C /home/razinka/cenjas/CENOP commit -m "fix(turbine): build vectorized deterrence vector from grid displacement" -m "calculate_aggregate_deterrence_vectorized emitted the deterrence vector from metre displacement (dx_m = grid_disp * cell_size), making it ~cell_size (400x) too large vs DEPONS (Porpoise.java:1290-1292) and CENOP's own scalar calculate_deterrence_vector path (grid units). dist_m stays in metres for TL/range. Divide dx_m/dy_m by cell_size before scaling by strength*deter_coeff so vectorized == scalar == DEPONS. Replaced the weak dx>1.0 assertion with a scalar-oracle magnitude-parity test (threshold lowered so porpoises are in-range, plus a non-vacuity guard) and an explicit grid-unit magnitude test." -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`. Expected: one commit on branch CENOP-JASMINE.


> **Note (turbine-units):** BEHAVIOR-CHANGING correctness fix: turbine deterrence push vectors shrink ~400x (metres->grid units), matching DEPONS + CENOP's scalar path. Downstream impact to review after merge: (1) `deter_strength` (L2 magnitude of the combined deterrence vector) and the `is_disturbed` threshold (0.01) are derived from these vectors in population.py, so turbine-scenario `is_disturbed` counts and movement displacement drop substantially — regenerate the Kattegat TURBINE baseline (`output/kattegat_ref_turbines/`) at end of phase (see Global Constraints for the exact command). (2) `_turbine_deter_strength` is `np.hypot(turb_dx, turb_dy)` (population.py:2553) — the L2 magnitude of THIS turbine vector — so it ALSO shrinks ~400x. The dispersal-deactivation decision is nonetheless unchanged because the gate is a strict `> 0` test (population.py:3073, mirrored in jax_kernels.py:843), and 400x scaling preserves the sign/zero of every value. (3) The JAX path reads that same scalar `_turbine_deter_strength` and also gates on `> 0`, so no separate JAX vector fix is needed. Run the slow multi-year turbine tests (`-m slow`) before release since population trajectories under turbines may shift.


> DECISION REQUIRED: This section fixes Finding #1 by bringing the injected movement MODULE to DEPONS parity with the validated inline CRW path (the module path does NOT return None / fall back to the inline path). Because the Shiny web app ALWAYS injects a movement module (`simulation_controller.py:245`), the production movement path is `_update_movement_jasmine`, which today bypasses the validated inline CRW + reference-memory + SSLogis dispersal. After this fix those behaviours actually apply in production, so production movement output CHANGES and the committed Kattegat reference baselines WILL shift. Before treating any downstream diff as a regression, regenerate the baselines via `scripts/run_kattegat_reference.py` and review them. Confirm this approach (module-to-parity, deterrence folded into the heading composition, SSLogis dispersal override) before executing.

### Task 2: Extract validated CRW generation + composition into a shared crw_core module

Refactor the two load-bearing NumPy blocks of the validated inline movement path
(`PorpoisePopulation._update_movement`) into pure functions so the inline path AND the
injected movement module can call the *same* code — guaranteeing identical RNG draw order
and arithmetic (the only way to get bit-parity given the reject-and-redraw branches). This
task is behavior-preserving (same buffers, same dtypes, same operations, just relocated).

**Files:**
- Create: `src/cenop/movement/crw_core.py`
- Modify: `src/cenop/agents/population.py` (add a top-level import with the other `from cenop...` imports — see Step 4; replace inline NumPy generation body at lines 934-1019; replace inline NumPy composition body at lines 1064-1103)
- Test: `tests/test_crw_core.py` (new)

**Interfaces:**
- Produces: `generate_crw_angle_step(rng, prev_angle, prev_log_mov, depths, salinity, mask, params, pres_angle, log_mov, env_mod_angle, rand_angle, rand_len) -> None` (writes pres_angle f64, log_mov f64; DEPONS reject-and-redraw + distance-dependent second angle loop + step-length reject-and-redraw)
- Produces: `compose_movement(heading, pres_angle, log_mov, ve_total, vt_x, vt_y, d_dx, d_dy, is_dispersing, mask, inertia_const, disp_step, rads, dx, dy, step_dist) -> None` (mutates heading f32, dx f32, dy f32, step_dist f32; ref-mem-aware heading composition `totalD = dir*crwContrib + vt + deter`, `crwContrib = inertia + presMov*veTotal`. Note: `pres_angle` is passed for signature symmetry with the generation function but is not read — heading is assumed already turned by pres_angle before the call.)
- Consumes: none

- [ ] **Step 1: Write the failing tests** (`tests/test_crw_core.py`)
```python
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

    pres = np.zeros(count); logm = np.zeros(count)
    envm = np.zeros(count, dtype=np.float32)
    ra = np.zeros(count); rl = np.zeros(count)
    generate_crw_angle_step(np.random.default_rng(2024), prev_angle, prev_log_mov,
                            depths, salinity, mask, params, pres, logm, envm, ra, rl)

    rng2 = np.random.default_rng(2024)
    exp_ra = rng2.normal(params.r2_mean, params.r2_sd, count)
    em = np.zeros(count, dtype=np.float32)
    np.multiply(params.corr_angle_bathy, depths, out=em)
    em += params.corr_angle_salinity * salinity
    em += params.corr_angle_base_sd
    exp_pres = (params.corr_angle_base * prev_angle + exp_ra) * em
    assert np.all(np.abs(exp_pres) <= 180)  # regime has no rejection
    exp_rl = rng2.normal(params.r1_mean, params.r1_sd, count)
    exp_logm = (params.corr_logmov_length * prev_log_mov
                + params.corr_logmov_bathy * depths
                + params.corr_logmov_salinity * salinity + exp_rl)

    np.testing.assert_allclose(pres, exp_pres, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(logm, exp_logm, rtol=1e-5, atol=1e-5)

def test_compose_movement_simple_no_attraction():
    """ve_total=vt=deterrence=0 -> heading direction is preserved and the step magnitude
    equals 10**log_mov / 4 (DEPONS 400 m cell conversion)."""
    from cenop.movement.crw_core import compose_movement

    count = 5
    heading = np.array([0., 90., 180., 270., 45.], dtype=np.float32)
    pres = np.zeros(count)
    logm = np.full(count, 0.9)
    ve = np.zeros(count, np.float32); vtx = np.zeros(count, np.float32)
    vty = np.zeros(count, np.float32)
    ddx = np.zeros(count); ddy = np.zeros(count)
    disp = np.zeros(count, bool); mask = np.ones(count, bool)
    rads = np.zeros(count, np.float32); dx = np.zeros(count, np.float32)
    dy = np.zeros(count, np.float32); step = np.zeros(count, np.float32)

    compose_movement(heading, pres, logm, ve, vtx, vty, ddx, ddy, disp, mask,
                     0.001, 5.0, rads, dx, dy, step)

    exp_step = (10.0 ** 0.9) / 4.0
    np.testing.assert_allclose(step, exp_step, rtol=1e-4)
    np.testing.assert_allclose(dx, np.sin(np.radians(heading)) * exp_step,
                               rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(dy, np.cos(np.radians(heading)) * exp_step,
                               rtol=1e-3, atol=1e-4)
```

- [ ] **Step 2: Run — expect FAIL.**
`cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_crw_core.py -q`
Expected: both tests error/fail at runtime with `ModuleNotFoundError: No module named 'cenop.movement.crw_core'` (the imports are inside the test bodies, so this is a per-test error, not a pytest collection error). 0 passed.

- [ ] **Step 3: Create `src/cenop/movement/crw_core.py`** (verbatim extraction of the validated inline NumPy path; identical operations/dtypes so behavior is preserved)
```python
"""Shared DEPONS CRW turning-angle + step-length generation and heading composition.

Extracted verbatim from PorpoisePopulation._update_movement (NumPy fallback branch) so
the inline path and the injected movement module produce identical RNG draws and results.
Java ref: Porpoise.java move() (332-397 rejection loops, 556-589 composition).
"""
from __future__ import annotations

import numpy as np

def generate_crw_angle_step(rng, prev_angle, prev_log_mov, depths, salinity, mask, params,
                            pres_angle, log_mov, env_mod_angle, rand_angle, rand_len):
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

def compose_movement(heading, pres_angle, log_mov, ve_total, vt_x, vt_y, d_dx, d_dy,
                     is_dispersing, mask, inertia_const, disp_step,
                     rads, dx, dy, step_dist):
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
```

- [ ] **Step 4: Route the inline path through the shared functions** (`src/cenop/agents/population.py`).
Add a TOP-LEVEL import with the other `from cenop...` imports at the top of the module. There are NO existing top-level `from cenop.movement` imports (the current movement imports are all lazy inside methods), so add it directly after the existing top-level import line `from cenop.behavior.sound import calculate_received_level, response_probability_from_rl` (line 18). Verified: this does NOT introduce a circular import.
```python
from cenop.movement.crw_core import generate_crw_angle_step, compose_movement
```
Replace the inline NumPy generation `else:` body (current lines 934-1019, i.e. everything from `# --- Turning Angle (NumPy fallback) ---` through `self.prev_log_mov[mask] = self._log_mov[mask]`) with:
```python
        else:
            generate_crw_angle_step(
                self.rng, self.prev_angle, self.prev_log_mov,
                self._depths, self._salinity_vals, mask, self.params,
                self._pres_angle, self._log_mov, self._env_mod_angle,
                self._rand_angle, self._rand_len,
            )
            self.prev_log_mov[mask] = self._log_mov[mask]
```
Replace the inline NumPy composition `else:` body (current lines 1064-1103, from `# Save dispersal heading before CRW composition overwrites it` through the final `self._dy *= self._step_dist`) with:
```python
        else:
            compose_movement(
                self.heading, self._pres_angle, self._log_mov,
                self._ve_total, self._vt_x, self._vt_y, d_dx, d_dy,
                self.is_dispersing, mask, self.params.inertia_const, disp_step,
                self._rads, self._dx, self._dy, self._step_dist,
            )
```

- [ ] **Step 5: Run — expect PASS + no inline regression.**
`cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_crw_core.py tests/test_depons_movement.py tests/test_backend_equivalence.py -q`
Expected: all pass (crw_core formula tests green; existing DEPONS-movement + backend-equivalence determinism unchanged because the extraction is byte-identical to the prior inline code).

- [ ] **Step 6: Commit** (from within `CENOP/`)
`git -C /home/razinka/cenjas/CENOP add src/cenop/movement/crw_core.py src/cenop/agents/population.py tests/test_crw_core.py && git commit -m "refactor(movement): extract validated CRW generation + composition into crw_core"`

### Task 3: Bring DEPONSCRWMovement(Vectorized).compute_step to DEPONS parity

Rewrite the injected module's `compute_step` to use `crw_core` (reject-and-redraw + distance
second-loop instead of clip), to carry ref-memory attraction inputs through the state, to put
deterrence into the heading composition (not the displacement), and to expose the raw
`pres_angle`/`log_mov` so the population glue can reproduce the inline composition exactly. Remove
the `*0.3` dispersal shortcut from the module path (the SSLogis override is applied by the caller).

**Files:**
- Modify: `src/cenop/movement/base.py` (add optional fields to `MovementState` after the `dispersal_heading` field declaration at line 55, and to `MovementResult` after the `turning_angle` field declaration at line 127)
- Modify: `src/cenop/movement/depons_crw.py` (rewrite `DEPONSCRWMovement.compute_step` body, lines 121-232; delete the `DEPONSCRWMovementVectorized.compute_step` override, lines 293-384, so it inherits the parity implementation)
- Test: `tests/test_movement_modules.py` (add two tests to class `TestDEPONSCRWMovementVectorized`)

**Interfaces:**
- Consumes: `generate_crw_angle_step`, `compose_movement` (from Task: Extract validated CRW ...)
- Produces: `MovementState.ve_total/vt_x/vt_y` (Optional[np.ndarray]=None); `MovementResult.pres_angle/log_mov` (Optional[np.ndarray]=None, raw DEPONS turning angle f64 and log10-step f64)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_movement_modules.py`; the file already imports `DEPONSCRWMovementVectorized`, `MovementState`, `EnvironmentContext`, and `SimulationParameters` at the top)
```python
    def test_result_exposes_raw_pres_angle_and_log_mov(self):
        """Vectorized module must expose raw pres_angle/log_mov so the population can
        reproduce the inline heading composition; step_distance = 10**log_mov / 4."""
        params = SimulationParameters(porpoise_count=16)
        mod = DEPONSCRWMovementVectorized(params, rng=np.random.default_rng(7))
        state = MovementState.create(16, rng=np.random.default_rng(7))
        env = EnvironmentContext.create_homogeneous(16)
        x = np.full(16, 150.0, np.float32); y = np.full(16, 150.0, np.float32)
        mask = np.ones(16, dtype=bool)

        res = mod.compute_step(x, y, state, env, mask)

        assert res.pres_angle is not None
        assert res.log_mov is not None
        np.testing.assert_allclose(
            res.step_distance[mask], (np.power(10.0, res.log_mov) / 4.0)[mask], rtol=1e-4
        )

    def test_uses_reject_redraw_not_clip(self):
        """Large env_mod would make the raw angle exceed 180; the OLD code clipped to
        exactly +/-180. Reject-and-redraw must keep |angle| strictly below 180."""
        params = SimulationParameters(porpoise_count=200)
        mod = DEPONSCRWMovementVectorized(params, rng=np.random.default_rng(3))
        state = MovementState.create(200, rng=np.random.default_rng(3))
        state.prev_angle[:] = 150.0
        env = EnvironmentContext.create_homogeneous(200, depth=30.0, salinity=200.0)
        x = np.full(200, 150.0, np.float32); y = np.full(200, 150.0, np.float32)
        mask = np.ones(200, dtype=bool)

        res = mod.compute_step(x, y, state, env, mask)

        assert not np.any(np.abs(res.turning_angle) == 180.0)
        assert np.all(np.abs(res.turning_angle) <= 180.0)
```

- [ ] **Step 2: Run — expect FAIL.**
`cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest "tests/test_movement_modules.py::TestDEPONSCRWMovementVectorized" -q`
Expected: `test_result_exposes_raw_pres_angle_and_log_mov` fails with `AttributeError: 'MovementResult' object has no attribute 'pres_angle'`; `test_uses_reject_redraw_not_clip` fails on `assert not np.any(np.abs(res.turning_angle) == 180.0)` (current code clips saturated angles to exactly 180 via `np.clip(angle_tmp*env_mod, -180, 180)`; env_mod = -0.008*30 + 0.93*200 - 14.0 ~= 171.8 and angle_tmp ~= -3.6, so many products land far outside +/-180 and clip to exactly +/-180).

- [ ] **Step 3a: Add optional fields to the dataclasses** (`src/cenop/movement/base.py`).
First widen the import at line 16 from `from dataclasses import dataclass` to `from dataclasses import dataclass, field`. Then, in `MovementState`, immediately after `dispersal_heading: np.ndarray  # Target heading for dispersal` (line 55), add the three fields below. **They MUST use `field(default=None, kw_only=True)`, not a plain `= None`.** `MovementState` is subclassed by `JASMINEMovementState` (`jasmine_physics.py:40`), which adds NON-default fields (`vx, vy, vz, z, ax, ay`); a plain default here would put the subclass's non-default fields after a defaulted parent field and raise `TypeError: non-default argument 'vx' follows default argument` at *import time* (the `cenop.movement` package imports `jasmine_physics` unconditionally, so every `from cenop.movement import ...` — including this task's own tests — would error at collection). `kw_only=True` (Python 3.10+; this repo runs 3.13) keeps them out of positional ordering, so `jasmine_physics.py` needs no change. `Optional` is already imported at line 18:
```python
    # Reference-memory attraction inputs (populated by the population glue; None -> zeros)
    ve_total: Optional[np.ndarray] = field(default=None, kw_only=True)
    vt_x: Optional[np.ndarray] = field(default=None, kw_only=True)
    vt_y: Optional[np.ndarray] = field(default=None, kw_only=True)
```
In `MovementResult`, immediately after the field declaration `turning_angle: np.ndarray  # Turning angle applied` (line 127):
```python
    # Raw DEPONS draws (before ref-mem/deterrence composition) for parity with inline path
    pres_angle: Optional[np.ndarray] = None
    log_mov: Optional[np.ndarray] = None
```

- [ ] **Step 3b: Rewrite `DEPONSCRWMovement.compute_step`** (`src/cenop/movement/depons_crw.py`, replace the entire method body currently at lines 121-232):
```python
    def compute_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        state: MovementState,
        environment: EnvironmentContext,
        mask: np.ndarray,
        deterrence_dx: Optional[np.ndarray] = None,
        deterrence_dy: Optional[np.ndarray] = None,
    ) -> MovementResult:
        """DEPONS CRW step via the shared validated core.

        Generates the turning angle (reject-and-redraw + distance second loop) and step
        length, then composes the heading with reference-memory attraction and deterrence
        (deterrence enters the heading vector, NOT the raw displacement). Dispersal heading
        override is the caller's responsibility. Exposes raw pres_angle/log_mov for parity.
        """
        from cenop.movement.crw_core import generate_crw_angle_step, compose_movement

        count = len(x)
        depths = np.asarray(environment.depth, dtype=np.float64)
        salinity = np.asarray(environment.salinity, dtype=np.float64)
        prev_angle = np.asarray(state.prev_angle, dtype=np.float64)
        prev_log_mov = np.asarray(state.prev_log_mov, dtype=np.float64)

        pres_angle = np.zeros(count, dtype=np.float64)
        log_mov = np.zeros(count, dtype=np.float64)
        env_mod = np.zeros(count, dtype=np.float32)
        rand_angle = np.zeros(count, dtype=np.float64)
        rand_len = np.zeros(count, dtype=np.float64)

        generate_crw_angle_step(
            self.rng, prev_angle, prev_log_mov, depths, salinity, mask, self.params,
            pres_angle, log_mov, env_mod, rand_angle, rand_len,
        )

        # Turn heading (dispersal override handled by the caller)
        heading = np.asarray(state.heading, dtype=np.float32).copy()
        heading[mask] = (heading[mask] + pres_angle[mask]) % 360.0

        ve_total = (state.ve_total if state.ve_total is not None
                    else np.zeros(count, dtype=np.float32))
        vt_x = state.vt_x if state.vt_x is not None else np.zeros(count, dtype=np.float32)
        vt_y = state.vt_y if state.vt_y is not None else np.zeros(count, dtype=np.float32)

        d_dx = (np.asarray(deterrence_dx, dtype=np.float64) if deterrence_dx is not None
                else np.zeros(count, dtype=np.float64))
        d_dy = (np.asarray(deterrence_dy, dtype=np.float64) if deterrence_dy is not None
                else np.zeros(count, dtype=np.float64))

        rads = np.zeros(count, dtype=np.float32)
        dx = np.zeros(count, dtype=np.float32)
        dy = np.zeros(count, dtype=np.float32)
        step_dist = np.zeros(count, dtype=np.float32)

        disp_step = getattr(self.params, 'mean_disp_dist', 1.6) / 0.4
        compose_movement(
            heading, pres_angle, log_mov, ve_total, vt_x, vt_y, d_dx, d_dy,
            state.is_dispersing, mask, self.params.inertia_const, disp_step,
            rads, dx, dy, step_dist,
        )

        inactive = ~mask
        dx[inactive] = 0.0
        dy[inactive] = 0.0
        step_dist[inactive] = 0.0

        turning_angle = np.zeros(count, dtype=np.float32)
        turning_angle[mask] = pres_angle[mask].astype(np.float32)

        new_heading = np.asarray(state.heading, dtype=np.float32).copy()
        new_heading[mask] = heading[mask]

        return MovementResult(
            dx=dx, dy=dy, new_heading=new_heading,
            step_distance=step_dist, turning_angle=turning_angle,
            pres_angle=pres_angle, log_mov=log_mov,
        )
```
Then DELETE the `DEPONSCRWMovementVectorized.compute_step` override (current lines 293-384) so `DEPONSCRWMovementVectorized` inherits this parity implementation. Leave `apply_dispersal_modulation`, `get_mode`, `get_name`, `__init__`, and `_ensure_work_arrays` intact (`apply_dispersal_modulation` is still required by the abstract base and stays defined even though the parity `compute_step` no longer calls it — the caller applies the SSLogis dispersal override).

- [ ] **Step 4: Run — expect PASS.**
`cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_movement_modules.py -q`
Expected: all tests in the file pass (new parity assertions green; existing `test_deterrence_affects_movement`, `test_movement_produces_displacement`, `test_masked_agents_not_updated`, `test_heading_wraps_correctly`, reproducibility tests still pass — deterrence in the heading vector dominates the tiny `inertia_const` crw_contrib, so it still increases mean dx; magnitude comes from step_dist).

- [ ] **Step 5: Commit**
`git -C /home/razinka/cenjas/CENOP add src/cenop/movement/base.py src/cenop/movement/depons_crw.py tests/test_movement_modules.py && git commit -m "fix(movement): DEPONS-faithful module compute_step (reject-redraw, ref-mem, deterrence-in-heading)"`

### Task 4: Rewrite _update_movement_jasmine to match the inline reference exactly

Make the production (module-injected) path a faithful mirror of the validated inline path:
sample environment at cached cells (with Kattegat salinity override), update reference memory
(previously skipped entirely), drive the module for the CRW draws off the *population* RNG,
apply the SSLogis dispersal override, fold deterrence + social into the heading composition, and
write back `prev_angle` as the DEPONS total turn (not the raw module angle). Add an end-to-end
differential test proving the module path equals the `movement_module=None` path for a fixed seed.

**Files:**
- Modify: `src/cenop/agents/population.py` (replace `_update_movement_jasmine`, current lines 1166-1248)
- Test: `tests/test_movement_parity.py` (new)

**Interfaces:**
- Consumes: `MovementResult.pres_angle/log_mov`, `MovementState.ve_total/vt_x/vt_y` (from the module parity task); `compose_movement` (top-level import added in the crw_core task); the corrected deterrence vector (from the turbine-units deterrence task) — this path folds `deterrence_vectors` into the heading composition, so it must receive the corrected vector.
- Produces: none

- [ ] **Step 1: Write the failing differential tests** (`tests/test_movement_parity.py`)
```python
"""End-to-end parity: the injected-module movement path must equal the validated inline
(movement_module=None) path for a fixed seed. Forces the NumPy CRW branch so both paths use
the same crw_core generation.

Precondition for parity: no memory_module is injected (Simulation is built without one), so
self._avoidance_result stays None in both sims and the module path's memory-avoidance folding
is inert — matching the inline path, which applies no memory avoidance to movement."""
import numpy as np
import pytest

import cenop.agents.population as popmod
from cenop.core.simulation import Simulation
from cenop.parameters import SimulationParameters
from cenop.movement import DEPONSCRWMovementVectorized

def _build(seed, with_module):
    params = SimulationParameters(porpoise_count=40, landscape="Homogeneous",
                                  sim_years=1, random_seed=seed)
    mod = DEPONSCRWMovementVectorized(params) if with_module else None
    return Simulation(params, movement_module=mod)

def _assert_parity(pa, pb, idx=slice(None)):
    np.testing.assert_allclose(pb.heading[idx], pa.heading[idx], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(pb.x[idx], pa.x[idx], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(pb.y[idx], pa.y[idx], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(pb.prev_angle[idx], pa.prev_angle[idx], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(pb.prev_log_mov[idx], pa.prev_log_mov[idx],
                               rtol=1e-4, atol=1e-3)

def test_module_path_matches_inline_reference_one_tick(monkeypatch):
    monkeypatch.setattr(popmod, "_HAS_KERNELS", False)
    a = _build(2024, with_module=False)
    b = _build(2024, with_module=True)
    a.step(); b.step()
    pa, pb = a.population_manager, b.population_manager

    _assert_parity(pa, pb)
    # Reference memory MUST be updated by the module path (it was skipped before the fix)
    np.testing.assert_array_equal(pb._mem_ptr, pa._mem_ptr)
    np.testing.assert_array_equal(pb._mem_count, pa._mem_count)
    np.testing.assert_allclose(pb._ve_total, pa._ve_total, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pb._vt_x, pa._vt_x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(pb._vt_y, pa._vt_y, rtol=1e-4, atol=1e-4)
    assert np.any(pa._mem_count > 0)  # ref-mem genuinely populated (test not vacuous)

def test_module_path_matches_inline_reference_multi_tick(monkeypatch):
    monkeypatch.setattr(popmod, "_HAS_KERNELS", False)
    a = _build(101, with_module=False)
    b = _build(101, with_module=True)
    for _ in range(8):
        a.step(); b.step()
    _assert_parity(a.population_manager, b.population_manager)

def test_module_path_matches_inline_reference_dispersal(monkeypatch):
    """Exercises the SSLogis dispersal override (module previously used a *0.3 shortcut
    and skipped _apply_dispersal_heading)."""
    monkeypatch.setattr(popmod, "_HAS_KERNELS", False)
    a = _build(77, with_module=False)
    b = _build(77, with_module=True)
    for sim in (a, b):
        pm = sim.population_manager
        pm.is_dispersing[:5] = True
        pm.dispersal_target_distance[:5] = 20.0
        pm.dispersal_start_x[:5] = pm.x[:5]
        pm.dispersal_start_y[:5] = pm.y[:5]
        pm._prev_step_heading[:5] = 45.0
    a.step(); b.step()
    _assert_parity(a.population_manager, b.population_manager, idx=slice(0, 5))
```

- [ ] **Step 2: Run — expect FAIL.**
`cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_movement_parity.py -q`
Expected FAIL: on today's code the pre-fix `_update_movement_jasmine` draws from the module's own (unsynced) RNG and never calls `_update_reference_memory`, so `test_..._one_tick` fails FIRST on the `heading`/`x`/`y` `np.testing.assert_allclose` in `_assert_parity` (called before the `_mem_ptr` check) — the module path uses an independently-seeded RNG and omits ref-mem attraction, uses `*0.3` dispersal / deterrence-in-displacement, and sets `prev_angle` from the raw module angle instead of the total turn — with the `assert_array_equal(pb._mem_ptr, pa._mem_ptr)` divergence (module stays 0 while inline advances to 1 — verified: a Homogeneous landscape is non-None and `_mem_count` reaches 1 after one inline step) as a secondary failure. The multi-tick and dispersal tests fail on the same `heading`/`x`/`y` divergence.

- [ ] **Step 3: Rewrite `_update_movement_jasmine`** (`src/cenop/agents/population.py`, replace the entire method currently at lines 1166-1248):
```python
    def _update_movement_jasmine(
        self,
        mask: np.ndarray,
        deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]],
        ambient_rl: Optional[np.ndarray],
    ) -> None:
        """JASMINE movement path — parity mirror of the inline NumPy CRW path.

        The injected module supplies only the DEPONS turning-angle + step-length draws
        (via crw_core, off the population RNG); everything else (environment sampling,
        reference-memory update, SSLogis dispersal override, deterrence/social heading
        composition, prev_angle/prev_log_mov bookkeeping) is done here identically to
        _update_movement, so results match movement_module=None for a fixed seed.
        """
        from cenop.movement.base import EnvironmentContext

        # 1. Environment at current cells (identical to inline, incl. Kattegat override)
        if self.landscape is not None:
            np.copyto(self._depths, self.landscape.get_depths_vectorized(
                None, xi=self._cell_xi, yi=self._cell_yi))
            np.copyto(self._salinity_vals, self.landscape.get_salinities_vectorized(
                None, xi=self._cell_xi, yi=self._cell_yi))
            if getattr(self.landscape, 'landscape_name', '') == 'Kattegat':
                self._salinity_vals[:] = 34.069105813295
        else:
            self._depths.fill(30.0)
            self._salinity_vals.fill(30.0)

        # 2. Reference memory FIRST: computes veTotal/vt from the pre-store buffer and stores
        #    the current (not-yet-moved) position. It is RNG-free and reads neither heading
        #    nor the generation outputs, so running it before generation gives identical
        #    ve_total/vt/stored-position AND leaves the RNG stream order unchanged vs inline.
        self._update_reference_memory(mask)

        # 3. Drive the module off the population RNG with full-precision f64 inputs
        self._movement_module.rng = self.rng
        state = self._movement_state
        state.heading = self.heading
        state.prev_angle = self.prev_angle          # f64 reference (no f32 rounding)
        state.prev_log_mov = self.prev_log_mov       # f64 reference
        state.is_dispersing = self.is_dispersing
        state.ve_total = self._ve_total
        state.vt_x = self._vt_x
        state.vt_y = self._vt_y
        env = EnvironmentContext(depth=self._depths, salinity=self._salinity_vals)

        # 4. Module produces the CRW draws (angle + step). Deterrence is NOT passed here —
        #    it enters via the heading composition below (matches inline).
        result = self._movement_module.compute_step(self.x, self.y, state, env, mask)
        np.copyto(self._pres_angle, result.pres_angle)
        np.copyto(self._log_mov, result.log_mov)
        self.prev_log_mov[mask] = self._log_mov[mask]

        # 5. Turn heading, then SSLogis dispersal override (identical order to inline)
        np.copyto(self._pre_heading, self.heading)
        self.heading[mask] += self._pres_angle[mask]
        self.heading[mask] %= 360.0
        self._apply_dispersal_heading(mask)

        # 6. Deterrence (+ memory avoidance + social) folded into the heading composition.
        #    NOTE: the inline reference path does NOT apply memory avoidance to movement;
        #    _avoidance_result is None unless a memory_module is injected, so this block is
        #    inert (and thus mirrors inline) whenever no memory module is present — which is
        #    the case in the parity test. It is retained here to preserve the module path's
        #    memory-avoidance feature in production runs that DO inject a memory module.
        if deterrence_vectors is not None:
            d_dx, d_dy = deterrence_vectors
            self.deter_strength[mask] = np.hypot(d_dx[mask], d_dy[mask])
            self._was_deterred |= (self.deter_strength > 0) & mask
        else:
            d_dx = self._zero_f64
            d_dy = self._zero_f64
            self.deter_strength[mask] = 0.0

        if self._avoidance_result is not None:
            av = self._avoidance_result
            d_dx = d_dx + av.avoidance_dx * av.avoidance_strength
            d_dy = d_dy + av.avoidance_dy * av.avoidance_strength

        if self._comm_enabled:
            soc_dx, soc_dy = self._compute_social_vectors(mask, ambient_rl)
            d_dx = d_dx + soc_dx
            d_dy = d_dy + soc_dy

        disp_step = getattr(self.params, 'mean_disp_dist', 1.6) / 0.4

        # 7. Heading composition + displacement (shared validated NumPy path)
        compose_movement(
            self.heading, self._pres_angle, self._log_mov,
            self._ve_total, self._vt_x, self._vt_y, d_dx, d_dy,
            self.is_dispersing, mask, self.params.inertia_const, disp_step,
            self._rads, self._dx, self._dy, self._step_dist,
        )

        # 8. Dispersal distance, prev_angle (total turn), prev_log_mov (identical to inline)
        dispersing = mask & self.is_dispersing
        if dispersing.any():
            self.dispersal_distance_traveled[dispersing] += self._step_dist[dispersing]
        total_turn = (self.heading - self._pre_heading + 180) % 360 - 180
        self.prev_angle[mask] = total_turn[mask]
        self.prev_log_mov[mask] = self._log_mov[mask]
```

- [ ] **Step 4: Run — expect PASS (parity + no movement-module regressions).**
`cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_movement_parity.py tests/test_movement_modules.py tests/test_depons_movement.py tests/test_backend_equivalence.py -q`
Expected: all pass — the module path now matches the inline reference on heading/x/y/prev_angle/prev_log_mov and on reference-memory state (`_mem_ptr`, `_mem_count`, `_ve_total`, `_vt_x`, `_vt_y`), including the dispersing subset via SSLogis.

- [ ] **Step 5: Commit**
`git -C /home/razinka/cenjas/CENOP add src/cenop/agents/population.py tests/test_movement_parity.py && git commit -m "fix(movement): parity of injected-module path with validated inline CRW (ref-mem, dispersal, deterrence)"`


## Phase 2 — HIGH: default-path correctness & reproducibility

Bugs on the validated headless path and/or the interactive app that bias every multi-year run or break seed-reproducibility.


### Task 5: Fix double daily food regrowth in Simulation.step()

**Files:**
- Modify: `src/cenop/core/simulation.py` (lines 561-572 — remove the inline duplicate `replenish_food` block inside `step()`; keep the single call in `_daily_tasks()` at lines 620-626)
- Test: `tests/test_simulation.py` (add new `TestDailyFoodReplenishment` class near the existing `TestSimulation` class)

**Interfaces:**
- Consumes: `cenop.core.simulation.Simulation.step()`, `Simulation._daily_tasks()`, `Simulation._cell_data` (a `CellData` with `.replenish_food(rate, max_u, regrowth_qualifier)`), `TimeManager.is_day_boundary()` (day boundary at `tick % 48 == 0` and `tick > 0`).
- Produces: no signature changes. Behavioral contract: `cell_data.replenish_food` is invoked exactly once per simulated day boundary from `step()` (via `_daily_tasks()` only), on both scalar and vectorized paths.

TDD steps:

- [ ] **Step 1: Write the failing regression test.** Add to `tests/test_simulation.py` (matches this file's plain-pytest / lazy-import style; no fixtures needed). It spies on the real `replenish_food` with `wraps=` (preserves real regrowth) and steps a full day (48 ticks = exactly one day boundary at tick 48). Construction pattern mirrors the existing `test_simulation_step` (proven to build a vectorized homogeneous sim where both `_cell_data` and `population_manager` are non-None, so both call sites fire today):

```python
class TestDailyFoodReplenishment:
    """Regression: food regrows exactly once per simulated day (DEPONS parity)."""

    def test_replenish_food_called_once_per_day_boundary(self):
        """step() must invoke cell_data.replenish_food exactly once per day.

        Guards against the double-regrowth bug where step() called
        replenish_food both inside _daily_tasks() AND again inline at the
        day boundary, regrowing food twice per simulated day on the
        vectorized path.
        """
        from unittest.mock import patch
        from cenop import Simulation, SimulationParameters
        from cenop.landscape import create_homogeneous_landscape

        params = SimulationParameters(
            porpoise_count=10,
            sim_years=1,
            landscape="Homogeneous",
        )
        landscape = create_homogeneous_landscape()
        sim = Simulation(params, landscape)

        # One day == 48 ticks; is_day_boundary() is (tick > 0 and tick % 48 == 0),
        # so exactly one day boundary occurs across 48 steps (at tick 48).
        with patch.object(
            sim._cell_data,
            "replenish_food",
            wraps=sim._cell_data.replenish_food,
        ) as spy:
            for _ in range(48):
                sim.step()

        assert sim.state.tick == 48
        assert spy.call_count == 1, (
            f"food should regrow once per day, got {spy.call_count} calls"
        )
```

- [ ] **Step 2: Run it, expect FAIL.** Command:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_simulation.py::TestDailyFoodReplenishment -q`
  Expected: 1 failed — `AssertionError: food should regrow once per day, got 2 calls` (today `step()` calls `replenish_food` twice at the day boundary: once inside `_daily_tasks()` and once in the inline block; both `_cell_data is not None` and `population_manager is not None` hold because construction auto-initializes when `cell_data` is passed).

- [ ] **Step 3: Minimal implementation — delete the inline duplicate block in `step()`.** In `src/cenop/core/simulation.py`, replace the current block (lines 561-572):

```python
        # 9. Daily tasks (at day boundary)
        if self.time_manager.is_day_boundary():
            self._daily_tasks()
            # Replenish food for vectorized path (_daily_tasks only does this
            # for scalar _porpoises which is empty in vectorized mode)
            if self._cell_data is not None and self.population_manager is not None:
                # Daily food replenishment (DEPONS 3.2 logistic regrowth)
                self._cell_data.replenish_food(
                    rate=self.params.food_growth_rate,
                    max_u=self.params.max_u,
                    regrowth_qualifier=self.params.regrowth_food_qualifier,
                )
```

with:

```python
        # 9. Daily tasks (at day boundary)
        # Food replenishment (DEPONS 3.2 logistic regrowth) runs inside
        # _daily_tasks(); it is gated only on `_cell_data is not None`, so it
        # covers BOTH the scalar and vectorized paths. Do NOT replenish again
        # here or food regrows twice per simulated day.
        if self.time_manager.is_day_boundary():
            self._daily_tasks()
```

  (The single surviving call is inside `_daily_tasks()` at lines 620-626, which runs `self._cell_data.replenish_food(rate=..., max_u=..., regrowth_qualifier=...)` whenever `_cell_data is not None`. Verified it executes on the vectorized path: `_daily_tasks()` is called unconditionally at the day boundary, and its trailing replenish block gates only on `_cell_data is not None` — the empty `_porpoises` loop above it is a no-op in vectorized mode but does not skip the replenish. Optionally clarify the comment at line 620 from "Replenish food across landscape (DEPONS 3.2 logistic regrowth)" to note it is the single daily-regrowth source for both paths; no code change there.)

- [ ] **Step 4: Run tests, expect PASS.** Commands:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_simulation.py::TestDailyFoodReplenishment -q`
  Expected: 1 passed (`spy.call_count == 1`).
  Then run the food-regrowth + broader simulation suites to confirm no regression:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_simulation.py tests/test_food_regrowth.py -q`
  Expected: all passed (these existing tests call `CellData.replenish_food(...)` directly, not through `step()`, so they are unaffected by removing the duplicate).

- [ ] **Step 5: Commit (from within the nested CENOP repo).**
  `git -C /home/razinka/cenjas/CENOP add src/cenop/core/simulation.py tests/test_simulation.py && git commit -m "fix(simulation): regrow food once per day (remove duplicate replenish_food)"`
  Message body should note: the day-boundary path called `replenish_food` twice (inline block + `_daily_tasks()`), doubling DEPONS 3.2 logistic regrowth per day; removed the inline block so `_daily_tasks()` is the single source on both scalar and vectorized paths. Ends with the required `Co-Authored-By:` trailer per repo convention.


> **Note (food-regrow):** BEHAVIOR CHANGE / baseline risk: this halves daily food regrowth on the production vectorized path (2x/day -> 1x/day), which is the DEPONS-correct behavior (food regrows once per daily task in DEPONS). Effect is toward parity, but it materially reduces food availability and will shift downstream energy/population trajectories. ACTIONS FOR ASSEMBLER: (1) run the MANUAL slow tier after this fix -- `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/ -m slow -q` -- because population-stability assertions in tests/test_validation.py and tests/test_depons_physiology.py could shift and may need tolerance/expectation updates; (2) the Kattegat reference baselines (output/kattegat_ref*/) will need REGENERATION via scripts/run_kattegat_reference.py since per-day food levels change. Fast tier is unaffected: existing replenish_food tests (test_food_regrowth.py, test_reproduction.py) call CellData.replenish_food directly rather than through step(), so they neither guard nor break on this. New test is fast (not @slow): 48 vectorized steps on a 10-porpoise Homogeneous landscape (first step incurs Numba JIT warmup, still sub-second-to-few-seconds, comparable to existing test_simulation_step). Only one replenish_food call site remains after the fix (grep-confirmed: simulation.py had exactly two -- the removed inline block and the surviving _daily_tasks call).


### Task 6: Reset recycled slot state for weaned calves (Finding #4)

**Files:**
- Modify: `src/cenop/agents/population.py` — weaning block `_update_pregnancy_status` (insert one call after line 2155, `self.mating_day[new_slots] = -99`); add new helper method `_reset_recycled_slots` between `_update_pregnancy_status` (ends line 2165) and `_step_jax` (starts line 2167).
- Test: `tests/test_reproduction.py` — add new class `TestWeanedCalfSlotReset` (append at end of file).

**Interfaces:**
- Consumes: `PorpoisePopulation._update_pregnancy_status(mask: np.ndarray)` (existing); the SoA per-agent arrays declared in `__init__` (`prev_log_mov`, `prev_angle`, `_prev_step_heading`, `_stored_util`, `_pos_history_x/_y`, `_mem_ptr`, `_mem_count`, `_ve_total`, `_vt_x/_vt_y`, `psm_buffer`, `_energy_history`, `_energy_ticks_today`, `_energy_consumed_today`, `energy_consumed_daily`, `_energy_level_sum`, `is_dispersing`, `days_declining_energy`, `dispersal_target_x/_y`, `dispersal_target_distance`, `dispersal_distance_traveled`, `dispersal_start_x/_y`, `deter_strength`, `_turbine_deter_strength`, `_was_deterred`, `_prev_x`, `_prev_y`, `x`, `y`).
- Produces: new method `PorpoisePopulation._reset_recycled_slots(self, slots: np.ndarray) -> None`.

Rationale (verified in source): `_check_mortality` (def at population.py:1938; sets `active_mask[all_deaths] = False` at population.py:2009) does NOT clear any per-agent state. The weaning block (population.py:2142-2155) reuses `inactive_slots` for new calves but resets only 12 fields (`active_mask, x, y, heading, age, is_female, energy, pregnancy_status, with_calf, days_since_mating, days_since_birth, mating_day`). Every other persistent per-agent array (reference memory, PSM grid, CRW headings `prev_log_mov`/`prev_angle`, dispersal target/flags, energy history, deterrence, `_prev_x`/`_prev_y`) is inherited from the dead occupant. Newborn defaults confirmed at population.py:124-125 (`prev_log_mov=0.8`, `prev_angle=10.0`), :202 (`_prev_step_heading=0`), :154-161/:174-175/:189-191/:194-201/:224/:145-150 (all zero/False). `_prev_x`/`_prev_y` are declared at population.py:278-279.

- [ ] **Step 1: Write the failing test.** Append this class to `tests/test_reproduction.py` (imports `numpy as np`, `PorpoisePopulation`, `SimulationParameters` already present at top of file, lines 1-7):

```python
class TestWeanedCalfSlotReset:
    """Finding #4: a calf recycled into a dead agent's slot must NOT inherit the
    dead occupant's memory / dispersal / CRW / deterrence / prev-position state.
    """

    class _DetRNG:
        """Deterministic RNG stub: forces calf creation (random > 0.5) and a
        fixed energy draw, so the weaning path is fully reproducible."""

        def random(self, n):
            return np.ones(n, dtype=np.float64)

        def normal(self, mean, sd, n):
            return np.full(n, 10.0, dtype=np.float64)

    def test_weaned_calf_does_not_inherit_dead_slot_state(self):
        params = SimulationParameters()
        pop = PorpoisePopulation(count=2, params=params)

        # Slot 0 = mother ready to wean a calf this day.
        pop.is_female[0] = True
        pop.active_mask[0] = True
        pop.with_calf[0] = True
        pop.pregnancy_status[0] = 2
        pop.days_since_birth[0] = params.nursing_time
        pop.mating_day[0] = -99  # not on a mating day -> skip conceive branch
        pop.age[0] = 6.0
        pop.x[0] = 40.0
        pop.y[0] = 55.0
        pop.heading[0] = 90.0

        # Slot 1 = a DEAD agent carrying distinctive persistent state.
        pop.active_mask[1] = False
        pop.is_dispersing[1] = True
        pop.dispersal_target_x[1] = 999.0
        pop.dispersal_target_y[1] = 888.0
        pop.dispersal_target_distance[1] = 123.0
        pop.dispersal_distance_traveled[1] = 45.0
        pop.dispersal_start_x[1] = 12.0
        pop.dispersal_start_y[1] = 34.0
        pop.days_declining_energy[1] = 7
        pop.prev_log_mov[1] = 5.5
        pop.prev_angle[1] = 123.0
        pop._prev_step_heading[1] = 77.0
        pop._stored_util[1, :] = 7.0
        pop._pos_history_x[1, :] = 3.0
        pop._pos_history_y[1, :] = 4.0
        pop._mem_count[1] = 50
        pop._mem_ptr[1] = 33
        pop._ve_total[1] = 9.0
        pop._vt_x[1] = 1.0
        pop._vt_y[1] = 2.0
        pop.psm_buffer[1, :, :, :] = 6.0
        pop._energy_history[1, :] = 8.0
        pop._energy_ticks_today[1] = 3.0
        pop._energy_consumed_today[1] = 5.0
        pop.energy_consumed_daily[1] = 6.0
        pop._energy_level_sum[1] = 7.0
        pop.deter_strength[1] = 0.9
        pop._turbine_deter_strength[1] = 0.8
        pop._was_deterred[1] = True
        pop._prev_x[1] = -1.0
        pop._prev_y[1] = -2.0

        # Deterministic weaning: force calf creation + fixed energy draw.
        pop.rng = self._DetRNG()
        pop._day_of_year = 0  # current_day = 0, != mating_day(-99): no conceive

        pop._update_pregnancy_status(pop.active_mask.copy())

        # The calf was recycled into slot 1.
        assert pop.active_mask[1], "calf should have been created into the dead slot"

        # Dispersal state must be newborn defaults, not inherited.
        assert pop.is_dispersing[1] == False
        assert pop.days_declining_energy[1] == 0
        assert pop.dispersal_target_x[1] == 0.0
        assert pop.dispersal_target_y[1] == 0.0
        assert pop.dispersal_target_distance[1] == 0.0
        assert pop.dispersal_distance_traveled[1] == 0.0
        assert pop.dispersal_start_x[1] == 0.0
        assert pop.dispersal_start_y[1] == 0.0

        # CRW movement state = newborn defaults.
        assert pop.prev_log_mov[1] == 0.8
        assert pop.prev_angle[1] == 10.0
        assert pop._prev_step_heading[1] == 0.0

        # Reference memory cleared.
        assert pop._mem_count[1] == 0
        assert pop._mem_ptr[1] == 0
        assert np.all(pop._stored_util[1] == 0.0)
        assert np.all(pop._pos_history_x[1] == 0.0)
        assert np.all(pop._pos_history_y[1] == 0.0)
        assert pop._ve_total[1] == 0.0
        assert pop._vt_x[1] == 0.0
        assert pop._vt_y[1] == 0.0

        # PSM grid + energy history cleared.
        assert np.all(pop.psm_buffer[1] == 0.0)
        assert np.all(pop._energy_history[1] == 0.0)
        assert pop._energy_ticks_today[1] == 0.0
        assert pop._energy_consumed_today[1] == 0.0
        assert pop.energy_consumed_daily[1] == 0.0
        assert pop._energy_level_sum[1] == 0.0

        # Deterrence status cleared.
        assert pop.deter_strength[1] == 0.0
        assert pop._turbine_deter_strength[1] == 0.0
        assert pop._was_deterred[1] == False

        # Prev-position anchored to the calf's (mother-copied) location.
        assert pop._prev_x[1] == pop.x[1]
        assert pop._prev_y[1] == pop.y[1]
        assert pop.x[1] == 40.0 and pop.y[1] == 55.0
```

Note (do NOT "clean up" the `== False` assertions): the repo's post-edit hook runs `ruff check --fix` with `select = ["E", ...]`, which flags E712 on `== False`. Its autofix is unsafe-gated, so `--fix` (no `--unsafe-fixes`) leaves these lines untouched — verified empirically. Keep `== False` as written: `pop.is_dispersing[1]` is a numpy scalar (`np.False_`), for which `== False` is truthy-correct, whereas an `is False` rewrite would spuriously fail. Do not manually convert these to `is False`.

- [ ] **Step 2: Run it, expect FAIL.** Command:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_reproduction.py::TestWeanedCalfSlotReset -q`
  Expected: FAIL at `assert pop.is_dispersing[1] == False` with `AssertionError: assert np.True_ == False` — on current code the recycled slot still holds the dead agent's `is_dispersing = True` (and downstream, `prev_log_mov == 5.5`, `_mem_count == 50`, etc.). The preceding `assert pop.active_mask[1]` passes (the calf IS created into the dead slot), so the failure isolates the state-inheritance defect. (Empirically confirmed against the current source.)

- [ ] **Step 3: Minimal implementation.** In `src/cenop/agents/population.py`, insert the reset call inside the weaning block, immediately after `self.mating_day[new_slots] = -99` (line 2155, 20-space indent, still inside `if slots_to_use > 0:`):

```python
                    self.mating_day[new_slots] = -99
                    self._reset_recycled_slots(new_slots)
```

Then add the helper method between the end of `_update_pregnancy_status` (after line 2165, `self.days_since_birth[lactating] += 1`) and the start of `_step_jax` (line 2167), at 4-space method indent:

```python
    def _reset_recycled_slots(self, slots: np.ndarray) -> None:
        """Reset all persistent per-agent state for recycled (reused) slots to
        newborn defaults.

        Dead slots keep the previous occupant's memory / dispersal / CRW /
        deterrence / prev-position state (`_check_mortality` only clears
        `active_mask`). When a weaned calf reuses such a slot it must start from
        clean newborn state, otherwise it inherits the dead porpoise's reference
        memory, dispersal target, CRW headings, etc. Identity / position /
        energy are set by the caller before this runs; here we clear the state
        the caller does NOT reset. `_prev_x`/`_prev_y` are anchored to the
        (already-set) calf position.
        """
        # CRW movement state (newborn defaults, see __init__ lines 124-125, 202)
        self.prev_log_mov[slots] = 0.8
        self.prev_angle[slots] = 10.0
        self._prev_step_heading[slots] = 0.0

        # Reference memory circular buffers (__init__ lines 154-161)
        self._stored_util[slots, :] = 0.0
        self._pos_history_x[slots, :] = 0.0
        self._pos_history_y[slots, :] = 0.0
        self._mem_ptr[slots] = 0
        self._mem_count[slots] = 0
        self._ve_total[slots] = 0.0
        self._vt_x[slots] = 0.0
        self._vt_y[slots] = 0.0

        # Persistent spatial memory grid (__init__ line 224)
        self.psm_buffer[slots, :, :, :] = 0.0

        # Energy history / daily accumulators (__init__ lines 174-175, 189-191)
        self._energy_history[slots, :] = 0.0
        self._energy_ticks_today[slots] = 0.0
        self._energy_consumed_today[slots] = 0.0
        self.energy_consumed_daily[slots] = 0.0
        self._energy_level_sum[slots] = 0.0

        # Dispersal state (__init__ lines 194-201)
        self.is_dispersing[slots] = False
        self.days_declining_energy[slots] = 0
        self.dispersal_target_x[slots] = 0.0
        self.dispersal_target_y[slots] = 0.0
        self.dispersal_target_distance[slots] = 0.0
        self.dispersal_distance_traveled[slots] = 0.0
        self.dispersal_start_x[slots] = 0.0
        self.dispersal_start_y[slots] = 0.0

        # Deterrence status (__init__ lines 145-150)
        self.deter_strength[slots] = 0.0
        self._turbine_deter_strength[slots] = 0.0
        self._was_deterred[slots] = False

        # Previous positions anchored to the calf's (already-set) location
        self._prev_x[slots] = self.x[slots]
        self._prev_y[slots] = self.y[slots]
```

- [ ] **Step 4: Run tests, expect PASS.** Commands:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_reproduction.py::TestWeanedCalfSlotReset -q`
  Expected: 1 passed. (Empirically confirmed.)
  Then run the reproduction + phase5 suites to confirm no regression in existing weaning/dispersal behavior:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_reproduction.py tests/test_phase5.py -q`
  Expected: all pass (existing `test_weaning_creates_female_calf` and phase-5 dispersal tests unaffected — the reset only touches freshly-recycled calf slots). (Empirically confirmed: 42 passed.)

- [ ] **Step 5: Commit.** From within the nested CENOP repo:
  `git -C /home/razinka/cenjas/CENOP add src/cenop/agents/population.py tests/test_reproduction.py && git commit -m "fix: reset recycled slot state for weaned calves (Finding #4)

Weaned calves reused dead agents' SoA slots but only 12 fields were reset,
so a calf inherited the dead occupant's reference memory, PSM grid, CRW
headings (prev_log_mov/prev_angle), dispersal target/flags, energy history,
deterrence status and prev positions. Add _reset_recycled_slots() to clear
all persistent per-agent state to newborn defaults; call it from the weaning
block after the calf's position/energy are set.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`


> **Note (calf-reset):** BEHAVIOR CHANGE — may require reference-baseline regeneration. This fixes a real simulation-behavior defect: weaned calves that recycle a dead agent's slot previously inherited stale memory/dispersal/CRW/deterrence state; now they start clean. In multi-year runs where deaths + weanings overlap (any Kattegat reference run), calf trajectories will differ, so the committed Kattegat reference baselines (output/kattegat_ref*, PorpoiseStatistics/Dispersal) may shift and should be re-validated/regenerated via scripts/run_kattegat_reference.py after this lands. Flag for a slow-tier run before merge: `micromamba run -n shiny python3 -m pytest tests/ -m slow -q`. SCOPE: fix covers the SoA per-agent arrays named in the finding plus deterrence. Deliberately OUT OF SCOPE (documented, not defects for this finding): (1) `_psm_instances[slot]` PersistentSpatialMemory objects — per population.py:2798-2799 they are kept ONLY for `preferred_distance` config, memory data lives in psm_buffer which IS reset; a recycled calf keeping the dead agent's preferred_distance is a benign config carry-over, not inherited memory. (2) Optional JASMINE module states `_behavior_state`/`_memory_state`/`_movement_state` (created only when the corresponding module is passed; `_energy_state.energy` already aliases self.energy) — production SoA path does not depend on them for the arrays in this finding; if a follow-up wants full parity, reset them guarded by `is not None`. No new dependencies; single-file src change plus one test class. Consider mirroring the reset into `_check_mortality` (zero-at-death) as an alternative/defense-in-depth in a later pass, but the weaning-site reset is sufficient and localized for this finding.


### Task 7: Thread seeded RNG and plumb psm_dist_mean/sd into PSM construction

Fixes finding #6 (non-reproducible `preferred_distance` — production builds every agent's PSM at `population.py:215` with no `rng`, so `PersistentSpatialMemory.__init__` falls back to `np.random.default_rng()` OS entropy) and finding #14 (`params.psm_dist_mean`/`psm_dist_sd` default 350/100 exist but are never read; PSM hardcodes mean=300). After the fix, same-seed runs produce identical dispersal `preferred_distance` sequences and the distribution centres on `params.psm_dist_mean` (350, DEPONS 3.2), not 300.

**Files:**
- Modify: `src/cenop/behavior/psm.py` (`PersistentSpatialMemory.__init__` lines 59-93; `generate_preferred_distance` lines 95-109; `copy_for_calf` lines 300-324)
- Modify: `src/cenop/agents/population.py` (PSM list construction lines 214-216)
- Test: `tests/test_dispersal.py` (append new class `TestPSMReproducibility`)

**Interfaces:**
- Consumes: `SimulationParameters.psm_dist_mean: float` (default 350.0), `SimulationParameters.psm_dist_sd: float` (default 100.0), `PorpoisePopulation.rng: np.random.Generator`
- Produces: `PersistentSpatialMemory.__init__(..., rng=None, pref_dist_mean: float = 350.0, pref_dist_sd: float = 100.0)`; new instance attrs `PersistentSpatialMemory.pref_dist_mean: float`, `PersistentSpatialMemory.pref_dist_sd: float`; `generate_preferred_distance(mean: float = 350.0, sd: float = 100.0, rng=None)`

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_dispersal.py`:
```python
class TestPSMReproducibility:
    """PSM preferred_distance must be seeded (finding #6) and centre on
    params.psm_dist_mean=350 not the hardcoded 300 (finding #14)."""

    def test_preferred_distance_reproducible_same_seed(self):
        """Two populations built with the same random_seed must produce an
        identical preferred_distance sequence across all agents."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters.simulation_params import SimulationParameters

        # Small world keeps per-agent psm_buffer tiny (12x12 grid).
        params_a = SimulationParameters(random_seed=12345, world_width=60, world_height=60)
        params_b = SimulationParameters(random_seed=12345, world_width=60, world_height=60)
        pop_a = PorpoisePopulation(count=30, params=params_a)
        pop_b = PorpoisePopulation(count=30, params=params_b)

        dists_a = [pop_a._psm_instances[i].preferred_distance for i in range(30)]
        dists_b = [pop_b._psm_instances[i].preferred_distance for i in range(30)]

        assert dists_a == dists_b, "same seed must give identical PSM distances"
        # Sanity: it is genuinely a distribution, not a constant.
        assert len(set(dists_a)) > 1

    def test_constructor_centres_on_pref_dist_mean_350(self):
        """PersistentSpatialMemory(pref_dist_mean=350) must sample ~N(350;100)."""
        from cenop.behavior.psm import PersistentSpatialMemory

        rng = np.random.default_rng(7)
        dists = np.array([
            PersistentSpatialMemory(
                100, 100, rng=rng, pref_dist_mean=350.0, pref_dist_sd=100.0
            ).preferred_distance
            for _ in range(2000)
        ])
        mean = float(dists.mean())
        assert 340.0 < mean < 360.0, f"expected ~350, got {mean}"

    def test_population_plumbs_params_psm_dist_mean(self):
        """Production PSM construction must read params.psm_dist_mean (350),
        not generate_preferred_distance's own default (was 300)."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters(random_seed=99, world_width=60, world_height=60)
        assert params.psm_dist_mean == 350.0  # guard the default
        pop = PorpoisePopulation(count=1000, params=params)
        dists = np.array([pop._psm_instances[i].preferred_distance for i in range(1000)])
        mean = float(dists.mean())
        # Pre-fix PSM ignores params and centres on 300 -> mean far below 335.
        assert 335.0 < mean < 365.0, f"expected ~350 (params.psm_dist_mean), got {mean}"
```

- [ ] **Step 2: Run the tests, expect FAIL.**
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_dispersal.py::TestPSMReproducibility -q`
  Expected failures on today's code:
  - `test_preferred_distance_reproducible_same_seed` → `AssertionError: same seed must give identical PSM distances` (PSM uses `np.random.default_rng()` OS entropy at `psm.py:80`, so the two lists differ).
  - `test_constructor_centres_on_pref_dist_mean_350` → `TypeError: __init__() got an unexpected keyword argument 'pref_dist_mean'` (constructor has no such param yet).
  - `test_population_plumbs_params_psm_dist_mean` → `AssertionError: expected ~350 ... got ~300` (PSM centres on the hardcoded 300 default).

- [ ] **Step 3: Implement the minimal fix.**
  In `src/cenop/behavior/psm.py`, extend the constructor signature (lines 59-66) and body:
```python
    def __init__(
        self,
        world_width: int,
        world_height: int,
        preferred_distance: Optional[float] = None,
        mem_cell_size: int = MEM_CELL_SIZE,
        rng: Optional[np.random.Generator] = None,
        pref_dist_mean: float = 350.0,
        pref_dist_sd: float = 100.0,
    ):
```
  Store the new params right after the `self.rng` assignment (after line 80):
```python
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pref_dist_mean = pref_dist_mean
        self.pref_dist_sd = pref_dist_sd
```
  Change the generation call (lines 90-91) to pass mean/sd:
```python
        if preferred_distance is None:
            self.preferred_distance = self.generate_preferred_distance(
                mean=self.pref_dist_mean, sd=self.pref_dist_sd, rng=self.rng
            )
        else:
            self.preferred_distance = preferred_distance
```
  Update the `generate_preferred_distance` default mean 300.0 → 350.0 (line 97):
```python
    @staticmethod
    def generate_preferred_distance(
        mean: float = 350.0,
        sd: float = 100.0,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
```
  Update `copy_for_calf` (lines 309-315) to preserve mean/sd:
```python
        new_psm = PersistentSpatialMemory(
            world_width=self.world_width,
            world_height=self.world_height,
            preferred_distance=self.generate_preferred_distance(
                mean=self.pref_dist_mean, sd=self.pref_dist_sd, rng=self.rng
            ),
            mem_cell_size=self.mem_cell_size,
            rng=self.rng,
            pref_dist_mean=self.pref_dist_mean,
            pref_dist_sd=self.pref_dist_sd,
        )
```
  In `src/cenop/agents/population.py`, thread the seeded rng and params into PSM construction (lines 214-216):
```python
        self._psm_instances: List[PersistentSpatialMemory] = [
            PersistentSpatialMemory(
                world_w,
                world_h,
                rng=self.rng,
                pref_dist_mean=self.params.psm_dist_mean,
                pref_dist_sd=self.params.psm_dist_sd,
            )
            for _ in range(count)
        ]
```

- [ ] **Step 4: Run the tests, expect PASS — including the full fast suite.**
  First confirm the three new tests pass:
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_dispersal.py::TestPSMReproducibility -q`
  Then run the whole fast suite, because threading `self.rng` into PSM construction inserts `count` normal draws between the energy draw (`population.py:132`) and the position/heading draws (`population.py:457-485`), shifting the seeded RNG stream for every seeded population — so any test that pins exact seeded positions/values would move:
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/ -q`
  Expected: the three new tests pass. The seeded fast tests (e.g. `tests/test_depons_movement.py`, `tests/test_ref_memory.py`, `tests/test_dispersal.py`, `tests/test_backend_equivalence.py`) assert statistical/structural properties (means, std, bounds, run-vs-run self-consistency) rather than hardcoded seeded values, so they should stay green. If any test asserts an EXACT pre-shift seeded number and now fails ONLY because the stream advanced (values still within the correct distribution/bounds), update that expected value to the new correct output and note it in the commit. Do NOT loosen a real behavioural assertion.
  Scope note: the committed Kattegat output reference files under `output/kattegat_ref*` also shift and are regenerated separately via `scripts/run_kattegat_reference.py` — that regeneration is out of scope for this pytest task and is tracked by the baseline-regeneration follow-up.

- [ ] **Step 5: Commit (from within the nested CENOP repo).**
  Command: `git -C /home/razinka/cenjas/CENOP add src/cenop/behavior/psm.py src/cenop/agents/population.py tests/test_dispersal.py && git commit -m "Fix PSM reproducibility and plumb psm_dist_mean/sd (findings #6, #14)"`
  Message body should note: PSM now uses the population's seeded generator (same-seed runs reproducible) and centres preferred_distance on params.psm_dist_mean (350, DEPONS 3.2) instead of a hardcoded 300; and that this shifts the seeded RNG stream so Kattegat reference baselines must be regenerated as a follow-up.

### Task 8: Set UI default and controller parse-fallback for psm_dist to N(350;100)

Completes finding #14: the `psm_dist` UI text input defaults to `N(300;100)` (`settings.py:360`) and the controller parse-fallback defaults `psm_dist_mean = 300.0` (`simulation_controller.py:96`). DEPONS 3.2 uses N(350;100); with Task 1 the model now reads these values, so the UI/controller defaults must match to avoid silently reintroducing 300.

**Files:**
- Modify: `src/cenop/ui/tabs/settings.py` (tooltip line 38; `ui.input_text` default line 360)
- Modify: `src/cenop/server/simulation_controller.py` (parse comment line 94; fallback `psm_dist_mean` line 96)
- Test: `tests/test_dispersal.py` (append new class `TestPSMDistDefaults`)

**Interfaces:**
- Consumes: none
- Produces: none (string/default constants only)

- [ ] **Step 1: Write the failing test.** Append to `tests/test_dispersal.py`:
```python
class TestPSMDistDefaults:
    """UI default and controller parse-fallback must be N(350;100) (finding #14)."""

    def test_ui_and_controller_defaults_are_350(self):
        import inspect
        import cenop.ui.tabs.settings as settings_mod
        import cenop.server.simulation_controller as ctrl_mod

        settings_src = inspect.getsource(settings_mod)
        ctrl_src = inspect.getsource(ctrl_mod)

        assert 'ui.input_text("psm_dist", None, value="N(350;100)")' in settings_src
        assert 'value="N(300;100)"' not in settings_src
        assert "psm_dist_mean = 350.0" in ctrl_src
        assert "psm_dist_mean = 300.0" not in ctrl_src
```

- [ ] **Step 2: Run the test, expect FAIL.**
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_dispersal.py::TestPSMDistDefaults -q`
  Expected: `AssertionError` — today `settings.py:360` contains `value="N(300;100)"` and `simulation_controller.py:96` contains `psm_dist_mean = 300.0`, so the first `assert ... in settings_src` (and the `not in` guards) fail.

- [ ] **Step 3: Implement the minimal fix.**
  In `src/cenop/ui/tabs/settings.py`, update the input default (line 360):
```python
                    ui.input_text("psm_dist", None, value="N(350;100)"),
```
  And the tooltip text (line 38) for consistency:
```python
    "psm_dist": "Preferred dispersal distance distribution. Format: N(mean;std) in km. DEPONS default: N(350;100) = mean 350km, std 100km.",
```
  In `src/cenop/server/simulation_controller.py`, update the fallback comment (line 94) and default (line 96):
```python
    # Parse PSM Dist string "N(350;100)"
    psm_dist_str = input.psm_dist()
    psm_dist_mean = 350.0
    psm_dist_sd = 100.0
```

- [ ] **Step 4: Run the test, expect PASS.**
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_dispersal.py::TestPSMDistDefaults -q`
  Expected: passes.

- [ ] **Step 5: Commit (from within the nested CENOP repo).**
  Command: `git -C /home/razinka/cenjas/CENOP add src/cenop/ui/tabs/settings.py src/cenop/server/simulation_controller.py tests/test_dispersal.py && git commit -m "Set psm_dist UI/controller defaults to N(350;100) (finding #14)"`


> **Note (psm-rng-distance):** BASELINE REGENERATION REQUIRED. Threading self.rng into PSM construction (population.py:215) inserts `count` normal draws into the population's RNG stream that previously came from a separate OS-entropy generator. Those draws now occur during __init__ (between the energy draw at population.py:132 and position/heading draws at ~457-485), which shifts ALL downstream seeded outputs (positions, headings, mating days, etc.). Additionally the mean default changes 300->350, altering dispersal target distances. Therefore any seeded reference baselines (Kattegat refs under CENOP/output/kattegat_ref*) will change and must be regenerated via scripts/run_kattegat_reference.py, and any existing test that pins exact seeded positions/values may need updating. This is the intended, correct fix per findings #6/#14.

Scope note: verified there is NO separate calf/birth PSM construction to fix — calves reuse pre-allocated inactive slots (population.py:2136-2155) and their existing _psm_instances[slot] object; _psm_instances is a fixed-length list built only at line 214-216, and copy_for_calf is not invoked on the production birth path (still fixed for correctness/consistency). params.psm_dist_mean/sd already exist (simulation_params.py:81-82, defaults 350/100) and are populated by the controller (simulation_controller.py:195-196), so Task 1 only needs to READ them at construction.

Memory sizing: population tests use small worlds (world_width/height=60 -> 12x12 psm grid) because default 1000x1000 world makes per-agent psm_buffer ~640KB; count=1000 at default world would allocate ~640MB, count=2000 ~1.3GB. The centring test with N=2000 samples constructs bare PersistentSpatialMemory objects (no psm_buffer) so it is cheap. All new tests are fast (no @pytest.mark.slow needed).

Task 2 uses an inspect.getsource source-scan test because the psm_dist UI default is a rendered Shiny component and the controller fallback is a function-local literal — neither is cleanly unit-callable; the source-scan deterministically captures the 300->350 change and guards against regression.


### Task 9: Fix stale rS/rR energy-panel UI defaults (0.04 -> 0.03)

**Context / root cause (verified in source):**
- `src/cenop/parameters/simulation_params.py:88-89` already holds the correct DEPONS 3.2 defaults: `r_s: float = 0.03`, `r_r: float = 0.03`.
- `src/cenop/ui/tabs/settings.py:399` and `:407` declare the Shiny numeric inputs with the STALE DEPONS-3.0 value: `ui.input_numeric("param_rS", None, value=0.04, step=0.01)` and `ui.input_numeric("param_rR", None, value=0.04, step=0.01)`.
- `src/cenop/server/simulation_controller.py:201-202` reads them verbatim: `r_s=input.param_rS(), r_r=input.param_rR()`, so the stale UI default (0.04) OVERRIDES the correct dataclass default (0.03) for every simulation launched from the UI.
- Tooltips at `settings.py:43-44` also state "DEPONS default: 0.04" (wrong).
- Empirically confirmed: rendering the panel via `_energy_settings_panel().content.tagify()` shows `param_rS -> 0.04`, `param_rR -> 0.04`; and building a Simulation through the controller with those UI defaults yields `sim.params.r_s == 0.04`, `sim.params.r_r == 0.04`.

**Files:**
- Create: `tests/test_settings_defaults.py`
- Modify: `src/cenop/ui/tabs/settings.py` (lines 43-44 tooltips; line 399 param_rS input; line 407 param_rR input)

**Interfaces:**
- Consumes: `cenop.ui.tabs.settings._energy_settings_panel() -> shiny.ui._navs.NavPanel` (has `.content` TagList that `.tagify()`s to HTML); `cenop.ui.tabs.settings.TOOLTIPS: dict[str, str]`; `cenop.server.simulation_controller.create_simulation_from_inputs(input) -> Simulation` (reads `input.param_rS()`/`input.param_rR()`, exposes result as `sim.params.r_s` / `sim.params.r_r`).
- Produces: none (fix changes literal default values + tooltip text only; no new symbols).

**TDD steps:**

- [ ] **Step 1: Write the failing test.** Create `tests/test_settings_defaults.py` with this exact content. It (a) extracts the declared numeric defaults from the rendered Energy panel, (b) checks the tooltip text, and (c) drives the real `create_simulation_from_inputs` with a full mock Shiny input whose `param_rS`/`param_rR` return the ACTUAL UI defaults — so the controller pass-through is genuinely exercised, not hardcoded.

```python
"""Finding #5: Energy-panel rS/rR UI defaults must match DEPONS 3.2 parameters.xml (0.03)."""
import re
from types import SimpleNamespace

import pytest

def _energy_panel_numeric_default(input_id: str) -> float:
    """Return the declared ``value=`` of a numeric input in the Energy settings panel."""
    from cenop.ui.tabs.settings import _energy_settings_panel

    html = str(_energy_settings_panel().content.tagify())
    tag_match = re.search(rf'<input[^>]*id="{input_id}"[^>]*>', html)
    assert tag_match, f"input {input_id!r} not found in Energy panel HTML"
    value_match = re.search(r'value="([^"]*)"', tag_match.group(0))
    assert value_match, f"no value attribute on input {input_id!r}"
    return float(value_match.group(1))

def _make_mock_input(param_rS: float, param_rR: float) -> SimpleNamespace:
    """Shiny-input stand-in returning UI defaults; rS/rR are parameterized."""
    def const(v):
        return lambda: v

    values = dict(
        random_seed=1, psm_dist="N(300;100)", porpoise_count=5, sim_years=1,
        simulation_mode="DEPONS", time_mode_override="", movement_mode_override="",
        fsm_mode_override="", energy_mode_override="", memory_mode_override="",
        jasmine_mass_kg=50.0, jasmine_drag_coeff=0.01, jasmine_max_thrust=100.0,
        jasmine_current_weight=0.5, jasmine_bmr_scale=1.0, jasmine_activity_cost=2.0,
        jasmine_disturbance_cost=1.5, jasmine_memory_decay_rate=0.001,
        jasmine_avoidance_strength=0.8, jasmine_avoidance_radius=20.0,
        landscape="Homogeneous", turbines="off", ships_enabled=False,
        weston_flux_percell=False, dispersal="off", tracked_porpoise_count=1,
        tdisp=3, psm_log=0.6, psm_tol=5.0, psm_angle=40.0,
        param_rS=param_rS, param_rR=param_rR, param_rU=0.1, bycatch_prob=0.018,
        param_k=0.001, param_a0=0.35, param_a1=0.0005, param_a2=-0.02,
        param_b0=-0.024, param_b1=-0.008, param_b2=0.93, param_b3=-14.0,
        communication_enabled=False, communication_range_km=1.0,
        communication_source_level=130.0, communication_threshold=80.0,
        communication_response_slope=0.1, social_weight=0.3,
    )
    return SimpleNamespace(**{k: const(v) for k, v in values.items()})

class TestEnergyPanelDecayDefaults:
    """rS/rR must default to DEPONS 3.2 value 0.03 (was stale DEPONS-3.0 0.04)."""

    def test_rS_ui_default_is_003(self):
        assert _energy_panel_numeric_default("param_rS") == pytest.approx(0.03)

    def test_rR_ui_default_is_003(self):
        assert _energy_panel_numeric_default("param_rR") == pytest.approx(0.03)

    def test_rS_tooltip_states_003(self):
        from cenop.ui.tabs.settings import TOOLTIPS
        assert "0.03" in TOOLTIPS["param_rS"]
        assert "0.04" not in TOOLTIPS["param_rS"]

    def test_rR_tooltip_states_003(self):
        from cenop.ui.tabs.settings import TOOLTIPS
        assert "0.03" in TOOLTIPS["param_rR"]
        assert "0.04" not in TOOLTIPS["param_rR"]

    def test_controller_propagates_ui_defaults_to_params(self):
        """End-to-end: UI energy defaults flow through create_simulation_from_inputs."""
        from cenop.server.simulation_controller import create_simulation_from_inputs

        ui_rS = _energy_panel_numeric_default("param_rS")
        ui_rR = _energy_panel_numeric_default("param_rR")
        sim = create_simulation_from_inputs(_make_mock_input(ui_rS, ui_rR))
        assert sim.params.r_s == pytest.approx(0.03)
        assert sim.params.r_r == pytest.approx(0.03)
```

- [ ] **Step 2: Run the test, expect FAIL.**
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_settings_defaults.py -q`
  Expected: 5 failures. `test_rS_ui_default_is_003` / `test_rR_ui_default_is_003` fail with `assert 0.04 == 0.03 ± 3.0e-08`; `test_rS_tooltip_states_003` / `test_rR_tooltip_states_003` fail on `assert "0.04" not in TOOLTIPS[...]`; `test_controller_propagates_ui_defaults_to_params` fails with `assert 0.04 == 0.03 ± 3.0e-08` (the mock feeds the real 0.04 UI default, controller passes it verbatim into `sim.params.r_s`).

- [ ] **Step 3: Fix the UI defaults and tooltips in `src/cenop/ui/tabs/settings.py`.** Four exact edits:
  1. Line 43 tooltip — change trailing `0.04` to `0.03`:
     - from: `    "param_rS": "Satiation memory decay rate. Higher = faster forgetting of food satisfaction. DEPONS default: 0.04",`
     - to:   `    "param_rS": "Satiation memory decay rate. Higher = faster forgetting of food satisfaction. DEPONS default: 0.03",`
  2. Line 44 tooltip — change trailing `0.04` to `0.03`:
     - from: `    "param_rR": "Reference memory decay rate. Higher = faster forgetting of remembered food locations. DEPONS default: 0.04",`
     - to:   `    "param_rR": "Reference memory decay rate. Higher = faster forgetting of remembered food locations. DEPONS default: 0.03",`
  3. Line 399 input default:
     - from: `                    ui.input_numeric("param_rS", None, value=0.04, step=0.01),`
     - to:   `                    ui.input_numeric("param_rS", None, value=0.03, step=0.01),`
  4. Line 407 input default:
     - from: `                    ui.input_numeric("param_rR", None, value=0.04, step=0.01),`
     - to:   `                    ui.input_numeric("param_rR", None, value=0.03, step=0.01),`
  (Each `ui.input_numeric` line is unique by its `"param_rS"` / `"param_rR"` id string, so the edits are unambiguous. `param_rU` at line 415 stays `value=0.1` — it is correct. All lines remain <=100 chars.)

- [ ] **Step 4: Run tests, expect PASS.**
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_settings_defaults.py -q`
  Expected: `5 passed`.
  Regression guard (energy/param defaults unaffected): `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_codebase_review_r2.py -q` -> expected all pass (no change to dataclass defaults).

- [ ] **Step 5: Commit (from within the nested CENOP repo).**
  Commands:
  `git -C /home/razinka/cenjas/CENOP add tests/test_settings_defaults.py src/cenop/ui/tabs/settings.py`
  `git -C /home/razinka/cenjas/CENOP commit -m "Fix stale rS/rR energy UI defaults 0.04->0.03 (DEPONS 3.2 parity)"`
  Commit message body/footer:
  `The Energy panel declared param_rS/param_rR defaults of 0.04 (DEPONS 3.0);`
  `create_simulation_from_inputs reads them verbatim, overriding the correct`
  `SimulationParameters defaults (0.03, DEPONS 3.2 parameters.xml). Set UI`
  `defaults + tooltips to 0.03; add tests/test_settings_defaults.py covering`
  `the panel defaults, tooltips, and end-to-end controller propagation.`
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`


> **Note (rs-rr-defaults):** Verified live: _energy_settings_panel().content.tagify() renders param_rS/param_rR with value=0.04 today; mock-driven create_simulation_from_inputs yields sim.params.r_s/r_r == 0.04 (defect reproduced). Dataclass defaults in simulation_params.py:88-89 are already correct (0.03) — this is purely a UI-default + tooltip fix; no src change to the controller. NO reference-baseline regeneration needed: the Kattegat baselines are produced by scripts/run_kattegat_reference.py using SimulationParameters dataclass defaults (0.03), not the Shiny UI, so committed baselines are unaffected. Test is fast (single small homogeneous Simulation build, porpoise_count=5, no ticks run) and unmarked, so it runs in the default suite. The full mock lists every input.* accessed by the controller (note the digit-suffixed param_a0/a1/a2/b0-b3 which a naive grep misses); landscape='Homogeneous' avoids landscape-file IO via create_homogeneous_landscape(). Risk: low; behavior of production UI-launched sims changes (rS/rR now 0.03 instead of 0.04) which is the intended correctness fix.


### Task 10: Strip parallel=True from six small-N Numba kernels

Fixes Finding #8 (HIGH, performance). Six `@njit(parallel=True)` kernels in
`src/cenop/optimizations/kernels.py` parallelize over the ~N-agent axis (a few hundred
elements). No code anywhere sets `numba.set_num_threads`/`NUMBA_NUM_THREADS` (verified:
`grep -rn "set_num_threads\|NUMBA_NUM_THREADS" src/ tests/` returns nothing), so every
per-tick call forks the full thread pool over a tiny loop → measured 6.7–8.5× slower
whole-tick vs serial. Writes are distinct-index (no data race), so stripping `parallel=True`
is behavior-preserving. `regrow_food_kernel` (~1e5 cells) keeps `parallel=True`.

Approach (a) from the finding — strip `parallel=True` and turn `prange`→`range` in the six
small-N kernels only; keep `regrow_food_kernel` parallel. Chosen over option (b)
(`set_num_threads` at import) for determinism and zero global state. Then re-measure the
committed tick baseline.

**Files:**
- Modify: `src/cenop/optimizations/kernels.py`
  - decorators to change (drop `, parallel=True`): line 42 `reflect_boundaries_kernel`,
    line 193 `turn_position_kernel`, line 384 `depons_bmr_cost_kernel`,
    line 477 `compute_ve_total_kernel`, line 507 `compute_attraction_kernel`,
    line 670 `heading_position_reflect_kernel`
  - `prange`→`range`: lines 63, 77 (reflect), 213 (turn_position), 405 (depons_bmr),
    493 (compute_ve_total), 527 (compute_attraction), 702 (heading_position_reflect)
  - UNCHANGED: line 460 `regrow_food_kernel` decorator + line 467 `prange(len(food))`;
    the `from numba import njit, prange` import stays (regrow still uses `prange`)
- Test: `tests/test_numba_kernels.py` (add two test classes at end of file)

**Interfaces:**
- Consumes: `cenop.optimizations.kernels.NUMBA_AVAILABLE` (bool);
  numba `CPUDispatcher.targetoptions` dict — key `"parallel"` is `True` only when the
  kernel was decorated `parallel=True`; for a serial kernel the key is PRESENT with value
  `None` (verified on numba 0.63.1: `seed_numba_rng`/`crw_angle_step_kernel` report
  `parallel=None`), so `opts.get("parallel") is True` cleanly distinguishes the two
  (`None is True` → False);
  `turn_position_kernel(x, y, heading, step_dist, turn_delta, world_w, world_h, out_x,
  out_y, out_heading, out_xi, out_yi)` (signature unchanged by this fix).
- Produces: the six kernels compiled serial (`targetoptions["parallel"]` is `None`);
  identical numeric outputs and identical call signatures. `regrow_food_kernel` still
  parallel (`targetoptions["parallel"] is True`).

- [ ] **Step 1: Write the failing guard test + an equivalence guard.**
  Append to `tests/test_numba_kernels.py`:
  ```python
  class TestKernelParallelFlags:
      """Small-N kernels must NOT be parallel=True.

      Forking the full thread pool over a few hundred agents measured 6.7-8.5x
      slower whole-tick than serial. Only regrow_food_kernel (~1e5 cells) keeps
      parallel=True. See Finding #8.
      """

      SERIAL_KERNELS = [
          "reflect_boundaries_kernel",
          "turn_position_kernel",
          "depons_bmr_cost_kernel",
          "compute_ve_total_kernel",
          "compute_attraction_kernel",
          "heading_position_reflect_kernel",
      ]

      def test_small_n_kernels_not_parallel(self):
          from cenop.optimizations import kernels as k
          if not k.NUMBA_AVAILABLE:
              pytest.skip("numba not installed — njit is a no-op passthrough")
          offenders = []
          for name in self.SERIAL_KERNELS:
              opts = getattr(getattr(k, name), "targetoptions", {})
              if opts.get("parallel") is True:
                  offenders.append(name)
          assert not offenders, (
              f"still parallel=True (strip it — over-forks the pool): {offenders}"
          )

      def test_regrow_food_kernel_stays_parallel(self):
          from cenop.optimizations import kernels as k
          if not k.NUMBA_AVAILABLE:
              pytest.skip("numba not installed — njit is a no-op passthrough")
          opts = getattr(k.regrow_food_kernel, "targetoptions", {})
          assert opts.get("parallel") is True, (
              "regrow_food_kernel (~1e5 cells) should keep parallel=True"
          )

  class TestTurnPositionEquivalence:
      """Guard: turn_position_kernel output must match a pure-NumPy reference so
      the parallel->serial refactor provably changes no numbers. Passes before
      AND after the fix (behavioral-equivalence guard)."""

      @staticmethod
      def _numpy_reference(x, y, heading, step_dist, turn_delta, world_w, world_h):
          max_x = float(world_w - 1)
          max_y = float(world_h - 1)
          h = (heading + turn_delta) % 360.0
          rads = h * np.pi / 180.0
          nx = x + np.sin(rads) * step_dist
          ny = y + np.cos(rads) * step_dist
          # mirror the kernel's if/elif reflect (below wins over above), then clamp
          nx = np.where(nx < 0.0, -nx,
                        np.where(nx > max_x, 2.0 * max_x - nx, nx))
          nx = np.clip(nx, 0.0, max_x)
          ny = np.where(ny < 0.0, -ny,
                        np.where(ny > max_y, 2.0 * max_y - ny, ny))
          ny = np.clip(ny, 0.0, max_y)
          xi = np.clip(nx.astype(np.int32), 0, world_w - 1)
          yi = np.clip(ny.astype(np.int32), 0, world_h - 1)
          return h, nx, ny, xi, yi

      def test_turn_position_matches_numpy(self):
          from cenop.optimizations.kernels import turn_position_kernel
          rng = np.random.default_rng(2024)
          n = 300
          world_w = world_h = 200
          x = rng.uniform(0, 199, n).astype(np.float64)
          y = rng.uniform(0, 199, n).astype(np.float64)
          heading = rng.uniform(0, 360, n).astype(np.float64)
          step = rng.uniform(0, 30, n).astype(np.float64)  # << max_x so no double-reflect
          turn_delta = 15.0
          out_x = np.empty(n, dtype=np.float64)
          out_y = np.empty(n, dtype=np.float64)
          out_h = np.empty(n, dtype=np.float64)
          out_xi = np.empty(n, dtype=np.int32)
          out_yi = np.empty(n, dtype=np.int32)
          turn_position_kernel(x, y, heading, step, turn_delta, world_w, world_h,
                               out_x, out_y, out_h, out_xi, out_yi)
          ref_h, ref_x, ref_y, ref_xi, ref_yi = self._numpy_reference(
              x, y, heading, step, turn_delta, world_w, world_h)
          np.testing.assert_allclose(out_x, ref_x, atol=1e-9)
          np.testing.assert_allclose(out_y, ref_y, atol=1e-9)
          np.testing.assert_allclose(out_h, ref_h, atol=1e-9)
          np.testing.assert_array_equal(out_xi, ref_xi)
          np.testing.assert_array_equal(out_yi, ref_yi)
  ```
  (`import numpy as np` and `import pytest` are already at the top of the file.)

- [ ] **Step 2: Run the new tests — expect FAIL on the flag guard.**
  ```
  cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest \
    tests/test_numba_kernels.py::TestKernelParallelFlags \
    tests/test_numba_kernels.py::TestTurnPositionEquivalence -q
  ```
  Expected: `test_small_n_kernels_not_parallel` FAILS with
  `AssertionError: still parallel=True (strip it ...): ['reflect_boundaries_kernel',
  'turn_position_kernel', 'depons_bmr_cost_kernel', 'compute_ve_total_kernel',
  'compute_attraction_kernel', 'heading_position_reflect_kernel']`.
  `test_regrow_food_kernel_stays_parallel` and `test_turn_position_matches_numpy` PASS.
  (1 failed, 2 passed.)

- [ ] **Step 3: Strip `parallel=True` and `prange`→`range` in the six kernels.**
  Six decorator edits (each decorator line + its `def` line is a unique anchor). Edit
  `src/cenop/optimizations/kernels.py`, changing `@njit(cache=True, parallel=True)` to
  `@njit(cache=True)` immediately above each of these defs:
  ```python
  @njit(cache=True)
  def reflect_boundaries_kernel(
  ```
  ```python
  @njit(cache=True)
  def turn_position_kernel(
  ```
  ```python
  @njit(cache=True)
  def depons_bmr_cost_kernel(
  ```
  ```python
  @njit(cache=True)
  def compute_ve_total_kernel(
  ```
  ```python
  @njit(cache=True)
  def compute_attraction_kernel(
  ```
  ```python
  @njit(cache=True)
  def heading_position_reflect_kernel(
  ```
  Then three `prange`→`range` edits (leaving `regrow_food_kernel`'s `prange(len(food))`
  at line 467 untouched). All target loops are at 4-space (function-body) indentation —
  verified with `cat -A` (lines 63/77/213/405 = `    for i in prange(n):`;
  lines 493/527 = `    for ai in prange(len(active_indices)):`;
  line 702 = `    for i in prange(len(heading)):`):
  - replace_all `    for i in prange(n):` → `    for i in range(n):`
    (this token appears ONLY in reflect ×2 / turn_position / depons_bmr — all now serial;
    regrow uses `prange(len(food))`, heading uses `prange(len(heading))` — neither matches)
  - replace_all `    for ai in prange(len(active_indices)):`
    → `    for ai in range(len(active_indices)):`
    (compute_ve_total + compute_attraction; regrow uses `prange(len(food))`, unaffected)
  - `    for i in prange(len(heading)):` → `    for i in range(len(heading)):`
    (heading_position_reflect_kernel — single occurrence, unique)
  Leave the `from numba import njit, prange` import as-is — `regrow_food_kernel` still uses
  `prange`. Numba's file cache keys on source, so it auto-recompiles; no manual cache clear.

- [ ] **Step 4: Re-run the new tests — expect PASS.**
  ```
  cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest \
    tests/test_numba_kernels.py::TestKernelParallelFlags \
    tests/test_numba_kernels.py::TestTurnPositionEquivalence -q
  ```
  Expected: 3 passed.

- [ ] **Step 5: Run the full kernel + tick-equivalence suites — expect all green.**
  ```
  cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest \
    tests/test_numba_kernels.py tests/test_tick_performance.py -q
  ```
  Expected: all passed (existing `TestReflectBoundariesKernel`,
  `TestO3CachedActiveIdx`, `TestO4ReduceNpAny`, `TestO5FusedHeadingKernel`,
  `TestO9SeparateBMR`, etc. still verify byte-for-byte trajectory equivalence via
  `assert_states_match` — this fix changes only threading, not numbers).

- [ ] **Step 6: Re-measure the committed tick baseline (was ~2.22 ms/tick).**
  The old committed baseline was produced with the over-forked pool; it MUST be
  re-measured. A timing assertion is flaky in CI, so this is a one-off measurement, not a
  committed test. The landscape/population/measurement helpers below are a verbatim copy of
  the working helpers in `tests/test_tick_performance.py` (lines 15-52). Write this to the
  scratchpad and run it:
  `/tmp/claude-1000/-home-razinka-cenjas/a92d6051-339e-40c9-ac93-a0b68673a463/scratchpad/measure_tick.py`
  ```python
  import time
  import numpy as np
  from cenop.agents.population import PorpoisePopulation
  from cenop.parameters.simulation_params import SimulationParameters
  from cenop.landscape.cell_data import CellData, LandscapeMetadata

  def make_landscape(w=200, h=200):
      cd = CellData.__new__(CellData)
      cd.landscape_name = "Homogeneous"
      cd.data_dir = ""
      cd.metadata = LandscapeMetadata(ncols=w, nrows=h, xllcorner=0.0, yllcorner=0.0)
      cd._depth = np.full((h, w), 30.0, dtype=np.float32)
      cd._dist_to_coast = np.full((h, w), 5000.0, dtype=np.float32)
      cd._sediment = np.full((h, w), 5.0, dtype=np.float32)
      cd._food_prob = np.ones((h, w), dtype=np.float32)
      cd._food_value = np.full((h, w), 50.0, dtype=np.float32)
      cd._blocks = np.zeros((h, w), dtype=np.int32)
      cd._entropy = np.full((12, h, w), 50.0, dtype=np.float32)
      cd._salinity = np.full((12, h, w), 30.0, dtype=np.float32)
      cd._demand_grid = np.zeros((h, w), dtype=np.float32)
      cd._current_month = 1
      cd._loaded = True
      return cd

  np.random.seed(42)
  params = SimulationParameters(porpoise_count=500, world_width=200,
                                world_height=200, random_seed=42)
  pop = PorpoisePopulation(500, params, landscape=make_landscape())
  pop._skip_land_avoidance = True
  for _ in range(50):      # warmup (JIT compile + steady state)
      pop.step()
  t0 = time.perf_counter()
  for _ in range(200):
      pop.step()
  print("ms/tick:", round((time.perf_counter() - t0) / 200 * 1000, 3))
  ```
  Run:
  ```
  cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 \
    /tmp/claude-1000/-home-razinka-cenjas/a92d6051-339e-40c9-ac93-a0b68673a463/scratchpad/measure_tick.py
  ```
  Expected: a single `ms/tick: <value>` line, and `<value>` should be markedly LOWER than
  the over-forked baseline (finding: 6.7–8.5× faster whole-tick serial). Record the printed
  number for the commit message and hand it to the human to update the `2.22 ms/tick`
  figure in MEMORY.md and `docs/superpowers/specs/2026-03-19-jax-jit-tick-design.md`
  (those live in the parent `cenjas` repo, not this nested repo).

- [ ] **Step 7: Commit (from within the nested CENOP repo, branch CENOP-JASMINE).**
  ```
  git -C /home/razinka/cenjas/CENOP add src/cenop/optimizations/kernels.py tests/test_numba_kernels.py
  ```
  Then commit, substituting the Step-6 measured value into `<value>`:
  ```
  git -C /home/razinka/cenjas/CENOP commit -m "perf: strip parallel=True from six small-N Numba kernels

reflect_boundaries/turn_position/depons_bmr_cost/compute_ve_total/
compute_attraction/heading_position_reflect forked the full thread pool
over only a few hundred agents each tick (nothing sets NUMBA_NUM_THREADS/
set_num_threads) -> 6.7-8.5x slower whole-tick than serial. Writes are
distinct-index so this is behavior-preserving; guarded by
TestKernelParallelFlags (targetoptions.parallel) + TestTurnPositionEquivalence
and the existing byte-for-byte trajectory tests. regrow_food_kernel (~1e5
cells) keeps parallel=True. Re-measured N=500 homogeneous tick: <value> ms/tick
(was ~2.22 ms with the over-forked pool).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```


> **Note (numba-threads):** Standalone perf-only fix; no cross-task deps. BASELINE RE-MEASURE REQUIRED: the committed 2.22 ms/tick figure was produced under the over-forked pool and is now stale — Step 6 re-measures it and Step 7 records the new value in the commit; the human must then update MEMORY.md + docs/superpowers/specs/2026-03-19-jax-jit-tick-design.md (both in the parent cenjas repo, NOT the nested CENOP repo). Behavioral equivalence: writes are distinct-index (no data race) and prange==range in serial numba, so outputs are bit-identical; the existing assert_states_match trajectory tests in tests/test_tick_performance.py plus the new TestTurnPositionEquivalence guard this. The parallel-flag test relies on numba CPUDispatcher.targetoptions (verified present on numba 0.63.1 in the shiny env); it skips cleanly if NUMBA_AVAILABLE is False. Slow tier unaffected (these tests are fast). Commit from within CENOP/ on branch CENOP-JASMINE (nested repo).


### Task 11: Introduce _WorkerHandle (fresh-per-run stop_event + queue, join old worker)

> NOTE: main.py is under concurrent edit — cited line numbers reflect the file at review time (2026-07-06) but may drift. The quoted anchor text in each Edit is authoritative; match on it, not on line numbers.

**Files:**
- Modify: `src/cenop/server/main.py` (insert new module-level class `_WorkerHandle` after `run_simulation_loop`, which ends at line 280, and before the `# Helper functions for testability` comment header at line 283 that precedes `def _build_landscape_table_rows(landscapes):` at line 287).
- Test: `tests/test_server_lifecycle.py` (new file)

**Interfaces:**
- Consumes: `run_simulation_loop` (existing module-level worker at `src/cenop/server/main.py:103`), module-level `logger` (line 52), stdlib `threading`, `queue` (already imported at lines 14-15).
- Produces: `cenop.server.main._WorkerHandle` with public attributes `thread: threading.Thread | None`, `stop_event: threading.Event`, `result_queue: queue.Queue`, and methods `is_alive() -> bool`, `stop_and_join(timeout: float = 5.0) -> bool`, `new_run(timeout: float = 5.0) -> tuple[threading.Event, queue.Queue]`, `start(target, args, daemon: bool = True) -> threading.Thread`.

**TDD steps:**

- [ ] **Step 1: Write the failing tests.** Create `tests/test_server_lifecycle.py`:

```python
"""Lifecycle tests for the background simulation worker (Finding #7).

A single stop_event / result_queue / thread used to be shared across runs, so a
Stop-then-Start could re-arm a still-alive worker via ``stop_event.clear()`` and
let two workers interleave output on one queue. ``_WorkerHandle`` gives each run
a FRESH stop_event + result_queue and joins the previous worker first.
"""

import queue
import threading
import time

import pytest

def test_new_run_installs_fresh_objects():
    from cenop.server.main import _WorkerHandle

    h = _WorkerHandle()
    old_event = h.stop_event
    old_queue = h.result_queue

    new_event, new_queue = h.new_run()

    assert new_event is not old_event
    assert new_queue is not old_queue
    assert new_event is h.stop_event
    assert new_queue is h.result_queue
    # A fresh event must NOT be pre-set. The old bug cleared a SHARED event,
    # which re-armed a still-alive old worker; a fresh event cannot do that.
    assert not new_event.is_set()

def test_stop_and_join_joins_cooperative_worker():
    from cenop.server.main import _WorkerHandle

    h = _WorkerHandle()

    def cooperative():
        # Re-checks stop_event only between "batches", like the real worker.
        while not h.stop_event.is_set():
            h.result_queue.put(("tick", threading.get_ident()))
            time.sleep(0.01)

    t = h.start(target=cooperative, args=())
    assert t.is_alive()
    time.sleep(0.05)  # let it produce a few items

    ok = h.stop_and_join(timeout=5.0)

    assert ok is True
    assert not t.is_alive()
    assert h.thread is None
    assert h.stop_event.is_set()

def test_new_run_isolates_stubborn_worker():
    from cenop.server.main import _WorkerHandle

    h = _WorkerHandle()
    hard_kill = threading.Event()

    def stubborn():
        # Ignores stop_event entirely; only hard_kill releases it.
        hard_kill.wait(timeout=10.0)

    old_thread = h.start(target=stubborn, args=())
    old_queue = h.result_queue

    # new_run signals + joins (join times out because the worker ignores
    # stop_event) but STILL swaps to fresh objects so the new run is isolated.
    new_event, new_queue = h.new_run(timeout=0.2)

    try:
        assert old_thread.is_alive()          # stubborn worker survived the join
        assert new_queue is not old_queue      # new run got a fresh queue anyway
        assert new_event is h.stop_event
        assert h.result_queue is new_queue
        assert not new_event.is_set()          # fresh event, not the (set) old one
    finally:
        hard_kill.set()
        old_thread.join(timeout=5.0)
    assert not old_thread.is_alive()

# ---------------------------------------------------------------------------
# Integration: drive the REAL production worker (run_simulation_loop) through a
# _WorkerHandle across a Stop-then-Start with the first worker still alive.
# ---------------------------------------------------------------------------

class _StubState:
    year = 1

class _StubSim:
    def __init__(self):
        self.state = _StubState()

class _StubRunner:
    """Minimal stand-in exposing only what run_simulation_loop touches."""

    def __init__(self, worker_id, complete_after=None, park=None, ready=None):
        self.worker_id = worker_id
        self.complete_after = complete_after
        self.park = park
        self.ready = ready
        self.is_complete = False
        self.should_update_map = False
        self.tick = 0
        self.calls = 0
        self.max_ticks = 1000
        self.progress_percent = 0.0
        self.total_births = 0
        self.total_deaths = 0
        self.sim = _StubSim()

    def set_ticks_per_update(self, n):
        pass

    def step_ticks(self):
        self.tick += 1
        self.calls += 1
        if self.complete_after is not None and self.calls >= self.complete_after:
            self.is_complete = True
        if self.park is not None and self.calls >= 2:
            if self.ready is not None:
                self.ready.set()
            self.park.wait(timeout=5.0)  # park so the worker is definitely alive
        return {
            "population": 100 * self.worker_id,
            "year": 1,
            "day": self.tick % 360,
            "worker_id": self.worker_id,
        }

def _loop_args(runner, q, ev):
    # throttle=1.0 -> current_speed>=0.99 -> no sleeps, fast loop.
    return (
        runner, q, ev,
        [1.0], threading.Lock(),          # throttle_value, throttle_lock
        [48], threading.Lock(),           # ticks_per_update_value, ticks_lock
        [False], [2], threading.Lock(),   # trace_enabled, trace_length, trace_lock
        [False], threading.Lock(),        # skip_viz_value, skip_viz_lock
    )

def _drain_worker_ids(q):
    ids = set()
    has_complete = False
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            break
        if msg.get("type") == "update":
            ids.add(msg["entry"]["worker_id"])
        elif msg.get("type") == "complete":
            has_complete = True
    return ids, has_complete

def test_stop_then_start_no_interleaving_and_old_worker_joined():
    from cenop.server.main import run_simulation_loop, _WorkerHandle

    h = _WorkerHandle()

    # --- start run #1 (production start_simulation path) ---
    ev1, q1 = h.new_run()
    runner1 = _StubRunner(
        worker_id=1, park=threading.Event(), ready=threading.Event()
    )
    t1 = h.start(target=run_simulation_loop, args=_loop_args(runner1, q1, ev1))

    # Wait until worker #1 is parked mid-run so it is DEFINITELY still alive.
    assert runner1.ready.wait(timeout=5.0)
    assert t1.is_alive()

    # --- stop (production stop_simulation) ---
    h.stop_event.set()
    runner1.park.set()  # let worker #1 finish its current batch and observe stop

    # --- start run #2 (production start_simulation again) ---
    ev2, q2 = h.new_run(timeout=5.0)  # joins worker #1, installs fresh objects
    assert not t1.is_alive()          # old worker was joined / finished
    assert q2 is not q1
    assert ev2 is not ev1
    assert not ev2.is_set()

    runner2 = _StubRunner(worker_id=2, complete_after=3)
    t2 = h.start(target=run_simulation_loop, args=_loop_args(runner2, q2, ev2))
    t2.join(timeout=5.0)
    assert not t2.is_alive()

    # The fresh queue must contain ONLY worker #2 output (single producer).
    ids2, complete2 = _drain_worker_ids(q2)
    assert ids2 == {2}
    assert complete2 is True
    # Worker #1's output stayed isolated on the OLD queue.
    ids1, _ = _drain_worker_ids(q1)
    assert ids1 == {1}
```

- [ ] **Step 2: Run the tests, expect FAIL.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_server_lifecycle.py -q`
  Expected: all 4 tests ERROR/FAIL at collection/first line with `ImportError: cannot import name '_WorkerHandle' from 'cenop.server.main'` (the class does not exist yet).

- [ ] **Step 3: Minimal implementation.** In `src/cenop/server/main.py`, insert the class at module level between the end of `run_simulation_loop` and the `# Helper functions for testability` comment header that precedes `def _build_landscape_table_rows(landscapes):`. Anchor the edit on the end of `run_simulation_loop` + the comment header + the next def (this comment header MUST be part of the anchor — it sits between the except block and the def in the real file):

  Replace:
```python
    except Exception as e:
        logger.error("Simulation error: %s", e, exc_info=True)
        result_queue.put({"type": "error", "message": str(e)})


# =========================================================================
# Helper functions for testability (defined at module level)
# =========================================================================

def _build_landscape_table_rows(landscapes):
```
  (Note the TWO blank lines between the `result_queue.put(...)` line and the `# ===` comment header — the live `main.py` has two blank lines there; the exact-string Edit will fail if you collapse them to one.)
  with:
```python
    except Exception as e:
        logger.error("Simulation error: %s", e, exc_info=True)
        result_queue.put({"type": "error", "message": str(e)})


class _WorkerHandle:
    """Owns the background simulation worker's thread plus its per-run
    ``stop_event`` and ``result_queue``.

    Every run gets a FRESH ``stop_event`` and ``result_queue``. This makes it
    impossible to re-arm a still-alive previous worker (the old code cleared a
    SHARED event) or to interleave two workers' output on one shared queue.
    Callers must go through :meth:`new_run` before starting a worker.
    """

    def __init__(self):
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.result_queue: queue.Queue = queue.Queue()

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def stop_and_join(self, timeout: float = 5.0) -> bool:
        """Signal the current worker to stop and join it.

        Returns True if no live worker remains afterward (none existed, or it
        stopped within ``timeout``); False if a worker is still alive after the
        join timed out (its reference is kept so a later call can retry).
        """
        if self.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                logger.warning(
                    "Simulation worker did not stop within %.1fs; abandoning "
                    "it on a stale queue",
                    timeout,
                )
                return False
        self.thread = None
        return True

    def new_run(self, timeout: float = 5.0):
        """Stop/join any prior worker and install a FRESH stop_event + queue.

        Fresh objects are installed regardless of whether the old worker
        actually joined: a stubborn old worker keeps writing to its OLD queue
        and observing its OLD (already-set) event, so the new run is isolated.
        Returns the fresh ``(stop_event, result_queue)`` pair.
        """
        self.stop_and_join(timeout)
        self.stop_event = threading.Event()
        self.result_queue = queue.Queue()
        return self.stop_event, self.result_queue

    def start(self, target, args, daemon: bool = True) -> threading.Thread:
        """Start a new worker thread and record it as the current worker."""
        self.thread = threading.Thread(target=target, args=args, daemon=daemon)
        self.thread.start()
        return self.thread

# =========================================================================
# Helper functions for testability (defined at module level)
# =========================================================================

def _build_landscape_table_rows(landscapes):
```

- [ ] **Step 4: Run the tests, expect PASS.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_server_lifecycle.py -q`
  Expected: `4 passed` (3 unit + 1 integration). Then confirm no regression in the fast suite:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/ -q`
  Expected: all fast tests pass (slow tier auto-deselected).

- [ ] **Step 5: Commit** (nested CENOP repo; use `git -C` per repo shell rules):
  `git -C /home/razinka/cenjas/CENOP add src/cenop/server/main.py tests/test_server_lifecycle.py`
  `git -C /home/razinka/cenjas/CENOP commit -m "Add _WorkerHandle: fresh stop_event+queue per run, join old worker (Finding #7)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

### Task 12: Wire start/stop/reset/poll closures to _WorkerHandle (drop shared clear())

> NOTE: main.py is under concurrent edit — cited line numbers reflect the file at review time (2026-07-06) but may drift. The quoted anchor text in each Edit is authoritative; match on it, not on line numbers.

**Files:**
- Modify: `src/cenop/server/main.py` — replace the three shared locals at lines 517-520 with a `_WorkerHandle`; rewrite `start_simulation` (lines 1143-1212); snapshot the queue in `poll_simulation` (before the drain loop at line 1279); rewrite `stop_simulation` (lines 1369-1372) and `reset_simulation` (lines 1377-1386).
- Test: `tests/test_server_lifecycle.py` (append one regression-guard test)

**Interfaces:**
- Consumes: `cenop.server.main._WorkerHandle` (from Task 11); existing `run_simulation_loop`; the shared mutable config lists/locks already in scope (`throttle_value/throttle_lock`, `ticks_per_update_value/ticks_lock`, `trace_enabled_value/trace_length_value/trace_lock`, `skip_viz_value/skip_viz_lock`).
- Produces: none (internal wiring only; the closures now delegate all thread/queue/event lifecycle to `worker`).

**TDD steps:**

- [ ] **Step 1: Write the failing regression guard.** Append to `tests/test_server_lifecycle.py`:

```python
def test_server_closures_use_worker_handle_and_drop_shared_clear():
    """Pin the Finding #7 fix in the Shiny server closures.

    The closures live inside server(input, output, session) and can't be
    invoked without a full reactive session, so guard the fix at the source
    level: the shared-event clear() must be gone and the closures must
    delegate to the fresh-per-run _WorkerHandle.
    """
    import inspect
    import cenop.server.main as main_mod

    src = inspect.getsource(main_mod)

    # The buggy pattern (re-arming a SHARED event) must be gone entirely.
    assert "stop_event.clear()" not in src
    # The server must own a fresh-per-run handle and route lifecycle through it.
    assert "worker = _WorkerHandle()" in src
    assert "worker.new_run(" in src          # start_simulation + reset_simulation
    assert "worker.stop_event.set()" in src  # stop_simulation
    assert "worker.start(" in src            # start_simulation
    assert "result_queue = worker.result_queue" in src  # poll snapshots the queue
```

- [ ] **Step 2: Run it, expect FAIL.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_server_lifecycle.py::test_server_closures_use_worker_handle_and_drop_shared_clear -q`
  Expected: FAIL on `assert "stop_event.clear()" not in src` — the current `start_simulation` still calls `stop_event.clear()` at line 1174 and the server still uses the three shared locals (no `worker = _WorkerHandle()`).

- [ ] **Step 3: Minimal implementation.** Four edits in `src/cenop/server/main.py`:

  **(a) Replace the shared locals (lines 517-520).** Replace:
```python
    # Internal state for background thread management
    sim_thread: threading.Thread | None = None
    stop_event = threading.Event()
    result_queue = queue.Queue()
```
  with:
```python
    # Internal state for background thread management.
    # _WorkerHandle owns the worker thread plus a FRESH stop_event + result_queue
    # per run, so a Stop-then-Start can never re-arm or interleave an old worker.
    worker = _WorkerHandle()
```

  **(b) start_simulation.** Remove the `nonlocal` and the drain+clear block, and start via the handle. First replace (lines 1144-1146):
```python
        """Start the simulation in a background thread."""
        nonlocal sim_thread
        logger.info("start_simulation() TRIGGERED")
```
  with:
```python
        """Start the simulation in a background thread."""
        logger.info("start_simulation() TRIGGERED")
```
  Then replace the drain/clear block (lines 1168-1174):
```python
        # Reset queue and event - use idiomatic pattern to avoid TOCTOU race
        try:
            while True:
                result_queue.get_nowait()
        except queue.Empty:
            pass
        stop_event.clear()
```
  with:
```python
        # Stop/join any prior worker and install a FRESH stop_event + result_queue.
        # Never clear() a shared event: a still-alive old worker may observe it.
        stop_event, result_queue = worker.new_run()
```
  Then replace the thread-start block (lines 1196-1209):
```python
        # Start background thread
        sim_thread = threading.Thread(
            target=run_simulation_loop,
            args=(
                runner, result_queue, stop_event,
                throttle_value, throttle_lock,
                ticks_per_update_value, ticks_lock,
                trace_enabled_value, trace_length_value, trace_lock,
                skip_viz_value, skip_viz_lock,
            ),
            daemon=True,
        )
        sim_thread.start()
        logger.info("Simulation thread started")
```
  with:
```python
        # Start background thread on the fresh queue/event
        worker.start(
            target=run_simulation_loop,
            args=(
                runner, result_queue, stop_event,
                throttle_value, throttle_lock,
                ticks_per_update_value, ticks_lock,
                trace_enabled_value, trace_length_value, trace_lock,
                skip_viz_value, skip_viz_lock,
            ),
        )
        logger.info("Simulation thread started")
```

  **(c) poll_simulation.** Snapshot the current run's queue so a mid-poll restart cannot swap it. Replace (lines 1279-1282):
```python
        # Drain queue - process all available messages
        while True:
            try:
                msg = result_queue.get_nowait()
```
  with:
```python
        # Snapshot the current run's queue so a mid-poll restart can't swap it.
        result_queue = worker.result_queue
        # Drain queue - process all available messages
        while True:
            try:
                msg = result_queue.get_nowait()
```

  **(d) stop_simulation and reset_simulation.** Replace (lines 1369-1386):
```python
    def stop_simulation():
        """Stop the running simulation."""
        stop_event.set()
        state.running.set(False)

    
    @reactive.effect
    @reactive.event(input.reset_sim)
    def reset_simulation():
        """Reset the simulation."""
        stop_event.set()
        # Clear queue to release refs - use idiomatic pattern to avoid TOCTOU race
        try:
            while True:
                result_queue.get_nowait()
        except queue.Empty:
            pass
        state.reset()
```
  with:
```python
    def stop_simulation():
        """Stop the running simulation."""
        worker.stop_event.set()
        state.running.set(False)

    
    @reactive.effect
    @reactive.event(input.reset_sim)
    def reset_simulation():
        """Reset the simulation."""
        # Stop/join any live worker and install fresh objects before resetting.
        worker.new_run()
        state.reset()
```

- [ ] **Step 4: Run tests, expect PASS.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_server_lifecycle.py -q`
  Expected: `5 passed`. Confirm the module still imports and the full fast suite is green:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -c "import cenop.server.main"`
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/ -q`
  Expected: import prints nothing (success); fast suite all pass.

- [ ] **Step 5: Commit:**
  `git -C /home/razinka/cenjas/CENOP add src/cenop/server/main.py tests/test_server_lifecycle.py`
  `git -C /home/razinka/cenjas/CENOP commit -m "Wire server start/stop/reset/poll to _WorkerHandle; drop shared stop_event.clear() (Finding #7)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`


> **Note (server-thread-race):** Server-lifecycle/threading fix only — NO model math or DEPONS-parity changes, so no Kattegat baseline regeneration needed. The four closures (start/poll/stop/reset_simulation) live inside server(input, output, session) and cannot be invoked without a full Shiny reactive session; Task 12's wiring is therefore pinned by a source-introspection guard (asserts stop_event.clear() is gone and worker.new_run/start/stop_event.set delegation is present), while Task 11's integration test proves the actual runtime behavior by driving the REAL production worker run_simulation_loop through the handle across a Stop-then-Start (fresh queue, old thread joined, no interleaving). Confirmed exact current line numbers against source: shared locals at 499-501, start_simulation 1124-1190 (drain+clear at 1149-1155, Thread start at 1177-1190), poll queue read at 1263, stop_simulation 1341-1344, reset_simulation 1349-1358. All tests are fast (sub-second, small sleeps) so they run under the default `-m "not slow"` addopts. Minor risk: worker.new_run() default join timeout is 5.0s — on a stuck real worker a Stop-then-Start could block the reactive thread up to 5s before swapping to fresh objects (still correct, just a brief UI stall); acceptable and documented in the class docstring.


## Phase 3 — MEDIUM: default-path parity & output fidelity

Includes two DECISION-REQUIRED tasks (see the callout above) — confirm intent before implementing them.


> **DECISION REQUIRED:** Before implementing this task, confirm the intended semantics of the two extra energy terms in the DEPONS energy path. `DEPONSEnergyModule.compute_bmr_cost` / `compute_energy_update` / `depons_bmr_cost_kernel` currently drain, on top of BMR, a **swimming/activity** cost (`current_speed * 0.0001 * scaling`) and a **disturbance** cost (`0.002 * deter_magnitude * scaling`). Authoritative DEPONS has `E_USE_PER_KM = 0.0` (no swimming term) and **no** disturbance energy term, and the headless reference path (`energy_module is None`, `agents/population.py:1921`) is BMR-only and matches DEPONS. So the interactive/server path (which always builds the module) over-drains energy vs DEPONS. **This task is authored for the DEPONS-purity default:** the two terms are gated behind new params — `e_use_per_km` (float, default `0.0`) and `jasmine_disturbance_energy` (bool, default `False`) — so **DEPONS mode == inline headless == DEPONS**, while a JASMINE research configuration can opt in. Confirm this is the desired resolution (restore purity, gate for JASMINE) rather than either (a) keeping the terms unconditionally, or (b) deleting them outright with no opt-in. If confirmed, proceed exactly as below.

### Task 13: Gate DEPONS swimming + disturbance energy terms behind params (restore DEPONS BMR-only purity)

**Files:**
- Modify: `src/cenop/parameters/simulation_params.py` (Energetics block, after line 114 `e_warm`)
- Modify: `src/cenop/physiology/energy_budget.py` (`DEPONSEnergyModule.__init__` lines 289-299; `compute_energy_update` activity/disturbance lines 335-343; `compute_bmr_cost` lines 438-464)
- Modify: `src/cenop/optimizations/kernels.py` (`depons_bmr_cost_kernel` signature+body lines 385-420; `warmup_kernels` call line 873)
- Test: `tests/test_energy_budget.py` (add class `TestBMRCostDEPONSPurity`; rewrite `TestDEPONSEnergyModule::test_disturbance_increases_cost` lines 152-164)
- Test: `tests/test_numba_kernels.py` (update the 4 direct `depons_bmr_cost_kernel` call sites in `TestDEPONSBmrCostKernel` lines 486-489, 507-510, 553-556 and `test_bmr_cost_parallel_deterministic` lines 733-734 to the new signature)

**Interfaces:**
- Consumes: `SimulationParameters.e_use_per_30_min: float`, `SimulationParameters.e_lact: float`, `EnergyContext.current_speed/is_disturbed/deterrence_magnitude/is_lactating: np.ndarray`
- Produces: `SimulationParameters.e_use_per_km: float = 0.0`, `SimulationParameters.jasmine_disturbance_energy: bool = False`; `DEPONSEnergyModule.e_use_per_km`, `DEPONSEnergyModule.jasmine_disturbance_energy` attributes; extended `depons_bmr_cost_kernel(..., e_use_per_30_min, e_lact, e_use_per_km, disturbance_coeff)` signature (2 new trailing float args)

- [ ] **Step 1: Write the failing test.** Append this class to `tests/test_energy_budget.py` (imports `DEPONSEnergyModule`, `EnergyState`, `EnergyContext`, `SimulationParameters` already exist at the top of the file):

```python
class TestBMRCostDEPONSPurity:
    """Finding #10: DEPONS compute_bmr_cost must be BMR-only by default.

    Authoritative DEPONS has E_USE_PER_KM=0.0 (no swimming term) and no
    disturbance energy term. The headless inline reference
    (population._apply_bmr_cost, energy_module is None) is BMR-only:
        total_cost = 0.001 * scaling * e_use_per_30_min
    The module path must match it under DEPONS defaults; the swimming +
    disturbance drains are JASMINE opt-ins gated behind params.
    """

    def _ctx(self, count=8, month=1):
        ctx = EnergyContext.create_default(count, month=month)
        # Non-lactating so BMR carries no e_lact multiplier (matches inline ref).
        ctx.is_lactating[:] = False
        # Nonzero speed + active deterrence: the (pre-fix) activity + disturbance
        # terms would fire here if they were still added unconditionally.
        ctx.current_speed[:] = 2.0
        ctx.is_disturbed[:] = True
        ctx.deterrence_magnitude[:] = 0.5
        return ctx

    def test_depons_default_params_exist(self):
        params = SimulationParameters(porpoise_count=8)
        assert params.e_use_per_km == 0.0
        assert params.jasmine_disturbance_energy is False

    def test_depons_default_is_bmr_only(self):
        params = SimulationParameters(porpoise_count=8)
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(8, initial_energy=10.0)
        ctx = self._ctx(8, month=1)  # winter -> scaling 1.0
        mask = np.ones(8, dtype=bool)

        cost = module.compute_bmr_cost(state, ctx, mask)

        expected_bmr = 0.001 * 1.0 * params.e_use_per_30_min
        np.testing.assert_allclose(cost, expected_bmr, rtol=1e-6)

    def test_matches_inline_headless_reference(self):
        # Inline path (agents/population.py:1921): 0.001 * scaling * e_use_per_30_min
        params = SimulationParameters(porpoise_count=8)
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(8, initial_energy=10.0)
        ctx = self._ctx(8, month=6)  # warm -> scaling e_warm (1.3)
        mask = np.ones(8, dtype=bool)

        cost = module.compute_bmr_cost(state, ctx, mask)

        inline_reference = 0.001 * params.e_warm * params.e_use_per_30_min
        np.testing.assert_allclose(cost, inline_reference, rtol=1e-6)

    def test_jasmine_flags_enable_extra_terms(self):
        params = SimulationParameters(
            porpoise_count=8, e_use_per_km=0.0001, jasmine_disturbance_energy=True
        )
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(8, initial_energy=10.0)
        ctx = self._ctx(8, month=1)
        mask = np.ones(8, dtype=bool)

        cost = module.compute_bmr_cost(state, ctx, mask)

        scaling = 1.0
        bmr = 0.001 * scaling * params.e_use_per_30_min
        activity = 2.0 * 0.0001 * scaling
        disturbance = 0.002 * 0.5 * scaling
        expected = bmr + activity + disturbance
        np.testing.assert_allclose(cost, expected, rtol=1e-5)
        assert float(cost[0]) > bmr + 1e-9

    def test_combined_path_activity_disturbance_zero_by_default(self):
        # Legacy combined path (compute_energy_update) must be gated too, so the
        # two DEPONS paths never diverge.
        params = SimulationParameters(porpoise_count=8)
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(8, initial_energy=10.0)
        ctx = self._ctx(8, month=1)
        mask = np.ones(8, dtype=bool)

        result = module.compute_energy_update(state, ctx, mask)

        assert np.all(result.energy_activity == 0.0)
        assert np.all(result.energy_disturbance == 0.0)
```

- [ ] **Step 2: Run it — expect FAIL.**
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_energy_budget.py::TestBMRCostDEPONSPurity -q`
  Expected failure: `test_depons_default_params_exist` fails on its assertion line with `AttributeError: 'SimulationParameters' object has no attribute 'e_use_per_km'` (the dataclass has no such field yet); `test_jasmine_flags_enable_extra_terms` errors earlier with `TypeError: __init__() got an unexpected keyword argument 'e_use_per_km'` (same missing field, passed as a kwarg); `test_depons_default_is_bmr_only` / `test_matches_inline_headless_reference` FAIL because today `compute_bmr_cost` returns `bmr + speed*0.0001*scaling + 0.002*deter*scaling` (≈0.0045 + 0.0002 + 0.001) instead of BMR-only 0.0045; `test_combined_path_activity_disturbance_zero_by_default` FAILS because `energy_activity`/`energy_disturbance` are currently nonzero when disturbed with speed>0.

- [ ] **Step 3: Add the two params.** In `src/cenop/parameters/simulation_params.py`, in the `# === Energetics ===` block, insert immediately after the `e_warm` line (currently line 114):

```python
    e_warm: float = 1.3                # Warm water energy multiplier
    # Finding #10 — swimming + disturbance drains are JASMINE opt-ins.
    # DEPONS has E_USE_PER_KM=0.0 (no swimming term) and no disturbance energy term.
    e_use_per_km: float = 0.0          # Swimming activity coefficient (JASMINE opt-in, e.g. 0.0001)
    jasmine_disturbance_energy: bool = False  # If True, drain disturbance energy during deterrence
```

- [ ] **Step 4: Gate the terms in the module.** In `src/cenop/physiology/energy_budget.py`, extend `DEPONSEnergyModule.__init__` — after `self.e_warm = params.e_warm` (line 295) add:

```python
        self.e_warm = params.e_warm
        # Finding #10: activity (swimming) + disturbance drain are JASMINE-only.
        self.e_use_per_km = params.e_use_per_km
        self.jasmine_disturbance_energy = params.jasmine_disturbance_energy
```

  In `compute_energy_update`, replace the activity/disturbance block (currently lines 335-343):

```python
            # Activity cost (swimming) — JASMINE opt-in; DEPONS E_USE_PER_KM=0.0
            energy_activity[mask] = context.current_speed[mask] * self.e_use_per_km * scaling

            # Disturbance cost — JASMINE opt-in; DEPONS has no disturbance energy term
            disturbance_coeff = 0.002 if self.jasmine_disturbance_energy else 0.0
            energy_disturbance[mask] = np.where(
                context.is_disturbed[mask],
                disturbance_coeff * context.deterrence_magnitude[mask] * scaling,
                0.0
            ).astype(np.float32)
```

  In `compute_bmr_cost`, replace the kernel-call + fallback body (currently lines 438-464) so the new coefficients flow to both paths:

```python
            disturbance_coeff = 0.002 if self.jasmine_disturbance_energy else 0.0

            try:
                from cenop.optimizations.kernels import depons_bmr_cost_kernel
                full_scaling = np.ones(count, dtype=np.float32)
                full_scaling[mask] = scaling.astype(np.float32)
                depons_bmr_cost_kernel(
                    context.current_speed, full_scaling,
                    context.is_lactating, context.is_disturbed,
                    context.deterrence_magnitude,
                    mask, total_cost,
                    self.e_use_per_30_min, self.e_lact,
                    self.e_use_per_km, disturbance_coeff,
                )
                return total_cost
            except ImportError:
                pass

            bmr = 0.001 * scaling * self.e_use_per_30_min
            bmr = np.where(context.is_lactating[mask], bmr * self.e_lact, bmr)

            activity = context.current_speed[mask] * self.e_use_per_km * scaling

            disturbance = np.where(
                context.is_disturbed[mask],
                disturbance_coeff * context.deterrence_magnitude[mask] * scaling,
                0.0
            ).astype(np.float32)

            total_cost[mask] = bmr + activity + disturbance
```

- [ ] **Step 5: Gate the Numba kernel symmetrically.** In `src/cenop/optimizations/kernels.py`, extend `depons_bmr_cost_kernel`. Add two trailing params to the signature (after `e_lact`, line 394):

```python
    e_use_per_30_min,   # float — BMR parameter
    e_lact,             # float — lactation multiplier
    e_use_per_km,       # float — swimming activity coefficient (0.0 in DEPONS)
    disturbance_coeff,  # float — disturbance energy coefficient (0.0 in DEPONS)
):
```

  and replace the `activity`/`disturbance` lines in the loop body (currently lines 414-418):

```python
        activity = speed[i] * e_use_per_km * scaling[i]

        disturbance = 0.0
        if is_disturbed[i]:
            disturbance = disturbance_coeff * deter_magnitude[i] * scaling[i]
```

  Update the warmup call in `warmup_kernels` (line 873) to pass the two new args:

```python
    depons_bmr_cost_kernel(spd, scl, lac, dis, dmg, msk, cost, 4.5, 1.4, 0.0001, 0.002)
```

- [ ] **Step 6: Run the new class — expect PASS.**
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_energy_budget.py::TestBMRCostDEPONSPurity -q`
  Expected: `5 passed`.

- [ ] **Step 7: Run the whole energy-budget file — expect ONE regression to fix.**
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_energy_budget.py -q`
  Expected failure: `TestDEPONSEnergyModule::test_disturbance_increases_cost` now FAILS at `assert np.all(result2.energy_disturbance > result1.energy_disturbance)` — under DEPONS purity both are `0.0`, so `0 > 0` is False. This test encoded the pre-fix (non-DEPONS) behavior and must be corrected.

- [ ] **Step 8: Correct the stale energy-budget test.** In `tests/test_energy_budget.py`, replace `TestDEPONSEnergyModule::test_disturbance_increases_cost` (currently lines 152-164, note the changed fixture args `params` instead of `module`):

```python
    def test_disturbance_increases_cost(self, params, state, context, mask):
        """DEPONS has no disturbance energy term by default (Finding #10);
        the JASMINE opt-in flag re-enables it."""
        context.is_disturbed[:] = True
        context.deterrence_magnitude[:] = 5.0

        # DEPONS default: disturbance adds nothing.
        depons_module = DEPONSEnergyModule(params)
        result_default = depons_module.compute_energy_update(state, context, mask)
        assert np.all(result_default.energy_disturbance == 0.0)

        # JASMINE opt-in flag: disturbance now drains energy.
        jasmine_params = SimulationParameters(
            porpoise_count=20, jasmine_disturbance_energy=True
        )
        flagged_module = DEPONSEnergyModule(jasmine_params)
        result_flagged = flagged_module.compute_energy_update(state, context, mask)
        assert np.all(result_flagged.energy_disturbance > 0.0)
```

- [ ] **Step 9: Update the direct-kernel tests broken by the new signature.** The Numba kernel now takes 2 extra REQUIRED positional args; four tests in `tests/test_numba_kernels.py` (class `TestDEPONSBmrCostKernel` + `test_bmr_cost_parallel_deterministic`) call it with the old 9-arg form and will raise a numba `TypingError` until updated. Make these exact edits:

  In `test_basic_cost` (currently lines 486-489) — keep activity+disturbance exercised by passing nonzero coefficients:

```python
        depons_bmr_cost_kernel(
            speed, scaling, is_lactating, is_disturbed, deter_magnitude,
            mask, out_cost, 4.5, 1.4, 0.0001, 0.002,
        )
```

  In `test_mask_skips_inactive` (currently lines 507-510):

```python
        depons_bmr_cost_kernel(
            speed, scaling, is_lactating, is_disturbed, deter_magnitude,
            mask, out_cost, 4.5, 1.4, 0.0001, 0.002,
        )
```

  In `test_equivalence_with_python` (currently lines 553-556) — must mirror the module's actual coefficients so it stays equivalent to the now-BMR-only `compute_bmr_cost` under default params (`module.e_use_per_km == 0.0`, `jasmine_disturbance_energy is False`):

```python
        depons_bmr_cost_kernel(
            speed, scaling, is_lact, is_dist, deter_mag,
            mask, nb_cost, module.e_use_per_30_min, module.e_lact,
            module.e_use_per_km,
            0.002 if module.jasmine_disturbance_energy else 0.0,
        )
```

  In `test_bmr_cost_parallel_deterministic` (currently lines 733-734):

```python
        depons_bmr_cost_kernel(speed, scaling, is_lact, is_dist, deter_mag, mask, out1, 4.5, 1.4, 0.0001, 0.002)
        depons_bmr_cost_kernel(speed, scaling, is_lact, is_dist, deter_mag, mask, out2, 4.5, 1.4, 0.0001, 0.002)
```

- [ ] **Step 10: Run energy-budget + numba-kernel files + the DEPONS physiology suite — expect PASS.**
  Command: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_energy_budget.py tests/test_numba_kernels.py tests/test_depons_physiology.py -q`
  Expected: all pass (no failures); `test_numba_kernels.py::TestDEPONSBmrCostKernel::test_equivalence_with_python` passes because both the kernel and the module are BMR-only under default params; the fast `test_depons_physiology.py` cases run, slow ones auto-deselected by `addopts="-m 'not slow'"`.

- [ ] **Step 11: Commit (from within the nested CENOP repo).**
  Command: `git -C /home/razinka/cenjas/CENOP add src/cenop/parameters/simulation_params.py src/cenop/physiology/energy_budget.py src/cenop/optimizations/kernels.py tests/test_energy_budget.py tests/test_numba_kernels.py && git commit -m "fix(energy): gate DEPONS swimming+disturbance drain behind params (restore BMR-only purity)"`
  Commit message body should note: DEPONS default (e_use_per_km=0.0, jasmine_disturbance_energy=False) now matches the headless inline reference; JASMINE can opt in. End the message with the required `Co-Authored-By` trailer.


> **Note (energy-terms):** DECISION-REQUIRED: task authored for DEPONS-purity default (activity + disturbance terms gated OFF via new params e_use_per_km=0.0 and jasmine_disturbance_energy=False); user must confirm this over (a) keep-unconditional or (b) delete-with-no-opt-in before implementing. Kattegat reference baselines are NOT affected: the headless runner uses energy_module=None (inline path population.py:1921) which is already BMR-only and unchanged; the fix only brings the interactive/server module path into line with it. One existing test (TestDEPONSEnergyModule::test_disturbance_increases_cost) encodes the pre-fix behavior and is corrected within this task (Step 8). The JASMINE-class test at test_energy_budget.py:220-230 uses JASMINEEnergyModule (a separate, untouched code path) and is unaffected. Numba kernel signature changes (2 new trailing float args) — cache=True means the first run after the edit recompiles depons_bmr_cost_kernel; warmup call updated to match. If any downstream JASMINE research config previously relied on the DEPONS module's implicit swimming/disturbance drain, it must now set e_use_per_km/jasmine_disturbance_energy explicitly. **THIRD path — JAX:** the JAX kernel `jax_bmr_cost` (jax_kernels.py) applies the SAME `speed*0.0001*scaling` + `0.002*deter*scaling` terms unconditionally. This task gates only the NumPy (`compute_bmr_cost`) and Numba (`depons_bmr_cost_kernel`) paths; to keep the (default-off) JAX DEPONS energy path equal to the corrected reference, the Phase-5 JAX work must extend `jax_bmr_cost` with the same `e_use_per_km`/disturbance params defaulting to 0.0 — noted here so it is not missed.


### Task 14: Make turbine deterrence deterministic by default (DEPONS parity), gate probabilistic scaling behind JASMINE opt-in

> **DECISION REQUIRED:** Confirm before implementing. In `TurbineManager.calculate_aggregate_deterrence_vectorized` (`src/cenop/agents/turbine.py:452`) turbine deterrence strength is currently multiplied by a logistic response probability whenever `params.deter_probabilistic` is `True` — and `True` is the current default, so **production runs (and the committed Kattegat turbine baseline) were generated with probabilistic scaling ON**. Authoritative DEPONS (`Porpoise.java` `deterPorpoise`) applies the **full** turbine deterrence strength (`RL − threshold`) deterministically once `RL` exceeds the threshold; **only ships** draw a Bernoulli reaction. This plan treats the probabilistic turbine scaling as JASMINE drift and makes the default **DEPONS-pure deterministic** by flipping `SimulationParameters.deter_probabilistic` default `True → False`, leaving the existing gate intact so JASMINE can re-enable probabilistic turbine scaling with `SimulationParameters(deter_probabilistic=True)`. You must confirm: (a) deterministic turbine deterrence is the intended default (DEPONS parity), and (b) the `deter_probabilistic=True` opt-in is acceptable for JASMINE. **Consequence to acknowledge:** the committed Kattegat turbine baseline (`output/kattegat_ref_turbines/`, generated via `scripts/run_kattegat_reference.py --turbines`) must be regenerated after this change, and the slow tier (`pytest tests/ -m slow`) must be re-run before release since deterministic strengths are larger than the previously-scaled ones. If instead JASMINE-probabilistic-by-default is desired, do NOT implement this task.

**Files:**
- Modify: `src/cenop/parameters/simulation_params.py` (lines 131-133 — flip `deter_probabilistic` default + rewrite comment)
- Modify: `src/cenop/agents/turbine.py` (lines 451-452 — tighten the code comment to document DEPONS parity + JASMINE opt-in; no logic change — the existing `else: p = None` branch already delivers deterministic behavior)
- Test: `tests/test_depons_deterrence.py` (add new class `TestTurbineDeterministicDeterrence` at end of file)

**Interfaces:**
- Consumes: `cenop.parameters.simulation_params.SimulationParameters` (fields `deter_probabilistic: bool`, `deter_response_slope: float`, `beta_hat`, `alpha_hat`, `deter_threshold`, `deter_coeff`, `deter_max_distance`); `cenop.agents.turbine.Turbine`, `cenop.agents.turbine.TurbineManager`, `cenop.agents.turbine.TurbinePhase`; `TurbineManager.calculate_aggregate_deterrence_vectorized(porpoise_x, porpoise_y, params, cell_size=400.0) -> Tuple[np.ndarray, np.ndarray]`.
- Produces: `SimulationParameters.deter_probabilistic` default changed to `False` (DEPONS-parity deterministic turbine deterrence). No signature changes. Behavior change: with default params, in-range turbine deterrence returns full un-scaled strength `(RL − threshold)`.

TDD steps:

- [ ] **Step 1: Write the failing tests.** Append this class to the end of `tests/test_depons_deterrence.py` (the file already imports `numpy as np`, `pytest`, `SimulationParameters`, and `from cenop.agents.turbine import Turbine, TurbinePhase`; `TurbineManager` is imported locally inside the helper below):

```python
class TestTurbineDeterministicDeterrence:
    """DEPONS parity: turbine deterrence applies FULL strength deterministically.

    DEPONS (Porpoise.java deterPorpoise) applies the full turbine deterrence
    strength (RL - threshold) whenever RL exceeds the threshold; only SHIPS draw a
    Bernoulli reaction. CENOP previously scaled turbine strength by a logistic
    response probability whenever params.deter_probabilistic was True (the old default),
    diverging from DEPONS. The default is now DEPONS-pure (deterministic); JASMINE can
    opt back into probabilistic scaling via SimulationParameters(deter_probabilistic=True).
    """

    def _make_manager(self):
        from cenop.agents.turbine import Turbine, TurbineManager, TurbinePhase

        t = Turbine(id=0, x=50.0, y=50.0, impact=200.0, phase=TurbinePhase.CONSTRUCTION)
        t._is_active = True
        mgr = TurbineManager([t])
        mgr.phase = TurbinePhase.CONSTRUCTION
        return mgr, t

    def _expected_deterministic(self, t, px, py, params, cell_size=400.0):
        # DEPONS deterministic turbine vector: raw displacement * (RL - threshold) * coeff
        dx_m = (px - t.x) * cell_size
        dy_m = (py - t.y) * cell_size
        dist_m = np.maximum(np.hypot(dx_m, dy_m), 1.0)
        tl = params.beta_hat * np.log10(dist_m) + params.alpha_hat * dist_m
        rl = t.get_source_level() - tl
        strength = np.where(rl - params.deter_threshold > 0, rl - params.deter_threshold, 0.0)
        return dx_m * strength * params.deter_coeff, dy_m * strength * params.deter_coeff

    def test_default_turbine_deterrence_is_deterministic(self):
        """With DEPONS defaults an in-range turbine yields FULL, un-scaled strength."""
        params = SimulationParameters()  # default must be DEPONS parity (deterministic)
        mgr, t = self._make_manager()

        # Porpoise 3 cells (1200 m) east of the turbine: in range, strength > 0.
        # RL ~= 154.35 dB, strength ~= 2.350, deterministic dx ~= 33.85;
        # probabilistic scaling (p ~= 0.615) would instead give dx ~= 20.83.
        px = np.array([53.0])
        py = np.array([50.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, params, cell_size=400.0)

        exp_dx, exp_dy = self._expected_deterministic(t, px, py, params)
        assert exp_dx[0] > 0.0, "sanity: porpoise must be in range with positive strength"
        np.testing.assert_allclose(dx, exp_dx, rtol=1e-6)
        np.testing.assert_allclose(dy, exp_dy, rtol=1e-6)

    def test_jasmine_can_opt_into_probabilistic_turbine_scaling(self):
        """JASMINE opt-in: deter_probabilistic=True attenuates strength by response prob (p<1)."""
        mgr, t = self._make_manager()
        px = np.array([53.0])
        py = np.array([50.0])

        det = SimulationParameters()                          # deterministic (DEPONS)
        prob = SimulationParameters(deter_probabilistic=True)  # JASMINE opt-in

        dx_det, _ = mgr.calculate_aggregate_deterrence_vectorized(px, py, det, cell_size=400.0)
        dx_prob, _ = mgr.calculate_aggregate_deterrence_vectorized(px, py, prob, cell_size=400.0)

        # Same push direction (east, away from turbine) but probabilistic path is smaller.
        assert dx_det[0] > 0.0 and dx_prob[0] > 0.0
        assert dx_prob[0] < dx_det[0], "probabilistic scaling (p<1) must reduce magnitude"
```

- [ ] **Step 2: Run the new tests, expect FAIL.** Command:

  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_depons_deterrence.py::TestTurbineDeterministicDeterrence -q`

  Expected failure (today `deter_probabilistic` defaults `True` so both tests exercise the probabilistic branch):
  - `test_default_turbine_deterrence_is_deterministic` fails at `np.testing.assert_allclose(dx, exp_dx, ...)` — actual `dx ≈ 20.83` (scaled by `p ≈ 0.6154`) vs expected `≈ 33.85` (mismatch ≈ 38%).
  - `test_jasmine_can_opt_into_probabilistic_turbine_scaling` fails at `assert dx_prob[0] < dx_det[0]` — with the old default both `det` and `prob` scale identically, so `dx_prob == dx_det` (assert False).

- [ ] **Step 3: Minimal implementation.** Two edits, no control-flow change.

  Edit 1 — `src/cenop/parameters/simulation_params.py`, replace the block at lines 131-133:

  ```python
      # Probabilistic deterrence response
      deter_probabilistic: bool = True  # Use sigmoid-based probability instead of binary threshold
      deter_response_slope: float = 0.2  # Steepness (per dB) of logistic response function
  ```

  with (flip default `True -> False`; the field is read only by the turbine deterrence path — sole reader is `turbine.py:452`):

  ```python
      # Probabilistic turbine deterrence response (JASMINE extension).
      # DEPONS applies FULL turbine deterrence strength deterministically once RL exceeds
      # the threshold (only ships draw a Bernoulli reaction), so the default is DEPONS-pure.
      # Set True to opt into logistic response-probability scaling of turbine strength (JASMINE).
      deter_probabilistic: bool = False
      deter_response_slope: float = 0.2  # Steepness (per dB) of logistic turbine response
  ```

  Edit 2 — `src/cenop/agents/turbine.py`, replace the comment at line 451 (keep the `if params.deter_probabilistic:` gate and the whole `if/else` untouched — the existing `else: p = None` branch already produces deterministic full strength):

  ```python
              # If probabilistic response enabled, compute probability and scale strength
              if params.deter_probabilistic:
  ```

  with:

  ```python
              # DEPONS parity: turbine deterrence is deterministic — full strength once
              # RL > threshold (only ships draw a Bernoulli reaction). JASMINE may opt into
              # logistic response-probability scaling via params.deter_probabilistic (default False).
              if params.deter_probabilistic:
  ```

- [ ] **Step 4: Run tests, expect PASS.** Run the new class, then the two existing deterrence suites to confirm no regression (the existing `test_no_normalization_vectorized_turbine` sets `deter_probabilistic = False` explicitly and stays green; the slow-tier `test_validation.py::...::test_deterrence_vector_magnitude` and `test_integration.py` turbine assertions only check `magnitude > 0` / `dx >= 0`, which deterministic-larger strengths still satisfy):

  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_depons_deterrence.py tests/test_deterrence.py tests/test_integration.py -q`

  Expected: all pass (including the 2 new tests). If you want to sanity-check the slow validation turbine test too: `micromamba run -n shiny python3 -m pytest "tests/test_validation.py::TestDeterrenceValidation::test_deterrence_vector_magnitude" -m "slow or not slow" -q` (note the `-m "slow or not slow"` selector is required for slow nodeids).

- [ ] **Step 5: Commit the src/tests change (do NOT regenerate the baseline here).** This change invalidates the Kattegat turbine baseline, but so do other Phase-3 fixes (energy terms) — per the Global Constraints, the turbine baseline is regenerated ONCE at the end of Phase 3, not per-task (regenerating here would burn a ~2-year run that the next fix immediately re-invalidates and commit a misleading interim artifact). So commit only the code + tests. Stage the specific files (NOT `git add -A`, which would sweep the stale baseline):

  `git -C /home/razinka/cenjas/CENOP add src/cenop/agents/turbine.py src/cenop/parameters/simulation_params.py tests/test_depons_deterrence.py`

  Write the commit message to a file (use the Write tool, not an inline heredoc/echo — repo shell hooks block heredocs/`$()`) at `/tmp/claude-1000/-home-razinka-cenjas/a92d6051-339e-40c9-ac93-a0b68673a463/scratchpad/turbine_deter_commit_msg.txt` with exactly this content:

  ```
  fix(deterrence): make turbine deterrence deterministic by default (DEPONS parity)

  DEPONS applies full turbine deterrence strength (RL - threshold) deterministically
  once RL exceeds the threshold; only ships draw a Bernoulli reaction. The vectorized
  turbine path scaled strength by a logistic response probability whenever
  deter_probabilistic was True (the old default), diverging from DEPONS. Flip the
  default to False (DEPONS parity); JASMINE opts into probabilistic turbine scaling
  via SimulationParameters(deter_probabilistic=True). Tests in
  tests/test_depons_deterrence.py. The Kattegat turbine baseline
  (output/kattegat_ref_turbines/) is regenerated at end of Phase 3, not here.

  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```

  Then commit with the message file:

  `git -C /home/razinka/cenjas/CENOP commit -F /tmp/claude-1000/-home-razinka-cenjas/a92d6051-339e-40c9-ac93-a0b68673a463/scratchpad/turbine_deter_commit_msg.txt`


> **Note (turbine-probabilistic):** DECISION-REQUIRED task — must be confirmed by the user before implementation (deterministic turbine deterrence as DEPONS-parity default vs. keeping JASMINE probabilistic). Design choice made: flip existing `SimulationParameters.deter_probabilistic` default True->False (Option B in the finding). Verified the flag's ONLY reader in src/ is turbine.py:452 (grep), so flipping the default affects only the turbine deterrence path and leaves no dead param; ships have a separate Bernoulli reaction path that is unaffected. BASELINE REGEN REQUIRED: committed Kattegat turbine baseline (output/kattegat_ref_turbines/, via scripts/run_kattegat_reference.py --turbines) was generated with scaling ON and must be regenerated at the END OF PHASE 3 (per Global Constraints); Step 5 commits only the code+tests, NOT the baseline. SLOW TIER: no fast test hard-couples to exact turbine magnitudes; slow-tier test_validation.py::test_deterrence_vector_magnitude and test_integration.py turbine tests only assert magnitude>0 / dx>=0 (deterministic strengths are larger, still pass), but run `pytest tests/ -m slow` before release per repo policy (no CI). Existing test_deterrence.py::test_no_normalization_vectorized_turbine sets deter_probabilistic=False explicitly and remains green. No signature/interface changes. Line length stays <=100.


### Task 15: Population manager exposes true per-tick birth/death counters

**Files:**
- Modify: `src/cenop/agents/population.py`
  - `PorpoisePopulation.__init__` — after `self.death_causes` init (~line 186): add 5 counter attributes.
  - `PorpoisePopulation.step()` — top of method after `mask = self.active_mask` (~line 2547): reset all 5 counters (covers NumPy, Cython and JAX, since `_step_jax` is only reached via `step()`).
  - `PorpoisePopulation._check_mortality()` — after `self.death_causes.extend(causes.tolist())` (~line 2007): accumulate per-cause death counts from the `causes` array.
  - Cython death-recording block — after `self.death_causes.extend(["starvation"] * len(dead_idx))` (~line 2642): accumulate starvation deaths.
  - `PorpoisePopulation._update_pregnancy_status()` — inside `if slots_to_use > 0:` weaning branch (~line 2138): accumulate births.
  - `PorpoisePopulation._step_jax()` — after `np.copyto(self.active_mask, np.asarray(new_active_mask))` (~line 2449): accumulate total deaths from the active-mask delta (no per-cause split — JAX is non-production and never populated `death_causes`).
- Test: `tests/test_simulation.py` — new class `TestPerTickPopulationCounters`.

**Interfaces:**
- Consumes: `PorpoisePopulation._check_mortality` `causes` ndarray (values `"starvation"`/`"old_age"`/`"bycatch"`); `_update_pregnancy_status` weaning `slots_to_use:int`; Cython `dead_idx` array.
- Produces: `PorpoisePopulation.last_step_births:int`, `.last_step_deaths:int`, `.last_step_deaths_starvation:int`, `.last_step_deaths_old_age:int`, `.last_step_deaths_bycatch:int` — all reset to 0 at the start of every `step()`, incremented in-tick; per-cause counts partition `last_step_deaths` exactly on the NumPy path.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_simulation.py`:
```python
class TestPerTickPopulationCounters:
    """Finding #11: population manager exposes true per-tick birth/death counts."""

    def _make_pop(self, count, food_prob=0.0):
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape

        params = SimulationParameters(random_seed=7)
        landscape = create_homogeneous_landscape(
            width=100, height=100, food_prob=food_prob
        )
        return PorpoisePopulation(count=count, params=params, landscape=landscape)

    def test_counters_initialized_to_zero(self):
        pop = self._make_pop(count=20)
        for attr in (
            "last_step_births",
            "last_step_deaths",
            "last_step_deaths_starvation",
            "last_step_deaths_old_age",
            "last_step_deaths_bycatch",
        ):
            assert getattr(pop, attr) == 0, attr

    def test_starvation_death_sets_per_cause_counter(self):
        pop = self._make_pop(count=40, food_prob=0.0)
        # No food available + zero energy => deterministic starvation this tick.
        pop.energy[:12] = 0.0
        zeros = (
            np.zeros(pop.count, dtype=np.float32),
            np.zeros(pop.count, dtype=np.float32),
        )
        pop.step(
            deterrence_vectors=zeros,
            ambient_rl=np.zeros(pop.count, dtype=np.float32),
        )
        assert pop.last_step_deaths >= 1
        assert pop.last_step_deaths_starvation >= 1
        # Tick 1 is not a day boundary => old-age/bycatch cannot fire.
        assert pop.last_step_deaths_old_age == 0
        assert pop.last_step_deaths_bycatch == 0
        # Per-cause counts partition the total deaths exactly.
        assert (
            pop.last_step_deaths_starvation
            + pop.last_step_deaths_old_age
            + pop.last_step_deaths_bycatch
        ) == pop.last_step_deaths

    def test_weaning_birth_increments_birth_counter(self):
        pop = self._make_pop(count=6, food_prob=0.0)
        # Free one slot so the weaned calf has a slot to occupy.
        pop.active_mask[5] = False
        pop._active_idx = np.flatnonzero(pop.active_mask)
        # One active female exactly at the weaning boundary; suppress every other
        # reproduction event so calf_roll is the ONLY random draw.
        pop.pregnancy_status[:] = 0
        pop.days_since_mating[:] = -99
        pop.mating_day[:] = -99  # no female is "ready" => no conceive draw
        pop.is_female[0] = True
        pop.with_calf[:] = False
        pop.with_calf[0] = True
        pop.days_since_birth[0] = pop.params.nursing_time
        pop.pregnancy_status[0] = 2  # ready-to-mate: not pregnant, no give-birth

        class _OnesRng:
            def random(self, n):
                return np.ones(n)  # calf_roll = 1.0 > 0.5 => calf created

            def normal(self, mean, sd, n):
                # New calf energy is drawn via self.rng.normal(...) in the
                # weaning branch (population.py ~line 2148). Return a
                # deterministic array so this stub covers that call too.
                return np.full(n, mean, dtype=float)

        pop.rng = _OnesRng()
        pop._handle_reproduction(pop.active_mask)
        assert pop.last_step_births == 1
        assert pop.active_mask[5]  # freed slot now holds the new calf
```

- [ ] **Step 2: Run it, expect FAIL.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_simulation.py::TestPerTickPopulationCounters -q`
  Expected: all 3 fail/error with `AttributeError: 'PorpoisePopulation' object has no attribute 'last_step_births'` (the counters do not exist yet).

- [ ] **Step 3: Minimal implementation.**
  (a) In `__init__`, immediately after the `self.death_causes: list = []` line:
```python
        self.death_causes: list = []     # Cause string per death

        # Finding #11: true per-tick birth/death counters, reset each step().
        # Simulation reads these instead of inferring births/deaths from the net
        # population delta (which hides co-occurring births + deaths and never
        # attributes per-cause mortality). death_causes still drives Mortality.txt.
        self.last_step_births = 0
        self.last_step_deaths = 0
        self.last_step_deaths_starvation = 0
        self.last_step_deaths_old_age = 0
        self.last_step_deaths_bycatch = 0
```
  (b) In `step()`, replace the opening `mask = self.active_mask` / early-return with a reset that runs before both the JAX dispatch and the early return:
```python
        mask = self.active_mask

        # Finding #11: reset per-tick counters. This runs before the early return
        # and before the JAX dispatch, so it covers the NumPy, Cython and JAX
        # paths (_step_jax is only reached from here).
        self.last_step_births = 0
        self.last_step_deaths = 0
        self.last_step_deaths_starvation = 0
        self.last_step_deaths_old_age = 0
        self.last_step_deaths_bycatch = 0

        if not mask.any():
            return
```
  (c) In `_check_mortality()`, right after `self.death_causes.extend(causes.tolist())`:
```python
            self.death_causes.extend(causes.tolist())

            # Finding #11: expose true per-cause death counts for this tick.
            # causes already partitions dead agents (starvation > old_age >
            # bycatch priority), so these sum to len(dead_indices) exactly.
            causes_arr = np.asarray(causes)
            self.last_step_deaths += int(dead_indices.size)
            self.last_step_deaths_starvation += int(
                np.count_nonzero(causes_arr == "starvation")
            )
            self.last_step_deaths_old_age += int(
                np.count_nonzero(causes_arr == "old_age")
            )
            self.last_step_deaths_bycatch += int(
                np.count_nonzero(causes_arr == "bycatch")
            )
```
  (d) In the Cython death block, right after `self.death_causes.extend(["starvation"] * len(dead_idx))`:
```python
                    self.death_causes.extend(["starvation"] * len(dead_idx))
                    # Finding #11: Cython path performs starvation-only deaths.
                    self.last_step_deaths += int(len(dead_idx))
                    self.last_step_deaths_starvation += int(len(dead_idx))
```
  (e) In `_update_pregnancy_status()`, as the first statement inside `if slots_to_use > 0:`:
```python
                if slots_to_use > 0:
                    self.last_step_births += int(slots_to_use)
                    new_slots = inactive_slots[:slots_to_use]
```
  (f) In `_step_jax()`, right after `np.copyto(self.active_mask, np.asarray(new_active_mask))`:
```python
        np.copyto(self.active_mask, np.asarray(new_active_mask))
        # Finding #11: JAX mortality happens inside jax_tick_energy with no
        # per-cause split (JAX also never populated death_causes). Count total
        # deaths from the active-mask delta (no births have occurred yet this
        # tick) so totals aren't lost; per-cause attribution is unavailable here.
        _jax_deaths = active_before - int(np.sum(self.active_mask))
        if _jax_deaths > 0:
            self.last_step_deaths += _jax_deaths
```

- [ ] **Step 4: Run tests, expect PASS.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_simulation.py::TestPerTickPopulationCounters -q`
  Expected: `3 passed`. Then guard against regressions:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_reproduction.py tests/test_simulation.py -q`
  Expected: all pass (0 failures).

- [ ] **Step 5: Commit** (from within the nested `CENOP/` repo):
  `git -C /home/razinka/cenjas/CENOP add src/cenop/agents/population.py tests/test_simulation.py && git commit -m "fix(population): expose true per-tick birth/death counters (Finding #11)"`

---

### Task 16: Simulation accumulates true per-tick births/deaths (replace net-delta inference)

**Files:**
- Modify: `src/cenop/core/simulation.py`
  - `Simulation.step()` — replace the net-delta statistics block (~lines 541-549) with a call to a new helper.
  - Add `Simulation._update_population_statistics()` method (insert after `step()` ends, before `_daily_tasks`, ~line 585).
- Test: `tests/test_simulation.py` — new class `TestPerTickSimulationStatistics`.

**Interfaces:**
- Consumes: `PorpoisePopulation.last_step_births/last_step_deaths/last_step_deaths_starvation/last_step_deaths_old_age/last_step_deaths_bycatch` (from prior task) and `population_manager.population_size`.
- Produces: `Simulation._update_population_statistics()`; correctly accumulates `state.births`, `state.deaths`, `state.deaths_starvation`, `state.deaths_old_age`, `state.deaths_bycatch`, and sets `state.population`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_simulation.py`:
```python
class TestPerTickSimulationStatistics:
    """Finding #11: Simulation accumulates true per-tick births/deaths."""

    def test_cooccurring_birth_and_death_not_netted_to_zero(self):
        from cenop.core.simulation import Simulation, SimulationState
        from cenop.parameters import SimulationParameters

        params = SimulationParameters(porpoise_count=20, landscape="Homogeneous")
        sim = Simulation(params)  # auto-initializes for Homogeneous
        sim.state = SimulationState()
        sim.state.population = 100

        class _PM:
            # net delta 0: exactly one birth cancels one death this tick
            population_size = 100
            last_step_births = 1
            last_step_deaths = 1
            last_step_deaths_starvation = 1
            last_step_deaths_old_age = 0
            last_step_deaths_bycatch = 0

        sim.population_manager = _PM()
        sim._update_population_statistics()
        assert sim.state.births == 1  # NOT hidden by the net delta
        assert sim.state.deaths == 1
        assert sim.state.deaths_starvation == 1
        assert sim.state.population == 100

    def test_step_records_starvation_into_state(self):
        from cenop.core.simulation import Simulation
        from cenop.parameters import SimulationParameters

        params = SimulationParameters(
            porpoise_count=30, landscape="Homogeneous", random_seed=11
        )
        sim = Simulation(params)
        pm = sim.population_manager
        # Starve a cohort deterministically: zero food everywhere + zero energy.
        pm.landscape._food_value[:] = 0.0
        pm.energy[:12] = 0.0
        before = sim.state.deaths_starvation
        sim.step()
        assert sim.state.deaths_starvation > before  # per-cause is now populated
        assert sim.state.deaths >= sim.state.deaths_starvation
```

- [ ] **Step 2: Run it, expect FAIL.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_simulation.py::TestPerTickSimulationStatistics -q`
  Expected: `test_cooccurring_birth_and_death_not_netted_to_zero` errors with `AttributeError: 'Simulation' object has no attribute '_update_population_statistics'`; `test_step_records_starvation_into_state` fails on `assert sim.state.deaths_starvation > before` (stays `0` because today's net-delta code never sets per-cause deaths). (This test also depends on the prior task's population-manager counters, which are committed before this task runs.)

- [ ] **Step 3: Minimal implementation.**
  (a) In `Simulation.step()`, replace the existing block:
```python
        # 6. Update Statistics
        current_pop = self.population_manager.population_size
        if current_pop != self.state.population:
            diff = current_pop - self.state.population
            if diff < 0:
                self.state.deaths += abs(diff)
            else:
                self.state.births += diff
            self.state.population = current_pop
```
  with:
```python
        # 6. Update Statistics — read true per-tick counts from the population
        # manager. The old net-delta inference (current - prev) hid co-occurring
        # births and deaths and left deaths_starvation/old_age/bycatch at 0 even
        # though they are written to DEPONS-format output. See Finding #11.
        self._update_population_statistics()
```
  (b) Add the helper method (insert after `step()` returns, before `_daily_tasks`):
```python
    def _update_population_statistics(self) -> None:
        """Accumulate true per-tick births/deaths from the population manager.

        In vectorized mode the scalar per-agent counters never fire (the legacy
        ``_porpoises`` list is empty), so births/deaths and per-cause mortality
        must come from the vectorized population manager's per-tick counters
        rather than from the net population delta. See Finding #11.
        """
        pm = self.population_manager
        self.state.births += int(getattr(pm, "last_step_births", 0))
        self.state.deaths += int(getattr(pm, "last_step_deaths", 0))
        self.state.deaths_starvation += int(
            getattr(pm, "last_step_deaths_starvation", 0)
        )
        self.state.deaths_old_age += int(getattr(pm, "last_step_deaths_old_age", 0))
        self.state.deaths_bycatch += int(getattr(pm, "last_step_deaths_bycatch", 0))
        self.state.population = int(pm.population_size)
```

- [ ] **Step 4: Run tests, expect PASS.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_simulation.py::TestPerTickSimulationStatistics -q`
  Expected: `2 passed`. Then broader regression check:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_simulation.py tests/test_reproduction.py -q`
  Expected: all pass (0 failures).

- [ ] **Step 5: Commit** (from within the nested `CENOP/` repo):
  `git -C /home/razinka/cenjas/CENOP add src/cenop/core/simulation.py tests/test_simulation.py && git commit -m "fix(simulation): accumulate true per-tick births/per-cause deaths (Finding #11)"`


> **Note (birth-death-counts):** Task 2 depends on Task 1 (its end-to-end test needs the population-manager counters). Verified against source: PorpoisePopulation._check_mortality builds a `causes` ndarray (values "starvation"/"old_age"/"bycatch", starvation>old_age>bycatch priority) that already partitions deaths; per-cause sums equal len(dead_indices). Births originate ONLY in _update_pregnancy_status weaning (slots_to_use newly-activated slots). Reset placed at top of step() covers NumPy/Cython/JAX (JAX only reached via step()). JAX path (default off) never populated death_causes and mortality is inside jax_tick_energy — plan counts total JAX deaths from active-mask delta (no per-cause split) to avoid regressing totals; this is a documented non-production limitation, not full parity. Output fidelity only — no model-behavior change, so Kattegat reference baselines are NOT affected and need no regeneration. The NumPy path (not Cython) runs in tests because `communication_enabled` defaults True (disqualifying the Cython fast path at population.py:2582) and `use_jax` defaults False — not because of the homogeneous landscape's land edges. Starvation determinism relies on food_prob=0 (=> _food_value 0 => no food gain) plus energy=0 (=> per-tick survival prob 0); tick 1 is not a day boundary so old_age/bycatch are skipped, making per-cause assertions exact. Birth test drives _handle_reproduction directly with a ones-returning rng stub and all mating_day=-99 so calf_roll is the only random draw. Line numbers are approximate (~) as required; implementer should anchor on the quoted surrounding code.


### Task 17: Exclude paused / no-buoy ships from deterrence candidate set

**Files:**
- Modify `src/cenop/agents/ship.py` — add `Ship.is_deterring` property (insert after `is_active`, currently lines 235-239); add `ShipManager.get_deterring_ships` (insert after `get_active_ships`, currently lines 445-449); switch the scalar oracle loop at line 492 and the vectorized loop at line 551 from `get_active_ships()` to `get_deterring_ships()`.
- Test `tests/test_ship_deterrence_port.py` — append a new test class `TestPausedShipsExcluded` (file currently 1172 lines).

**Interfaces:**
- Consumes: `Ship._is_active` (bool), `Ship.current_buoy_idx` (int, ship.py:212), `Ship.ticks_paused` (int, ship.py:213); `ShipManager.calculate_aggregate_deterrence_vectorized(porpoise_x, porpoise_y, params, ..., _force_u=None) -> Tuple[np.ndarray, np.ndarray]`; `ShipManager.calculate_aggregate_deterrence(porpoise_x, porpoise_y, params, ...) -> Tuple[float, float, float]` (max_magnitude, total_dx, total_dy).
- Produces: `Ship.is_deterring -> bool` (property); `ShipManager.get_deterring_ships() -> List[Ship]`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_ship_deterrence_port.py` (`import numpy as np` is already at the top of the file):
```python
class TestPausedShipsExcluded:
    """DEPONS Ship.deterPorpoise (Ship.java:197-203) returns early when
    currentBuoyIdx < 0 or ticksStillPaused > 0. A paused ship in CENOP stays
    _is_active with _prev == current, so all 30 interpolated sub-positions collapse
    to a stationary point and it would deter for the whole pause unless excluded."""

    def _params(self):
        from cenop.parameters.simulation_params import SimulationParameters
        return SimulationParameters()

    def _ship(self, ticks_paused=0, current_buoy_idx=0):
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True
        s.noise.base_source_level = 200.0
        s.ticks_paused = ticks_paused
        s.current_buoy_idx = current_buoy_idx
        mgr = ShipManager([s]); mgr.enabled = True
        return mgr

    def test_paused_ship_contributes_zero_moving_ship_nonzero(self):
        p = self._params()
        # porpoise 2 cells (800 m) north of the ship: in range (>100 m floor, <10 km cap);
        # SL 200 -> RL 157 dB > Tships 80. _force_u=0.0 forces a reaction, so the result
        # is DETERMINISTIC and the only causal difference is the pause flag.
        px = np.array([50.0]); py = np.array([52.0])

        mgr_moving = self._ship(ticks_paused=0)   # not paused -> deters
        mgr_paused = self._ship(ticks_paused=3)   # paused    -> excluded

        _, dy_moving = mgr_moving.calculate_aggregate_deterrence_vectorized(
            px, py, p, _force_u=0.0)
        _, dy_paused = mgr_paused.calculate_aggregate_deterrence_vectorized(
            px, py, p, _force_u=0.0)

        assert dy_moving[0] != 0.0   # precondition: identical unpaused ship DOES deter
        assert dy_paused[0] == 0.0   # paused ship must be excluded (DEPONS parity)

    def test_no_current_buoy_ship_contributes_zero(self):
        p = self._params()
        px = np.array([50.0]); py = np.array([52.0])
        mgr = self._ship(ticks_paused=0, current_buoy_idx=-1)   # no active buoy
        _, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        assert dy[0] == 0.0

    def test_scalar_oracle_also_excludes_paused(self):
        # The scalar oracle path draws np.random.random() ONCE per processed ship
        # (ship.py:508); it has no _force_u hook. The ship-response probability at this
        # geometry is only ~0.12, so the draw MUST be seeded for a deterministic result.
        # np.random.seed(9) -> first draw 0.0104 < prob, so the ship deterministically
        # reacts. On today's unfixed code the PAUSED ship would react too (it is in
        # get_active_ships()), so the final assert fails pre-fix; after the fix the paused
        # ship is excluded from the loop entirely -> (0, 0, 0) regardless of the RNG.
        p = self._params()
        np.random.seed(9)
        m_move, _, dy_move = self._ship(ticks_paused=0).calculate_aggregate_deterrence(
            50.0, 52.0, p)
        assert m_move > 0.0   # precondition: identical unpaused ship DOES deter

        np.random.seed(9)
        m_pause, _, dy_pause = self._ship(ticks_paused=3).calculate_aggregate_deterrence(
            50.0, 52.0, p)
        assert m_pause == 0.0 and dy_pause == 0.0   # paused excluded in oracle too
```
Both ships share identical geometry (ship at (50,50), porpoise at (50,52), SL=200, `_prev==current` for both); the ONLY causal difference is the pause / buoy flag, so the tests isolate the finding exactly. The two vectorized tests are deterministic via `_force_u=0.0`; the scalar-oracle test is made deterministic via `np.random.seed(9)` because the scalar path has no forced-draw hook.

- [ ] **Step 2: Run the test, expect FAIL.**
```
cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py::TestPausedShipsExcluded -q
```
Expected FAIL (verified empirically against today's code):
- `test_paused_ship_contributes_zero_moving_ship_nonzero` asserts `dy_paused[0] == 0.0`, but on today's code the paused ship is in `get_active_ships()`, its 30 collapsed sub-positions all evaluate at 800 m (SL 200 -> RL 157 dB > Tships 80), react (`_force_u=0`), and sum to `dy_paused[0] == 1.967` -> `AssertionError`.
- `test_no_current_buoy_ship_contributes_zero` fails the same way: `current_buoy_idx=-1` is ignored by `get_active_ships()`, so `dy[0] == 1.967 != 0.0`.
- `test_scalar_oracle_also_excludes_paused` is deterministic via `np.random.seed(9)` (first draw 0.0104 < the ~0.12 reaction prob): on today's code the paused ship reacts -> `m_pause == 26.23`, `dy_pause == 0.066` -> `AssertionError`. (Without the seed this test would silently PASS on buggy code ~88% of the time, since the ship reacts only ~12% of draws — the seed is required for it to capture the defect.)

- [ ] **Step 3: Minimal implementation.** In `src/cenop/agents/ship.py`, add the `is_deterring` property to `Ship` immediately after the `is_active` method (the `return self._is_active` block ending at line 239):
```python
    @property
    def is_deterring(self) -> bool:
        """DEPONS Ship.deterPorpoise gate (Ship.java:197-203): an active ship deters a
        porpoise only when it has a valid current buoy (current_buoy_idx >= 0) and is not
        paused (ticks_paused == 0). A paused ship stays _is_active with _prev == current,
        so without this gate its 30 interpolated sub-positions collapse to a stationary
        point and it would deter for the whole pause."""
        return self._is_active and self.current_buoy_idx >= 0 and self.ticks_paused <= 0
```
Add `get_deterring_ships` to `ShipManager` immediately after `get_active_ships` (the `return [s for s in self.ships if s._is_active]` block ending at line 449):
```python
    def get_deterring_ships(self) -> List[Ship]:
        """Active ships eligible to deter this tick: excludes paused ships and ships with
        no current buoy, mirroring DEPONS Ship.deterPorpoise (Ship.java:197-203)."""
        if not self.enabled:
            return []
        return [s for s in self.ships if s.is_deterring]
```
Switch the scalar oracle loop (currently `for ship in self.get_active_ships():` at line 492):
```python
        for ship in self.get_deterring_ships():
```
Switch the vectorized candidate set (currently `active_ships = self.get_active_ships()` at line 551):
```python
        active_ships = self.get_deterring_ships()
```
Leave `ambient_received_level_at_positions` (ship.py:680) and the ambient-RL loop in `src/cenop/core/simulation.py:520` unchanged — those model emitted noise for communication masking, which a paused ship still produces; DEPONS gates only `deterPorpoise`. Leave the `src/cenop/core/simulation.py:486` `len(get_active_ships()) > 0` skip-guard unchanged — it is only a cheap pre-check; the vectorized method now returns zeros internally (`get_deterring_ships()` -> empty -> early return) when every active ship is paused.

- [ ] **Step 4: Run tests, expect PASS.**
```
cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py tests/test_deterrence.py -q
```
Expected: all pass, including the 3 new `TestPausedShipsExcluded` tests (default ships have `current_buoy_idx=0`, `ticks_paused=0` so `is_deterring == _is_active`, leaving every existing test's ship eligible). Verified: the 75 tests in these two files pass with the fix applied. Then confirm no wider regression:
```
cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/ -q
```
Expected: previously-passing count + 3, zero failures (11 deselected slow, 1 xfailed unchanged).

- [ ] **Step 5: Commit (from within the nested CENOP repo).**
```
git -C /home/razinka/cenjas/CENOP add src/cenop/agents/ship.py tests/test_ship_deterrence_port.py && git commit -m "fix(ship): exclude paused/no-buoy ships from deterrence (DEPONS parity)

Ship deterrence candidate set was get_active_ships() (filters only _is_active).
A paused ship (ticks_paused>0) stays active with _prev==current, so its 30
interpolated sub-positions collapse to a stationary point and it deterred for the
whole pause. Mirror DEPONS Ship.deterPorpoise (Ship.java:197-203): add
Ship.is_deterring + ShipManager.get_deterring_ships gating on current_buoy_idx>=0
and ticks_paused==0; use it in both the vectorized production path and the scalar
oracle.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```


> **Note (paused-ships):** BASELINE REGEN LIKELY: this is a production-behavior change. Real Kattegat ship routes have paused buoys (pause_ticks), so excluding paused ships during their pause will shift ship-deterrence output; the committed ship reference baseline output/kattegat_ref_ships/ (deter_strength counts) will change. Regenerate via scripts/run_kattegat_reference.py --ships and run the slow tier (pytest tests/ -m slow) before merge. Single fix point: ship deterrence is computed once in ship.py and passed to both NumPy and JAX movement paths (simulation.py:489 -> population_manager.step deterrence_vectors), so no JAX-side change is needed. Second gate clause (current_buoy_idx<0) is inert on today's CENOP (index never goes negative) but included to mirror DEPONS and future-proof route/tick_end handling. Scope-limited: ambient-RL masking loops (ship.py ambient_received_level_at_positions, simulation.py:520) intentionally left unchanged — DEPONS gates only deterPorpoise, not sound emission. Existing tests unaffected (default ships: current_buoy_idx=0, ticks_paused=0 => is_deterring==_is_active).


## Phase 4 — LOW: default-path polish

Low-severity display, init-distribution, robustness, and dead-code items.


### Task 18: Extract pure population-stats snapshot builder + renderer helpers

**Files:**
- Create: `src/cenop/server/renderers/population_stats.py`
- Test: `tests/test_population_stats.py` (new)

**Interfaces:**
- Consumes: a live `sim` object exposing `sim.population_manager` (with `.active_mask`, `.age`, `.energy`, `.is_female`, `.with_calf` numpy arrays) and `sim.get_statistics() -> dict`; falls back to `sim.agents_df` (pandas DataFrame with `age`/`energy` columns) when `population_manager` is absent.
- Produces:
  - `build_population_stats_snapshot(sim) -> dict` with keys `"ages": list[float]`, `"energies": list[float]`, `"stats": dict` (an immutable, copied snapshot — no numpy views).
  - `render_age_histogram(snapshot: dict) -> shiny UI element` (`ui.HTML` chart or `ui.Tag` placeholder).
  - `render_energy_histogram(snapshot: dict) -> shiny UI element`.
  - `build_vital_stats_df(snapshot: dict) -> pandas.DataFrame` with columns `["Statistic", "Value"]`.

- [ ] **Step 1: Write the failing test.** Create `tests/test_population_stats.py`:

```python
"""Tests for the population-stats snapshot builder and renderer helpers.

These helpers publish an immutable stats snapshot from the background worker
thread and let the histogram/vital-stats renderers read that snapshot instead of
reaching into the live, concurrently-mutated Simulation (Finding #22 data-race).
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from cenop.server.renderers.population_stats import (
    build_population_stats_snapshot,
    render_age_histogram,
    render_energy_histogram,
    build_vital_stats_df,
)

def _fake_sim_with_pm():
    pm = SimpleNamespace(
        active_mask=np.array([True, True, False, True]),
        age=np.array([2.0, 4.0, 99.0, 6.0], dtype=np.float32),
        energy=np.array([10.0, 12.0, 0.0, 8.0], dtype=np.float32),
        is_female=np.array([True, False, True, True]),
        with_calf=np.array([True, False, False, False]),
    )
    sim = SimpleNamespace(
        population_manager=pm,
        get_statistics=lambda: {
            "tick": 48, "day": 1, "year": 0, "population": 3,
            "births_total": 0, "deaths_total": 0,
        },
    )
    return sim, pm

class TestBuildSnapshot:
    def test_snapshot_extracts_active_ages_and_energies(self):
        sim, _pm = _fake_sim_with_pm()
        snap = build_population_stats_snapshot(sim)
        assert snap["ages"] == [2.0, 4.0, 6.0]
        assert snap["energies"] == [10.0, 12.0, 8.0]

    def test_snapshot_stats_merge_active_aggregates(self):
        sim, _pm = _fake_sim_with_pm()
        stats = build_population_stats_snapshot(sim)["stats"]
        assert stats["population"] == 3
        assert stats["avg_age"] == 4.0
        assert stats["avg_energy"] == 10.0
        assert stats["females"] == 2
        assert stats["with_calf"] == 1

    def test_snapshot_is_decoupled_from_later_mutation(self):
        # The core defect: the snapshot must be an immutable copy taken at one
        # instant. Mutating the live arrays afterwards must NOT change the snapshot.
        sim, pm = _fake_sim_with_pm()
        snap = build_population_stats_snapshot(sim)
        pm.age[:] = 999.0
        pm.energy[:] = -1.0
        pm.active_mask[:] = False
        assert snap["ages"] == [2.0, 4.0, 6.0]
        assert snap["energies"] == [10.0, 12.0, 8.0]
        assert snap["stats"]["avg_age"] == 4.0

    def test_snapshot_agents_df_fallback(self):
        df = pd.DataFrame({"age": [1.0, 3.0], "energy": [5.0, 7.0]})
        sim = SimpleNamespace(
            population_manager=None,
            agents_df=df,
            get_statistics=lambda: {"population": 2},
        )
        snap = build_population_stats_snapshot(sim)
        assert snap["ages"] == [1.0, 3.0]
        assert snap["energies"] == [5.0, 7.0]
        assert snap["stats"]["population"] == 2

    def test_snapshot_empty_population(self):
        pm = SimpleNamespace(
            active_mask=np.zeros(3, dtype=bool),
            age=np.zeros(3, dtype=np.float32),
            energy=np.zeros(3, dtype=np.float32),
            is_female=np.zeros(3, dtype=bool),
            with_calf=np.zeros(3, dtype=bool),
        )
        sim = SimpleNamespace(population_manager=pm,
                              get_statistics=lambda: {"population": 0})
        snap = build_population_stats_snapshot(sim)
        assert snap["ages"] == []
        assert snap["energies"] == []
        assert "avg_age" not in snap["stats"]

class TestRenderHelpers:
    def test_age_histogram_from_snapshot_returns_html(self):
        out = render_age_histogram({"ages": [1.0, 2.0, 3.0], "energies": [], "stats": {}})
        assert type(out).__name__ == "HTML"

    def test_age_histogram_empty_returns_placeholder(self):
        out = render_age_histogram({"ages": [], "energies": [], "stats": {}})
        assert type(out).__name__ == "Tag"

    def test_energy_histogram_from_snapshot_returns_html(self):
        out = render_energy_histogram({"ages": [], "energies": [4.0, 5.0], "stats": {}})
        assert type(out).__name__ == "HTML"

    def test_energy_histogram_empty_returns_placeholder(self):
        out = render_energy_histogram({"ages": [], "energies": [], "stats": {}})
        assert type(out).__name__ == "Tag"

    def test_vital_stats_df_from_snapshot(self):
        df = build_vital_stats_df(
            {"ages": [], "energies": [],
             "stats": {"population": 3, "avg_age": 4.0}}
        )
        assert list(df.columns) == ["Statistic", "Value"]
        rows = {r["Statistic"]: r["Value"] for _, r in df.iterrows()}
        assert rows["population"] == "3"
        assert rows["avg_age"] == "4.00"

    def test_vital_stats_df_empty_stats(self):
        df = build_vital_stats_df({"ages": [], "energies": [], "stats": {}})
        assert df.empty

    def test_helpers_never_touch_a_live_sim(self):
        # Passing a plain dict (no Simulation) must fully work — proves the
        # renderers no longer depend on state.simulation().
        snap = {"ages": [2.0], "energies": [3.0], "stats": {"population": 1}}
        assert type(render_age_histogram(snap)).__name__ == "HTML"
        assert type(render_energy_histogram(snap)).__name__ == "HTML"
        assert not build_vital_stats_df(snap).empty
```

- [ ] **Step 2: Run it — expect FAIL.** Run:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_population_stats.py -q`
  Expected: collection/import error `ModuleNotFoundError: No module named 'cenop.server.renderers.population_stats'` (the module does not exist yet).

- [ ] **Step 3: Minimal implementation.** Create `src/cenop/server/renderers/population_stats.py`:

```python
"""Pure population-statistics snapshot builder and renderer helpers.

Finding #22 (data-race): the age/energy histograms and the vital-stats table
used to read ``pm.active_mask``/``age``/``energy``/``is_female``/``with_calf``
and ``sim.get_statistics()`` directly on the session thread while the background
worker mutated those arrays in place, producing torn (transiently wrong)
displays.  The worker now builds an immutable snapshot with
``build_population_stats_snapshot`` (called on the same thread that steps the
sim, so it is coherent) and publishes it over the result queue.  The renderers
consume that snapshot dict and never touch the live Simulation.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from .chart_helpers import create_histogram_chart, no_data_placeholder

def build_population_stats_snapshot(sim: Any) -> Dict[str, Any]:
    """Build an immutable population-stats snapshot from a live ``sim``.

    Must be called on the worker thread (the one that steps the sim) so the
    numpy reductions and ``.tolist()`` copies see a coherent, non-torn state.
    Returns ``{"ages": list, "energies": list, "stats": dict}`` — plain Python
    containers fully decoupled from the sim's arrays.
    """
    ages: list = []
    energies: list = []
    stats: Dict[str, Any] = {}

    pm = getattr(sim, "population_manager", None)
    active = getattr(pm, "active_mask", None) if pm is not None else None

    if active is not None:
        if hasattr(pm, "age") and np.any(active):
            ages = pm.age[active].tolist()
        if hasattr(pm, "energy") and np.any(active):
            energies = pm.energy[active].tolist()
    elif pm is None:
        df = getattr(sim, "agents_df", None)
        if df is not None and not df.empty:
            if "age" in df.columns:
                ages = df["age"].tolist()
            if "energy" in df.columns:
                energies = df["energy"].tolist()

    try:
        stats = dict(sim.get_statistics())
    except (AttributeError, TypeError, ValueError, KeyError):
        stats = {}

    if active is not None and np.any(active):
        stats["avg_age"] = float(np.mean(pm.age[active]))
        stats["avg_energy"] = float(np.mean(pm.energy[active]))
        stats["females"] = int(np.sum(pm.is_female[active]))
        stats["with_calf"] = int(np.sum(pm.with_calf[active]))

    return {"ages": ages, "energies": energies, "stats": stats}

def render_age_histogram(snapshot: Dict[str, Any]):
    """Render the age-distribution histogram from a stats snapshot dict."""
    ages = (snapshot or {}).get("ages") or []
    if not ages:
        return no_data_placeholder("No age data available.")
    return create_histogram_chart(
        data=ages,
        title="Porpoise Age Distribution",
        x_title="Age (years)",
        y_title="Count",
        x_range=(0, 30),
        nbins=30,
        color="red",
        height=300,
    )

def render_energy_histogram(snapshot: Dict[str, Any]):
    """Render the energy-level histogram from a stats snapshot dict."""
    energies = (snapshot or {}).get("energies") or []
    if not energies:
        return no_data_placeholder("No energy data available.")
    return create_histogram_chart(
        data=energies,
        title="Energy Level Distribution",
        x_title="Energy",
        y_title="Porpoise Count",
        x_range=(0, 20),
        nbins=20,
        color="red",
        height=300,
    )

def build_vital_stats_df(snapshot: Dict[str, Any]) -> pd.DataFrame:
    """Build the vital-stats DataFrame from a stats snapshot dict."""
    stats = (snapshot or {}).get("stats") or {}
    if not stats:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Statistic": k,
                "Value": f"{v:.2f}" if isinstance(v, float) else str(v),
            }
            for k, v in stats.items()
        ]
    )
```

- [ ] **Step 4: Run tests — expect PASS.** Run:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_population_stats.py -q`
  Expected: all tests pass (`12 passed`). The `test_snapshot_is_decoupled_from_later_mutation` and `test_helpers_never_touch_a_live_sim` tests concretely capture the fix.

- [ ] **Step 5: Commit.**
  `git -C /home/razinka/cenjas/CENOP add src/cenop/server/renderers/population_stats.py tests/test_population_stats.py`
  `git -C /home/razinka/cenjas/CENOP commit -m "Add immutable population-stats snapshot builder + pure renderer helpers (Finding #22)"`

### Task 19: Publish population snapshot from worker and read it in renderers

**Files:**
- Modify: `src/cenop/server/reactive_state.py` (add `population_snapshot` field near lines 45-48; reset near lines 80-81)
- Modify: `src/cenop/server/main.py` (import block lines 27-32; worker payload lines 158-238; poll drain lines 1311-1326; renderers `age_histogram` 1869-1905, `energy_histogram` 1907-1943, `vital_stats_table` 1999-2027)
- Test: `tests/test_population_stats.py` (extend with a state test + a worker-payload test)

**Interfaces:**
- Consumes: `build_population_stats_snapshot` (from Task 18); `run_simulation_loop(runner, result_queue, stop_event, throttle_value, throttle_lock, ticks_per_update_value, ticks_lock, trace_enabled_value, trace_length_value, trace_lock, skip_viz_value, skip_viz_lock)` (existing worker in `main.py`).
- Produces: `SimulationState.population_snapshot: reactive.Value` (default `None`); a new `"population_snapshot"` key in the worker's queued `update` dict; the three renderers (`age_histogram`, `energy_histogram`, `vital_stats_table`) now read `state.population_snapshot()` instead of `state.simulation()` / live `pm.*` arrays.

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_population_stats.py`:

```python
import queue
import threading

class TestReactiveStateSnapshot:
    def test_state_has_population_snapshot_default_none(self):
        from shiny import reactive
        from cenop.server.reactive_state import create_state

        s = create_state()
        with reactive.isolate():
            assert s.population_snapshot() is None

    def test_reset_clears_population_snapshot(self):
        from shiny import reactive
        from cenop.server.reactive_state import create_state

        s = create_state()
        s.population_snapshot.set({"ages": [1.0], "energies": [], "stats": {}})
        s.reset()
        with reactive.isolate():
            assert s.population_snapshot() is None

class _FakeRunner:
    """Minimal runner: steps exactly once then signals stop."""

    def __init__(self, sim, stop_event):
        self.sim = sim
        self._stop = stop_event
        self.is_complete = False
        self.max_ticks = 48
        self.tick = 48
        self.progress_percent = 10.0
        self.total_births = 0
        self.total_deaths = 0

    def set_ticks_per_update(self, n):
        pass

    def step_ticks(self):
        self._stop.set()  # stop the loop after this single iteration
        return {"year": 0, "day": 0, "population": 3}

    @property
    def should_update_map(self):
        return True

class TestWorkerPublishesSnapshot:
    def test_worker_update_includes_population_snapshot(self):
        from cenop.server.main import run_simulation_loop

        pm = SimpleNamespace(
            active_mask=np.array([True, True, False, True]),
            age=np.array([2.0, 4.0, 99.0, 6.0], dtype=np.float32),
            energy=np.array([10.0, 12.0, 0.0, 8.0], dtype=np.float32),
            is_female=np.array([True, False, True, True]),
            with_calf=np.array([True, False, False, False]),
        )
        sim = SimpleNamespace(
            population_manager=pm,
            get_statistics=lambda: {"population": 3},
            get_porpoise_positions=lambda: np.empty((0, 7)),
            _cell_data=None,
            params=SimpleNamespace(landscape="Homogeneous"),
            state=SimpleNamespace(year=0),
        )
        stop_event = threading.Event()
        runner = _FakeRunner(sim, stop_event)
        result_queue = queue.Queue()

        run_simulation_loop(
            runner, result_queue, stop_event,
            [1.0], threading.Lock(),        # throttle
            [48], threading.Lock(),         # ticks_per_update
            [False], [7], threading.Lock(), # trace enabled / length / lock
            [False], threading.Lock(),      # skip_viz
        )

        updates = []
        while True:
            try:
                msg = result_queue.get_nowait()
            except queue.Empty:
                break
            if msg.get("type") == "update":
                updates.append(msg)

        assert len(updates) == 1
        snap = updates[0]["population_snapshot"]
        assert snap is not None
        assert snap["ages"] == [2.0, 4.0, 6.0]
        assert snap["energies"] == [10.0, 12.0, 8.0]
        assert snap["stats"]["population"] == 3
```

- [ ] **Step 2: Run it — expect FAIL.** Run:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_population_stats.py -q -k "ReactiveStateSnapshot or WorkerPublishesSnapshot"`
  Expected FAIL: `test_state_has_population_snapshot_default_none` fails with `AttributeError: 'SimulationState' object has no attribute 'population_snapshot'`, and `test_worker_update_includes_population_snapshot` fails with `KeyError: 'population_snapshot'` (worker's `update` dict has no such key today).

- [ ] **Step 3a: Add the reactive value.** In `src/cenop/server/reactive_state.py`, after line 48 (`trail_time` field) add:

```python
    # Latest immutable population-stats snapshot (ages/energies/stats) for the
    # Population-tab renderers — decoupled from the live, worker-mutated sim.
    population_snapshot: reactive.Value = field(default_factory=lambda: reactive.Value(None))
```

  And in `reset()`, after line 81 (`self.porpoise_trails.set([])`) add:

```python
        self.population_snapshot.set(None)
```

- [ ] **Step 3b: Import helpers in main.py.** Replace the import block at `src/cenop/server/main.py` lines 27-32 with:

```python
from .renderers.chart_helpers import (
    create_time_series_chart,
    create_histogram_chart,
    create_svg_chart,
    no_data_placeholder
)
from .renderers.population_stats import (
    build_population_stats_snapshot,
    render_age_histogram,
    render_energy_histogram,
    build_vital_stats_df,
)
```

- [ ] **Step 3c: Build + queue the snapshot in the worker.** In `src/cenop/server/main.py`, immediately before the `update = {` dict (currently line 227), insert:

```python
            # Publish an immutable population-stats snapshot on the same cadence
            # as porpoise_positions (map updates), built on THIS worker thread so
            # it is coherent — the renderers read this instead of the live sim
            # (Finding #22 data-race).
            population_snapshot = None
            if runner.should_update_map and not viz_skipped:
                try:
                    population_snapshot = build_population_stats_snapshot(runner.sim)
                except (ValueError, TypeError, AttributeError, KeyError) as e:
                    logger.warning("Population stats snapshot failed: %s", e)
                    population_snapshot = None

```

  Then add the key to the `update` dict (after the `"trail_time": trail_time_counter,` line, currently line 236):

```python
                "population_snapshot": population_snapshot,
```

- [ ] **Step 3d: Set the snapshot in the poll drain.** In `src/cenop/server/main.py`, inside the `if msg["should_update_map"]:` block, after the `porpoise_trails` handler (currently ending line 1326) add:

```python
                    if msg.get("population_snapshot") is not None:
                        try:
                            state.population_snapshot.set(
                                msg.get("population_snapshot")
                            )
                        except (AttributeError, TypeError) as e:
                            logger.warning(
                                "Could not update population_snapshot: %s", e
                            )
```

- [ ] **Step 3e: Rewrite the three renderers to read the snapshot.** Replace `age_histogram` (lines 1869-1905) with:

```python
    @render.ui
    def age_histogram():
        """Age distribution histogram (reads the immutable worker snapshot)."""
        try:
            snapshot = state.population_snapshot()
            if snapshot is None:
                return no_data_placeholder("Run simulation to see age distribution.")
            return render_age_histogram(snapshot)
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.error("age_histogram error: %s", e, exc_info=True)
            return no_data_placeholder("Error rendering age histogram.")
```

  Replace `energy_histogram` (lines 1907-1943) with:

```python
    @render.ui
    def energy_histogram():
        """Energy level histogram (reads the immutable worker snapshot)."""
        try:
            snapshot = state.population_snapshot()
            if snapshot is None:
                return no_data_placeholder("Run simulation to see energy distribution.")
            return render_energy_histogram(snapshot)
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.error("energy_histogram error: %s", e, exc_info=True)
            return no_data_placeholder("Error rendering energy histogram.")
```

  Replace `vital_stats_table` (lines 1999-2027) with:

```python
    @render.data_frame
    def vital_stats_table():
        # Read the immutable population snapshot published by the worker instead
        # of reaching into the live, concurrently-mutated sim (Finding #22).
        snapshot = state.population_snapshot()
        if snapshot is None:
            return pd.DataFrame()
        try:
            return build_vital_stats_df(snapshot)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Vital stats table rendering failed: %s", e)
            return pd.DataFrame()
```

- [ ] **Step 4: Run tests — expect PASS.** Run the targeted file then the broader server/renderer suite:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_population_stats.py tests/test_map_layers.py -q`
  Expected: all pass (the new state + worker-payload tests now green — `test_population_stats.py` is `15 passed`; `test_map_layers.py` unaffected). Then run the fast suite to confirm no regression:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/ -q`
  Expected: full fast suite passes (slow tests auto-deselected), no new failures.

- [ ] **Step 5: Commit.**
  `git -C /home/razinka/cenjas/CENOP add src/cenop/server/reactive_state.py src/cenop/server/main.py tests/test_population_stats.py`
  `git -C /home/razinka/cenjas/CENOP commit -m "Publish population-stats snapshot from worker; renderers read it not live sim (Finding #22)"`


> **Note (renderer-snapshot):** UI-only fix; no model-behavior change, so NO reference-baseline regeneration is needed and the slow tier is unaffected. Cadence decision: the snapshot is published on the `should_update_map and not viz_skipped` gate (~daily, tick%48==0), mirroring porpoise_positions exactly as the finding requests. Consequence: the three renderers now re-render on `state.population_snapshot()` changes (map cadence) rather than every `population_history()` append — a slight change in refresh frequency, but each refresh is now coherent instead of torn. The fix relies on the fact that `build_population_stats_snapshot` runs on the worker thread (same thread as `sim.step()`), so the copy is taken between steps and is race-free; the immutable dict then crosses the queue to the session thread. Verified in-env: `create_histogram_chart` returns `ui.HTML` (type name "HTML"), `no_data_placeholder` returns `ui.Tag`; reading a `reactive.Value` in tests requires `reactive.isolate()`; `cenop.server.main` and `run_simulation_loop` import cleanly under the pytest conftest shiny_deckgl mock. Line numbers in this task (Task 19) are from the current main.py (2187 lines) and will shift by the ~11 inserted worker lines after Step 3c — locate edits by the quoted anchor code, not absolute line numbers, when applying 3d/3e.


### Task 20: Restore dropped age-1 weight in AGE_DISTRIBUTION_FREQUENCY (DEPONS parity)

**Context (verified against source):**
- CENOP `AGE_DISTRIBUTION_FREQUENCY` (`src/cenop/parameters/demography.py:13-34`) has **311 entries with 54 ones** (verified: `len==311`, `count(1)==54`, `index(2)==135`).
- DEPONS-3.2 `ageDistribution[]` (`DEPONS-3.2/src/dk/au/bios/porpoise/PorpoiseSimBuilder.java:249-258`) has **312 entries with 55 ones** (verified by parsing the actual Java literal: `len==312`, `count(1)==55`, `array[135]==1`).
- Empirical diff (verified): the two arrays are identical for indices 0..134; the **first and only divergence is at index 135** — DEPONS has `1`, CENOP has `2`. All other value counts match exactly. Inserting one `1` at index 135 makes CENOP bit-identical to DEPONS (`patched == depons` → `True`, confirmed against the parsed Java array).
- Concretely this is the young-adults line `src/cenop/parameters/demography.py:22`, currently `15 ones + 5 twos`; it must become `16 ones + 5 twos`.
- Sampled by value/index uniformly at `src/cenop/agents/population.py:491` (`self.age = self.rng.choice(AGE_DISTRIBUTION_FREQUENCY, size=self.count)`, call spans lines 491–494) and `src/cenop/core/simulation.py:240-241` (`np.random.choice(AGE_DISTRIBUTION_FREQUENCY)`). Both pick an element uniformly, so `P(value==1)` is currently `54/311`; must be `55/312` to match DEPONS `nextAgeDistrib(0, ageDistribution.length)`.
- **Repo-specific gotcha (verified):** the PostToolUse hook in `/home/razinka/cenjas/.claude/settings.json` runs `black --quiet --line-length 100` on every edited `.py` file. `demography.py` has no `# fmt` guard and is not excluded, so `black` will otherwise **explode the 312-element literal into ~300 one-int-per-line rows** on save — burying this one-value fix in a huge reformat and violating the repo's minimal-diff convention. The fix below adds `# fmt: off` / `# fmt: on` guards around the literal (verified: `black --check` then reports the file unchanged) so the change stays a surgical 3-line diff.

**Files:**
- Modify: `src/cenop/parameters/demography.py` (add `# fmt: off`/`# fmt: on` guards around the literal; insert one `1` in the young-adults `1…2` transition row)
- Test: `tests/test_demography.py` (Create — no demography test exists today; confirmed absent)

**Interfaces:**
- Consumes: `cenop.parameters.demography.AGE_DISTRIBUTION_FREQUENCY: List[int]`
- Produces: none (data-only fix; same symbol, corrected contents — length 312, 55 ones)

- [ ] **Step 1: Write the failing parity test.** Create `tests/test_demography.py`. It asserts the length, the exact count of age-1 weights, and — as a true differential/parity guard — bit-equality against the DEPONS-3.2 `ageDistribution[]` array transcribed literally from `PorpoiseSimBuilder.java:249-258`.

```python
"""Parity tests for demographic init parameters vs DEPONS-3.2 source.

Guards AGE_DISTRIBUTION_FREQUENCY (src/cenop/parameters/demography.py) against the
authoritative DEPONS ageDistribution[] table in
DEPONS-3.2/src/dk/au/bios/porpoise/PorpoiseSimBuilder.java (lines 249-258).
"""

from collections import Counter

from cenop.parameters.demography import AGE_DISTRIBUTION_FREQUENCY

# Transcribed verbatim from DEPONS-3.2 PorpoiseSimBuilder.java:249-258 (int[] ageDistribution).
DEPONS_AGE_DISTRIBUTION = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13,
    13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 18, 18, 19, 19, 21, 22,
]

def test_age_distribution_length_matches_depons():
    """DEPONS ageDistribution[] has 312 entries; the port must match."""
    assert len(AGE_DISTRIBUTION_FREQUENCY) == 312

def test_age_distribution_age1_weight_count():
    """55 entries equal 1 in DEPONS; one was dropped in the port (was 54)."""
    assert AGE_DISTRIBUTION_FREQUENCY.count(1) == 55

def test_age_distribution_bit_identical_to_depons():
    """Full parity: every index must equal the DEPONS reference table."""
    assert AGE_DISTRIBUTION_FREQUENCY == DEPONS_AGE_DISTRIBUTION

def test_age_distribution_value_histogram_matches_depons():
    """Per-value frequency histogram must match DEPONS exactly."""
    assert Counter(AGE_DISTRIBUTION_FREQUENCY) == Counter(DEPONS_AGE_DISTRIBUTION)
```

- [ ] **Step 2: Run the test — expect FAIL.** Run:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_demography.py -q`
  Expected FAIL. Concrete reasons on today's code: `test_age_distribution_length_matches_depons` fails with `assert 311 == 312`; `test_age_distribution_age1_weight_count` fails with `assert 54 == 55`; `test_age_distribution_bit_identical_to_depons` fails (lists differ starting at index 135: CENOP `2` vs DEPONS `1`); `test_age_distribution_value_histogram_matches_depons` fails (histograms differ only in the count of value `1`: `{1: 54}` vs `{1: 55}`; the count of `0` is `81` in both).

- [ ] **Step 3: Apply the minimal fix (fmt-guarded so the auto-format hook cannot explode the array).** Make these three edits to `src/cenop/parameters/demography.py`, **in this order** — the `# fmt: off` guard must go in first so that when the PostToolUse `black` hook runs after each edit it leaves the 312-element literal untouched (a lone `# fmt: off` disables formatting through EOF until the matching `# fmt: on`):

  **Edit 3a — add `# fmt: off` immediately above the declaration.** Change:
  ```python
  AGE_DISTRIBUTION_FREQUENCY: List[int] = [
  ```
  to:
  ```python
  # fmt: off
  AGE_DISTRIBUTION_FREQUENCY: List[int] = [
  ```

  **Edit 3b — insert the missing `1` in the young-adults transition row** (15 ones + 5 twos → 16 ones + 5 twos). This is the only row that transitions from `1`s to `2`s, so the match is unique. Change this exact line:
  ```python
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
  ```
  to:
  ```python
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
  ```
  This inserts the missing age-1 weight at index 135. Ones on the three young-adult rows become 19 (`0`+19 ones) + 20 + 16 = **55** total.

  **Edit 3c — add `# fmt: on` immediately after the closing bracket.** Change (the closing bracket is the line that is just `]`):
  ```python
      21, 22
  ]
  ```
  to:
  ```python
      21, 22
  ]
  # fmt: on
  ```

  After these edits the file is bit-identical to DEPONS in values, and `black --check --line-length 100` reports it unchanged (verified) — the diff is exactly 3 lines, not a ~300-line reformat.

- [ ] **Step 4: Run the tests — expect PASS.** Run:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_demography.py -q`
  Expected: `4 passed`. Then run the neighbouring suite to confirm no regression in code that samples this table:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_reproduction.py tests/test_demography.py -q`
  Expected: all passed (the array length/contents change introduces no failures). Also confirm the auto-format hook did not reflow the table: `git -C /home/razinka/cenjas/CENOP diff --stat src/cenop/parameters/demography.py` should show a small change (~3–4 lines), not ~300.

- [ ] **Step 5: Commit (from within the nested CENOP repo, branch CENOP-JASMINE).** Run:
  `git -C /home/razinka/cenjas/CENOP add src/cenop/parameters/demography.py tests/test_demography.py`
  then:
  `git -C /home/razinka/cenjas/CENOP commit -m "fix(demography): restore dropped age-1 weight in AGE_DISTRIBUTION_FREQUENCY" -m "Port had 311 entries / 54 ones vs DEPONS-3.2 ageDistribution[] 312 entries / 55 ones (PorpoiseSimBuilder.java:249-258). One age-1 weight was dropped in the young-adults block, biasing P(age=1) to 54/311 instead of 55/312 on the production init path (population.py:491, simulation.py:241). Insert the missing 1 at index 135 so the table is bit-identical to DEPONS. Wrap the literal in fmt:off/fmt:on so the black auto-format hook keeps the hand-laid grid and a minimal diff. Add tests/test_demography.py parity guards." -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`


> **Note (age-table):** Verified empirically: DEPONS ageDistribution[] = 312 entries / 55 ones; CENOP = 311 / 54; sole divergence at index 135 (DEPONS 1, CENOP 2). Inserting one 1 at index 135 makes the arrays bit-identical (confirmed `patched == depons`). Fix is data-only (single line 22 edit) — no interface/signature change. RISK: this alters the age-initialization sampling distribution on the production init path (population.py:491 rng.choice, simulation.py:240-241 np.random.choice), so any golden/reference baseline that fixes a random_seed and depends on initial ages (e.g. Kattegat reference outputs) will shift slightly and may need regeneration; the change moves CENOP TOWARD DEPONS parity, so this is the correct direction. No slow-tier test is required to capture the defect (the parity tests are fast and deterministic), but run `pytest tests/ -m slow` before release per repo policy since model-behavior init changed. Standalone task, no dependencies.


### Task 21: Clamp non-positive length in jomopans_spl (defense-in-depth)

**Files:**
- Modify: `src/cenop/behavior/jomopans_spl.py` (add `_MIN_LENGTH_M` constant after line 70; add clamp after line 91; edit the `math.log10(length_m / L_REF)` call at line 111)
- Test: `tests/test_jomopans.py` (add `import math` at top; add test methods to `TestJomopansSPL`)

**Interfaces:**
- Consumes: none (signature of `jomopans_spl(vessel_class, speed_knots, length_m, band=DEFAULT_BAND)` is unchanged)
- Produces: `_MIN_LENGTH_M: float` module constant in `cenop.behavior.jomopans_spl`; `jomopans_spl` now returns a finite float for any `length_m <= 0.0` (speed != 0) instead of raising `ValueError`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_jomopans.py`. Also add `import math` to the imports block at the top of the file (currently only `import pytest` and the two `from cenop...` lines).

```python
# add to imports at top of tests/test_jomopans.py:
import math

# add inside class TestJomopansSPL:
    def test_non_positive_length_does_not_raise(self):
        """A non-positive length (malformed ship file / ships.json) must NOT crash
        the per-tick source-level call — it is clamped to a positive floor."""
        for bad_len in (0.0, -5.0, -100.0):
            spl = jomopans_spl(VesselClass.CARGO, speed_knots=10.0, length_m=bad_len)
            assert math.isfinite(spl), f"length_m={bad_len} produced non-finite SPL {spl}"

    def test_non_positive_length_clamps_to_minimum(self):
        """A non-positive length is clamped to the 1.0 m floor, so its SPL equals
        the SPL computed at length_m=1.0 (and positive lengths are untouched)."""
        clamped = jomopans_spl(VesselClass.CARGO, speed_knots=10.0, length_m=-5.0)
        floor = jomopans_spl(VesselClass.CARGO, speed_knots=10.0, length_m=1.0)
        assert clamped == floor
        # A valid positive length is NOT altered by the clamp.
        assert jomopans_spl(VesselClass.CARGO, 10.0, 150.0) != floor
```

- [ ] **Step 2: Run it, expect FAIL.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_jomopans.py::TestJomopansSPL::test_non_positive_length_does_not_raise tests/test_jomopans.py::TestJomopansSPL::test_non_positive_length_clamps_to_minimum -q`
  Expected FAIL: `test_non_positive_length_does_not_raise` errors with `ValueError: math domain error` (raised at `jomopans_spl.py:111` from `math.log10(length_m / L_REF)` when `length_m <= 0`), and the clamp test errors the same way. (Verified today: `jomopans_spl(VesselClass.CARGO, 10.0, -5.0)` and `... 0.0` both raise `ValueError: math domain error`.)

- [ ] **Step 3: Minimal implementation.** In `src/cenop/behavior/jomopans_spl.py`, add the floor constant immediately after the `L_REF` definition (line 70):

```python
# Reference length: 300 ft in meters
L_REF = 300.0 / 3.28084

# Minimum vessel length (m). A non-positive length from a malformed ship file / ships.json
# is clamped to this floor so the math.log10(length / L_REF) term below cannot raise a
# "math domain error" and crash the per-tick source-level call (defense-in-depth; load-time
# validation in ShipManager.load_from_json is the primary guard).
_MIN_LENGTH_M = 1.0
```

Then add the clamp right after the zero-speed early return (currently lines 90-91):

```python
    if speed_knots == 0:
        return 0.0

    # Defense-in-depth: clamp a non-positive length to a positive floor before the
    # log10(length_eff / L_REF) term so a bad ship file cannot crash the simulation.
    length_eff = length_m if length_m > 0.0 else _MIN_LENGTH_M
```

Finally change the last term of the `sp` expression (line 111) from `length_m` to `length_eff`:

```python
          + 60.0 * math.log10(speed_knots / d_vc)
          + 20.0 * math.log10(length_eff / L_REF))
```

- [ ] **Step 4: Run tests, expect PASS.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_jomopans.py -q`
  Expected: all `TestJomopansSPL` tests pass (existing 6 + 2 new = 8). The existing `test_spl_increases_with_length` and `test_all_classes_produce_valid_spl` still pass because positive lengths are untouched by the `length_m > 0.0` guard.

- [ ] **Step 5: Commit (from within the nested CENOP repo, branch CENOP-JASMINE).**
  `git -C /home/razinka/cenjas/CENOP add src/cenop/behavior/jomopans_spl.py tests/test_jomopans.py && git -C /home/razinka/cenjas/CENOP commit -m "fix(jomopans): clamp non-positive vessel length to avoid log10 math domain crash"`

### Task 22: Validate vessel length at load time in ShipManager.load_from_json

**Files:**
- Modify: `src/cenop/agents/ship.py` (`load_from_json`, replace `length_m = ship_data.get("length", 100.0)` at line 957 with a numeric + positivity check)
- Test: `tests/test_file_parsers.py` (add `import json` and `import logging` to the imports block at top; add a new `TestShipJsonLengthValidation` class)

**Interfaces:**
- Consumes: `ShipManager.load_from_json(json_file, utm_origin_x=..., utm_origin_y=..., cell_size=400.0)` (signature unchanged); reads the per-ship `length` field from the JSON.
- Produces: `ShipManager.load_from_json` now substitutes the default `100.0` m and logs a `CENOP`-logger `WARNING` when a ship's `length` is non-numeric or `<= 0`, so `Ship.vessel_length` (and hence `ShipNoise.length`) is always positive.

- [ ] **Step 1: Write the failing test.** Add to `tests/test_file_parsers.py`. Add `import json` and `import logging` to the top imports block (file currently imports `pytest`, `numpy as np`, `tempfile`, `os`, plus `from cenop.agents.ship import ShipManager, Route, Buoy`).

```python
# add to imports at top of tests/test_file_parsers.py:
import json
import logging

class TestShipJsonLengthValidation:
    """Ship JSON loader must reject non-positive vessel length (Finding #26)."""

    def _write_json(self, obj) -> str:
        fd, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f)
        return path

    def _ship_json(self, length_val):
        return {
            "routes": [{"name": "r1", "route": [
                {"x": 3976618.0, "y": 3363923.0, "speed": 10.0},
                {"x": 3977018.0, "y": 3364323.0, "speed": 10.0},
            ]}],
            "ships": [{"name": "bad", "type": "Cargo",
                       "length": length_val, "route": "r1", "start": 0}],
        }

    def test_negative_length_substitutes_default_and_warns(self, caplog):
        path = self._write_json(self._ship_json(-5.0))
        try:
            manager = ShipManager()
            with caplog.at_level(logging.WARNING, logger="CENOP"):
                manager.load_from_json(path)
            assert len(manager.ships) == 1
            ship = manager.ships[0]
            # Load-time validation replaces the bad length with the 100 m default.
            assert ship.vessel_length == 100.0
            assert ship.noise.length == 100.0
            assert "length" in caplog.text.lower()
            # And the per-tick source-level call must not raise.
            assert ship.get_source_level() == ship.get_source_level()
        finally:
            os.unlink(path)

    def test_zero_length_substitutes_default(self):
        path = self._write_json(self._ship_json(0.0))
        try:
            manager = ShipManager()
            manager.load_from_json(path)
            assert manager.ships[0].vessel_length == 100.0
        finally:
            os.unlink(path)

    def test_positive_length_preserved(self):
        path = self._write_json(self._ship_json(180.0))
        try:
            manager = ShipManager()
            manager.load_from_json(path)
            assert manager.ships[0].vessel_length == 180.0
        finally:
            os.unlink(path)
```

- [ ] **Step 2: Run it, expect FAIL.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_file_parsers.py::TestShipJsonLengthValidation -q`
  Expected FAIL: `test_negative_length_substitutes_default_and_warns` fails at `assert ship.vessel_length == 100.0` because today `load_from_json` sets `vessel_length` to the raw `-5.0` (no validation); `test_zero_length_substitutes_default` fails the same way (raw `0.0`). `test_positive_length_preserved` already passes (regression guard).

- [ ] **Step 3: Minimal implementation.** In `src/cenop/agents/ship.py`, replace the single line inside `load_from_json` (line 957):

```python
            length_m = ship_data.get("length", 100.0)
```

with a numeric + positivity check:

```python
            length_m = ship_data.get("length", 100.0)
            try:
                length_m = float(length_m)
            except (TypeError, ValueError):
                logger.warning(
                    "Ship '%s': non-numeric length %r — using default 100 m.",
                    name, length_m,
                )
                length_m = 100.0
            if length_m <= 0.0:
                logger.warning(
                    "Ship '%s': non-positive length %s m — using default 100 m.",
                    name, length_m,
                )
                length_m = 100.0
```

- [ ] **Step 4: Run tests, expect PASS.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_file_parsers.py -q`
  Expected: all `TestShipParser` + `TestShipJsonLengthValidation` tests pass. Then run the ship/deterrence suites to confirm no regression in the JSON load path:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_file_parsers.py tests/test_jomopans.py tests/test_ship_deterrence_port.py -q`
  Expected: all pass.

- [ ] **Step 5: Commit (from within the nested CENOP repo, branch CENOP-JASMINE).**
  `git -C /home/razinka/cenjas/CENOP add src/cenop/agents/ship.py tests/test_file_parsers.py && git -C /home/razinka/cenjas/CENOP commit -m "fix(ship): reject non-positive vessel length in load_from_json (Finding #26)"`


> **Note (jomopans-length):** Verified defect: jomopans_spl(VesselClass.CARGO, speed_knots=10.0, length_m<=0) raises ValueError: math domain error at jomopans_spl.py:111 (math.log10(length_m/L_REF)); confirmed for both -5.0 and 0.0. ShipNoise.get_source_level (sound.py:228) calls jomopans_spl on the default path (base_source_level None), invoked per in-range ship per tick, and length flows unvalidated from load_from_json (ship.py:957). Two independent, order-independent tasks: Task 1 (clamp in jomopans_spl) is defense-in-depth; Task 2 (validate-at-load in load_from_json) is the primary guard — its assertion (vessel_length==100.0) is specific to load-time sanitization and does NOT overlap Task 1's clamp, so either can land first. Both fixes are additive guards on malformed input only; real Kattegat ships have positive length so the reference baselines (output/kattegat_ref_ships/) are NOT affected and need no regeneration. All tests are fast (not @pytest.mark.slow) and run in the default suite. Line length stays under 100. Commit from within the nested CENOP git repo (branch CENOP-JASMINE), not the parent cenjas repo. Note: the txt-file loader _load_ships (ship.py:817) already does float(parts[2]) but does not check positivity — out of scope for this finding (which cites the JSON/default JOMOPANS path); could be a follow-up if txt ship files with non-positive length are a concern.


### Task 23: Remove the effect-free blade rAF loop and orphaned server/UI plumbing

**Files:**
- Modify: `src/cenop/ui/layout.py` (delete the `cenop_blade_animation` custom message handler, lines 969-1001)
- Modify: `src/cenop/server/main.py` (lines 1650-1653 and 1685-1688 call-site `client_animated=`/`_safe_input` args; delete `_manage_blade_animation` effect lines 1846-1863)
- Modify: `src/cenop/ui/tabs/dashboard.py` (delete `input_switch("blade_animation", ...)` lines 131-134)
- Test: `tests/test_map_layers.py` (append a new `TestBladeRafLoopRemoved` class)

**Interfaces:**
- Consumes: none
- Produces: none (removes the `cenop_blade_animation` Shiny custom-message channel, the `input.blade_animation` reactive input, and the ~60/s `setProps` render loop). `build_turbine_blade_layer` is now called with positional args only; its signature (including the still-present `client_animated` default param) is UNCHANGED by this task and removed in the next one. Blades still render at `angle=0` (production `rotation` was always 0), so no visual change.

TDD steps:

- [ ] **Step 1: Write the failing structural test.** Append to `tests/test_map_layers.py`:

```python
class TestBladeRafLoopRemoved:
    """The effect-free requestAnimationFrame blade loop and its plumbing are gone."""

    @staticmethod
    def _read_src(relpath):
        import pathlib
        import cenop
        return (pathlib.Path(cenop.__file__).parent / relpath).read_text()

    def test_layout_has_no_blade_raf_loop(self):
        src = self._read_src("ui/layout.py")
        assert "cenop_blade_animation" not in src
        assert "_cenopBladeRotation" not in src
        assert "animateBlades" not in src

    def test_main_does_not_send_or_wire_blade_animation(self):
        src = self._read_src("server/main.py")
        assert "cenop_blade_animation" not in src
        assert "client_animated" not in src
        assert "blade_animation" not in src

    def test_dashboard_has_no_blade_switch(self):
        src = self._read_src("ui/tabs/dashboard.py")
        assert "blade_animation" not in src
```

- [ ] **Step 2: Run the structural test, expect FAIL.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_map_layers.py -q -k "BladeRafLoopRemoved"`
  Expected: 3 failed. `test_layout_has_no_blade_raf_loop` fails because `layout.py:970` still registers `cenop_blade_animation` with `animateBlades`/`_cenopBladeRotation`; `test_main_does_not_send_or_wire_blade_animation` fails because `main.py:1859` still sends `cenop_blade_animation` and lines 1652/1687/1856 still pass `client_animated`; `test_dashboard_has_no_blade_switch` fails because `dashboard.py:132` still defines the switch.

- [ ] **Step 3: Minimal implementation.**
  (a) In `src/cenop/ui/layout.py`, delete the entire `cenop_blade_animation` handler. Replace this exact block:

```
        });

        /* Blade animation handler — replaces generic eval_js */
        Shiny.addCustomMessageHandler('cenop_blade_animation', function(payload) {
            var action = (typeof payload === 'object') ? payload.action : payload;
            if (action === 'start') {
                window._cenopBladeAnimRunning = false;
                window._cenopBladeRotation = window._cenopBladeRotation || 0;
                setTimeout(function() {
                    window._cenopBladeAnimRunning = true;
                    function animateBlades() {
                        if (!window._cenopBladeAnimRunning) return;
                        window._cenopBladeRotation =
                            (window._cenopBladeRotation + 1.5) % 360;
                        var inst = (window.__deckgl_instances || {})['sim_map'];
                        if (inst) {
                            if (inst.lastLayers) {
                                var idx = inst.lastLayers.findIndex(function(l) {
                                    return l.id === 'turbine-blades';
                                });
                                if (idx >= 0) {
                                    inst.overlay.setProps(
                                        {layers: inst.overlay.props.layers}
                                    );
                                }
                            }
                        }
                        requestAnimationFrame(animateBlades);
                    }
                    requestAnimationFrame(animateBlades);
                }, 20);
            } else {
                window._cenopBladeAnimRunning = false;
            }
        });
    })();
```
  with:

```
        });
    })();
```
  (The leading `        });` is the close of the preceding `cenop_legend_update` handler and is preserved.)

  (b) In `src/cenop/server/main.py`, simplify the two call sites. Replace lines 1650-1653:

```python
            animate = _safe_input(input, "blade_animation", True)
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(
                turbine_data, client_animated=animate
            )
```
  with:

```python
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(turbine_data)
```
  and replace lines 1685-1688:

```python
            animate = _safe_input(input, "blade_animation", True)
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(
                updated, client_animated=animate
            )
```
  with:

```python
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(updated)
```

  (c) In `src/cenop/server/main.py`, delete the entire `_manage_blade_animation` effect (lines 1846-1863, including its two decorators). Remove this block:

```python
    @reactive.effect
    @reactive.event(input.blade_animation, state.turbine_load_counter)
    async def _manage_blade_animation():
        """Start or stop client-side blade animation based on toggle."""
        animate = input.blade_animation()
        raw = _layer_cache.get("_turbine_data_raw", [])
        has_operational = any(t.get("phase") == "operational" for t in raw)

        if animate and has_operational:
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(
                raw, client_animated=True
            )
            await _push_dynamic_layers("turbine-blades")
            await session.send_custom_message("cenop_blade_animation", {"action": "start"})
        else:
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(raw, rotation=0)
            await _push_dynamic_layers("turbine-blades")
            await session.send_custom_message("cenop_blade_animation", {"action": "stop"})
```

  (d) In `src/cenop/ui/tabs/dashboard.py`, remove the orphaned switch. Replace the `ui.div` at lines 129-136:

```python
                ui.div(
                    "Spatial Distribution",
                    ui.input_switch(
                        "blade_animation", "Animate blades",
                        value=True,
                    ),
                    style="display: flex; justify-content: space-between; align-items: center; width: 100%;",
                ),
```
  with:

```python
                ui.div(
                    "Spatial Distribution",
                    style="display: flex; justify-content: space-between; align-items: center; width: 100%;",
                ),
```

- [ ] **Step 4: Run structural tests + full map-layers suite + compile the touched UI/server modules, expect PASS.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_map_layers.py -q`
  Expected: all pass (including the 3 `TestBladeRafLoopRemoved` tests; the pre-existing `TestBladeAnimation` tests still pass because `build_turbine_blade_layer` still accepts `client_animated` and the `BLADE_ANIMATION_*` constants still exist — both are removed in the next task).
  Then verify the edited source modules still parse:
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m py_compile src/cenop/server/main.py src/cenop/ui/layout.py src/cenop/ui/tabs/dashboard.py`
  Expected: no output, exit 0 (no leftover reference to `input.blade_animation` or `client_animated` in main.py).

- [ ] **Step 5: Commit (from within the nested CENOP repo).**
  `git -C /home/razinka/cenjas/CENOP add src/cenop/ui/layout.py src/cenop/server/main.py src/cenop/ui/tabs/dashboard.py tests/test_map_layers.py && git commit -m "Remove effect-free blade rAF loop and orphaned blade-animation plumbing"`
  Commit message body line: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

### Task 24: Remove dead blade-animation constants and ineffective client_animated parameter

**Files:**
- Modify: `src/cenop/server/map_layers.py` (lines 288-323 `build_turbine_blade_layer`; delete dead constants lines 326-363 `BLADE_ANIMATION_JS` / `BLADE_ANIMATION_STOP_JS`)
- Test: `tests/test_map_layers.py` (lines 12-22 imports; lines 115-141 `TestBladeAnimation`)

**Interfaces:**
- Consumes: none
- Produces: `build_turbine_blade_layer(turbine_data: list, rotation: float = 0) -> dict` (the `client_animated` keyword is removed). Module-level names `BLADE_ANIMATION_JS` and `BLADE_ANIMATION_STOP_JS` are deleted from `cenop.server.map_layers`. (Safe: the previous task already removed every caller that passed `client_animated`, so no call site breaks.)

TDD steps:

- [ ] **Step 1: Write the failing regression tests.** Append two new tests to `tests/test_map_layers.py` (inside the existing `TestBladeAnimation` class or as new methods). These capture the defect on the current code (the dead constants still exist; the effect-free `client_animated` param still exists):

```python
    def test_no_dead_animation_constants(self):
        """Dead client-side animation constants must not exist on the module."""
        import cenop.server.map_layers as ml
        assert not hasattr(ml, "BLADE_ANIMATION_JS")
        assert not hasattr(ml, "BLADE_ANIMATION_STOP_JS")

    def test_client_animated_param_removed(self):
        """The ineffective client_animated parameter must be gone."""
        import inspect
        from cenop.server.map_layers import build_turbine_blade_layer
        params = inspect.signature(build_turbine_blade_layer).parameters
        assert "client_animated" not in params
```

- [ ] **Step 2: Run the new tests, expect FAIL.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_map_layers.py -q -k "no_dead_animation_constants or client_animated_param_removed"`
  Expected: 2 failed. `test_no_dead_animation_constants` fails with `assert not hasattr(...)` because `BLADE_ANIMATION_JS` is still defined at `map_layers.py:326`; `test_client_animated_param_removed` fails because `client_animated` is still a parameter at `map_layers.py:289` (the previous task did not touch map_layers.py).

- [ ] **Step 3: Minimal implementation.**
  (a) In `src/cenop/server/map_layers.py` replace the function signature + docstring (lines 288-296) so the `client_animated` param and its docstring line are gone:

```python
def build_turbine_blade_layer(turbine_data: list, rotation: float = 0) -> dict:
    """Build turbine blade icon layer with rotation angle.

    Args:
        turbine_data: List of turbine position dicts.
        rotation: Server-side rotation angle applied to operational turbines.
    """
```
  The body is unchanged (the `for t in turbine_data:` loop keeps `t["angle"] = rotation if t.get("phase") == "operational" else 0`).

  (b) Delete the dead constants block entirely (current lines 326-363), i.e. remove both `BLADE_ANIMATION_JS = """ ... """` and `BLADE_ANIMATION_STOP_JS = """ ... """` so the file goes straight from the end of `build_turbine_blade_layer` (the `return icon_layer(...)` block) to `def compute_grid_bounds(...)`. (Any extra blank lines left behind are normalized to two by the black hook.)

  (c) In `tests/test_map_layers.py`, drop the now-invalid imports (lines 19-20) so the top import block reads:

```python
from cenop.server.map_layers import (  # noqa: E402
    build_porpoise_layer,
    build_porpoise_trails_layer,
    build_noise_construction_layer,
    build_noise_operational_layer,
    build_turbine_pole_layer,
    build_turbine_blade_layer,
    GIS_COLOR_SCHEMES,
)
```
  (d) In `tests/test_map_layers.py`, rewrite the `TestBladeAnimation` class body so it no longer references removed symbols. Change `test_server_side_rotation` to drop the `client_animated=False` kwarg, and delete `test_client_side_animation`, `test_client_animated_empty_data`, and `test_animation_js_constants_exist` (they exercise the removed param/constants). Keep the two new tests from Step 1. Final class:

```python
class TestBladeAnimation:
    def test_server_side_rotation(self):
        data = [{"position": [21.0, 55.5], "radius": 300, "phase": "operational"}]
        layer = build_turbine_blade_layer(data, rotation=90.0)
        assert "90.0" in str(layer)
        assert "window._cenopBladeRotation" not in str(layer)

    def test_no_dead_animation_constants(self):
        """Dead client-side animation constants must not exist on the module."""
        import cenop.server.map_layers as ml
        assert not hasattr(ml, "BLADE_ANIMATION_JS")
        assert not hasattr(ml, "BLADE_ANIMATION_STOP_JS")

    def test_client_animated_param_removed(self):
        """The ineffective client_animated parameter must be gone."""
        import inspect
        from cenop.server.map_layers import build_turbine_blade_layer
        params = inspect.signature(build_turbine_blade_layer).parameters
        assert "client_animated" not in params
```

- [ ] **Step 4: Run the full map-layers suite, expect PASS.**
  `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_map_layers.py -q`
  Expected: all tests pass (the module imports cleanly with the stale `BLADE_ANIMATION_*` imports gone; the two regression tests pass; `test_server_side_rotation` still passes since operational turbines get `angle=90.0`, embedded in `str(layer)` via the conftest `icon_layer` mock; the `TestBladeRafLoopRemoved` class from the previous task still passes).

- [ ] **Step 5: Commit (from within the nested CENOP repo).**
  `git -C /home/razinka/cenjas/CENOP add src/cenop/server/map_layers.py tests/test_map_layers.py && git commit -m "Remove dead blade-animation constants and ineffective client_animated param"`
  Commit message body line: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`


> **Note (blade-animation):** Pure UI/perf cleanup — NO src/cenop model-behavior change, so NO Kattegat baseline regeneration and NO slow-tier run required. Finding #30 is LOW/cosmetic; the recommended REMOVAL path is used (blades were already always static at angle=0 in production since rotation is always 0 and client_animated was never read, so removing the machinery changes nothing visually while eliminating a ~60/s setProps CPU burn). Verified against source: client_animated is never referenced in build_turbine_blade_layer; BLADE_ANIMATION_JS/STOP are imported only by the test file; the live rAF loop is the duplicate cenop_blade_animation handler in layout.py (the map_layers.py constants are a second, fully-dead copy). Task 2 also removes the now-orphaned input.blade_animation switch (dashboard.py) and _manage_blade_animation effect (main.py) — grep confirms input.blade_animation has no other consumers. Tests are structural/asset assertions (inspect.signature + hasattr + source-text scan via cenop.__file__ path), exactly the light check the finding calls for; import cenop is cheap (core-only __init__) so the source-read tests avoid importing heavy shiny UI modules. Task 1 must edit map_layers.py AND test_map_layers.py in the same impl step because the test file's top-level import of BLADE_ANIMATION_JS/STOP would otherwise break module import; the two new regression tests are the guards. Line length 100 respected. Commit from within CENOP/ (nested git repo).


## Phase 5 — JAX backend parity (Track B: repair)

Bring the opt-in JAX backend to bit-parity with the validated NumPy/Numba reference.


### Task 25: JAX dispersal heading — match NumPy reference (Finding #15)

**Files:**
- Modify `src/cenop/optimizations/jax_kernels.py` (`jax_heading_composition`: signature ~lines 263–284; dispersal-override block lines 320–349)
- Modify `src/cenop/optimizations/tick_jax.py` (`jax_tick_movement`: signature lines 68–95; RNG split ~line 153; `jax_heading_composition` call lines 165–186)
- Modify `src/cenop/agents/population.py` (`_step_jax` → `jax_tick_movement` call, lines 2286–2331)
- Test `tests/test_jax_tick.py` (`_make_heading_inputs` lines 537–561; existing `test_tick_movement_returns_valid_positions` call lines 1341–1365; two new tests)

**Interfaces:**
- Consumes: `x, y, dispersal_start_x, dispersal_start_y` (float64[n]), `dispersal_target_distance` (float64[n]), `prev_step_heading` (float64[n]), `psm_angle`/`psm_log` (float scalars = `getattr(params,'psm_type2_random_angle',20.0)` / `getattr(params,'psm_log',0.6)`), `disp_key` (jax PRNGKey).
- Produces: `jax_heading_composition(...)` new signature ending `..., inertia_const, mean_disp_dist, dispersal_start_x, dispersal_start_y, psm_angle, psm_log, disp_key`; `jax_tick_movement(...)` gains `dispersal_start_x, dispersal_start_y` (after `prev_step_heading`) and `psm_angle, psm_log` (after `mean_disp_dist`). Dispersal heading now = `prev_step_heading + U(-psm_angle,psm_angle)*SSLogis(3*distPct-1.5, 1, 0, psm_log)`, `distPct` from distance TRAVELLED from `dispersal_start`.

Note: Edits are string-anchored (unique anchors verified), so the exact line numbers may drift as sibling tasks land; the anchor text still matches.

- [ ] **Step 1: Write the failing tests.** Add to `tests/test_jax_tick.py`. First test (kernel-level parity) goes in class `TestJaxHeadingAndPosition`; second (composed-level threading) in class `TestJaxTickComposition`.

```python
    # --- add inside class TestJaxHeadingAndPosition ---
    def test_dispersal_heading_matches_reference(self):
        """JAX dispersal heading matches NumPy _apply_dispersal_heading formula.

        Reference (population.py _apply_dispersal_heading / DispersalPSMType2.java):
            distPct  = dist_travelled_from_start / dispersal_target_distance
            distLogX = 3*distPct - 1.5
            logistic = 1 / (1 + exp((0 - distLogX)/psm_log))   # SSLogis phi3 = psm_log
            delta    = U(-psm_angle, +psm_angle) * logistic
            heading  = prev_step_heading + delta               (mod 360)
        The old kernel used max_angle=120, distance-to-TARGET, phi3=1.0 and a
        deterministic sign-toward-target turn — all wrong.
        """
        from cenop.optimizations.jax_kernels import jax_heading_composition

        n = 4000
        psm_angle, psm_log, prev = 20.0, 0.6, 100.0
        inputs = _make_heading_inputs(n=n, seed=21)
        inputs["is_dispersing"] = jnp.ones(n, dtype=bool)
        inputs["mask"] = jnp.ones(n, dtype=bool)
        inputs["prev_step_heading"] = jnp.full(n, prev, dtype=jnp.float64)
        inputs["psm_angle"] = psm_angle
        inputs["psm_log"] = psm_log
        inputs["disp_key"] = jax.random.PRNGKey(7)
        # Travelled 100 of a 100-cell dispersal from start (0,0) -> distPct = 1.0.
        inputs["dispersal_start_x"] = jnp.zeros(n, dtype=jnp.float64)
        inputs["dispersal_start_y"] = jnp.zeros(n, dtype=jnp.float64)
        inputs["x"] = jnp.full(n, 100.0, dtype=jnp.float32)
        inputs["y"] = jnp.zeros(n, dtype=jnp.float32)
        inputs["dispersal_target_distance"] = jnp.full(n, 100.0, dtype=jnp.float64)
        # Target far away: old (distance-to-target) code would steer ~120 deg here.
        inputs["dispersal_target_x"] = jnp.full(n, 5000.0, dtype=jnp.float64)
        inputs["dispersal_target_y"] = jnp.full(n, 5000.0, dtype=jnp.float64)

        new_heading = np.asarray(jax_heading_composition(**inputs)[0])

        dist_log_x = 3.0 * 1.0 - 1.5
        logistic = 1.0 / (1.0 + np.exp((0.0 - dist_log_x) / psm_log))  # ~0.9242
        max_dev = psm_angle * logistic                                # ~18.48
        dev = (new_heading - prev + 180.0) % 360.0 - 180.0

        assert np.max(np.abs(dev)) <= max_dev + 1e-4, (
            f"max|dev|={np.max(np.abs(dev)):.3f} > psm_angle*logistic={max_dev:.3f} "
            "(max_angle should be psm_angle, not 120)"
        )
        assert np.min(dev) < -0.5 * max_dev, f"no negative deltas (not random): {np.min(dev):.3f}"
        assert np.max(dev) > 0.5 * max_dev, f"no positive deltas (not random): {np.max(dev):.3f}"
        # Exact scale pins distance-TRAVELLED (18.48) vs distance-to-target (~20) and phi3=0.6.
        np.testing.assert_allclose(np.max(dev), max_dev, atol=0.2)
        np.testing.assert_allclose(np.min(dev), -max_dev, atol=0.2)
        assert abs(np.mean(dev)) < 0.5, f"mean dev {np.mean(dev):.3f} not ~0 (asymmetric turn)"

    # --- add inside class TestJaxTickComposition ---
    def test_tick_movement_threads_dispersal_params(self):
        """Composed jax_tick_movement threads dispersal_start/psm params into the
        heading kernel: dispersing agents' heading stays within psm_angle*logistic."""
        from cenop.optimizations.tick_jax import jax_tick_movement

        n = 200
        world_w = world_h = 300
        mem = 20
        psm_angle, psm_log = 20.0, 0.6
        work_table = jnp.array([np.exp(-i * 0.01) for i in range(mem)], dtype=jnp.float64)
        depth_grid = jnp.full((world_h, world_w), 30.0, dtype=jnp.float32)

        try:
            result = jax_tick_movement(
                jnp.full(n, 150.0, dtype=jnp.float32),          # x
                jnp.full(n, 150.0, dtype=jnp.float32),          # y
                jnp.zeros(n, dtype=jnp.float32),                # heading
                jnp.zeros(n, dtype=jnp.float64),                # prev_angle
                jnp.full(n, 1.0, dtype=jnp.float64),            # prev_log_mov
                jnp.ones(n, dtype=bool),                        # active_mask
                jnp.zeros((n, mem), dtype=jnp.float32),         # stored_util
                jnp.zeros((n, mem), dtype=jnp.float32),         # pos_hist_x
                jnp.zeros((n, mem), dtype=jnp.float32),         # pos_hist_y
                jnp.zeros(n, dtype=jnp.int32),                  # mem_ptr
                jnp.zeros(n, dtype=jnp.int32),                  # mem_count
                work_table,                                     # work_mem_table
                jnp.zeros(n, dtype=jnp.float64),                # deter_dx
                jnp.zeros(n, dtype=jnp.float64),                # deter_dy
                jnp.zeros(n, dtype=jnp.float32),                # social_dx
                jnp.zeros(n, dtype=jnp.float32),                # social_dy
                jnp.ones(n, dtype=bool),                        # is_dispersing
                jnp.full(n, 5000.0, dtype=jnp.float32),         # dispersal_target_x
                jnp.full(n, 5000.0, dtype=jnp.float32),         # dispersal_target_y
                jnp.full(n, 100.0, dtype=jnp.float32),          # dispersal_target_distance
                jnp.zeros(n, dtype=jnp.float32),                # dispersal_distance_traveled
                jnp.zeros(n, dtype=jnp.float32),                # prev_step_heading
                jnp.full(n, 150.0, dtype=jnp.float32),          # dispersal_start_x (== pos -> travelled 0)
                jnp.full(n, 150.0, dtype=jnp.float32),          # dispersal_start_y
                jnp.full(n, 30.0, dtype=jnp.float64),           # depths
                jnp.full(n, 30.0, dtype=jnp.float64),           # salinity
                depth_grid,
                -0.024, -0.008, 0.93, -14.0,
                0.35, 0.0005, -0.02, 1.73,
                0.0, 4.0, 0.0, 0.15,
                0.00001,
                0.001, 2.0,                                     # inertia_const, mean_disp_dist
                psm_angle, psm_log,
                1.0, world_w, world_h,                          # min_depth, world dims
                jax.random.PRNGKey(0),
            )
        except Exception as e:  # noqa: BLE001 - classify GPU OOM as environmental
            if any(k in str(e) for k in ("RESOURCE_EXHAUSTED", "OUT_OF_MEMORY")):
                pytest.skip(f"JAX GPU OOM (environmental): {e}")
            raise

        disp_heading = np.asarray(result[8])  # new_prev_step_heading = dispersal heading
        dist_log_x = 3.0 * 0.0 - 1.5          # travelled 0 -> distPct 0
        logistic = 1.0 / (1.0 + np.exp((0.0 - dist_log_x) / psm_log))  # ~0.0759
        max_dev = psm_angle * logistic                                # ~1.52
        dev = (disp_heading - 0.0 + 180.0) % 360.0 - 180.0
        assert np.max(np.abs(dev)) <= max_dev + 1e-3, f"max|dev|={np.max(np.abs(dev)):.4f} > {max_dev:.4f}"
        assert dev.min() < 0.0 < dev.max(), "dispersal turn is not random (one-sided)"
```

- [ ] **Step 2: Run — expect FAIL.** `cd /home/razinka/cenjas/CENOP && JAX_PLATFORMS=cpu micromamba run -n shiny python3 -m pytest tests/test_jax_tick.py::TestJaxHeadingAndPosition::test_dispersal_heading_matches_reference tests/test_jax_tick.py::TestJaxTickComposition::test_tick_movement_threads_dispersal_params -q`
  Expected: both ERROR/FAIL with `TypeError: jax_heading_composition() got an unexpected keyword argument 'dispersal_start_x'` (kernel test) and `TypeError: jax_tick_movement() takes N positional arguments but M were given` (composed test) — the new signature does not exist yet.

- [ ] **Step 3: Implement.** Five coordinated edits (must land together — the signature change breaks the positional callers until all are updated).

  (3a) `src/cenop/optimizations/jax_kernels.py` — extend the signature. Replace:
```python
    inertia_const,
    mean_disp_dist,
):
```
  with:
```python
    inertia_const,
    mean_disp_dist,
    dispersal_start_x,
    dispersal_start_y,
    psm_angle,
    psm_log,
    disp_key,
):
```

  (3a-doc) `src/cenop/optimizations/jax_kernels.py` — the signature AND dispersal behaviour of `jax_heading_composition` just changed, so update its OWN docstring (allowed — this is a function you are modifying, so it does not violate "no docstrings on unchanged code"). Revise the line that currently reads `Dispersing agents get a SSLogis-based heading override toward their target.` (~line 288) to describe the new PSM-Type2 behaviour — a logistic-scaled **uniform-random** turn from `previous_step_heading` (NOT steered toward the target) — and add Parameters entries for the five new args (`dispersal_start_x`, `dispersal_start_y`, `psm_angle`, `psm_log`, `disp_key`).

  (3b) `src/cenop/optimizations/jax_kernels.py` — replace the dispersal-override block. Replace lines 320–349 (from `    # 2. Dispersal heading override (SSLogis formula)` through `    dispersal_heading = (prev_step_heading + angle_delta * sign_of_turn) % 360.0`) with:
```python
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
```
  (`dispersal_target_x`/`dispersal_target_y` remain in the signature but are now unused by the heading — leave them; other call sites still pass them.)

  (3c) `src/cenop/optimizations/tick_jax.py` — extend `jax_tick_movement` signature. Edit 1, replace:
```python
    dispersal_distance_traveled,
    prev_step_heading,
    # Environment (per-agent)
```
  with:
```python
    dispersal_distance_traveled,
    prev_step_heading,
    dispersal_start_x,
    dispersal_start_y,
    # Environment (per-agent)
```
  Edit 2, replace:
```python
    inertia_const,
    mean_disp_dist,
    min_depth,
```
  with:
```python
    inertia_const,
    mean_disp_dist,
    psm_angle,
    psm_log,
    min_depth,
```
  Edit 3, add the dispersal RNG split — replace:
```python
    vt_x = jnp.where(has_history, vt_x, 0.0)
    vt_y = jnp.where(has_history, vt_y, 0.0)

    # --- Phase 3: Heading composition + position update ---
```
  with:
```python
    vt_x = jnp.where(has_history, vt_x, 0.0)
    vt_y = jnp.where(has_history, vt_y, 0.0)

    # Dispersal random-turn key (PSM-Type2)
    key, disp_key = jax.random.split(key)

    # --- Phase 3: Heading composition + position update ---
```
  Edit 4, pass the new args into the kernel call — replace:
```python
        x.astype(jnp.float64),
        y.astype(jnp.float64),
        inertia_const,
        mean_disp_dist,
    )
```
  with:
```python
        x.astype(jnp.float64),
        y.astype(jnp.float64),
        inertia_const,
        mean_disp_dist,
        dispersal_start_x.astype(jnp.float64),
        dispersal_start_y.astype(jnp.float64),
        psm_angle,
        psm_log,
        disp_key,
    )
```

  (3d) `src/cenop/agents/population.py` — pass the new args from `_step_jax`. Edit 1, replace:
```python
            jnp.asarray(self._prev_step_heading),
            jnp.asarray(self._depths),
```
  with:
```python
            jnp.asarray(self._prev_step_heading),
            jnp.asarray(self.dispersal_start_x),
            jnp.asarray(self.dispersal_start_y),
            jnp.asarray(self._depths),
```
  Edit 2, replace:
```python
            float(self.params.mean_disp_dist),
            float(min_depth),
```
  with:
```python
            float(self.params.mean_disp_dist),
            float(getattr(self.params, 'psm_type2_random_angle', 20.0)),
            float(getattr(self.params, 'psm_log', 0.6)),
            float(min_depth),
```
  (`getattr(...,'psm_type2_random_angle',20.0)` intentionally mirrors the NumPy reference: `SimulationParameters` has no such attribute, so both paths use 20.0 — matching `_apply_dispersal_heading` exactly rather than the unrelated `params.psm_angle`=40.0.)

  (3e) `tests/test_jax_tick.py` — keep the existing callers working. Update `_make_heading_inputs`, replace:
```python
        inertia_const=0.001,
        mean_disp_dist=2.0,
    )
```
  with:
```python
        inertia_const=0.001,
        mean_disp_dist=2.0,
        dispersal_start_x=jnp.zeros(n, dtype=jnp.float64),
        dispersal_start_y=jnp.zeros(n, dtype=jnp.float64),
        psm_angle=20.0,
        psm_log=0.6,
        disp_key=jax.random.PRNGKey(seed),
    )
```
  And update the existing `test_tick_movement_returns_valid_positions` positional call. Edit 1, replace:
```python
            jnp.zeros(n, dtype=jnp.float32),
            jnp.array(rng.uniform(5.0, 50.0, n), dtype=jnp.float64),
```
  with:
```python
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.zeros(n, dtype=jnp.float32),
            jnp.array(rng.uniform(5.0, 50.0, n), dtype=jnp.float64),
```
  Edit 2, replace:
```python
            0.00001,
            0.001, 2.0, 1.0,
            world_w, world_h,
```
  with:
```python
            0.00001,
            0.001, 2.0, 20.0, 0.6, 1.0,
            world_w, world_h,
```

- [ ] **Step 4: Run — expect PASS.** `cd /home/razinka/cenjas/CENOP && JAX_PLATFORMS=cpu micromamba run -n shiny python3 -m pytest tests/test_jax_tick.py::TestJaxHeadingAndPosition tests/test_jax_tick.py::TestJaxTickComposition tests/test_backend_equivalence.py -q`
  Expected: all pass (both new tests green; existing heading/composition/determinism tests still green — `_make_heading_inputs` + `test_tick_movement_returns_valid_positions` updated to the new signature).

- [ ] **Step 5: Commit.** `git -C /home/razinka/cenjas/CENOP add src/cenop/optimizations/jax_kernels.py src/cenop/optimizations/tick_jax.py src/cenop/agents/population.py tests/test_jax_tick.py` then `git -C /home/razinka/cenjas/CENOP commit -m "fix(jax): match NumPy reference in JAX dispersal heading (Finding #15)" -m "Use psm_type2_random_angle (20) not 120, distance TRAVELLED from dispersal_start not distance-to-target, SSLogis phi3=psm_log (0.6), and a uniform random delta instead of deterministic sign-toward-target steering. Thread dispersal_start_x/y + psm_angle/psm_log + a split RNG key through jax_tick_movement into jax_heading_composition." -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

### Task 26: JAX food floor 0.01 in _step_jax, not u_min (Finding #16)

**Files:**
- Modify `src/cenop/agents/population.py` (`_step_jax` → `jax_tick_energy` `min_food` arg, line 2438)
- Test `tests/test_jax_tick.py` (new class `TestJaxStepFoodFloor`)

**Interfaces:**
- Consumes: `PorpoisePopulation(count, params, landscape)` with `params.use_jax=True`; `landscape._food_value`.
- Produces: `_step_jax` passes literal `0.01` (DEPONS `ADD_ARTIFICIAL_FOOD`) as `min_food` to `jax_tick_energy`, matching the NumPy/Numba/kernel paths (`cell_data.py:274`, `kernels.py:329/381`).

- [ ] **Step 1: Write the failing test.** Add to `tests/test_jax_tick.py`:
```python
class TestJaxStepFoodFloor:
    """_step_jax must floor grazed cells at 0.01 (DEPONS ADD_ARTIFICIAL_FOOD),
    not params.u_min (0.001)."""

    def test_grazed_cell_floors_at_0_01(self):
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation

        params = SimulationParameters(porpoise_count=40)
        params.random_seed = 7
        params.use_jax = True
        assert params.u_min == 0.001, "guard: u_min must differ from the 0.01 floor"
        land = create_homogeneous_landscape(width=60, height=60, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=40, params=params, landscape=land)
        assert pop._use_jax, "JAX backend not active; NumPy path floors at 0.01 and the test would pass vacuously"

        # Uniform food just above the floor; starving agents (frac=0.99) deplete
        # every grazed cell below 0.01 in one tick, so the floor binds.
        pop.landscape._food_value[:] = 0.02
        pop.energy[:] = 0.0

        try:
            pop.step()
        except Exception as e:  # noqa: BLE001 - classify GPU OOM as environmental
            if any(k in str(e) for k in ("RESOURCE_EXHAUSTED", "OUT_OF_MEMORY")):
                pytest.skip(f"JAX GPU OOM (environmental): {e}")
            raise

        food = np.asarray(pop.landscape._food_value)
        # Grazed cells -> 0.01 (floor), ungrazed cells stay 0.02, so the grid min is 0.01.
        assert np.isclose(food.min(), 0.01, atol=1e-6), (
            f"food floor = {food.min()} (expected 0.01 ADD_ARTIFICIAL_FOOD, not u_min=0.001)"
        )
```

- [ ] **Step 2: Run — expect FAIL.** `cd /home/razinka/cenjas/CENOP && JAX_PLATFORMS=cpu micromamba run -n shiny python3 -m pytest tests/test_jax_tick.py::TestJaxStepFoodFloor -q`
  Expected: FAIL — `food floor = 0.001 (expected 0.01 ...)`. (Empirically verified on current code: grid min after step = 0.001.) `JAX_PLATFORMS=cpu` forces execution instead of the GPU-OOM skip.

- [ ] **Step 3: Implement.** In `src/cenop/agents/population.py`, in the `jax_tick_energy(...)` call, replace:
```python
            float(self.params.e_lact),
            float(getattr(self.params, 'u_min', 0.001)),
            float(self._m_mort_prob_const),
```
  with:
```python
            float(self.params.e_lact),
            0.01,  # DEPONS ADD_ARTIFICIAL_FOOD floor (matches NumPy/Numba/kernels)
            float(self._m_mort_prob_const),
```

- [ ] **Step 4: Run — expect PASS.** `cd /home/razinka/cenjas/CENOP && JAX_PLATFORMS=cpu micromamba run -n shiny python3 -m pytest tests/test_jax_tick.py::TestJaxStepFoodFloor tests/test_jax_tick.py::TestJaxFoodKernel tests/test_jax_tick.py::TestJaxTickComposition::test_tick_energy_conserves_food -q`
  Expected: all pass — grid min now 0.01; the direct `jax_eat_food`/`jax_tick_energy` kernel tests (which pass their own `min_food`) are unaffected.

- [ ] **Step 5: Commit.** `git -C /home/razinka/cenjas/CENOP add src/cenop/agents/population.py tests/test_jax_tick.py` then `git -C /home/razinka/cenjas/CENOP commit -m "fix(jax): floor grazed food at 0.01 in _step_jax, not u_min (Finding #16)" -m "jax_tick_energy's min_food is DEPONS ADD_ARTIFICIAL_FOOD (0.01), matching cell_data.py and the Numba kernels; _step_jax was passing params.u_min (0.001)." -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

### Task 27: Refresh _active_idx in _step_jax so dead slots skip ref-mem (Finding #28)

**Files:**
- Modify `src/cenop/agents/population.py` (`_step_jax`, right after `self._global_tick += 1`, ~line 2185)
- Test `tests/test_jax_tick.py` (new class `TestJaxStepRefMem`)

**Interfaces:**
- Consumes: `self.active_mask` (bool[n]).
- Produces: `self._active_idx = np.flatnonzero(self.active_mask)` set at the top of `_step_jax` (mirrors the Numba path at `step()` line 2566), so `_update_reference_memory` no longer writes `_stored_util`/`_pos_history`/`_mem_ptr`/`_mem_count` for inactive (dead) slots.

- [ ] **Step 1: Write the failing test.** Add to `tests/test_jax_tick.py`:
```python
class TestJaxStepRefMem:
    """_step_jax must exclude dead slots from reference-memory updates."""

    def test_dead_slot_ref_mem_not_advanced(self):
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation

        params = SimulationParameters(porpoise_count=30)
        params.random_seed = 7
        params.use_jax = True
        land = create_homogeneous_landscape(width=60, height=60, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=30, params=params, landscape=land)

        dead, live = 5, 0
        pop.active_mask[dead] = False
        mem_ptr_before = pop._mem_ptr.copy()
        mem_count_before = pop._mem_count.copy()

        try:
            pop.step()
        except Exception as e:  # noqa: BLE001 - classify GPU OOM as environmental
            if any(k in str(e) for k in ("RESOURCE_EXHAUSTED", "OUT_OF_MEMORY")):
                pytest.skip(f"JAX GPU OOM (environmental): {e}")
            raise

        # Dead slot's circular ref-mem buffer must not advance.
        assert pop._mem_ptr[dead] == mem_ptr_before[dead], "dead slot ref-mem pointer advanced"
        assert pop._mem_count[dead] == mem_count_before[dead], "dead slot ref-mem count advanced"
        # Non-vacuity: a live slot DID advance this tick, proving ref-mem ran.
        assert pop._mem_count[live] == mem_count_before[live] + 1, (
            "live slot ref-mem did not advance -> test is vacuous"
        )
```

- [ ] **Step 2: Run — expect FAIL.** `cd /home/razinka/cenjas/CENOP && JAX_PLATFORMS=cpu micromamba run -n shiny python3 -m pytest tests/test_jax_tick.py::TestJaxStepRefMem -q`
  Expected: FAIL — `dead slot ref-mem pointer advanced` (empirically verified: dead slot `_mem_ptr` 0→1, `_mem_count` 0→1, because `_active_idx` stays `arange(count)` and still contains the dead slot).

- [ ] **Step 3: Implement.** In `src/cenop/agents/population.py` `_step_jax`, replace:
```python
        mask = self.active_mask
        active_before = int(np.sum(mask))
        self._global_tick += 1

        if self._global_tick == 1:
```
  with:
```python
        mask = self.active_mask
        active_before = int(np.sum(mask))
        self._global_tick += 1

        # Refresh cached active indices each tick so dead slots are excluded from
        # reference-memory updates (mirrors the Numba path in step()).
        self._active_idx = np.flatnonzero(self.active_mask)

        if self._global_tick == 1:
```

- [ ] **Step 4: Run — expect PASS.** `cd /home/razinka/cenjas/CENOP && JAX_PLATFORMS=cpu micromamba run -n shiny python3 -m pytest tests/test_jax_tick.py::TestJaxStepRefMem tests/test_backend_equivalence.py::test_jax_backend_deterministic tests/test_jax_tick.py::TestJaxFullTick -q`
  Expected: all pass — dead slot unchanged, live slot advanced by 1; determinism and full-tick behavior intact (excluding dead slots only removes writes to inactive rows).

- [ ] **Step 5: Commit.** `git -C /home/razinka/cenjas/CENOP add src/cenop/agents/population.py tests/test_jax_tick.py` then `git -C /home/razinka/cenjas/CENOP commit -m "fix(jax): refresh _active_idx in _step_jax so dead slots skip ref-mem (Finding #28)" -m "_step_jax never refreshed self._active_idx (stayed arange(count) from __init__), so _update_reference_memory kept writing dead slots. Set _active_idx = flatnonzero(active_mask) at the top, mirroring the Numba path." -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`


> **Note (jax-parity):** All three defects empirically reproduced on current code via the JAX CPU backend (GPU is OOM in this env — the same environmental condition the existing test_jax_backend_deterministic skips on; the new integration tests replicate that RESOURCE_EXHAUSTED/OUT_OF_MEMORY skip guard, and the run commands prepend JAX_PLATFORMS=cpu so they actually execute rather than skip). Confirmed: #28 dead slot _mem_ptr/_mem_count advance 0->1; #16 grid food min = 0.001 (should be 0.01). **Docstring:** Step 3b changes `jax_heading_composition`'s signature (adds `dispersal_start_x/_y`, `psm_angle`, `psm_log`, `disp_key`) and inverts its dispersal behaviour, so also update that function's own docstring — line ~288 ("SSLogis-based heading override toward their target" is now a PSM-Type2 logistic-scaled uniform-random turn from `previous_step_heading`, NOT toward the target) and add Parameters entries for the 5 new args. Updating the docstring of a function you are changing does not violate the "no docstrings on unchanged code" rule.

The three fixes are independent and edit non-overlapping, string-anchored regions of population.py (#28 ~line 2185, #16 line 2438, #15 lines 2308/2326), so they can land in any order in separate commits; anchors are unique (verified). Task #15 is the only multi-file/signature change — its five edits (jax_kernels.py, tick_jax.py, population.py, and two spots in tests/test_jax_tick.py: _make_heading_inputs + the existing test_tick_movement_returns_valid_positions positional call) MUST land in one commit or the positional callers break.

BASELINE / BEHAVIOR RISK: #15 adds a jax.random.split in jax_tick_movement, shifting the downstream JAX RNG stream (land-avoidance key), and changes dispersal-heading semantics; #16 raises the JAX food floor 0.001->0.01. Both alter JAX-path trajectories, but there is NO committed JAX reference baseline (JAX is the experimental backend; Kattegat baselines are Numba-path) so no baseline regeneration is required. The Numba/NumPy production paths are untouched. Statistical guards (TestJaxFullTick, test_jax_vs_numba_statistical_equivalence) still bound the JAX path and should stay green. Note per CLAUDE.md: getattr(params,'psm_type2_random_angle',20.0) resolves to 20.0 (attribute absent on SimulationParameters) — intentional, to bit-match the NumPy reference _apply_dispersal_heading rather than params.psm_angle=40.0.


## Phase 6 — Cython backend parity (Track B: repair)

Repair the four Cython defects and flip the xfail-strict equivalence test green.


### Task 28: Repair Cython post-CRW (food dtype, land rollback, seeded mortality) and flip the equivalence xfail green

Fixes Findings #18 (float32 `food_grid` dtype crash), #19 (missing post-move land rollback → ~3.6-cell divergence), and #21 (mortality drawn from global `np.random`). These are exactly the three reasons enumerated in the `test_cython_postcrw_matches_reference` xfail docstring, so completing this task removes that xfail.

**Files:**
- Modify: `src/cenop/optimizations/tick_cython.pyx` — signature (lines 42-46), cdef block (lines 76-79), move/reflect/eat-cell section (lines 92-132)
- Modify: `src/cenop/agents/population.py` — Cython call site (food_grid build lines 2592-2596; kernel call lines 2599-2626)
- Modify: `tests/test_backend_equivalence.py` — remove the `@pytest.mark.xfail` decorator (lines 101-105; KEEP the `@pytest.mark.skipif` on line 100); add `test_cython_food_grid_dtype_no_crash` and `test_cython_mortality_uses_seeded_rng`
- Modify: `tests/test_cython_tick.py` — update the 3 existing kernel calls (lines 52-58, 80-96, 111-127) for the new signature
- Regenerate (build artifact): `src/cenop/optimizations/tick_cython.cpython-313-x86_64-linux-gnu.so` via `setup_cython.py`

**Interfaces:**
- Consumes: `CellData._food_value` (float64 ndarray, homogeneous/ASC), `CellData._depth` (float64 ndarray, land = -10.0), `PorpoisePopulation.rng` (`np.random.Generator`), existing gate `_HAS_CYTHON and _energy_module is None and _skip_land_avoidance and not _comm_enabled`.
- Produces: extended kernel signature `cython_depons_post_crw(..., food_grid: f32[:, :], depth_grid: f64[:, :], out_food_gained, dispersal_distance_traveled, rand_mort: f64[:], ...)`; post-move land rollback matching `_apply_positions`; seeded mortality (`self.rng.random(count)`) with in-place float32 food write-back.

**Reference facts (verified in-repo):**
- `create_homogeneous_landscape` (`cell_data.py:806-829`) builds float64 `_depth` with land (-10.0) at the 5%/2% row and 3%/3% col edges and float64 `_food_value` → the kernel's `f32` `food_grid` param raises `ValueError: Buffer dtype mismatch, expected 'f32' but got 'double'` today.
- `_apply_positions` (`population.py:1516-1528`) rolls agents whose post-move cell has `depth <= 0` back to the pre-move position, gated on `landscape is not None` (NOT `_skip_land_avoidance`). `_skip_land_avoidance` is set True by landscape NAME ("Homogeneous", `population.py:388-389`) even though that landscape has land edges, so the Cython gate engages and the reference still rolls agents off the land edges — the divergence the parity test measures.
- Reference mortality draw is `self.rng.random(self.count)` (`population.py:1963`); Numba CRW `np.random.*` calls are inside `@njit` (isolated RNG), and the ONLY global-CPython `np.random` in `population.py` is `__init__` (`line 101`), so the ONLY global-CPython `np.random` consumer in the Cython step path is `.pyx:79`.
- Seed 11 / 150 agents / 120×120: 0 pre-move cell collisions and 0 initial land occupancy (verified), so the food proportional-vs-sequential path does not diverge and single-tick mortality has no deaths (energy 7.86–12.57, step_surv ≈ 1).

TDD steps:

- [ ] **Step 1 (Cycle A test — #18): add the dtype-crash guard.** Append to `tests/test_backend_equivalence.py` (after `test_cython_gate_is_engaged`):
```python
@pytest.mark.skipif(not getattr(pop_mod, "_HAS_CYTHON", False), reason="Cython not built")
def test_cython_food_grid_dtype_no_crash():
    """Cython post-CRW must accept a float64 landscape food grid (homogeneous/ASC
    landscapes store float64) without a buffer-dtype ValueError (Finding #18)."""
    p = _build_cy(11)
    p.step()  # must NOT raise ValueError: Buffer dtype mismatch
    assert int(p.active_mask.sum()) > 0
```

- [ ] **Step 2 (run — expect FAIL): ** `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_backend_equivalence.py::test_cython_food_grid_dtype_no_crash -q` → FAILS with `ValueError: Buffer dtype mismatch, expected 'f32' but got 'double'` raised inside `PorpoisePopulation.step` (the Cython call site passes float64 `self.landscape._food_value`).

- [ ] **Step 3 (impl — #18, call-site only, no recompile): ** In `src/cenop/agents/population.py`, replace the `food_grid` build (lines 2592-2596) and add a write-back after the kernel call. Replace:
```python
            food_grid = (
                self.landscape._food_value
                if self.landscape
                else np.full((world_h, world_w), 50.0, dtype=np.float32)
            )

            self._cython_food_gained.fill(0.0)
```
with:
```python
            # food_grid must be float32 (kernel buffer dtype); homogeneous/ASC
            # landscapes store float64. Cast to float32 and write the in-place
            # depletion back afterwards so cross-tick food consumption is kept.
            if self.landscape is not None:
                _food_src = self.landscape._food_value
                food_grid = (
                    _food_src if _food_src.dtype == np.float32
                    else np.ascontiguousarray(_food_src, dtype=np.float32)
                )
            else:
                _food_src = None
                food_grid = np.full((world_h, world_w), 50.0, dtype=np.float32)

            self._cython_food_gained.fill(0.0)
```
Then, immediately after the kernel call's closing `)` (line 2626), before the `# Post-Cython housekeeping` comment, insert:
```python
            # Write float32 food depletion back into the landscape store (if copied).
            if _food_src is not None and food_grid is not _food_src:
                _food_src[:] = food_grid

```

- [ ] **Step 4 (run — expect PASS): ** `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_backend_equivalence.py tests/test_cython_tick.py -q` → `test_cython_food_grid_dtype_no_crash` PASSES; the parity test STILL xfails (move divergence not yet fixed); expected roughly `8 passed, 1 skipped, 1 xfailed` (JAX skipped if unavailable).

- [ ] **Step 5 (commit): ** `git -C /home/razinka/cenjas/CENOP add -A && git -C /home/razinka/cenjas/CENOP commit -m "fix(cython): cast food_grid to float32 with depletion write-back (Finding #18)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

- [ ] **Step 6 (Cycle B tests — #19 + #21): remove the xfail and add the mortality-RNG guard.** In `tests/test_backend_equivalence.py`, delete the `@pytest.mark.xfail(...)` decorator (lines 101-105) so `test_cython_postcrw_matches_reference` becomes a normal test (keep its `@pytest.mark.skipif(not _HAS_CYTHON ...)` on line 100), and update its docstring to drop the "Currently xfails" paragraph so it reads only `"""SINGLE-TICK Cython post-CRW == Numba/NumPy reference, given identical Numba CRW."""`. Then append:
```python
@pytest.mark.skipif(not getattr(pop_mod, "_HAS_CYTHON", False), reason="Cython not built")
def test_cython_mortality_uses_seeded_rng():
    """Cython post-CRW mortality must draw from self.rng, not global np.random
    (Finding #21). The Numba CRW path uses njit-isolated RNG, so the only global
    np.random consumer in this step is the mortality draw: after the fix the
    global CPython MT19937 stream is untouched by a Cython tick."""
    assert pop_mod._HAS_KERNELS, "needs njit CRW so global np.random is untouched by CRW"
    p = _build_cy(11)
    np.random.seed(1234)
    s0 = np.random.get_state()
    p.step()
    s1 = np.random.get_state()
    assert s1[2] == s0[2] and np.array_equal(s1[1], s0[1]), \
        "Cython tick consumed global np.random; mortality must use self.rng"
```

- [ ] **Step 7 (run — expect FAIL x2): ** `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_backend_equivalence.py -q` → `2 failed`: `test_cython_postcrw_matches_reference` fails `AssertionError: Not equal ... Max absolute difference` on `cy.x`/`cy.y` (agents that stepped onto a land-edge cell were not rolled back, ~3.6 cells); `test_cython_mortality_uses_seeded_rng` fails the `get_state` assertion (kernel's `np.random.random(n)` advanced the global stream).

- [ ] **Step 8 (impl — #19 + #21 in the kernel + call site + recompile).**
  1. In `src/cenop/optimizations/tick_cython.pyx`, add the two new array params. Replace:
```cython
    # Landscape (read-write for food consumption)
    np.ndarray[f32, ndim=2] food_grid,
    # OUTPUT arrays (caller pre-allocates)
    np.ndarray[f32, ndim=1] out_food_gained,
    np.ndarray[f32, ndim=1] dispersal_distance_traveled,
    # Scalar parameters
```
with:
```cython
    # Landscape (read-write for food consumption)
    np.ndarray[f32, ndim=2] food_grid,
    np.ndarray[f64, ndim=2] depth_grid,
    # OUTPUT arrays (caller pre-allocates)
    np.ndarray[f32, ndim=1] out_food_gained,
    np.ndarray[f32, ndim=1] dispersal_distance_traveled,
    np.ndarray[f64, ndim=1] rand_mort,
    # Scalar parameters
```
  2. Replace the cdef/`rand_mort` block. Replace:
```cython
    cdef int xi_c, yi_c, deaths = 0

    # Pre-generate mortality random draws (vectorized NumPy, fast)
    cdef np.ndarray[f64, ndim=1] rand_mort = np.random.random(n)

    for i in range(n):
```
with:
```cython
    cdef int xi_c, yi_c, deaths = 0
    cdef int post_xi, post_yi
    cdef double pre_x, pre_y

    for i in range(n):
```
  3. Rewrite the move/eat-cell section (add `pre_x`/`pre_y` and the post-move land rollback). Replace:
```cython
        rad = new_h * DEG2RAD
        ddx = sin(rad) * step
        ddy = cos(rad) * step
        nx = x[i] + ddx
        ny = y[i] + ddy

        if nx < 0:
            nx = -nx
        elif nx > max_x:
            nx = 2.0 * max_x - nx
        if nx < 0:
            nx = 0.0
        elif nx > max_x:
            nx = max_x
        if ny < 0:
            ny = -ny
        elif ny > max_y:
            ny = 2.0 * max_y - ny
        if ny < 0:
            ny = 0.0
        elif ny > max_y:
            ny = max_y

        # Pre-move cell index — DEPONS eats at the cell just left
        # (Porpoise.updEnergeticStatus → posList.get(1)), so derive the eat
        # cell from the position BEFORE write-back, not the post-move position.
        xi_c = <int>x[i]
        if xi_c < 0: xi_c = 0
        if xi_c >= world_w: xi_c = world_w - 1
        yi_c = <int>y[i]
        if yi_c < 0: yi_c = 0
        if yi_c >= world_h: yi_c = world_h - 1

        x[i] = <f32>nx
        y[i] = <f32>ny
```
with:
```cython
        rad = new_h * DEG2RAD
        ddx = sin(rad) * step
        ddy = cos(rad) * step
        pre_x = x[i]
        pre_y = y[i]
        nx = pre_x + ddx
        ny = pre_y + ddy

        if nx < 0:
            nx = -nx
        elif nx > max_x:
            nx = 2.0 * max_x - nx
        if nx < 0:
            nx = 0.0
        elif nx > max_x:
            nx = max_x
        if ny < 0:
            ny = -ny
        elif ny > max_y:
            ny = 2.0 * max_y - ny
        if ny < 0:
            ny = 0.0
        elif ny > max_y:
            ny = max_y

        # Pre-move cell index — DEPONS eats at the cell just left
        # (Porpoise.updEnergeticStatus → posList.get(1)), so derive the eat
        # cell from the position BEFORE write-back, not the post-move position.
        xi_c = <int>pre_x
        if xi_c < 0: xi_c = 0
        if xi_c >= world_w: xi_c = world_w - 1
        yi_c = <int>pre_y
        if yi_c < 0: yi_c = 0
        if yi_c >= world_h: yi_c = world_h - 1

        # Post-move land rollback (reference _apply_positions, gated on landscape
        # present): if the destination cell is land (depth <= 0), restore the
        # pre-move position. Even 'Homogeneous' has land (depth -10) at the edges.
        post_xi = <int>nx
        if post_xi < 0: post_xi = 0
        if post_xi >= world_w: post_xi = world_w - 1
        post_yi = <int>ny
        if post_yi < 0: post_yi = 0
        if post_yi >= world_h: post_yi = world_h - 1
        if depth_grid[post_yi, post_xi] <= 0.0:
            nx = pre_x
            ny = pre_y

        x[i] = <f32>nx
        y[i] = <f32>ny
```
  4. In `src/cenop/agents/population.py`, build `depth_grid` and draw `rand_mort`, and add both to the kernel call. Immediately before `self._cython_food_gained.fill(0.0)` insert:
```python
            # depth_grid drives the post-move land rollback (reference
            # _apply_positions: destination on land -> restore pre-move cell).
            if self.landscape is not None and getattr(self.landscape, '_depth', None) is not None:
                depth_grid = np.ascontiguousarray(self.landscape._depth, dtype=np.float64)
            else:
                depth_grid = np.full((world_h, world_w), 20.0, dtype=np.float64)

            # Mortality draws from the seeded generator (matches reference
            # _check_mortality: self.rng.random(count)) for reproducibility.
            rand_mort = self.rng.random(self.x.shape[0])

```
     Then in the kernel call, change `food_grid,` to `food_grid,\n                depth_grid,` and change the `self.dispersal_distance_traveled,` line to be followed by `rand_mort,`. The argument list must read: `... self._vt_x, self._vt_y, food_grid, depth_grid, self._cython_food_gained, self.dispersal_distance_traveled, rand_mort, self.params.inertia_const, ...`.
  5. In `tests/test_cython_tick.py`, update the 3 existing kernel calls for the new signature. In `TestCythonFullPostCRW.test_deterministic_output.run_once`, after `disp_dist = np.zeros(n, dtype=np.float32)` add `depth = np.full((200, 200), 20.0, dtype=np.float64)` and `rand_mort = np.random.random(n)`, and change the call line `food, out_food, disp_dist,` to `food, depth, out_food, disp_dist, rand_mort,`. In `test_food_gained_output_populated`, after its `disp_dist = np.zeros(n, dtype=np.float32)` add `depth = np.full((200, 200), 20.0, dtype=np.float64)` and `rand_mort = np.zeros(n, dtype=np.float64)`, and change `food, out_food, disp_dist,` to `food, depth, out_food, disp_dist, rand_mort,`. In `test_mortality_kills_starving`, after its `disp_dist = np.zeros(n, dtype=np.float32)` add `depth = np.full((200, 200), 20.0, dtype=np.float64)` and `rand_mort = np.full(n, 0.5, dtype=np.float64)` (all > 0 so every starving agent dies), and change `food, out_food, disp_dist,` to `food, depth, out_food, disp_dist, rand_mort,`.
  6. Recompile: `cd /home/razinka/cenjas/CENOP/src/cenop/optimizations && rm -f tick_cython.c && rm -rf build && micromamba run -n shiny python3 setup_cython.py build_ext --inplace` (expect `building 'tick_cython' extension` — a real recompile, not just a cached copy).

- [ ] **Step 9 (run — expect PASS): ** `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_backend_equivalence.py tests/test_cython_tick.py -q` → `test_cython_postcrw_matches_reference` PASSES (x/y/energy/active_mask within `rtol=1e-4, atol=1e-3`), `test_cython_mortality_uses_seeded_rng` PASSES, all 4 `test_cython_tick` calls PASS with the new signature; expected roughly `10 passed, 1 skipped` (no xfailed remaining; JAX skipped if unavailable).

- [ ] **Step 10 (commit): ** `git -C /home/razinka/cenjas/CENOP add -A && git -C /home/razinka/cenjas/CENOP commit -m "fix(cython): post-move land rollback + seeded mortality; flip equivalence xfail green (Findings #19, #21)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

### Task 29: Recompute Cython heading after boundary reflection (DEPONS forward())

Fixes Finding #29: boundary-reflected agents keep their outward heading, whereas the reference recomputes heading from the sign-flipped displacement (`reflect_boundaries_kernel` flips dx/dy; `_handle_land_avoidance` sets `heading = degrees(arctan2(dx, dy)) % 360`). This is a correctness refinement on top of Task 28 (it does not affect the seed-11 parity test, where no world-boundary reflection occurs, so it needs its own targeted test).

**Files:**
- Modify: `src/cenop/optimizations/tick_cython.pyx` — cdef block (add `reflected`) and the reflection section rewritten in Task 28
- Modify: `tests/test_cython_tick.py` — add `test_cython_reflection_recomputes_heading`
- Regenerate (build artifact): `tick_cython.cpython-313-x86_64-linux-gnu.so`

**Interfaces:**
- Consumes: kernel signature finalized in Task 28 (`food_grid`, `depth_grid`, `rand_mort`); `heading` (f32 array, written in place); reference `src/cenop/agents/population.py` (NOT the DEPONS Java `population.py`).
- Produces: reflected agents get `heading = fmod(atan2(flipped_ddx, flipped_ddy) * RAD2DEG, 360)` normalized to `[0, 360)`, matching `population.py:1280-1282`.

- [ ] **Step 1 (test): add a direct-kernel reflection test.** Append to `tests/test_cython_tick.py` (module level, guarded like the existing tests):
```python
@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
def test_cython_reflection_recomputes_heading():
    """After a world-boundary bounce the kernel must recompute heading from the
    sign-flipped displacement (DEPONS forward()), not keep the outward heading."""
    from cenop.optimizations.tick_cython import cython_depons_post_crw

    W = H = 100
    x = np.array([1.0], dtype=np.float32)
    y = np.array([50.0], dtype=np.float32)
    heading = np.array([270.0], dtype=np.float32)          # points toward -x
    prev_angle = np.zeros(1, dtype=np.float64)
    prev_log_mov = np.zeros(1, dtype=np.float64)
    energy = np.array([10.0], dtype=np.float32)
    active = np.ones(1, dtype=np.uint8)
    is_disp = np.ones(1, dtype=np.uint8)                   # step = disp_step
    with_calf = np.zeros(1, dtype=np.uint8)
    pres_angle = np.zeros(1, dtype=np.float64)
    log_mov = np.zeros(1, dtype=np.float64)
    ve_total = np.zeros(1, dtype=np.float32)
    vt_x = np.zeros(1, dtype=np.float32)
    vt_y = np.zeros(1, dtype=np.float32)
    food = np.zeros((H, W), dtype=np.float32)
    depth = np.full((H, W), 20.0, dtype=np.float64)        # all water -> no rollback
    out_food = np.zeros(1, dtype=np.float32)
    disp_dist = np.zeros(1, dtype=np.float32)
    rand_mort = np.zeros(1, dtype=np.float64)              # rand=0 -> no death

    cython_depons_post_crw(
        x, y, heading, prev_angle, prev_log_mov, energy,
        active, is_disp, with_calf,
        pres_angle, log_mov, ve_total, vt_x, vt_y,
        food, depth, out_food, disp_dist, rand_mort,
        0.0, 3.0, 0.0, 1.0, 1.0, 0.4, 1.0, W, H,   # disp_step=3.0 -> step=3
    )
    # ddx = sin(270deg)*3 = -3 -> nx = 1-3 = -2 -> reflect -> nx = 2, ddx -> +3
    assert abs(float(x[0]) - 2.0) < 1e-4
    # heading recomputed from flipped displacement: atan2(+3, 0) -> 90 deg
    assert abs(float(heading[0]) - 90.0) < 1e-4
```

- [ ] **Step 2 (run — expect FAIL): ** `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_cython_tick.py::test_cython_reflection_recomputes_heading -q` → FAILS on `assert abs(float(heading[0]) - 90.0) < 1e-4` (heading is still `270.0`; the kernel never writes `heading[i]` after reflection). The `x[0] == 2.0` assertion already passes (position reflection works).

- [ ] **Step 3 (impl — #29): ** In `src/cenop/optimizations/tick_cython.pyx`, add `reflected` to the cdef block. Replace:
```cython
    cdef int xi_c, yi_c, deaths = 0
    cdef int post_xi, post_yi
    cdef double pre_x, pre_y
```
with:
```cython
    cdef int xi_c, yi_c, deaths = 0
    cdef int post_xi, post_yi, reflected
    cdef double pre_x, pre_y
```
Then rewrite the reflection block (the Task-28 version) to flip `ddx`/`ddy` and recompute heading. Replace:
```cython
        nx = pre_x + ddx
        ny = pre_y + ddy

        if nx < 0:
            nx = -nx
        elif nx > max_x:
            nx = 2.0 * max_x - nx
        if nx < 0:
            nx = 0.0
        elif nx > max_x:
            nx = max_x
        if ny < 0:
            ny = -ny
        elif ny > max_y:
            ny = 2.0 * max_y - ny
        if ny < 0:
            ny = 0.0
        elif ny > max_y:
            ny = max_y

        # Pre-move cell index — DEPONS eats at the cell just left
```
with:
```cython
        nx = pre_x + ddx
        ny = pre_y + ddy

        reflected = 0
        if nx < 0:
            nx = -nx
            ddx = -ddx
            reflected = 1
        elif nx > max_x:
            nx = 2.0 * max_x - nx
            ddx = -ddx
            reflected = 1
        if nx < 0:
            nx = 0.0
        elif nx > max_x:
            nx = max_x
        if ny < 0:
            ny = -ny
            ddy = -ddy
            reflected = 1
        elif ny > max_y:
            ny = 2.0 * max_y - ny
            ddy = -ddy
            reflected = 1
        if ny < 0:
            ny = 0.0
        elif ny > max_y:
            ny = max_y

        # DEPONS forward(): after a boundary bounce recompute heading from the
        # sign-flipped displacement (reflect_boundaries flips dx/dy; heading =
        # degrees(arctan2(dx, dy)) % 360). Non-reflected agents keep their heading.
        if reflected:
            new_h = fmod(atan2(ddx, ddy) * RAD2DEG, 360.0)
            if new_h < 0.0:
                new_h += 360.0
            heading[i] = <f32>new_h

        # Pre-move cell index — DEPONS eats at the cell just left
```
(`atan2`, `fmod`, `M_PI` are already imported at line 10; `RAD2DEG` is defined at line 17. `ddx`/`ddy` are used only for `nx`/`ny` and this heading recompute, so flipping them does not perturb any other computation.) Recompile: `cd /home/razinka/cenjas/CENOP/src/cenop/optimizations && rm -f tick_cython.c && rm -rf build && micromamba run -n shiny python3 setup_cython.py build_ext --inplace`.

- [ ] **Step 4 (run — expect PASS): ** `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_cython_tick.py tests/test_backend_equivalence.py -q` → `test_cython_reflection_recomputes_heading` PASSES; all Task-1 tests still PASS (seed-11 parity unaffected — no boundary reflection there); expected roughly `11 passed, 1 skipped`.

- [ ] **Step 5 (commit): ** `git -C /home/razinka/cenjas/CENOP add -A && git -C /home/razinka/cenjas/CENOP commit -m "fix(cython): recompute heading after boundary reflection (DEPONS forward(), Finding #29)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`


> **Note (cython-parity):** RECOMPILE REQUIRED after every .pyx edit: `cd /home/razinka/cenjas/CENOP/src/cenop/optimizations && rm -f tick_cython.c && rm -rf build && micromamba run -n shiny python3 setup_cython.py build_ext --inplace` (Cython 3.2.4 confirmed working in env `shiny`; a stale build/ can silently copy the old .so instead of recompiling, hence the rm). Production impact is nil: the Cython fast path is gated to homogeneous + `_energy_module is None` + `_skip_land_avoidance` + `not _comm_enabled`, and `communication_enabled` defaults True, so these findings are latent and NO reference baselines (Kattegat, etc.) need regenerating. The main deliverable is removing the `xfail(strict=True)` on `tests/test_backend_equivalence.py::test_cython_postcrw_matches_reference`; keep the `test_cython_gate_is_engaged` non-vacuity guard untouched. Empirically confirmed the seed-11 scenario has 0 pre-move cell collisions and 0 initial land occupancy, so the food path does not diverge and single-tick mortality yields no deaths (energy 7.86–12.57) — the parity test hinges on Findings #18 (crash) + #19 (land rollback); #21 and #29 are captured by their own dedicated tests. Task 28 splits into two commits (Cycle A #18 call-site-only no recompile; Cycle B #19+#21 kernel signature change + recompile + update 3 existing test_cython_tick.py calls); Task 29 is the heading recompute (Finding #29). CLAUDE.md: use `git -C <path>` (never `cd && git`); line length 100. Minor accepted perf cost in #18: the float64→float32 write-back copies the full food grid per Cython tick when the landscape is float64 (correctness over speed); the alternative (store `_food_value` as float32 in cell_data.py constructors) was rejected to avoid perturbing the reference/JAX dtype.


---

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task with two-stage review between tasks (superpowers:subagent-driven-development). Suits this plan: tasks are independent and each ends in a committable, testable deliverable.
2. **Inline Execution** — execute in-session with checkpoints (superpowers:executing-plans).

Suggested sequencing: settle the two DECISION-REQUIRED tasks (13, 14) and the movement-parity approach (Tasks 2-4) with the maintainer first, then run phases in order. Regenerate the Kattegat baselines and run the slow tier at the end of Phases 1, 2, and 3.
