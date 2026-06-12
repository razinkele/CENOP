# Tier 0 — Restore Validation Suite + Backend Determinism/Equivalence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the long-running biological-validation tests runnable on demand (a `slow` tier) so emergent demographics are actually tested, recover the *fast* tests currently excluded alongside them, and add per-backend determinism guards plus a documented backend-equivalence matrix.

**Architecture:** Two parts. Part A (0a) introduces a pytest `slow` marker, marks only the long-tick-loop tests in `test_validation.py` / `test_depons_physiology.py`, stops excluding those files wholesale (so their fast tests rejoin the suite), and runs + triages the slow tier. Part B (0b) adds per-backend determinism guards (same seed → identical result), a single-tick Cython↔reference post-CRW equivalence differential (the one cross-backend deterministic comparison the RNG architecture permits — currently `xfail` because deep-review execution proved the Cython backend is broken; the defects are handed to Track B), and a markdown matrix documenting what is and is not cross-backend comparable (see Findings).

**Tech Stack:** Python 3, pytest, NumPy, Numba; optional Cython/JAX backends. Run from `/home/razinka/cenjas/CENOP/`, prefix Python/pytest with `micromamba run -n shiny`. CENOP is a nested git repo — commit from inside `CENOP/` (use `git -C /home/razinka/cenjas/CENOP` if your shell CWD is the parent). Branch off `CENOP-JASMINE`. Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. **Do not put backticks in commit messages** (bash command-substitution will mangle them — write the message to a file and use `git commit -F` if it must contain shell metacharacters).

**Spec:** `docs/superpowers/specs/2026-06-12-cenop-roadmap-design.md` (Tier 0, §4).

---

## Planning findings that refine the roadmap (read before starting)

1. **The validation tests are SLOW, not hanging.** `test_validation.py::TestPopulationDynamicsValidation::test_population_stability_one_year` runs to completion in ~22 s and passes. The prior "hangs" note was inaccurate. So 0a is *test hygiene* (mark + run), not deadlock-debugging.
2. **Both files are fast/slow MIXES.** `test_validation.py` and `test_depons_physiology.py` each contain fast formula/API/parameter tests *and* slow multi-year simulation loops. Marking only the slow tests means the fast ones rejoin the default suite. (Note: these files are NOT excluded in `CLAUDE.md` — its command is `pytest tests/ -x -q`, which already runs the slow tests today. The `--ignore=...` form lives only in the dev-habit/auto-memory command. So this plan does not introduce a slow default suite; it *removes* the slowness once the slow tests are marked + deselected.)
3. **Full-trajectory *stochastic* cross-backend bit-identical equivalence is architecturally impossible — but the deterministic post-CRW stage IS comparable.** The Numba RNG is seeded from a *separate* stream (`population.py:917` `_seed_numba_rng(self.rng.integers(0, 2**31))`) and CRW draws happen inside the kernel, so the NumPy-fallback / Numba / JAX paths consume different random streams and their *stochastic* trajectories diverge by design — no tolerance fixes this. **However**, the Cython fast path (`cython_depons_post_crw`) replaces only the *deterministic* post-CRW pipeline (heading composition, move, food, energy, mortality) and consumes already-computed CRW outputs with **no internal RNG**. So with identical Numba CRW (same `random_seed`, both `_HAS_KERNELS=True`), a Cython-vs-reference post-CRW **single-tick** differential is valid — built in Task 7. (It must be single-tick: the Cython path draws mortality from the *global* `np.random` while the reference draws from `self.rng`, so after tick 1 the two `self.rng` states desync and the per-tick Numba CRW seed diverges — see Finding #4.) For the small Numba kernels, only `reflect_boundaries` has a pure NumPy counterpart (`Population._reflect_boundaries`, existing test `test_numba_kernels.py::TestReflectBoundariesKernel::test_equivalence_with_numpy_version`); `_compute_turn_position` and the other fused kernels dispatch to / inline the kernels, so they are covered by behavioral unit tests rather than NumPy-equivalence. **Therefore 0b = per-backend determinism guards (Task 6) + a single-tick Cython↔reference post-CRW differential that currently documents a real divergence as `xfail` (Task 7) + a documented matrix (Task 8).** The stochastic-trajectory exclusion is the spec's explicitly-anticipated "legitimately cannot match bit-for-bit → document why" case (§4, 0b).

4. **Empirical execution (deep in-loop review of this plan, both parts run end-to-end) confirmed Part A and exposed three real Cython-backend defects.** Part A runs clean: the measured slow set is exactly the 11 candidates (6 in `test_validation.py`, 5 in `test_depons_physiology.py`), and de-excluding the two files surfaces **zero** fast-test failures (724 pass, fast). Part B: the Numba/NumPy determinism guards pass. But running the Task 7 Cython differential revealed the Cython fast path (gated to homogeneous + `communication_enabled=False` + no energy module — i.e. **off in production**, where comm defaults True) is broken three ways: **(a)** it crashes on a float64 `food_grid` (`cython_depons_post_crw` expects float32 but `population.py` passes `self.landscape._food_value` uncast; homogeneous landscapes store float64 at `cell_data.py:118`); **(b)** after that, its post-CRW `x`/`y` math diverges from the reference by ~3.6 cells at a single tick *with bit-identical CRW inputs* — a genuine formula/units bug in `tick_cython.pyx`; **(c)** its mortality uses unseeded global `np.random` (`tick_cython.pyx:79`), so the Cython path is non-reproducible. **These are NOT fixed here** — repairing/removing the Cython backend is Track B (backend-fate) work, now with hard evidence. Tier 0's job is to *guard and document* the divergence: Task 7 encodes it as a strict `xfail` (it flips to a failure-that-must-be-investigated the moment someone fixes Cython), and Task 8's matrix records all three defects for Track B.

## File Structure

- `pyproject.toml` — add `[tool.pytest.ini_options]` (marker registration + default `-m "not slow"`). One responsibility: pytest configuration.
- `tests/test_validation.py` — add `@pytest.mark.slow` to long-tick-loop tests only. No logic changes.
- `tests/test_depons_physiology.py` — same.
- `tests/test_backend_equivalence.py` — **new**: per-backend determinism guards (Task 6) + Cython↔reference post-CRW differential (Task 7). One responsibility: backend reproducibility & equivalence.
- `docs/backend-equivalence.md` — **new**: the comparison matrix + RNG-architecture note (Task 8). One responsibility: documenting equivalence scope.
- `CLAUDE.md` — update the documented test invocation (drop the `--ignore` flags, add the slow-tier commands).

---

# Part A — Restore the validation suite (0a)

### Task 1: Add the `slow` marker + default deselection

**Files:**
- Modify: `pyproject.toml` (append a `[tool.pytest.ini_options]` table)

- [ ] **Step 1: Check there is no existing pytest table**

Run: `micromamba run -n shiny python3 -c "import tomllib,pathlib;print('tool.pytest.ini_options' in tomllib.loads(pathlib.Path('pyproject.toml').read_text()).get('tool',{}))"`
Expected: prints `False` (no existing pytest config — confirmed during planning). If it prints `True`, STOP and report — merge into the existing table instead of appending a duplicate.

- [ ] **Step 2: Append the pytest configuration**

Append to the end of `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: long-running simulation tests (1-2 simulated years per test); deselected by default. Run with `-m slow`, or run everything with `-m 'slow or not slow'`.",
]
addopts = "-m 'not slow'"
```

- [ ] **Step 3: Verify the marker is registered and default-deselected**

Run: `micromamba run -n shiny python3 -m pytest --markers 2>/dev/null | grep -A1 "slow:"`
Expected: the `@pytest.mark.slow` description prints (marker registered, no "unknown mark" warnings).

Run: `micromamba run -n shiny python3 -m pytest tests/test_deterrence.py -q 2>&1 | tail -3`
Expected: PASS — a fast file still collects and runs normally under the new config.

- [ ] **Step 4: Commit**

```bash
git -C /home/razinka/cenjas/CENOP add pyproject.toml
git -C /home/razinka/cenjas/CENOP commit -m "test: register slow marker, deselect slow tests by default

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Mark the slow tests in `test_validation.py`

Mark only tests whose runtime is dominated by multi-year tick loops. Use **measured durations** as the source of truth.

**Files:**
- Modify: `tests/test_validation.py`

- [ ] **Step 1: Measure per-test durations**

Run: `micromamba run -n shiny python3 -m pytest tests/test_validation.py -m "slow or not slow" -p no:cacheprovider --durations=0 -q 2>&1 | tail -40`
Expected: a duration table. **The measured durations are authoritative**; the planning candidate list below is illustrative and may be incomplete. Mark every test whose own runtime exceeds **5 s** (the multi-year tick-loop tests; in practice they run tens of seconds while fast tests are sub-second, so the threshold is unambiguous — record the observed gap in the Task 5 results note as the justification). Planning candidates to expect (confirm/extend against the measured table, do not treat as exhaustive): `test_population_stability_one_year`, `test_birth_rate_realistic`, `test_mortality_rate_age_dependent`, `test_age_distribution_remains_valid`, `test_energy_distribution_stable`, `test_annual_mortality_rate_realistic`.

- [ ] **Step 2: Add `@pytest.mark.slow` to each measured-slow test**

For each test function exceeding 5 s, add the decorator on the line directly above its `def` (keep existing decorators). `pytest` is already imported in this file. Example shape:

```python
    @pytest.mark.slow
    def test_population_stability_one_year(self):
        ...
```

- [ ] **Step 3: Verify the fast subset is now fast and green**

Run: `micromamba run -n shiny python3 -m pytest tests/test_validation.py -q 2>&1 | tail -8`
Expected: completes in a few seconds (slow tests deselected by the default `-m 'not slow'`), all selected (fast) tests PASS. These fast tests were previously excluded by the dev-habit `--ignore` flag, so some may surface pre-existing failures. **Triage rule (bounds the blast radius):** if **≤ 2** fast tests fail, fix a real regression, or — only if the expectation is genuinely outdated — update it with a one-line justification in the commit. If **> 2** fast tests fail, STOP, list them, and report: a wholesale recovery of pre-existing failures is its own work item, out of scope for a marking task. **Never** mark a failing fast test `slow` to hide it.

- [ ] **Step 4: Verify the slow tests are collectable under `-m slow`**

Run: `micromamba run -n shiny python3 -m pytest tests/test_validation.py -m slow --collect-only -q 2>&1 | tail -10`
Expected: exactly the tests you marked are collected.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/cenjas/CENOP add tests/test_validation.py
git -C /home/razinka/cenjas/CENOP commit -m "test: mark multi-year validation tests slow; fast tests rejoin suite

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Mark the slow tests in `test_depons_physiology.py`

**Files:**
- Modify: `tests/test_depons_physiology.py`

- [ ] **Step 1: Measure per-test durations**

Run: `micromamba run -n shiny python3 -m pytest tests/test_depons_physiology.py -m "slow or not slow" -p no:cacheprovider --durations=0 -q 2>&1 | tail -40`
Expected: a duration table. **The measured durations are authoritative**; mark every test exceeding **5 s**. Planning candidates (illustrative, known-incomplete — a measured run showed `test_population_stability_short_term` at ~38 s also qualifies, so confirm the full set against the table): `test_population_stability_short_term`, `test_age_distribution_evolves`, `test_annual_mortality_rate`, `test_energy_dynamics_over_year`, `test_female_reproduction_rate`.

- [ ] **Step 2: Add `@pytest.mark.slow` to each measured-slow test**

Same decorator pattern as Task 2 Step 2. Confirm `import pytest` is present at the top of the file; if absent, add it.

- [ ] **Step 3: Verify the fast subset is fast and green**

Run: `micromamba run -n shiny python3 -m pytest tests/test_depons_physiology.py -q 2>&1 | tail -8`
Expected: completes quickly, fast tests PASS. Triage any fast failure using the same bounded rule as Task 2 Step 3 (≤ 2 failures: fix/justify; > 2: STOP and report).

- [ ] **Step 4: Verify slow collection**

Run: `micromamba run -n shiny python3 -m pytest tests/test_depons_physiology.py -m slow --collect-only -q 2>&1 | tail -10`
Expected: exactly the marked tests collected.

- [ ] **Step 5: Commit**

```bash
git -C /home/razinka/cenjas/CENOP add tests/test_depons_physiology.py
git -C /home/razinka/cenjas/CENOP commit -m "test: mark multi-year physiology tests slow; fast tests rejoin suite

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Confirm the default fast suite (no `--ignore`) is green

With the slow tests marked + deselected by default, the two files' fast tests now run in the default suite (and the multi-year tests no longer do).

**Files:** none (verification + docs).

- [ ] **Step 1: Run the full default (fast) suite**

First record the prior fast count for comparison: `micromamba run -n shiny python3 -m pytest tests/ -m "not slow" --co -q 2>&1 | tail -1`.
Then run: `micromamba run -n shiny python3 -m pytest tests/ -q 2>&1 | tail -8`
Expected: PASS, completing fast (well under ~2 min) — the multi-year tests are now deselected, and the two files' fast tests are included. The collected count should be ≥ the count this suite had before this plan (do not hardcode a number; compare to the just-recorded baseline). If any test fails, apply the bounded triage rule from Task 2 Step 3 before continuing.

- [ ] **Step 2: Update the documented test command in `CLAUDE.md`**

In `CLAUDE.md`, under `## Testing`, replace the line:

```
- `cd /home/razinka/cenjas/CENOP && python3 -m pytest tests/ -x -q`
```

with:

```
- Fast suite (default; slow sim tests deselected): `cd /home/razinka/cenjas/CENOP && python3 -m pytest tests/ -x -q`
- Slow tier (multi-year validation/physiology): `python3 -m pytest tests/ -m slow -q`
- Everything: `python3 -m pytest tests/ -m "slow or not slow" -q`
- NOTE: a default `addopts = -m "not slow"` is active. To run a single SLOW test you must add a selector, e.g. `pytest tests/test_validation.py::Cls::test_x -m "slow or not slow"` — a bare nodeid (or bare `-k`) silently reports it as `deselected`, not run.
- The slow tier is MANUAL (this repo has no CI). Run `pytest tests/ -m slow` before releasing/merging model-behavior changes. Any older `--ignore=...test_validation.py --ignore=...test_depons_physiology.py` alias is now obsolete — it re-hides the fast tests this change recovered; drop it.
```

- [ ] **Step 3: Commit**

```bash
git -C /home/razinka/cenjas/CENOP add CLAUDE.md
git -C /home/razinka/cenjas/CENOP commit -m "docs: document fast/slow/all pytest invocations

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Run the slow tier and record/triage results

This is the point of 0a — actually exercise emergent demographics.

**Files:** none (verification; create a results note).

- [ ] **Step 1: Run the slow tier (allow several minutes)**

Run: `micromamba run -n shiny python3 -m pytest tests/ -m slow -q --durations=0 2>&1 | tee /tmp/tier0_slow_results.txt | tail -30`
Expected: completes (minutes, not hanging — confirmed feasible in planning). Record pass/fail counts.

- [ ] **Step 2: Triage every failure**

For each failing slow test, decide and document:
- **Real regression** — open it up; if it reveals a model bug, STOP and escalate (do not paper over a demographic regression).
- **Outdated/over-tight expectation** — adjust the assertion with an explicit justification (e.g. a tolerance band that matches DEPONS-documented ranges), never a silent number swap.

If all pass, record that.

- [ ] **Step 3: Write a short results note**

Create `docs/validation-suite-status.md` summarizing: which tests are in the slow tier, total runtime, pass/fail, and any expectation changes made (with rationale). Keep it factual and brief.

**Include a prominent MANUAL-RUN warning** (this is a real gap surfaced in review): the repo has **no CI** (`.github/`, `.gitlab-ci.yml`, pre-commit — none exist), so with the slow tier deselected by default, these demographic-validation tests will **only run when someone types `-m slow`**. State explicitly: "⚠️ No CI — the slow validation tier is MANUAL. Run `pytest tests/ -m slow` before every release/merge of model-behavior changes. If CI is ever added, it MUST include a slow-tier job, or these tests silently never run." (Deselecting-by-default is the spec's chosen design for fast iteration; this note is the compensating control.)

- [ ] **Step 4: Commit**

```bash
git -C /home/razinka/cenjas/CENOP add docs/validation-suite-status.md tests/test_validation.py tests/test_depons_physiology.py
git -C /home/razinka/cenjas/CENOP commit -m "test: run + record slow validation tier status

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(If no test files changed in triage, drop them from the `git add`.)

---

# Part B — Backend determinism guard + equivalence matrix (0b)

### Task 6: Per-backend determinism differential test

Guard against accidental nondeterminism: the same `random_seed` must reproduce the same trajectory **within a single backend**. (Cross-backend identity is intentionally NOT asserted — see Findings #3.)

**Files:**
- Create: `tests/test_backend_equivalence.py`

- [ ] **Step 1: Write the determinism test**

Create `tests/test_backend_equivalence.py`:

```python
"""Backend determinism guards.

Each backend (NumPy-fallback, Numba, JAX) must be reproducible: identical
construction + seed -> identical per-tick trajectory. Cross-backend bit-identity
is NOT asserted because the Numba/JAX RNG streams are seeded independently of the
NumPy stream (population.py:917, :406), so their trajectories diverge by design.
See docs/backend-equivalence.md for the full comparison matrix.
"""
import numpy as np
import pytest

from cenop.parameters.simulation_params import SimulationParameters
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.agents.population import PorpoisePopulation
import cenop.agents.population as pop_mod


def _build(seed, use_jax=False):
    params = SimulationParameters(porpoise_count=120)
    params.random_seed = seed
    params.use_jax = use_jax
    land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
    return PorpoisePopulation(count=120, params=params, landscape=land)


def _run(pop, ticks=40):
    for _ in range(ticks):
        pop.step()
    return (pop.x.copy(), pop.y.copy(), pop.energy.copy(),
            pop.age.copy(), pop.active_mask.copy())


def _assert_same(a, b):
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)


def test_numba_backend_deterministic():
    """Production Numba path: same seed -> identical trajectory across two runs.

    `communication_enabled` defaults True, so the Cython post-CRW fast path is gated
    off (`not _comm_enabled` is False) and this exercises the Numba/NumPy post-CRW
    pipeline, not Cython. (Cython is covered separately in Task 7.)
    """
    assert pop_mod._HAS_KERNELS, "expected Numba kernels available in test env"
    _assert_same(_run(_build(7)), _run(_build(7)))


def test_numpy_fallback_deterministic(monkeypatch):
    """Pure-NumPy path (all kernels + Cython forced off): same seed -> identical trajectory.

    Disable `_HAS_CYTHON` explicitly so the guarantee does not depend on the
    `communication_enabled` default happening to gate Cython off.
    """
    monkeypatch.setattr(pop_mod, "_HAS_KERNELS", False)
    monkeypatch.setattr(pop_mod, "_HAS_LAND_KERNEL", False)
    monkeypatch.setattr(pop_mod, "_HAS_CYTHON", False)
    _assert_same(_run(_build(7)), _run(_build(7)))


@pytest.mark.skipif(not getattr(pop_mod, "_HAS_JAX", False), reason="JAX not installed")
def test_jax_backend_deterministic():
    """JAX path: same seed -> identical trajectory across two runs.

    JAX may be installed but bound to a memory-constrained GPU; a RESOURCE_EXHAUSTED /
    OOM at runtime is environmental, not a determinism bug, so skip on it. A failing
    determinism *assertion* is NOT caught here -> it still reports as a real failure.
    """
    try:
        a = _run(_build(7, use_jax=True))
        b = _run(_build(7, use_jax=True))
    except Exception as e:  # noqa: BLE001 - inspect the message to classify
        msg = str(e)
        if any(k in msg for k in ("RESOURCE_EXHAUSTED", "OUT_OF_MEMORY")) \
                or "Runtime" in type(e).__name__:
            pytest.skip(f"JAX runtime/OOM (environmental, not a determinism failure): {e}")
        raise
    _assert_same(a, b)
```

- [ ] **Step 2: Run the determinism tests**

Run: `micromamba run -n shiny python3 -m pytest tests/test_backend_equivalence.py -q 2>&1 | tail -15`
Expected: PASS (JAX test skips if JAX is unavailable). **If a backend is nondeterministic, the test FAILS — that is a real bug; STOP and escalate** (a reproducibility defect, not a test to relax). Confirm the attribute name `pop_mod._HAS_JAX` exists; if the JAX flag has a different name, grep `population.py` for the JAX availability global and use it (skip cleanly if absent).

- [ ] **Step 3: Commit**

```bash
git -C /home/razinka/cenjas/CENOP add tests/test_backend_equivalence.py
git -C /home/razinka/cenjas/CENOP commit -m "test: per-backend determinism guards (numpy/numba/jax)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Cython post-CRW equivalence differential

The Cython fast path (`cython_depons_post_crw`) replaces only the *deterministic* post-CRW pipeline and consumes CRW outputs computed upstream by the Numba kernel. With identical Numba CRW (same `random_seed`, both `_HAS_KERNELS=True`), a **single-tick** Cython-vs-reference comparison isolates the Cython post-CRW math. The gate (`population.py:2577`) requires `_HAS_CYTHON and _energy_module is None and _skip_land_avoidance and not _comm_enabled`, so the test uses a homogeneous landscape, `communication_enabled=False`, and no energy module.

**Deep-review execution (Finding #4) proved this comparison currently FAILS for real reasons** — the Cython path crashes on a float64 `food_grid`, and once that is bypassed its move math diverges ~3.6 cells at one tick with identical CRW inputs, and its mortality uses unseeded global `np.random`. **Tier 0 does NOT repair the Cython backend** (Track B owns that). Instead it lands the differential as a strict `xfail`, so the divergence is guarded and tracked — and the test flips to an attention-grabbing failure the moment Cython is fixed (prompting removal of the marker).

Why single-tick (not multi-tick): the Cython path draws mortality from global `np.random` while the reference draws from `self.rng`, so after tick 1 the two `self.rng` streams desync and the per-tick Numba CRW seed diverges. One tick shares a fresh `self.rng`, so CRW inputs are bit-identical and the comparison isolates the post-CRW math.

**Files:**
- Modify: `tests/test_backend_equivalence.py` (append)

- [ ] **Step 1: Write the (xfail) equivalence test + a non-vacuity gate test**

Append to `tests/test_backend_equivalence.py`:

```python
def _build_cy(seed):
    """Homogeneous, comm-off, no energy module -> the Cython post-CRW gate is eligible."""
    params = SimulationParameters(porpoise_count=150)
    params.random_seed = seed
    params.use_jax = False
    params.communication_enabled = False
    land = create_homogeneous_landscape(width=120, height=120, depth=20.0, food_prob=0.5)
    return PorpoisePopulation(count=150, params=params, landscape=land)


@pytest.mark.skipif(not getattr(pop_mod, "_HAS_CYTHON", False), reason="Cython not built")
def test_cython_gate_is_engaged():
    """Non-vacuity guard (separate from the xfail below, which errors before its own
    inline checks run): the Cython post-CRW gate must actually be satisfiable, else the
    equivalence comparison would prove nothing."""
    p = _build_cy(11)
    assert p._energy_module is None and p._skip_land_avoidance and not p._comm_enabled


@pytest.mark.skipif(not getattr(pop_mod, "_HAS_CYTHON", False), reason="Cython not built")
@pytest.mark.xfail(strict=True, raises=(AssertionError, ValueError, TypeError), reason=(
    "Cython post-CRW path is broken (Track B backend-fate work): (a) float64 food_grid "
    "dtype crash, (b) ~3.6-cell move-math divergence vs reference at one tick with "
    "identical CRW, (c) non-seeded global-np.random mortality. Flips to a hard failure "
    "(remove this xfail) once Cython is repaired."))
def test_cython_postcrw_matches_reference(monkeypatch):
    """SINGLE-TICK Cython post-CRW == Numba/NumPy reference, given identical Numba CRW.

    Currently xfails — documents the known Cython divergence (Finding #4). Do NOT 'fix'
    by loosening tolerance or deleting it; it is the guard that flips green when the
    Cython backend is repaired.
    """
    # Reference: Cython disabled -> Numba/NumPy post-CRW.
    monkeypatch.setattr(pop_mod, "_HAS_CYTHON", False)
    ref = _build_cy(11)
    ref.step()
    ref_x, ref_y, ref_e, ref_m = (ref.x.copy(), ref.y.copy(),
                                  ref.energy.copy(), ref.active_mask.copy())

    # Cython: enabled, same seed -> identical Numba CRW upstream at this first tick.
    monkeypatch.setattr(pop_mod, "_HAS_CYTHON", True)
    cy = _build_cy(11)
    cy.step()

    np.testing.assert_allclose(cy.x, ref_x, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(cy.y, ref_y, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(cy.energy, ref_e, rtol=1e-4, atol=1e-3)
    np.testing.assert_array_equal(cy.active_mask, ref_m)
```

- [ ] **Step 2: Run the tests**

Run: `micromamba run -n shiny python3 -m pytest tests/test_backend_equivalence.py -q 2>&1 | tail -20`
Expected: `test_cython_gate_is_engaged` **PASS**; `test_cython_postcrw_matches_reference` reported as **XFAIL** (the documented Cython divergence — `1 xfailed`). If it instead reports **XPASS** under `strict=True` (a *failure*), Cython has been fixed: STOP, delete the `xfail` marker so the test runs as a real green guard, and note it. Never loosen the tolerance or delete the test to silence it.

- [ ] **Step 3: Commit**

```bash
git -C /home/razinka/cenjas/CENOP add tests/test_backend_equivalence.py
git -C /home/razinka/cenjas/CENOP commit -m "test: single-tick Cython post-CRW equivalence (xfail: documents Cython divergence)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Document the backend-equivalence matrix

Record what is and is not comparable across backends, so future contributors do not chase impossible bit-identity — and point to the real equivalence/determinism tests that exist.

**Files:**
- Create: `docs/backend-equivalence.md`

- [ ] **Step 1: Confirm the existing kernel-equivalence test still passes**

Run: `micromamba run -n shiny python3 -m pytest "tests/test_numba_kernels.py::TestReflectBoundariesKernel::test_equivalence_with_numpy_version" -q 2>&1 | tail -3`
Expected: PASS — this is the one kernel with a pure NumPy counterpart (`Population._reflect_boundaries`), the template for any future kernel equivalence test.

- [ ] **Step 2: Write the matrix doc**

Create `docs/backend-equivalence.md`:

```markdown
# Backend equivalence & determinism — scope and limits

CENOP has four tick backends: NumPy reference, Numba (production), Cython
(`tick_cython.pyx`), and JAX (opt-in). This documents what we test for equivalence
and why some comparisons are impossible. (Code references use symbol names, not line
numbers, to avoid rot.)

## RNG architecture (why full *stochastic* cross-backend identity is impossible)

- The population's master stream is `self.rng = np.random.default_rng(random_seed)`.
- The Numba kernels are seeded from a *derived, separate* stream
  (`_seed_numba_rng(self.rng.integers(0, 2**31))`, called per step), and CRW draws
  happen *inside* the kernel. JAX is seeded from `random_seed` into its own PRNG.
- Consequence: the NumPy-fallback, Numba, and JAX paths consume different random
  streams in different orders. Even with the same `random_seed`, their stochastic
  trajectories (CRW headings, and everything downstream) diverge. This is a design
  property, not a tolerance issue — no `rtol/atol` makes them match.
- **Exception — the deterministic post-CRW stage IS comparable.** The Cython fast path
  replaces only the post-CRW pipeline (heading composition, move, food, energy,
  mortality) and consumes CRW outputs computed upstream with no internal RNG. Feeding
  both Cython and the reference identical Numba CRW makes a float-tolerance differential
  valid — see `test_cython_postcrw_matches_reference`.

## What we DO test

| Guard | Test | Scope |
|-------|------|-------|
| Per-backend determinism | `tests/test_backend_equivalence.py::test_{numba,numpy_fallback,jax}_backend_deterministic` | same seed -> identical full-tick trajectory, within each of NumPy-fallback / Numba / JAX (JAX skips on absence or GPU-OOM) |
| Cython post-CRW equivalence | `tests/test_backend_equivalence.py::test_cython_postcrw_matches_reference` | single-tick Cython post-CRW vs Numba/NumPy reference on identical CRW. **Currently `xfail(strict)`** — the Cython path is broken (see defects below); the test flips green when repaired |
| Cython kernel determinism | `tests/test_cython_tick.py::TestCythonFullPostCRW::test_deterministic_output` | same *fixed inputs* -> identical fused post-CRW kernel output (note: NOT same-seed reproducibility — Cython mortality uses unseeded global `np.random`) |
| Kernel vs NumPy reference | `tests/test_numba_kernels.py::TestReflectBoundariesKernel::test_equivalence_with_numpy_version` | `reflect_boundaries_kernel` == `Population._reflect_boundaries` (atol 1e-10) |

## Known Cython-backend defects (→ Track B backend-fate decision)

The Cython fast path is gated to homogeneous landscapes + `communication_enabled=False` +
no energy module, so it is **off in production** (comm defaults True). The Tier-0
equivalence work (running `test_cython_postcrw_matches_reference`) found it broken three
ways; all are tracked here for Track B (repair or remove the backend):

1. **Crash on float64 `food_grid`** — `cython_depons_post_crw` declares a float32
   `food_grid`, but `step()` passes `self.landscape._food_value` uncast and homogeneous
   landscapes store it float64 (`cell_data.py`); the path raises `ValueError: Buffer dtype
   mismatch` on its first tick. (One-line fix: cast at the call site or store float32.)
2. **Move-math divergence** — with bit-identical CRW inputs, Cython `x`/`y` diverge ~3.6
   cells from the Numba/NumPy reference at a single tick (energy matches). A real
   formula/units bug in `tick_cython.pyx`'s heading-composition/move section.
3. **Non-reproducible mortality** — Cython draws mortality from the *global* `np.random`
   (`tick_cython.pyx`), not the seeded `self.rng`, so the Cython path does not honor
   `random_seed` and desyncs `self.rng` relative to the reference across ticks.

## What we do NOT (and cannot) test

- **Full *stochastic* cross-backend trajectory identity** — impossible per the RNG
  architecture above (NumPy-fallback / Numba / JAX consume independent streams).
- **Per-kernel NumPy equivalence for the small fused kernels** — only
  `reflect_boundaries` has a pure NumPy counterpart. `_compute_turn_position` and the
  food/BMR/social kernels dispatch to or inline the kernels, so there is no independent
  NumPy reference without re-deriving the formula; they are covered by behavioral unit
  tests in `tests/test_numba_kernels.py`. (The *Cython* post-CRW stage is different — it
  mirrors a whole NumPy pipeline and IS equivalence-tested above.)
- **JAX numerical identity to Numba** — JAX runs float32 and uses a different
  (fixed-iteration batch-rejection) CRW algorithm; only statistical/range properties
  are meaningful, covered in `tests/test_jax_tick.py`.

## Adding a new kernel or backend path?

If a new path has a clean, independent reference (a pure NumPy kernel like
`reflect_boundaries`, or a deterministic stage fed identical RNG outputs like the Cython
post-CRW), add an equivalence test following those templates: identical inputs,
`np.testing.assert_allclose` (exact where integer/identical, documented tolerance where
float precision differs).
```

- [ ] **Step 3: Commit**

```bash
git -C /home/razinka/cenjas/CENOP add docs/backend-equivalence.md
git -C /home/razinka/cenjas/CENOP commit -m "docs: backend equivalence + determinism matrix

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (roadmap §4):**
- 0a "mark slow + run + triage" → Tasks 1-5 (marker infra, mark both files by measured duration, run fast suite, run + triage slow tier).
- 0a "fast suite unchanged" → Task 4 Step 1 (default suite green, no `--ignore`).
- 0b "differential test, exact where identical, tight rtol/atol where float reductions differ, document where bit-identity impossible" → Task 6 (per-backend determinism guards) + **Task 7 (single-tick Cython↔reference post-CRW differential — currently `xfail(strict)` documenting a real, deep-review-confirmed Cython divergence)** + Task 8 (matrix doc + Cython-defects section). Only the *stochastic* cross-backend trajectory comparison is dropped (Finding #3, spec-permitted). The deterministic cross-backend comparison the spec asked for exists as Task 7 and doubles as the regression guard that flips green when Cython is repaired.
- 0b "covers the four named backends / explicit skips with reasons" → NumPy-fallback/Numba/JAX determinism in Task 6 (JAX skips on absence OR GPU-OOM); Cython exercised through `step()` in Task 7 (skips when not built, xfails the known divergence). All Part B tests are fast.
- **Real-world gaps surfaced by empirically running both parts (Finding #4) and addressed:** JAX determinism made robust to GPU-OOM (Task 6); the no-CI / slow-tier-is-manual obligation flagged prominently (Task 5 status doc + CLAUDE.md); the stale `--ignore` dev alias called out for retirement (Task 4). Cython's three defects are NOT repaired here (out of Tier-0 scope) — they are guarded (xfail) and handed to Track B (Task 8 defects section).

**Placeholder scan:** no TBD/TODO. The only measured-and-confirm steps (Task 2/3 Step 1) give explicit candidate lists plus an objective >5 s rule; this is deliberate (durations are environment-dependent) and is not a placeholder.

**Type/name consistency:** `_HAS_KERNELS`, `_HAS_LAND_KERNEL`, `_HAS_JAX` are module globals in `cenop.agents.population` (verified at `population.py:53/60` and JAX seed at `:406`); Task 6 monkeypatches them on `pop_mod` and Task 6 Step 2 instructs verifying the JAX flag name before relying on it. `random_seed`, `use_jax`, `porpoise_count` are `SimulationParameters` fields (verified). `PorpoisePopulation(count, params, landscape)` and `.step()`, `.x/.y/.energy/.age/.active_mask` match usage in existing tests (`active_mask` verified at `population.py:115`).

**Deliberate scope note:** Part B does not attempt per-kernel NumPy equivalence for the *small fused* kernels beyond the existing `reflect_boundaries` test (they lack independent NumPy counterparts; covered by behavioral unit tests). It DOES add the Cython↔reference *post-CRW* differential (Task 7), the one cross-backend deterministic comparison that is cleanly buildable — landed as `xfail` because a second, *empirical* in-loop review (which ran both parts end-to-end) proved the Cython backend is genuinely broken; those defects are documented and handed to Track B rather than fixed in this foundation tier. Both the scope and the reasons are in `docs/backend-equivalence.md`. This plan was refined by two in-loop review rounds: a four-angle static review (round 1) and a four-angle review that executed the tasks (round 2).
