# CENOP Balanced Roadmap — Validation, Fidelity, Engineering Health

**Date:** 2026-06-12
**Status:** Approved roadmap (sequencing/priority layer; each tier gets its own spec + plan when picked up)
**Branch:** authored on `CENOP-JASMINE`

## 1. Purpose

CENOP's DEPONS 3.2 parity work and the ship-deterrence path are complete and merged,
and the Numba/Cython tick is fast. What is missing is **trust** (the model's emergent
biology is not routinely tested, and fidelity to the DEPONS Java reference is unverified)
and a **deliberate engineering strategy** (four parallel tick backends, an unresolved
2.6× speed gap to Java). This roadmap sequences that remaining work as a foundation tier
plus two parallel tracks, with explicit acceptance criteria, so any tier can be picked up
independently later.

This is a *roadmap*, not a single implementation spec. Each tier below is its own
sub-project: when started, it gets a dedicated design spec (`docs/superpowers/specs/`)
and implementation plan (`docs/superpowers/plans/`). This document is the layer above
them — what to do, in what order, and what "done" means for each.

## 2. Current state (facts this roadmap builds on)

- **Tests:** 680 fast tests pass (`pytest tests/ --ignore=test_depons_physiology.py
  --ignore=test_validation.py`). Those two excluded files are **not broken — they are
  slow**: they run 17,280–34,560-tick (1–2 year) simulations in Python loops to validate
  emergent demographics (lifespan, age structure, physiology). They are the
  scientifically most important tests and are currently never executed.
- **Parity:** DEPONS 3.2 sync complete (energy, reproduction/mortality, movement/memory,
  deterrence/dispersal); ship-deterrence faithful + sub-tick interpolated.
- **Performance:** Numba tick ≈ 2.22 ms/tick (N=500 Kattegat) vs DEPONS Java ≈ 0.84 ms
  (2.6× gap). JAX full-tick ≈ 11.7 ms/tick (≈5× slower than Numba; effectively shelved).
- **Backends:** four tick implementations exist — NumPy (reference), Numba (production
  fast path), Cython (`tick_cython.pyx`, wired into `step()`), JAX (`tick_jax.py` +
  `jax_kernels.py`, opt-in `use_jax=False`). The JAX path **consumes** ship-deterrence
  vectors computed by the shared NumPy aggregator (`ship.py`); it does not recompute them,
  so deterrence semantics stay consistent across backends.
- **Baselines:** committed Kattegat references for undisturbed / ships / turbines
  (`output/kattegat_ref{,_ships,_turbines}/`) — these prove CENOP **self-consistency**,
  not fidelity to DEPONS Java.
- **Reference Java present:** `DEPONS-3.2/` is a full Repast Simphony project (`DEPONS.rs`,
  `src/dk/au/bios/porpoise/`, batch launchers); Java 21 is available. Repast runtime libs
  are **not** vendored, and landscape data must be fetched — so running it headless is
  feasible but has real setup cost.

## 3. Structure

```
Tier 0 — Foundation (do first; unblocks both tracks)
        ├── 0a Restore the validation suite (slow tier) + run it
        └── 0b Backend-equivalence differential test
                    │
        ┌───────────┴───────────┐
   Track A (science)       Track B (engineering)
   A1 DEPONS-Java harness   B1 Decide JAX backend fate
   A2 Quantitative          B2 Profile + attack the
      cross-validation         2.6× Numba→Java gap
```

Tier 0 first. Then Track A and Track B run in parallel — A is the scientific payoff, B is
the engineering payoff; neither blocks the other. Within B, **B1 before B2** (do not
optimize a backend that might be cut).

## 4. Tier 0 — Foundation

**Rationale:** Establishes the safety net that makes every later change trustworthy. Cheap
relative to its value.

### 0a — Restore the validation suite as a `slow` tier
- Mark every long-running test in `tests/test_validation.py` and
  `tests/test_depons_physiology.py` with `@pytest.mark.slow` (register the marker in
  `pytest.ini`/`conftest.py`); default `pytest` deselects `slow`, a dedicated target runs
  only them.
- Actually execute them and record the outcome. Triage each failure as either a real
  regression (fix it) or an outdated expectation (update with justification, never
  silently). Reduce per-test tick counts only where the biological assertion still holds
  at the shorter horizon, and say so.
- **Acceptance:** `pytest -m slow` runs to completion; results documented (pass, or each
  failure triaged with a decision). The fast suite is unchanged.

### 0b — Backend-equivalence differential test
- A seeded test that runs the same short scenario through NumPy, Numba, Cython, and (where
  applicable) JAX and asserts identical per-tick outputs (positions, energy, deterrence,
  population) within a documented tolerance — exact where the math is integer/identical,
  tight `rtol/atol` where float reductions differ by backend.
- Where a backend legitimately cannot match bit-for-bit (e.g. JAX float32 vs float64),
  document the tolerance and *why*; do not loosen silently.
- **Acceptance:** the differential test is green across all available backends and is part
  of the fast suite; any backend it cannot cover is explicitly skipped with a reason.

## 5. Track A — Scientific fidelity (highest value)

**Rationale:** Self-consistent baselines do not prove the port reproduces DEPONS. This is
the evidence that makes CENOP defensible for research use.

### A1 — DEPONS-Java headless harness (feasibility spike first)
- Time-boxed spike: stand up a single headless Repast batch run of `DEPONS-3.2` (vendor
  Repast Simphony runtime, fetch the landscape(s), script a batch via the existing
  `batch/` config + `Batch DEPONS Model.launch`). Capture its `Statistics`/`Population`
  output.
- If the spike shows the setup cost is disproportionate, record that finding and fall back
  to comparing against **published DEPONS output** (TRACE documents / released result
  files) instead of a live run — still valid cross-validation, lower cost.
- **Acceptance:** either a reproducible headless DEPONS run producing population/energy
  output, or a documented decision to use published reference output, with the harness/
  procedure committed.

### A2 — Quantitative cross-validation
- Run CENOP and the DEPONS reference on identical scenario + seed (start with undisturbed
  Kattegat, then one disturbed scenario — ships or turbines).
- Compare distributions/trajectories statistically: KS test on per-day population, energy,
  and dispersal distributions; trajectory envelopes (mean ± band) overlaid. Define
  pass bands *before* running.
- **Acceptance:** a committed parity report covering ≥1 undisturbed + ≥1 disturbed
  scenario, with the statistical comparison and an explicit fidelity verdict per metric.

## 6. Track B — Engineering health & speed

**Rationale:** Reduce the maintenance tax of four backends and decide whether the 2.6×
gap is worth closing — but only after Tier 0b can prove changes preserve behavior.

### B1 — Decide the JAX **and Cython** backends' fate
- With 0b quantifying the cost of keeping JAX correct, choose for JAX: (i) invest (GPU
  batching to make it competitive for large N), or (ii) deprecate and remove it (cut the
  surface that every cross-cutting change must keep in sync — e.g. the sub-tick
  `turbine_deter_strength` plumbing).
- **Cython, now with hard evidence (Tier 0b):** the Cython fast path is broken in three
  ways — float64 `food_grid` dtype crash, ~3.6-cell post-CRW move-math divergence from the
  reference, and non-seeded global-`np.random` mortality (so it ignores `random_seed`). It
  is gated off in production (comm defaults True), so this is latent. Decide: repair it
  (fix the crash + move math + RNG source, then flip the Tier-0b `xfail` to a green guard),
  or remove it. See `docs/backend-equivalence.md` "Known Cython-backend defects" and
  `tests/test_backend_equivalence.py::test_cython_postcrw_matches_reference`.
- **Acceptance:** a short decision record per backend (context, options, choice, rationale)
  committed, and the chosen action executed (investment plan opened, or backend removed with
  tests/docs updated; if Cython is repaired, the `xfail` marker is removed).

### B2 — Profile and attack the 2.6× Numba→Java gap
- Use `/profile` on the N=500 Kattegat tick to rank hot spots (likely CRW, land avoidance,
  ref-mem). Address the top contributor; re-measure.
- Every change guarded by Tier 0a (validation) and 0b (equivalence) — behavior-preserving
  unless a divergence is explicitly intended and re-baselined.
- **Acceptance:** a committed profile, the top hot-spot either optimized (with before/after
  numbers) or documented as a reasoned "acceptable gap"; baselines regenerated only if a
  change is intentionally non-bit-identical.

## 7. Cross-cutting principles

- **Behavior-preserving by default.** Optimizations (Track B, and any 0a test-count
  reduction) must be bit-identical unless a divergence is intentional; intentional
  divergences re-baseline and say so. This is the discipline the recent ship-deterrence
  perf work already followed.
- **Each tier is independently shippable** and lands on `CENOP-JASMINE` via its own
  spec → plan → subagent-driven execution → review → merge cycle.
- **No silent scope creep.** A tier that uncovers adjacent work records it here as a
  future item rather than absorbing it.

## 8. Non-goals

- Re-opening settled DEPONS parity decisions (scalar-oracle sub-tick, buoy-teleport
  movement model, reproducing DEPONS global-RNG draw order — all documented non-goals).
- Building new scientific scenarios / sensitivity analyses (a separate research effort, not
  part of hardening the port).
- A unified "one backend to rule them all" rewrite — B1 decides backend strategy
  incrementally, it does not mandate consolidation.

## 9. Immediate next step

Tier 0 is the concrete starting point. It will get a dedicated implementation plan
(`writing-plans`) covering 0a and 0b, executed via subagent-driven development. Tracks A
and B get their own specs + plans when scheduled.
