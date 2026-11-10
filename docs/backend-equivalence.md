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
| Per-backend determinism | `tests/test_backend_equivalence.py::test_{numba_backend,numpy_fallback,jax_backend}_deterministic` | same seed -> identical full-tick trajectory, within each of NumPy-fallback / Numba / JAX (JAX skips on absence or GPU-OOM) |
| Cython post-CRW equivalence | `tests/test_backend_equivalence.py::test_cython_postcrw_matches_reference` | single-tick Cython post-CRW vs Numba/NumPy reference on identical CRW. Passing since the v2.2.0 repair (was `xfail(strict)` while the path was broken — see "Cython-backend defects" below) |
| Cython float64 food grid | `tests/test_backend_equivalence.py::test_cython_food_grid_dtype_no_crash` | a float64 landscape food grid runs a Cython tick without a buffer-dtype `ValueError` |
| Cython seeded mortality | `tests/test_backend_equivalence.py::test_cython_mortality_uses_seeded_rng` | Cython mortality draws come from the seeded `self.rng`; global `np.random` is untouched |
| Cython kernel determinism | `tests/test_cython_tick.py::TestCythonFullPostCRW::test_deterministic_output` | same *fixed inputs* -> identical fused post-CRW kernel output |
| Kernel vs NumPy reference | `tests/test_numba_kernels.py::TestReflectBoundariesKernel::test_equivalence_with_numpy_version` | `reflect_boundaries_kernel` == `Population._reflect_boundaries` (atol 1e-10) |

## Cython-backend defects — REPAIRED in v2.2.0

The Cython fast path is gated to homogeneous landscapes + `communication_enabled=False` +
no energy module, so it is **off in production** (comm defaults True). The Tier-0
equivalence work (running `test_cython_postcrw_matches_reference`) found it broken three
ways. Track B chose to **repair** it (2026-07 review-fixes, Phase 6); all three are fixed
and guarded by the tests in the table above:

1. **Crash on float64 `food_grid`** (Finding #18) — the kernel declares a float32
   `food_grid`, but homogeneous landscapes store food as float64, so the first tick raised
   `ValueError: Buffer dtype mismatch`. Fixed by casting to float32 at the call site and
   writing depletion back to the landscape grid (`a0c5c00`).
2. **~3.6-cell move divergence** (Finding #19) — the root cause was a missing post-move
   land/depth rollback, so agents could step onto land. Fixed by adding the rollback
   (`7beeaec`).
3. **Non-reproducible mortality** (Finding #21) — mortality drew from the *global*
   `np.random`. It now uses a draw from the seeded `self.rng` passed in by `step()`, so the
   path honors `random_seed` (`7beeaec`).

Also fixed: reflected (boundary-bounced) agents now get their heading recomputed after
reflection, matching DEPONS `forward()` (Finding #29, `cdfd8cf`).

The compiled extension is gitignored. After a fresh checkout, rebuild it from
`src/cenop/optimizations/` with `python3 setup_cython.py build_ext --inplace` (as CI does);
otherwise the Cython tests skip.

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
