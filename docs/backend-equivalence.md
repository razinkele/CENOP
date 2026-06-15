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
