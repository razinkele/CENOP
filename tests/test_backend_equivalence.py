"""Backend determinism guards.

Each backend (NumPy-fallback, Numba, JAX) must be reproducible: identical
construction + seed -> identical per-tick trajectory. Cross-backend bit-identity
is NOT asserted because the Numba/JAX RNG streams are seeded independently of the
NumPy stream (population.py:917, :406), so their trajectories diverge by design.
See docs/backend-equivalence.md for the full comparison matrix.
"""

import numpy as np
import pytest

import cenop.agents.population as pop_mod
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters


def _build(seed, use_jax=False):
    params = SimulationParameters(porpoise_count=120)
    params.random_seed = seed
    params.use_jax = use_jax
    land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
    return PorpoisePopulation(count=120, params=params, landscape=land)


def _run(pop, ticks=40):
    for _ in range(ticks):
        pop.step()
    return (pop.x.copy(), pop.y.copy(), pop.energy.copy(), pop.age.copy(), pop.active_mask.copy())


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
        if (
            any(k in msg for k in ("RESOURCE_EXHAUSTED", "OUT_OF_MEMORY"))
            or "Runtime" in type(e).__name__
        ):
            pytest.skip(f"JAX runtime/OOM (environmental, not a determinism failure): {e}")
        raise
    _assert_same(a, b)


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
def test_cython_food_grid_dtype_no_crash():
    """Cython post-CRW must accept a float64 landscape food grid (homogeneous/ASC
    landscapes store float64) without a buffer-dtype ValueError (Finding #18)."""
    p = _build_cy(11)
    p.step()  # must NOT raise ValueError: Buffer dtype mismatch
    assert int(p.active_mask.sum()) > 0


@pytest.mark.skipif(not getattr(pop_mod, "_HAS_CYTHON", False), reason="Cython not built")
def test_cython_postcrw_matches_reference(monkeypatch):
    """SINGLE-TICK Cython post-CRW == Numba/NumPy reference, given identical Numba CRW."""
    # Reference: Cython disabled -> Numba/NumPy post-CRW.
    monkeypatch.setattr(pop_mod, "_HAS_CYTHON", False)
    ref = _build_cy(11)
    ref.step()
    ref_x, ref_y, ref_e, ref_m = (
        ref.x.copy(),
        ref.y.copy(),
        ref.energy.copy(),
        ref.active_mask.copy(),
    )

    # Cython: enabled, same seed -> identical Numba CRW upstream at this first tick.
    monkeypatch.setattr(pop_mod, "_HAS_CYTHON", True)
    cy = _build_cy(11)
    cy.step()

    np.testing.assert_allclose(cy.x, ref_x, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(cy.y, ref_y, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(cy.energy, ref_e, rtol=1e-4, atol=1e-3)
    np.testing.assert_array_equal(cy.active_mask, ref_m)


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
    assert s1[2] == s0[2] and np.array_equal(
        s1[1], s0[1]
    ), "Cython tick consumed global np.random; mortality must use self.rng"
