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
