"""End-to-end parity: the injected-module movement path must equal the validated inline
(movement_module=None) path for a fixed seed. Forces the NumPy CRW branch so both paths use
the same crw_core generation.

Precondition for parity: no memory_module is injected (Simulation is built without one), so
self._avoidance_result stays None in both sims and the module path's memory-avoidance folding
is inert — matching the inline path, which applies no memory avoidance to movement."""

import numpy as np

import cenop.agents.population as popmod
from cenop.core.simulation import Simulation
from cenop.movement import DEPONSCRWMovementVectorized
from cenop.parameters import SimulationParameters


def _build(seed, with_module):
    params = SimulationParameters(
        porpoise_count=40, landscape="Homogeneous", sim_years=1, random_seed=seed
    )
    mod = DEPONSCRWMovementVectorized(params) if with_module else None
    return Simulation(params, movement_module=mod)


def _assert_parity(pa, pb, idx=slice(None)):
    np.testing.assert_allclose(pb.heading[idx], pa.heading[idx], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(pb.x[idx], pa.x[idx], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(pb.y[idx], pa.y[idx], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(pb.prev_angle[idx], pa.prev_angle[idx], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(pb.prev_log_mov[idx], pa.prev_log_mov[idx], rtol=1e-4, atol=1e-3)


def test_module_path_matches_inline_reference_one_tick(monkeypatch):
    monkeypatch.setattr(popmod, "_HAS_KERNELS", False)
    a = _build(2024, with_module=False)
    b = _build(2024, with_module=True)
    a.step()
    b.step()
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
        a.step()
        b.step()
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
    a.step()
    b.step()
    _assert_parity(a.population_manager, b.population_manager, idx=slice(0, 5))
