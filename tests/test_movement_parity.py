"""End-to-end parity: the injected-module movement path must equal the validated inline
(movement_module=None) path for a fixed seed. Forces the NumPy CRW branch so both paths use
the same crw_core generation.

Precondition for parity: no memory_module is injected (Simulation is built without one), so
self._avoidance_result stays None in both sims and the module path's memory-avoidance folding
is inert — matching the inline path, which applies no memory avoidance to movement."""

import numpy as np

import cenop.agents.population as popmod
from cenop.agents.population import PorpoisePopulation
from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import create_homogeneous_landscape
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
    """Dispersers take the same DEPONS dispersal step (_disperse_step: completion check, PSM
    turn, look-ahead stops) on the module path as on the inline path. The module path
    previously used a *0.3 shortcut and skipped _apply_dispersal_heading."""
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
    for sim in (a, b):
        # Non-vacuity: a disperser that stopped takes a CRW step, so check one continued
        assert sim.population_manager.is_dispersing[:5].any(), "all 5 dispersers stopped"
    _assert_parity(a.population_manager, b.population_manager, idx=slice(0, 5))


def _coast_pop(with_module, seed=11):
    """Renamed 200 x 200 grid with a coastline (land at y >= 100), so _skip_land_avoidance is
    False and the D15 land check runs. 30 agents just south of the coast with headings spread
    over 360 deg (tier turns), agent 30 boxed in a one-cell pocket inside the land with a
    remembered open-water position (beh 4), the rest in open water."""
    params = SimulationParameters(random_seed=seed)
    land = create_homogeneous_landscape(200, 200)
    land._depth[:] = 30.0
    land._depth[100:, :] = -10.0
    land._depth[150, 100] = 30.0  # the pocket
    land.landscape_name = "TestCoast"
    mod = DEPONSCRWMovementVectorized(params) if with_module else None
    pop = PorpoisePopulation(40, params, land, movement_module=mod)
    assert not pop._skip_land_avoidance
    pop.is_dispersing[:] = False
    pop.x[:30] = np.linspace(20.5, 179.5, 30)
    pop.y[:30] = 98.5
    pop.heading[:30] = np.linspace(0.0, 360.0, 30, endpoint=False)
    pop.x[30], pop.y[30] = 100.5, 150.5
    pop._pos_history_x[30, 0], pop._pos_history_y[30, 0] = 80.5, 60.5
    pop._mem_ptr[30], pop._mem_count[30] = 1, 1
    pop.x[31:] = np.linspace(30.5, 170.5, 9)
    pop.y[31:] = 40.5
    pop._recompute_cell_indices()
    return pop


def test_module_path_matches_inline_on_coast(monkeypatch):
    """D15: both CRW paths run the land check at the same point (after reference memory and
    the CRW turn, before composition) with the same RNG seeding, so they agree on a coast."""
    monkeypatch.setattr(popmod, "_HAS_KERNELS", False)
    seen = {}
    orig = PorpoisePopulation._depons_land_check

    def check(self, mask):
        x0, h0 = self.x.copy(), self.heading.copy()
        orig(self, mask)
        seen[id(self)] = (x0, h0, self.x.copy(), self.heading.copy())

    monkeypatch.setattr(PorpoisePopulation, "_depons_land_check", check)
    pa, pb = _coast_pop(with_module=False), _coast_pop(with_module=True)
    pa.step()
    pb.step()

    for p in (pa, pb):  # non-vacuous: tier turns on the coast, a beh-4 walk-back in the pocket
        x0, h0, x1, h1 = seen[id(p)]
        turn = np.abs((h1 - h0 + 180.0) % 360.0 - 180.0)
        assert turn[:30].max() >= 40.0
        assert x1[30] != x0[30]
    np.testing.assert_array_equal(pb.x, pa.x)
    np.testing.assert_array_equal(pb.y, pa.y)
    np.testing.assert_array_equal(pb.heading, pa.heading)
    np.testing.assert_array_equal(pb._enough_water_ahead, pa._enough_water_ahead)
    _assert_parity(pa, pb)
