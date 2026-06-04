import numpy as np
import pytest
from cenop.behavior.sound import ShipDeterrenceModel


class TestModelExpAndArrays:
    def test_magnitude_uses_exp_link_day(self):
        """DEPONS predictMag returns exp(Mag) (Ship.java:395), not the raw linear term."""
        m = ShipDeterrenceModel()
        std = m.STD_MAG_DAY
        ns = (100.0 - std['noise_mean']) / std['noise_sd']
        ds = (5.0 - std['dist_mean']) / std['dist_sd']
        linear = (m.cship_int_day + m.cship_noise_day * ns
                  + m.cship_dist_day * ds + m.cship_dist_x_noise_day * ns * ds)
        expected = np.exp(linear)
        got = m.calculate_deterrence_magnitude(100.0, 5.0, is_day=True)
        assert got == pytest.approx(expected), f"magnitude should be exp(Mag)={expected}, got {got}"

    def test_probability_and_magnitude_accept_arrays(self):
        """Both model functions must vectorize over arrays for the kernel."""
        m = ShipDeterrenceModel()
        rl = np.array([90.0, 110.0, 130.0])
        dist_km = np.array([1.0, 3.0, 8.0])
        p = m.calculate_deterrence_probability(rl, dist_km, is_day=True)
        mag = m.calculate_deterrence_magnitude(rl, dist_km, is_day=True)
        assert p.shape == (3,) and mag.shape == (3,)
        assert np.all((p >= 0.0) & (p <= 1.0))
        assert np.all(mag > 0.0)  # exp(x) > 0 always


class TestDeterrenceComponents:
    def _model(self):
        return ShipDeterrenceModel()

    def test_gate_excludes_rl_at_or_below_tships(self):
        m = self._model()
        rl = np.array([80.0, 80.0001])
        dist_m = np.array([2000.0, 2000.0])
        gdx = np.array([5.0, 5.0]); gdy = np.array([0.0, 0.0])
        u = np.array([0.0, 0.0])  # always reacts if gated & prob>0
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, is_day=True, u_draw=u, tships=80.0)
        assert react[0] == False and react[1] == True  # strict > 80

    def test_reacts_iff_u_below_prob(self):
        m = self._model()
        rl = np.array([130.0, 130.0]); dist_m = np.array([1500.0, 1500.0])
        gdx = np.array([3.0, 3.0]); gdy = np.array([0.0, 0.0])
        p = m.calculate_deterrence_probability(130.0, 1.5, True)
        u = np.array([float(p) - 1e-6, float(p) + 1e-6])
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, u, tships=80.0)
        assert react[0] == True and react[1] == False  # strict u < prob

    def test_vector_is_grid_disp_over_metre_distance_times_mag(self):
        """DEPONS Ship.java:231-235: unit vector = grid displacement / metre distance."""
        m = self._model()
        rl = np.array([130.0]); dist_m = np.array([2000.0])
        gdx = np.array([5.0]); gdy = np.array([0.0])  # 5 cells east
        u = np.array([0.0])
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, u, tships=80.0)
        assert react[0]
        assert vx[0] == pytest.approx((5.0 / 2000.0) * mag[0])
        assert vy[0] == pytest.approx(0.0)
        assert vx[0] > 0  # pushed east, away from ship

    def test_zero_vector_when_not_reacting(self):
        m = self._model()
        rl = np.array([130.0]); dist_m = np.array([2000.0])
        gdx = np.array([5.0]); gdy = np.array([0.0])
        u = np.array([1.0])  # never reacts
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, u, tships=80.0)
        assert react[0] == False
        assert vx[0] == 0.0 and vy[0] == 0.0
        assert mag[0] > 0.0  # magnitude still computed, just not applied


class TestVectorizedPath:
    def _mgr_with_ship(self, sx=50.0, sy=50.0, sl=170.0):
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        s = Ship(id=1, x=sx, y=sy, vessel_type=VesselClass.CARGO)
        s._is_active = True
        s.noise.base_source_level = sl
        mgr = ShipManager([s]); mgr.enabled = True
        return mgr, s

    def _params(self):
        from cenop.parameters.simulation_params import SimulationParameters
        return SimulationParameters()

    def test_10km_cap_boundary(self):
        """Default deter_max_distance=1000 km, but ships are capped at 10 km."""
        mgr, s = self._mgr_with_ship()
        p = self._params()
        near_x = np.array([50.0 + 9900.0 / 400.0]); y = np.array([50.0])
        far_x = np.array([50.0 + 10100.0 / 400.0])
        dxn, _ = mgr.calculate_aggregate_deterrence_vectorized(near_x, y, p, base_seed=1, tick=1)
        dxf, _ = mgr.calculate_aggregate_deterrence_vectorized(far_x, y, p, base_seed=1, tick=1)
        assert dxf[0] == 0.0

    def test_deter_max_distance_tightens_cap(self):
        mgr, s = self._mgr_with_ship()
        p = self._params(); p.deter_max_distance = 5.0  # km -> cap = 5 km
        x = np.array([50.0 + 6000.0 / 400.0]); y = np.array([50.0])  # 6 km
        dx, _ = mgr.calculate_aggregate_deterrence_vectorized(x, y, p, base_seed=1, tick=1)
        assert dx[0] == 0.0  # beyond 5 km cap

    def test_min_distance_floor(self):
        mgr, s = self._mgr_with_ship()
        p = self._params()  # deter_min_distance_ships = 0.1 km = 100 m
        x99 = np.array([50.0 + 99.0 / 400.0]); y = np.array([50.0])   # 99 m
        dx99, _ = mgr.calculate_aggregate_deterrence_vectorized(x99, y, p, base_seed=1, tick=1)
        assert dx99[0] == 0.0  # inside the 100 m floor -> excluded

    def test_deter_coeff_does_not_affect_ship_vector(self):
        """Ships must NOT use deter_coeff (turbine-only). _force_u=0 guarantees a reaction
        (the DEPONS ship prob caps ~0.2, so seeded draws can't be relied on to react)."""
        mgr, s = self._mgr_with_ship(sl=200.0)
        x = np.array([50.0 + 2000.0 / 400.0]); y = np.array([50.0])  # 2 km
        p1 = self._params(); p1.deter_coeff = 0.012
        p2 = self._params(); p2.deter_coeff = 0.5
        dx1, _ = mgr.calculate_aggregate_deterrence_vectorized(x, y, p1, _force_u=0.0)
        dx2, _ = mgr.calculate_aggregate_deterrence_vectorized(x, y, p2, _force_u=0.0)
        assert dx1[0] != 0.0                       # precondition: actually deterred (non-vacuous)
        assert dx1[0] == pytest.approx(dx2[0])     # deter_coeff has no effect on ships

    def test_loudest_ship_wins_not_sum(self):
        """Two ships near one porpoise -> only the higher-RL ship contributes (DEPONS recordStep)."""
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        p = self._params()
        loud = Ship(id=1, x=48.0, y=50.0, vessel_type=VesselClass.CARGO); loud._is_active = True
        loud.noise.base_source_level = 195.0
        quiet = Ship(id=2, x=60.0, y=50.0, vessel_type=VesselClass.CARGO); quiet._is_active = True
        quiet.noise.base_source_level = 175.0
        px = np.array([50.0]); py = np.array([50.0])
        mgr = ShipManager([loud, quiet]); mgr.enabled = True
        mgr_loud = ShipManager([loud]); mgr_loud.enabled = True
        mgr_quiet = ShipManager([quiet]); mgr_quiet.enabled = True
        dx_both, _ = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        dx_loud, _ = mgr_loud.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        dx_quiet, _ = mgr_quiet.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        assert dx_loud[0] != 0.0 and dx_quiet[0] != 0.0     # both would deter alone
        assert dx_both[0] == pytest.approx(dx_loud[0])      # loudest wins...
        assert dx_both[0] != pytest.approx(dx_loud[0] + dx_quiet[0])  # ...NOT a sum

    def test_order_and_membership_invariance(self):
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        p = self._params()
        a = Ship(id=1, x=49.0, y=50.0, vessel_type=VesselClass.CARGO); a._is_active = True; a.noise.base_source_level = 195.0
        b = Ship(id=2, x=51.0, y=50.0, vessel_type=VesselClass.CARGO); b._is_active = True; b.noise.base_source_level = 190.0
        far = Ship(id=3, x=500.0, y=900.0, vessel_type=VesselClass.CARGO); far._is_active = True; far.noise.base_source_level = 195.0
        px = np.array([50.0]); py = np.array([50.0])
        ab = ShipManager([a, b]); ab.enabled = True
        ba = ShipManager([b, a]); ba.enabled = True
        abfar = ShipManager([a, b, far]); abfar.enabled = True
        r_ab = ab.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        r_ba = ba.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        r_abfar = abfar.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        assert r_ab[0][0] != 0.0                            # precondition: non-vacuous
        assert r_ab[0][0] == pytest.approx(r_ba[0][0])      # order invariant
        assert r_ab[0][0] == pytest.approx(r_abfar[0][0])   # far out-of-range ship has no effect


class TestSimulationDeterminism:
    def _sim_with_ship(self):
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.core.simulation import Simulation
        from cenop.agents.ship import Ship, VesselClass
        params = SimulationParameters(porpoise_count=50, sim_years=1, random_seed=42, ships_enabled=True)
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        sim = Simulation(params=params, cell_data=land, seed=42)
        sim.initialize()
        # Replace any sample ship with one guaranteed near the porpoises
        loud = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        loud._is_active = True; loud.noise.base_source_level = 195.0
        sim._ship_manager.ships = [loud]; sim._ship_manager.enabled = True
        return sim

    def test_tick_varies_ship_draws_after_wiring(self):
        """Before wiring, every tick uses base_seed=0/tick=0, so the per-ship reaction
        draw is IDENTICAL every tick; after wiring (tick threaded) the draw varies.
        Hold porpoise positions fixed so the only source of variation is the seed/tick."""
        import numpy as np
        sim = self._sim_with_ship()
        pm = sim.population_manager
        n0 = int(pm.active_mask.sum())
        snaps = []
        for _ in range(30):
            # freeze porpoises so deter_strength changes ONLY if the ship draw changes
            pre_x, pre_y = pm.x.copy(), pm.y.copy()
            sim.step()
            pm.x[:] = pre_x; pm.y[:] = pre_y
            pm._recompute_cell_indices()
            # guard: no births/deaths, so any deter_strength change is purely the draw
            assert int(pm.active_mask.sum()) == n0
            snaps.append(pm.deter_strength.copy())
        # With tick threaded, at least two ticks must differ (draws vary by tick).
        assert any(not np.array_equal(snaps[0], s) for s in snaps[1:])

    def test_reproducible_across_sequential_runs(self):
        """Green guard: two identically-seeded sims (run SEQUENTIALLY to avoid global
        np.random cross-talk) produce identical ship deterrence."""
        import numpy as np
        s1 = self._sim_with_ship()
        for _ in range(5):
            s1.step()
        d1 = s1.population_manager.deter_strength.copy()
        s2 = self._sim_with_ship()   # constructed AFTER s1 finishes
        for _ in range(5):
            s2.step()
        d2 = s2.population_manager.deter_strength.copy()
        np.testing.assert_array_equal(d1, d2)
