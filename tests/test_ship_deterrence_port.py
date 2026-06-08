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


class TestScalarOracleConsistency:
    def test_scalar_matches_kernel(self):
        """Ship.calculate_deterrence must agree with the shared kernel for a fixed u."""
        import numpy as np
        from cenop.agents.ship import Ship, VesselClass
        from cenop.behavior.sound import ShipDeterrenceModel
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = 180.0
        px, py = 50.0 + 2000.0 / 400.0, 50.0  # 2 km east
        # Force u=0 by monkeypatching np.random in the scalar path
        orig = np.random.random
        np.random.random = lambda *a, **k: 0.0
        try:
            should, prob, mag, dkm = s.calculate_deterrence(px, py, p, is_day=True)
        finally:
            np.random.random = orig
        # Independently compute via kernel, using the SAME RL the implementation uses.
        # (ShipNoise.get_source_level returns the JOMOPANS band-12 SL when no override is set,
        #  so RL != base_source_level - TL; derive it from the ship's own received-level method.)
        m = ShipDeterrenceModel()
        gdx = np.array([px - 50.0]); gdy = np.array([0.0])
        dist_m = np.array([2000.0])
        rl = np.array([max(0.0, s.get_received_level(px, py, p.alpha_hat, p.beta_hat, 400.0))])
        _, _, kprob, kmag, kreact = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, np.array([0.0]), tships=p.deter_ships_min_db)
        assert should == bool(kreact[0])
        assert should is True  # u=0 + 180 dB ship at 2 km must react; guards against both-False pass
        assert prob == pytest.approx(float(kprob[0]))
        assert mag == pytest.approx(float(kmag[0]))

    def test_min_distance_boundary_strict(self):
        """DEPONS uses strict '>' at the min-distance floor (Ship.java:220)."""
        from cenop.agents.ship import Ship, VesselClass
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()  # min = 0.1 km = 100 m
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = 200.0
        at_floor = 50.0 + 100.0 / 400.0   # exactly 100 m
        should, *_ = s.calculate_deterrence(at_floor, 50.0, p, is_day=True)
        assert should == False  # 100 m is excluded (strict >)


class TestEdgeCases:
    def _model(self):
        return ShipDeterrenceModel()

    def test_u_equals_prob_does_not_react(self):
        """Strict '<': u == prob must NOT react."""
        m = self._model()
        rl = np.array([130.0]); dist_m = np.array([1500.0])
        p = float(m.calculate_deterrence_probability(130.0, 1.5, True))
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, np.array([3.0]), np.array([0.0]), True, np.array([p]), tships=80.0)
        assert react[0] == False

    def test_day_night_select_different_coefficients(self):
        """Night uses different std/coeffs; pship_noise_night=0 makes prob noise-independent."""
        m = self._model()
        p_lo = m.calculate_deterrence_probability(90.0, 2.0, is_day=False)
        p_hi = m.calculate_deterrence_probability(150.0, 2.0, is_day=False)
        assert float(p_lo) == pytest.approx(float(p_hi))  # night prob independent of RL
        # Day prob DOES depend on RL:
        d_lo = m.calculate_deterrence_probability(90.0, 2.0, is_day=True)
        d_hi = m.calculate_deterrence_probability(150.0, 2.0, is_day=True)
        assert float(d_hi) > float(d_lo)

    def test_porpoise_on_ship_gives_finite_zero_direction(self):
        """dist clamps to 1 m; grid disp 0 -> zero (defined, finite) vector."""
        m = self._model()
        vx, vy, *_ = m.deterrence_components(
            np.array([200.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]),
            True, np.array([0.0]), tships=80.0)
        assert np.isfinite(vx[0]) and vx[0] == 0.0 and vy[0] == 0.0

    def test_nodata_cell_gives_zero_rl_weston(self):
        """WestonFlux path: a NODATA cell -> RL 0 -> no deterrence (DEPONS Ship.java:300)."""
        import numpy as np
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        params = SimulationParameters(); params.weston_flux_percell = True
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        land._depth[:] = -9999.0  # all NODATA depth
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = 210.0
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([50.5]); py = np.array([50.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(
            px, py, params, cell_data=land, _force_u=0.0)
        assert dx[0] == 0.0 and dy[0] == 0.0  # NODATA -> RL 0 -> gate fails even with forced react


class TestIntegration:
    def _sim(self, source_level):
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.core.simulation import Simulation
        from cenop.agents.ship import Ship, VesselClass
        params = SimulationParameters(porpoise_count=200, sim_years=1, random_seed=42, ships_enabled=True)
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        sim = Simulation(params=params, cell_data=land, seed=42); sim.initialize()
        ship = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        ship._is_active = True; ship.noise.base_source_level = source_level
        sim._ship_manager.ships = [ship]; sim._ship_manager.enabled = True
        return sim

    def test_loud_ship_deters_some_porpoises(self):
        import numpy as np
        sim = self._sim(source_level=210.0)
        for _ in range(10):
            sim.step()
        assert float(np.max(sim.population_manager.deter_strength)) > 0.0

    def test_quiet_ship_below_tships_does_not_deter(self):
        import numpy as np
        sim = self._sim(source_level=70.0)  # RL stays below Tships=80 everywhere
        for _ in range(10):
            sim.step()
        assert float(np.max(sim.population_manager.deter_strength)) == 0.0


class TestCharacterizationSnapshot:
    """Pinned reference values for the kernel — locks the new behavior against drift."""

    def test_kernel_snapshot_day(self):
        import numpy as np
        from cenop.behavior.sound import ShipDeterrenceModel
        m = ShipDeterrenceModel()
        rl = np.array([160.0, 100.0, 79.0])          # high / mid / below-Tships
        dist_m = np.array([800.0, 3000.0, 500.0])
        gdx = np.array([2.0, -7.5, 1.25]); gdy = np.array([0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0, 0.0])                # force reaction where gated
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, is_day=True, u_draw=u, tships=80.0)
        # Sanity: 3rd porpoise gated out (RL 79 < 80).
        assert react.tolist() == [True, True, False]
        assert vx[0] > 0.0 and vx[1] < 0.0 and vx[2] == 0.0
        # PINNED reference values (computed 2026-06-04):
        np.testing.assert_allclose(vx, [0.0661084953394404, -0.054404584935397474, 0.0], rtol=1e-9)
        np.testing.assert_allclose(
            mag,
            [26.443398135776157, 21.76183397415899, 21.221962101845463],
            rtol=1e-9,
        )


class TestDeterStrengthL2:
    def test_deter_strength_is_euclidean(self):
        """DEPONS ShipDeterrence.java:75 uses sqrt(dx^2+dy^2), not |dx|+|dy|."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=1)
        land = create_homogeneous_landscape(width=50, height=50, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=1, params=params, landscape=land)
        d_dx = np.array([3.0], dtype=np.float64)
        d_dy = np.array([4.0], dtype=np.float64)
        pop.step(deterrence_vectors=(d_dx, d_dy))
        assert pop.deter_strength[0] == pytest.approx(5.0)  # hypot(3,4), not 7


class TestVectorizedPerfRefactor:
    def test_mixed_in_and_out_of_range_correct(self):
        """In-range porpoises deterred; out-of-range exactly zero; one vectorized call."""
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = 205.0
        mgr = ShipManager([s]); mgr.enabled = True
        # idx 0: in-range 2 km east; idx 1: out-of-range 50 km east; idx 2: in-range 1 km west
        px = np.array([50.0 + 2000.0/400.0, 50.0 + 50000.0/400.0, 50.0 - 1000.0/400.0])
        py = np.array([50.0, 50.0, 50.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        assert dx[0] > 0.0          # pushed east (in-range, reacting)
        assert dx[1] == 0.0 and dy[1] == 0.0   # out of 10 km cap -> exactly zero
        assert dx[2] < 0.0          # pushed west (in-range, reacting)

    def test_seed_order_invariance_still_holds(self):
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()
        def mk(sid, sx):
            s = Ship(id=sid, x=sx, y=50.0, vessel_type=VesselClass.CARGO)
            s._is_active = True; s.noise.base_source_level = 195.0
            return s
        a, b = mk(1, 49.0), mk(2, 51.0)
        px = np.array([50.0]); py = np.array([50.0])
        r1 = ShipManager([a, b]); r1.enabled = True
        r2 = ShipManager([b, a]); r2.enabled = True
        d1 = r1.calculate_aggregate_deterrence_vectorized(px, py, p, base_seed=3, tick=7)
        d2 = r2.calculate_aggregate_deterrence_vectorized(px, py, p, base_seed=3, tick=7)
        np.testing.assert_array_equal(d1[0], d2[0])
        np.testing.assert_array_equal(d1[1], d2[1])


class TestIsDisturbedThreshold:
    def test_ship_magnitude_reports_disturbed(self):
        """is_disturbed must fire for ship-scale deterrence (~0.04), matching the
        disturbance-memory threshold (>0.01), not the old >0.1."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=1)
        land = create_homogeneous_landscape(width=50, height=50, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=1, params=params, landscape=land)
        pop.deter_strength[0] = 0.04
        df = pop.to_dataframe()
        assert bool(df["is_disturbed"].iloc[0]) is True


class TestJomopansSourceLevel:
    def test_default_uses_jomopans(self):
        """With no base_source_level override, get_source_level == jomopans_spl band 12."""
        from cenop.behavior.sound import ShipNoise
        from cenop.behavior.jomopans_spl import jomopans_spl
        from cenop.agents.ship import VesselClass
        n = ShipNoise(vessel_class=VesselClass.CARGO, length=200.0, speed=12.0)
        expected = jomopans_spl(VesselClass.CARGO, 12.0, 200.0, band=12)
        assert n.get_source_level() == expected

    def test_explicit_override_wins(self):
        """An explicit base_source_level overrides JOMOPANS (ships.json impact / tests)."""
        from cenop.behavior.sound import ShipNoise
        from cenop.agents.ship import VesselClass
        n = ShipNoise(vessel_class=VesselClass.CARGO, length=200.0, speed=12.0,
                      base_source_level=170.0)
        assert n.get_source_level() == 170.0

    def test_speed_zero_silent(self):
        """JOMOPANS returns 0.0 for a stationary ship."""
        from cenop.behavior.sound import ShipNoise
        from cenop.agents.ship import VesselClass
        n = ShipNoise(vessel_class=VesselClass.CARGO, length=200.0, speed=0.0)
        assert n.get_source_level() == 0.0

    def test_class_dependence(self):
        """Different vessel classes give different SL (JOMOPANS, not a flat default)."""
        from cenop.behavior.sound import ShipNoise
        from cenop.agents.ship import VesselClass
        a = ShipNoise(vessel_class=VesselClass.CARGO, length=200.0, speed=12.0)
        b = ShipNoise(vessel_class=VesselClass.FISHING, length=200.0, speed=12.0)
        assert a.get_source_level() != b.get_source_level()

    def test_post_init_leaves_override_none(self):
        """Ship.__post_init__ must NOT seed base_source_level (so JOMOPANS is the default)."""
        from cenop.agents.ship import Ship, VesselClass
        s = Ship(id=0, x=0.0, y=0.0, vessel_type=VesselClass.CARGO, vessel_length=200.0)
        assert s.noise.base_source_level is None
        assert s.noise.vessel_class == VesselClass.CARGO


class TestShipJsonLoader:
    def test_type_string_mapping_all_kattegat_types(self):
        """All 12 Kattegat ships.json `type` strings map to a valid VesselClass."""
        from cenop.agents.ship import _vessel_class_from_type, VesselClass
        cases = {
            "Bulker": VesselClass.BULKER, "Containership": VesselClass.CONTAINER,
            "Tanker": VesselClass.TANKER, "Government/Research": VesselClass.GOVERNMENT,
            "Cruise": VesselClass.CRUISE, "Dredger": VesselClass.DREDGER,
            "Passenger": VesselClass.PASSENGER, "Tug": VesselClass.TUG,
            "Recreational": VesselClass.RECREATIONAL, "Fishing": VesselClass.FISHING,
            "Naval": VesselClass.NAVAL, "Other": VesselClass.OTHER,
        }
        for s, vc in cases.items():
            assert _vessel_class_from_type(s) == vc, s

    def test_unknown_type_raises(self):
        from cenop.agents.ship import _vessel_class_from_type
        import pytest
        with pytest.raises(ValueError):
            _vessel_class_from_type("Submarine")

    def test_none_or_empty_type_defaults_to_other(self):
        from cenop.agents.ship import _vessel_class_from_type, VesselClass
        # The loader passes `ship_data.get("type") or "Other"`, so None/"" -> "Other".
        assert _vessel_class_from_type("Other") == VesselClass.OTHER
        # And a simulated null-type ship_data dict resolves via the `or "Other"` guard:
        ship_data = {"type": None}
        assert _vessel_class_from_type(ship_data.get("type") or "Other") == VesselClass.OTHER

    def test_loader_reads_type_length_and_no_forced_impact(self):
        """Loader maps real type/length and does NOT force a 170 dB override when impact absent."""
        from cenop.agents.ship import ShipManager
        mgr = ShipManager()
        mgr.load_from_json("data/Kattegat/ships.json",
                           utm_origin_x=529473.0, utm_origin_y=5972242.0, cell_size=400.0)
        assert mgr.count > 0
        assert all(s.noise.base_source_level is None for s in mgr.ships)
        assert len({s.vessel_length for s in mgr.ships}) > 1
        assert len({s.vessel_type for s in mgr.ships}) > 1

    def test_loader_preserves_real_per_buoy_speed(self):
        """Route buoys keep the JSON per-waypoint speeds (not a hardcoded 10.0)."""
        from cenop.agents.ship import ShipManager
        mgr = ShipManager()
        mgr.load_from_json("data/Kattegat/ships.json",
                           utm_origin_x=529473.0, utm_origin_y=5972242.0, cell_size=400.0)
        speeds = {round(b.speed, 3) for s in mgr.ships for b in s.route.buoys}
        assert speeds != {10.0}
        assert len(speeds) > 1


class TestTurbineOnlyDispersal:
    def _pop(self):
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=1)
        land = create_homogeneous_landscape(width=50, height=50, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=1, params=params, landscape=land)
        pop.is_dispersing[0] = True
        pop.dispersal_start_x[0] = pop.x[0]; pop.dispersal_start_y[0] = pop.y[0]
        # CRITICAL: dispersal_target_distance defaults to 0.0, so the distance-completion
        # check (distances >= 0.95*target) would deactivate dispersal regardless of the
        # deterrence gate, masking what we're testing. Set it huge so only the gate can fire.
        pop.dispersal_target_distance[0] = 1e9
        return pop

    def test_ship_only_deterrence_does_not_deactivate_dispersal(self):
        import numpy as np
        pop = self._pop()
        d = (np.array([0.05], dtype=np.float64), np.array([0.0], dtype=np.float64))
        t = (np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64))
        pop.step(deterrence_vectors=d, turbine_deterrence_vectors=t)
        assert bool(pop.is_dispersing[0]) is True

    def test_turbine_deterrence_deactivates_dispersal(self):
        import numpy as np
        pop = self._pop()
        d = (np.array([0.05], dtype=np.float64), np.array([0.0], dtype=np.float64))
        t = (np.array([0.05], dtype=np.float64), np.array([0.0], dtype=np.float64))
        pop.step(deterrence_vectors=d, turbine_deterrence_vectors=t)
        assert bool(pop.is_dispersing[0]) is False


class TestTurbineOnlyDispersalJax:
    def test_jax_dispersal_uses_turbine_strength(self):
        import numpy as np, jax.numpy as jnp
        from cenop.optimizations.jax_kernels import jax_dispersal_update
        n = 2
        is_dispersing = jnp.array([True, True])
        zeros = jnp.zeros(n); ddt = jnp.zeros(n); dde = jnp.zeros(n, dtype=jnp.int32)
        x = jnp.zeros(n); y = jnp.zeros(n)
        eh = jnp.zeros((n, 8)); active = jnp.array([True, True])
        # turbine strength nonzero only for agent 0
        turb = jnp.array([0.05, 0.0])
        new_disp, _, _ = jax_dispersal_update(
            is_dispersing, zeros, zeros, jnp.full(n, 1e9), ddt, dde, x, y,
            turbine_deter_strength=turb, energy_history=eh, active_mask=active,
            is_day_boundary=False)
        assert bool(new_disp[0]) is False
        assert bool(new_disp[1]) is True

    def test_step_jax_ship_only_keeps_dispersing(self):
        """End-to-end JAX: ship-only deterrence (turbine zero) must NOT deactivate dispersal."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=1, use_jax=True)
        land = create_homogeneous_landscape(width=50, height=50, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=1, params=params, landscape=land)
        if not getattr(pop, "_use_jax", False):
            import pytest; pytest.skip("JAX not active")
        pop.is_dispersing[0] = True
        pop.dispersal_start_x[0] = pop.x[0]; pop.dispersal_start_y[0] = pop.y[0]
        pop.dispersal_target_distance[0] = 1e9
        d = (np.array([0.05], dtype=np.float64), np.array([0.0], dtype=np.float64))
        t = (np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64))
        pop.step(deterrence_vectors=d, turbine_deterrence_vectors=t)
        assert bool(pop.is_dispersing[0]) is True


class TestSharedReceivedLevel:
    def test_non_weston_uses_alpha_beta(self):
        import numpy as np
        from cenop.agents.ship import _ship_received_level
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()
        dist_m = np.array([2000.0])
        px = np.array([50.0]); py = np.array([50.0])
        rl = _ship_received_level(180.0, dist_m, px, py, p,
                                  cell_data=None, month=1, weston=False)
        expected = 180.0 - (p.beta_hat * np.log10(2000.0) + p.alpha_hat * 2000.0)
        assert rl[0] == pytest.approx(max(0.0, expected))

    def test_weston_nodata_gives_zero(self):
        import numpy as np
        from cenop.agents.ship import _ship_received_level
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        p = SimulationParameters(); p.weston_flux_percell = True
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        land._depth[:] = -9999.0  # all NODATA depth
        dist_m = np.array([2000.0])
        px = np.array([50.0]); py = np.array([50.0])
        rl = _ship_received_level(210.0, dist_m, px, py, p,
                                  cell_data=land, month=1, weston=True)
        assert rl[0] == 0.0

    def test_clamped_non_negative(self):
        import numpy as np
        from cenop.agents.ship import _ship_received_level
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()
        rl = _ship_received_level(10.0, np.array([9000.0]), np.array([0.0]),
                                  np.array([0.0]), p, cell_data=None, month=1, weston=False)
        assert rl[0] == 0.0


class TestScalarAggregatorTL:
    def _mgr(self, sl=205.0):
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = sl
        mgr = ShipManager([s]); mgr.enabled = True
        return mgr, s

    def test_scalar_uses_weston_when_enabled(self):
        """Scalar aggregator RL must use WestonFlux (per-cell) when enabled, NOT alpha/beta.

        Non-vacuous: the WestonFlux RL and the alpha/beta RL both gate in and react here
        (so 'dx != 0' alone cannot distinguish them). We assert the produced vector matches
        the WestonFlux-derived reference and DIFFERS from the alpha/beta-derived reference.
        At RED (scalar still uses alpha/beta) `dx_w != alphabeta_ref` fails."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.ship import _ship_received_level
        from cenop.behavior.sound import ShipDeterrenceModel
        mgr, s = self._mgr()
        p = SimulationParameters(); p.weston_flux_percell = True
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        px, py = 50.0 + 2000.0 / 400.0, 50.0
        orig = np.random.random
        np.random.random = lambda *a, **k: 0.0   # force reaction (u=0 < prob)
        try:
            mag_w, dx_w, dy_w = mgr.calculate_aggregate_deterrence(
                px, py, p, is_day=True, cell_data=land, month=1)
        finally:
            np.random.random = orig
        # Reference vectors from the two RL models (force react u=0):
        m = ShipDeterrenceModel()
        dist_m = np.array([2000.0]); gdx = np.array([px - 50.0]); gdy = np.array([0.0])
        rl_w = _ship_received_level(s.noise.get_source_level(), dist_m,
                                    np.array([px]), np.array([py]), p, land, 1, True)
        rl_ab = _ship_received_level(s.noise.get_source_level(), dist_m,
                                     np.array([px]), np.array([py]), p, None, 1, False)
        assert rl_w[0] > p.deter_ships_min_db and rl_ab[0] > p.deter_ships_min_db  # both gated in
        assert rl_w[0] != pytest.approx(rl_ab[0])                                  # models differ
        vx_w = float(m.deterrence_components(rl_w, dist_m, gdx, gdy, True,
                                             np.array([0.0]), p.deter_ships_min_db)[0][0])
        vx_ab = float(m.deterrence_components(rl_ab, dist_m, gdx, gdy, True,
                                              np.array([0.0]), p.deter_ships_min_db)[0][0])
        assert dx_w == pytest.approx(vx_w)         # scalar used WestonFlux RL
        assert dx_w != pytest.approx(vx_ab)        # ... NOT alpha/beta RL

    def test_scalar_without_celldata_uses_alpha_beta(self):
        """No cell_data -> alpha/beta TL (unchanged legacy behavior)."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.agents.ship import _ship_received_level
        mgr, s = self._mgr()
        p = SimulationParameters()
        px, py = 50.0 + 2000.0 / 400.0, 50.0
        orig = np.random.random
        np.random.random = lambda *a, **k: 0.0
        try:
            mag, dx, dy = mgr.calculate_aggregate_deterrence(px, py, p, is_day=True)
        finally:
            np.random.random = orig
        from cenop.behavior.sound import ShipDeterrenceModel
        rl = _ship_received_level(s.noise.get_source_level(), np.array([2000.0]),
                                  np.array([px]), np.array([py]), p, None, 1, False)
        assert rl[0] > p.deter_ships_min_db
        vx_ref = float(ShipDeterrenceModel().deterrence_components(
            rl, np.array([2000.0]), np.array([px - 50.0]), np.array([0.0]),
            True, np.array([0.0]), p.deter_ships_min_db)[0][0])
        assert dx == pytest.approx(vx_ref)        # matches alpha/beta reference exactly


class TestShipPrevPosition:
    def test_post_init_sets_prev_to_initial(self):
        from cenop.agents.ship import Ship, VesselClass
        s = Ship(id=0, x=7.0, y=9.0, vessel_type=VesselClass.CARGO)
        assert s._prev_x == 7.0 and s._prev_y == 9.0

    def test_update_records_pre_move_position(self):
        from cenop.agents.ship import Ship, Route, Buoy, VesselClass
        route = Route(buoys=[Buoy(x=0.0, y=0.0, speed=10.0),
                             Buoy(x=100.0, y=0.0, speed=10.0)])
        s = Ship(id=1, x=0.0, y=0.0, route=route, vessel_type=VesselClass.CARGO)
        s.tick_start = 0; s.tick_end = 100
        s.update(1)
        assert (s._prev_x, s._prev_y) == (0.0, 0.0)   # start-of-tick position
        assert s.x != 0.0                              # moved toward the next buoy

    def test_update_inactive_leaves_prev_equal_current(self):
        from cenop.agents.ship import Ship, VesselClass
        s = Ship(id=1, x=5.0, y=5.0, vessel_type=VesselClass.CARGO)
        s.tick_start = 10; s.tick_end = 20   # inactive at tick 1
        s.update(1)
        assert (s._prev_x, s._prev_y) == (5.0, 5.0)
        assert (s.x, s.y) == (5.0, 5.0)


class TestSubTickInterpolation:
    def _params(self):
        from cenop.parameters.simulation_params import SimulationParameters
        return SimulationParameters()

    def _ship(self, sid, x, y, prev=None, sl=205.0):
        from cenop.agents.ship import Ship, VesselClass
        s = Ship(id=sid, x=x, y=y, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = sl
        if prev is not None:
            s._prev_x, s._prev_y = prev
        else:
            s._prev_x, s._prev_y = x, y
        return s

    def _kernel_vec(self, s, px, py, p, sub_x, sub_y, cell=400.0):
        """Single-slot kernel vector for a ship sub-position (force react), using the same
        non-WestonFlux RL the implementation uses: source_level - (beta*log10(d) + alpha*d)."""
        import numpy as np
        gdx = np.array([px - sub_x]); gdy = np.array([py - sub_y])
        dist_m = np.array([max(float(np.hypot(gdx[0]*cell, gdy[0]*cell)), 1.0)])
        tl = p.beta_hat * np.log10(dist_m[0]) + p.alpha_hat * dist_m[0]
        rl = np.array([max(0.0, float(s.noise.get_source_level() - tl))])
        vx, vy, _, _, _ = s.deterrence_model.deterrence_components(
            rl, dist_m, gdx, gdy, True, np.array([0.0]), p.deter_ships_min_db)
        return float(vx[0]), float(vy[0])

    def test_stationary_ship_is_30x_single_position(self):
        """prev == cur -> 30 identical slots -> total == 30 x single-position vector (force_u=0)."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params()
        s = self._ship(1, 50.0, 50.0)   # prev == cur
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([50.0 + 2000.0/400.0]); py = np.array([50.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        vx1, vy1 = self._kernel_vec(s, px[0], py[0], p, 50.0, 50.0)
        assert dx[0] == pytest.approx(30.0 * vx1)
        assert dy[0] == pytest.approx(30.0 * vy1)

    def test_moving_ship_sums_distinct_substep_vectors(self):
        """Total equals the slot-wise sum over i=1..30 sub-positions (force_u=0)."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params()
        # Swept path 40->60 east along y=50; porpoise north at (50, 60).
        s = self._ship(1, 60.0, 50.0, prev=(40.0, 50.0))
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([50.0]); py = np.array([60.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        exp_x = exp_y = 0.0
        for i in range(1, 31):
            sub_x = 40.0 + (60.0 - 40.0) * i / 30.0
            sub_y = 50.0
            vx, vy = self._kernel_vec(s, px[0], py[0], p, sub_x, sub_y)
            exp_x += vx; exp_y += vy
        assert exp_y > 0.0                  # non-vacuous: some slots gated + reacting
        assert dx[0] == pytest.approx(exp_x)
        assert dy[0] == pytest.approx(exp_y)
        assert dy[0] > 0.0   # net push north, away from the east-west path

    def test_substep_endpoints_exclude_start_include_end(self):
        """i=1..30: first sub-position is start+delta/30, last is exactly the end position."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params()
        s = self._ship(1, 10.0, 0.0, prev=(0.0, 0.0))
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([10.0]); py = np.array([5.0])   # near the END (10,0), north
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        exp_x = exp_y = 0.0
        for i in range(1, 31):
            sub_x = 0.0 + 10.0 * i / 30.0
            vx, vy = self._kernel_vec(s, px[0], py[0], p, sub_x, 0.0)
            exp_x += vx; exp_y += vy
        assert dx[0] == pytest.approx(exp_x)
        assert dy[0] == pytest.approx(exp_y)

    def test_per_slot_max_rl_ship_wins(self):
        """Different ships win different slots (DEPONS recordStep). Aggregator must match a
        brute-force per-slot max-RL+sum reference, and the winner set must include BOTH ships."""
        import numpy as np
        from cenop.agents.ship import ShipManager, VesselClass
        p = self._params()
        # Asymmetric crossing so the distance curves cross inside i=1..30:
        #   A approaches  (dist_A = 6 - i/6, from ~5.83 down to 1 cell),
        #   B recedes     (dist_B = 1 + i/6, from ~1.17 up to 6 cells).
        # B is closer for i<15 (B wins), A is closer for i>15 (A wins) -> both win slots.
        A = self._ship(1, 49.0, 50.0, prev=(44.0, 50.0), sl=195.0)   # approaching
        B = self._ship(2, 56.0, 50.0, prev=(51.0, 50.0), sl=195.0)   # receding
        px = np.array([50.0]); py = np.array([50.0])
        mgr = ShipManager([A, B]); mgr.enabled = True
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)

        # Brute-force reference (true oracle): apply BOTH gates the impl applies, keep the
        # max-RL ship per slot, sum. No midpoint pre-cull (independent of the optimization).
        from cenop.agents.ship import MAX_DETER_DIST_M
        cell = 400.0
        min_m = p.deter_min_distance_ships * 1000.0
        max_m = min(MAX_DETER_DIST_M, p.deter_max_distance * 1000.0)
        def slot_rl_vec(s, i):
            sub_x = s._prev_x + (s.x - s._prev_x) * i / 30.0
            sub_y = s._prev_y + (s.y - s._prev_y) * i / 30.0
            gdx = np.array([px[0] - sub_x]); gdy = np.array([py[0] - sub_y])
            dist_m = np.array([max(float(np.hypot(gdx[0]*cell, gdy[0]*cell)), 1.0)])
            rl = max(0.0, float(s.noise.get_source_level()
                                - (p.beta_hat*np.log10(dist_m[0]) + p.alpha_hat*dist_m[0])))
            vx, vy, _, _, _ = s.deterrence_model.deterrence_components(
                np.array([rl]), dist_m, gdx, gdy, True, np.array([0.0]), p.deter_ships_min_db)
            return rl, float(vx[0]), float(vy[0]), float(dist_m[0])
        exp_x = exp_y = 0.0; winners = set()
        for i in range(1, 31):
            ra, vax, vay, da = slot_rl_vec(A, i)
            rb, vbx, vby, db = slot_rl_vec(B, i)
            a_ok = (min_m < da <= max_m) and ra > p.deter_ships_min_db
            b_ok = (min_m < db <= max_m) and rb > p.deter_ships_min_db
            # Lowest id wins ties (impl processes sorted by id, strict '>' keeps the
            # first-processed; A has id=1 so 'ra >= rb' favors A consistently).
            if a_ok and (not b_ok or ra >= rb):
                exp_x += vax; exp_y += vay; winners.add(1)
            elif b_ok:
                exp_x += vbx; exp_y += vby; winners.add(2)
        assert winners == {1, 2}                    # both ships win some slots
        assert dx[0] == pytest.approx(exp_x)
        assert dy[0] == pytest.approx(exp_y)

    def test_gated_nonreacting_winner_contributes_zero(self):
        """Characterization (passes at RED too): a gated ship that does NOT react stores a
        zero vector (DEPONS recordStep keeps the max-RL step with deterX=0 when
        reactingOrNot=0). With a uniform non-reacting draw the total is exactly zero even
        though every slot is gated in."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params()
        loud = self._ship(1, 49.0, 50.0, sl=205.0)   # gated in (RL >> Tships)
        px = np.array([50.0]); py = np.array([50.0])
        mgr = ShipManager([loud]); mgr.enabled = True
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=1.0)  # never react
        assert dx[0] == 0.0 and dy[0] == 0.0

    def test_louder_ship_fully_blocks_quieter(self):
        """Blocking (DEPONS recordStep): a ship with higher RL at every slot wins every slot,
        so the quieter ship is fully blocked. The two-ship result equals the loud-only result
        and is NOT their sum. Seeded draws (loud's own per-ship stream decides reactions)."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params(); seed, tick = 7, 3
        def loud(): return self._ship(1, 49.0, 50.0, sl=205.0)   # closer + louder -> higher RL
        def quiet(): return self._ship(2, 45.0, 50.0, sl=185.0)  # farther + quieter, still gated
        px = np.array([50.0]); py = np.array([50.0])
        both = ShipManager([loud(), quiet()]); both.enabled = True
        only_l = ShipManager([loud()]); only_l.enabled = True
        only_q = ShipManager([quiet()]); only_q.enabled = True
        b = both.calculate_aggregate_deterrence_vectorized(px, py, p, base_seed=seed, tick=tick)
        l = only_l.calculate_aggregate_deterrence_vectorized(px, py, p, base_seed=seed, tick=tick)
        q = only_q.calculate_aggregate_deterrence_vectorized(px, py, p, base_seed=seed, tick=tick)
        np.testing.assert_array_equal(b[0], l[0])   # quiet fully blocked: both == loud-only
        np.testing.assert_array_equal(b[1], l[1])
        if q[0][0] != 0.0:                           # quiet would deter alone -> result is not a sum
            assert b[0][0] != pytest.approx(l[0][0] + q[0][0])

    def test_porpoise_count_invariance_subtick(self):
        """Count/membership invariance (guards the (n, STEPS) draw layout): appending an
        out-of-range porpoise must not change an in-range porpoise's vector. FAILS if the
        draws are laid out (STEPS, n) instead of (n, STEPS)."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params()
        s = self._ship(1, 50.0, 50.0, prev=(45.0, 50.0), sl=205.0)  # moving ship
        mgr = ShipManager([s]); mgr.enabled = True
        px1 = np.array([52.0]); py1 = np.array([50.0])
        d1 = mgr.calculate_aggregate_deterrence_vectorized(px1, py1, p, base_seed=4, tick=9)
        px2 = np.array([52.0, 800.0]); py2 = np.array([50.0, 50.0])  # far porpoise appended
        d2 = mgr.calculate_aggregate_deterrence_vectorized(px2, py2, p, base_seed=4, tick=9)
        assert d2[0][0] == pytest.approx(d1[0][0])   # in-range porpoise unchanged by the far one
        assert d2[1][0] == pytest.approx(d1[1][0])
        assert d2[0][1] == 0.0 and d2[1][1] == 0.0   # far porpoise: no deterrence

    def test_moving_ship_weston_subtick_matches_reference(self):
        """Moving ship + WestonFlux: per-slot RL via WestonFlux at 30 DISTINCT positions,
        summed, matches an unculled brute-force reference using _ship_received_level(weston=True)."""
        import numpy as np
        from cenop.agents.ship import ShipManager, _ship_received_level, MAX_DETER_DIST_M
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        p = SimulationParameters(); p.weston_flux_percell = True
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        s = self._ship(1, 60.0, 50.0, prev=(40.0, 50.0), sl=205.0)  # sweep east along y=50
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([50.0]); py = np.array([55.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(
            px, py, p, cell_data=land, month=1, _force_u=0.0)
        cell = 400.0
        min_m = p.deter_min_distance_ships * 1000.0
        max_m = min(MAX_DETER_DIST_M, p.deter_max_distance * 1000.0)
        exp_x = exp_y = 0.0
        for i in range(1, 31):
            sub_x = 40.0 + 20.0 * i / 30.0; sub_y = 50.0
            gdx = np.array([px[0]-sub_x]); gdy = np.array([py[0]-sub_y])
            d = np.array([max(float(np.hypot(gdx[0]*cell, gdy[0]*cell)), 1.0)])
            if not (min_m < d[0] <= max_m):
                continue
            rl = _ship_received_level(s.noise.get_source_level(), d,
                                      np.array([px[0]]), np.array([py[0]]), p, land, 1, True)
            if rl[0] <= p.deter_ships_min_db:
                continue
            vx, vy, _, _, _ = s.deterrence_model.deterrence_components(
                rl, d, gdx, gdy, True, np.array([0.0]), p.deter_ships_min_db)
            exp_x += float(vx[0]); exp_y += float(vy[0])
        assert exp_x != 0.0 or exp_y != 0.0          # non-vacuous
        assert dx[0] == pytest.approx(exp_x)
        assert dy[0] == pytest.approx(exp_y)

    def test_vectorized_matches_bruteforce_multi(self):
        """Multi-ship, multi-porpoise, mixed-range, seeded: aggregator matches an independent
        brute-force per-(porpoise,slot,ship) reference (max-RL ship wins, lowest id on tie,
        winner's own per-ship draw decides reacting, sum over 30 slots)."""
        import numpy as np
        from cenop.agents.ship import ShipManager, MAX_DETER_DIST_M
        p = self._params(); seed, tick = 13, 6
        ships = [
            self._ship(1, 52.0, 50.0, prev=(48.0, 50.0), sl=200.0),
            self._ship(2, 49.0, 53.0, prev=(49.0, 47.0), sl=195.0),
            self._ship(3, 300.0, 300.0, sl=205.0),  # far -> out of range for all
        ]
        px = np.array([50.0, 51.0, 47.0, 500.0])     # last is far out of range
        py = np.array([50.0, 49.0, 52.0, 500.0])
        mgr = ShipManager(list(ships)); mgr.enabled = True
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(
            px, py, p, base_seed=seed, tick=tick)
        cell = 400.0
        min_m = p.deter_min_distance_ships * 1000.0
        max_m = min(MAX_DETER_DIST_M, p.deter_max_distance * 1000.0)
        n = px.shape[0]
        draws = {}
        for s in ships:
            rng = np.random.default_rng(np.random.SeedSequence([seed, tick, int(s.id)]))
            draws[int(s.id)] = rng.random((n, 30))
        exp_x = np.zeros(n); exp_y = np.zeros(n)
        for pi in range(n):
            for k in range(1, 31):
                best = -np.inf; bvx = bvy = 0.0
                for s in sorted(ships, key=lambda z: int(z.id)):
                    sx = s._prev_x + (s.x - s._prev_x) * k / 30.0
                    sy = s._prev_y + (s.y - s._prev_y) * k / 30.0
                    gdx = np.array([px[pi] - sx]); gdy = np.array([py[pi] - sy])
                    d = np.array([max(float(np.hypot(gdx[0]*cell, gdy[0]*cell)), 1.0)])
                    if not (min_m < d[0] <= max_m):
                        continue
                    rl = max(0.0, float(s.noise.get_source_level()
                                        - (p.beta_hat*np.log10(d[0]) + p.alpha_hat*d[0])))
                    if not (rl > p.deter_ships_min_db):
                        continue
                    if rl > best:   # strict: first-processed (lowest id) keeps ties
                        u = np.array([draws[int(s.id)][pi, k-1]])
                        vx, vy, _, _, _ = s.deterrence_model.deterrence_components(
                            np.array([rl]), d, gdx, gdy, True, u, p.deter_ships_min_db)
                        best = rl; bvx = float(vx[0]); bvy = float(vy[0])
                exp_x[pi] += bvx; exp_y[pi] += bvy
        np.testing.assert_allclose(dx, exp_x, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(dy, exp_y, rtol=1e-9, atol=1e-12)
        assert np.any(exp_x != 0.0) or np.any(exp_y != 0.0)   # non-vacuous
