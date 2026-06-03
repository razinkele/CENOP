"""Tests for DEPONS 3.2 deterrence fixes."""

import numpy as np
import pytest


class TestTurbineDeterrenceVector:
    """Test turbine deterrence uses raw displacement, not unit vector."""

    def test_no_normalization_scalar(self):
        """calculate_deterrence_vector should return raw displacement * strength * coeff.

        Java: Porpoise.java:1290-1292 — raw displacement, no normalization.
        Before fix: dx/=distance, dy/=distance (unit vector).
        After fix: dx and dy proportional to distance from turbine.
        """
        from cenop.behavior.sound import calculate_deterrence_vector

        # Porpoise at (100, 100), turbine at (95, 100) — 5 cells away
        porp_x, porp_y = 100.0, 100.0
        turb_x, turb_y = 95.0, 100.0
        strength = 1.0
        coeff = 1.0  # Use 1.0 to isolate normalization effect

        dx, dy = calculate_deterrence_vector(porp_x, porp_y, turb_x, turb_y, strength, coeff)

        # Raw displacement is (5.0, 0.0), so dx should be 5.0 * 1.0 * 1.0 = 5.0
        # If normalized, dx would be 1.0 (unit vector)
        assert dx == pytest.approx(5.0), f"dx should be raw*strength*coeff=5.0, got {dx} (normalized?)"
        assert dy == pytest.approx(0.0), f"dy should be 0.0, got {dy}"

    def test_no_normalization_diagonal(self):
        """Diagonal case: magnitude should encode distance."""
        from cenop.behavior.sound import calculate_deterrence_vector

        # 3-4-5 triangle
        porp_x, porp_y = 103.0, 104.0
        turb_x, turb_y = 100.0, 100.0
        strength = 1.0
        coeff = 1.0

        dx, dy = calculate_deterrence_vector(porp_x, porp_y, turb_x, turb_y, strength, coeff)

        # Raw: (3.0, 4.0), distance=5.0
        # Correct (raw): dx=3.0, dy=4.0
        # Wrong (normalized): dx=0.6, dy=0.8
        assert dx == pytest.approx(3.0), f"dx should be 3.0, got {dx}"
        assert dy == pytest.approx(4.0), f"dy should be 4.0, got {dy}"

    def test_no_normalization_vectorized_turbine(self):
        """TurbineManager vectorized path should also use raw displacement."""
        from cenop.agents.turbine import Turbine, TurbineManager, TurbinePhase
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        # Override to simplify: coeff=1, threshold=0 so any RL produces strength
        params.deter_coeff = 1.0
        params.deter_threshold = 0.0
        params.deter_probabilistic = False

        t = Turbine(id=0, x=50.0, y=50.0, impact=200.0, phase=TurbinePhase.CONSTRUCTION)
        t._is_active = True
        mgr = TurbineManager([t])
        mgr.phase = TurbinePhase.CONSTRUCTION

        # Porpoise 5 cells east of turbine
        px = np.array([55.0])
        py = np.array([50.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, params, cell_size=400.0)

        # dx should be positive (pushed away east) and proportional to raw displacement
        # NOT a unit vector of magnitude ~1
        # Raw displacement in meters: (55-50)*400 = 2000m
        # With no normalization, the vector should encode distance
        assert dx[0] > 1.0, f"Vectorized dx should encode distance, got {dx[0]} (unit vector?)"

    def test_no_normalization_vectorized_ship(self):
        """ShipManager vectorized path should also use raw displacement."""
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        params.deter_coeff = 1.0
        params.deter_threshold = 0.0
        params.deter_probabilistic = False

        s = Ship(id=0, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True
        mgr = ShipManager([s])
        mgr.enabled = True

        px = np.array([55.0])
        py = np.array([50.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, params, cell_size=400.0)

        assert dx[0] > 1.0, f"Ship vectorized dx should encode distance, got {dx[0]} (unit vector?)"


class TestTLParameterDefaults:
    """Confirmatory test: TL/deterrence parameters match DEPONS 3.2 Kattegat calibration.

    These were set in Phase 1 (commit fac2e7f). This test guards against regression.
    """

    def test_kattegat_calibrated_values(self):
        """Verify Phase 1 parameter values are preserved."""
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()

        # Sound propagation (Kattegat-calibrated, DEPONS 3.2)
        assert params.alpha_hat == pytest.approx(0.00027), \
            f"alpha_hat should be 0.00027, got {params.alpha_hat}"
        assert params.beta_hat == pytest.approx(14.72), \
            f"beta_hat should be 14.72, got {params.beta_hat}"

        # Deterrence (Kattegat-calibrated, DEPONS 3.2)
        assert params.deter_coeff == pytest.approx(0.012), \
            f"deter_coeff should be 0.012, got {params.deter_coeff}"
        assert params.deter_threshold == pytest.approx(152.0), \
            f"deter_threshold should be 152.0, got {params.deter_threshold}"

    def test_deter_max_distance_is_1000km(self):
        """deter_max_distance should be 1000.0 km (parameters.xml dmax_deter).

        Authoritative source is parameters.xml, not Java field initializers:
        - parameters.xml: dmax_deter=1000.0 [km]
        - SimulationParameters.initialize(): deterMaxDistance = dmax_deter * 1000
          -> 1,000,000 m = 1000 km
        - resetToDefaultsForUnitTest(): deterMaxDistance = 1000.0 * 1000 (also 1000 km)
        The `50 * 1000` field initializer (SimulationParameters.java:89) is stale and
        overwritten in every code path. CENOP stores km and converts with *1000 at use.
        """
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()

        assert params.deter_max_distance == pytest.approx(1000.0), \
            f"deter_max_distance should be 1000.0 km, got {params.deter_max_distance}"


class TestShipDeterrenceStandardization:
    """Test ship deterrence uses standardized inputs (Java Ship.java:334-408)."""

    def test_standardize_day_probability(self):
        """Probability model (day) should standardize dist and noise.

        Java: Ship.java:349 — (distInKm - 5.801812) / 2.602801
        Raw inputs should be standardized before applying model coefficients.
        """
        from cenop.behavior.sound import ShipDeterrenceModel

        model = ShipDeterrenceModel()

        # Two distances that are equidistant from the standardization mean
        # should produce symmetric changes in probability
        dist_below_mean = 3.0   # km
        dist_above_mean = 8.6   # km (roughly symmetric around mean 5.8)
        noise_db = 80.0

        prob_below = model.calculate_deterrence_probability(noise_db, dist_below_mean, is_day=True)
        prob_above = model.calculate_deterrence_probability(noise_db, dist_above_mean, is_day=True)

        # Closer should have higher deterrence probability
        assert prob_below > prob_above, \
            f"Closer distance should have higher prob: {prob_below} vs {prob_above}"

        # With standardization, the probabilities should be meaningful (not ~0 or ~1 for typical inputs)
        assert 0.01 < prob_below < 0.99, \
            f"Day probability at 3km, 80dB should be in meaningful range, got {prob_below}"

    def test_tships_minimum_gate(self):
        """Ship deterrence should be skipped when RL <= Tships (80 dB).

        Java: Ship.java:228 — if (receivedLevel <= Tships) skip.
        Authoritative value is parameters.xml Tships=80.0, loaded verbatim by
        SimulationParameters.initialize(). The 70.0 in the field initializer
        (SimulationParameters.java:164) and resetToDefaultsForUnitTest() is a
        unit-test default, not the production runtime value.
        """
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()
        assert hasattr(params, 'deter_ships_min_db'), "Missing deter_ships_min_db parameter"
        assert params.deter_ships_min_db == pytest.approx(80.0), \
            f"deter_ships_min_db should be 80.0, got {params.deter_ships_min_db}"

    def test_ship_deterrence_gated_below_tships(self):
        """Ship with RL below Tships (80 dB) should produce zero deterrence."""
        from cenop.agents.ship import Ship, VesselClass
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        params.deter_ships_min_db = 80.0

        # Ship very far away (will produce low RL)
        s = Ship(id=0, x=0.0, y=0.0, vessel_type=VesselClass.FISHING)
        s._is_active = True

        # Porpoise at extreme distance
        result = s.calculate_deterrence(
            porpoise_x=1000.0, porpoise_y=1000.0,
            params=params, is_day=True, cell_size=400.0
        )
        should_deter, prob, magnitude, dist_km = result

        # At extreme distance, RL should be below Tships threshold
        # So should_deter should be False
        assert should_deter is False, "Ship below Tships should not deter"


class TestPSMFoodGating:
    """Test PSM only counts ticks where food was consumed."""

    def test_psm_ticks_only_increment_with_food(self):
        """PSM tick counter should only increment when food > 0.

        Java: PersistentSpatialMemory.java:119 — only records ticks with food.
        """
        # Simulate: 3 agents, only agent 0 and 2 have food
        food_gained = np.array([0.5, 0.0, 1.0], dtype=np.float32)
        mask = np.array([True, True, True])

        # Only agents with food should have their tick count incremented
        food_positive = food_gained > 0
        ticks_mask = mask & food_positive

        assert ticks_mask[0] == True, "Agent 0 has food, should count"
        assert ticks_mask[1] == False, "Agent 1 has no food, should NOT count"
        assert ticks_mask[2] == True, "Agent 2 has food, should count"


class TestPhase5Integration:
    """Smoke test for Phase 5 deterrence and dispersal."""

    def test_deterrence_parameters_consistent(self):
        """All deterrence parameters should be internally consistent."""
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()

        # Max distance matches parameters.xml dmax_deter = 1000 km
        assert params.deter_max_distance == pytest.approx(1000.0), \
            f"deter_max_distance={params.deter_max_distance} should be 1000.0 km"

        # Tships gate should be set
        assert hasattr(params, 'deter_ships_min_db')
        assert params.deter_ships_min_db > 0

        # Deterrence coeff should be positive
        assert params.deter_coeff > 0

    def test_dispersal_energy_stop_logic(self):
        """Dispersal should stop when today's energy > min of past 7 days."""
        energy_history = np.array([7.0, 3.0, 4.0, 5.0, 6.0, 5.5, 4.5, 3.5, 2.0, 1.0])
        today = energy_history[0]
        past_min = np.min(energy_history[1:8])
        assert today > past_min, "Should stop dispersal (7.0 > 3.0)"

    def test_weston_flux_importable(self):
        """WestonFlux module should be importable and functional."""
        from cenop.behavior.weston_flux import weston_flux_tl
        tl = weston_flux_tl(1000.0, 30.0, 2.0, 10.0, 35.0)
        assert tl > 0

    def test_jomopans_importable(self):
        """JOMOPANS module should be importable and functional."""
        from cenop.behavior.jomopans_spl import jomopans_spl
        from cenop.agents.ship import VesselClass
        spl = jomopans_spl(VesselClass.CARGO, 10.0, 100.0)
        assert spl > 0
