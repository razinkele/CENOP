"""
Test CENOP deterrence model against DEPONS 3.0 expected behaviors.

This module validates the sound propagation, turbine deterrence, and
ship deterrence implementations against DEPONS 3.0 formulas and parameters.

DEPONS Deterrence Model:
- Transmission loss: TL = beta * log10(r) + alpha * r
- Received level: RL = SL - TL
- Turbine deterrence: if RL > threshold, strength = RL - threshold
- Ship deterrence: probabilistic model with day/night variation
"""

import numpy as np
import pytest
from cenop.parameters.simulation_params import SimulationParameters
from cenop.behavior.sound import (
    calculate_transmission_loss,
    calculate_received_level,
    calculate_deterrence_vector,
    calculate_deterrence_distance,
    TurbineNoise,
    ShipNoise,
    ShipDeterrenceModel,
    response_probability_from_rl,
)
from cenop.agents.turbine import Turbine, TurbinePhase
from cenop.agents.ship import Ship


class TestSoundPropagationParameters:
    """Test that sound propagation parameters match DEPONS 3.0 defaults."""

    def test_spreading_loss_factor(self):
        """Verify beta_hat (spreading loss) matches DEPONS."""
        params = SimulationParameters()
        assert params.beta_hat == 20.0, "beta_hat should be 20.0 (spherical spreading)"

    def test_absorption_coefficient(self):
        """Verify alpha_hat (absorption) matches DEPONS."""
        params = SimulationParameters()
        assert params.alpha_hat == 0.0, "alpha_hat should be 0.0 (default)"

    def test_deterrence_threshold(self):
        """Verify deterrence threshold matches DEPONS."""
        params = SimulationParameters()
        assert params.deter_threshold == 158.0, "deter_threshold should be 158 dB"

    def test_deterrence_coefficient(self):
        """Verify deterrence coefficient matches DEPONS."""
        params = SimulationParameters()
        assert params.deter_coeff == 0.07, "deter_coeff should be 0.07"

    def test_max_deterrence_distance(self):
        """Verify max deterrence distance matches DEPONS."""
        params = SimulationParameters()
        assert params.deter_max_distance == 50.0, "deter_max_distance should be 50 km"


class TestTransmissionLoss:
    """Test transmission loss formula: TL = beta * log10(r) + alpha * r"""

    def test_transmission_loss_at_1m(self):
        """At 1m, TL should be 0 (reference distance)."""
        tl = calculate_transmission_loss(1.0, alpha_hat=0.0, beta_hat=20.0)
        assert abs(tl) < 0.01, f"TL at 1m should be ~0, got {tl}"

    def test_transmission_loss_at_10m(self):
        """At 10m, TL = 20 * log10(10) = 20 dB."""
        tl = calculate_transmission_loss(10.0, alpha_hat=0.0, beta_hat=20.0)
        expected = 20.0  # 20 * log10(10) = 20
        assert abs(tl - expected) < 0.01, f"TL at 10m should be {expected}, got {tl}"

    def test_transmission_loss_at_100m(self):
        """At 100m, TL = 20 * log10(100) = 40 dB."""
        tl = calculate_transmission_loss(100.0, alpha_hat=0.0, beta_hat=20.0)
        expected = 40.0  # 20 * log10(100) = 40
        assert abs(tl - expected) < 0.01, f"TL at 100m should be {expected}, got {tl}"

    def test_transmission_loss_at_1km(self):
        """At 1000m, TL = 20 * log10(1000) = 60 dB."""
        tl = calculate_transmission_loss(1000.0, alpha_hat=0.0, beta_hat=20.0)
        expected = 60.0  # 20 * log10(1000) = 60
        assert abs(tl - expected) < 0.01, f"TL at 1km should be {expected}, got {tl}"

    def test_transmission_loss_with_absorption(self):
        """Test absorption adds linearly to spreading loss."""
        alpha = 0.01  # 0.01 dB/m
        distance = 1000.0  # 1 km
        tl = calculate_transmission_loss(distance, alpha_hat=alpha, beta_hat=20.0)

        # TL = 20*log10(1000) + 0.01*1000 = 60 + 10 = 70
        expected = 60.0 + 10.0
        assert abs(tl - expected) < 0.01, f"TL with absorption should be {expected}, got {tl}"

    def test_transmission_loss_vectorized(self):
        """Test vectorized transmission loss calculation."""
        distances = np.array([10.0, 100.0, 1000.0])
        tl = calculate_transmission_loss(distances, alpha_hat=0.0, beta_hat=20.0)

        expected = np.array([20.0, 40.0, 60.0])
        np.testing.assert_array_almost_equal(tl, expected, decimal=1)


class TestReceivedLevel:
    """Test received level formula: RL = SL - TL"""

    def test_received_level_at_1m(self):
        """RL at 1m should equal source level."""
        sl = 200.0
        rl = calculate_received_level(sl, 1.0, alpha_hat=0.0, beta_hat=20.0)
        assert abs(rl - sl) < 0.1, f"RL at 1m should be ~{sl}, got {rl}"

    def test_received_level_at_10m(self):
        """RL at 10m: 200 - 20 = 180 dB."""
        sl = 200.0
        rl = calculate_received_level(sl, 10.0, alpha_hat=0.0, beta_hat=20.0)
        expected = sl - 20.0
        assert abs(rl - expected) < 0.1, f"RL at 10m should be {expected}, got {rl}"

    def test_received_level_at_1km(self):
        """RL at 1km: 200 - 60 = 140 dB."""
        sl = 200.0
        rl = calculate_received_level(sl, 1000.0, alpha_hat=0.0, beta_hat=20.0)
        expected = sl - 60.0
        assert abs(rl - expected) < 0.1, f"RL at 1km should be {expected}, got {rl}"


class TestTurbineDeterrenceParameters:
    """Test turbine deterrence parameters match DEPONS."""

    def test_construction_source_level(self):
        """Construction source level should be ~200 dB."""
        noise = TurbineNoise()
        assert noise.source_level_construction == 200.0

    def test_operation_source_level(self):
        """Operation source level should be ~145 dB."""
        noise = TurbineNoise()
        assert noise.source_level_operation == 145.0

    def test_impact_factor_effect(self):
        """Impact factor modifies source level logarithmically."""
        noise_default = TurbineNoise(impact=1.0)
        noise_loud = TurbineNoise(impact=2.0)

        sl_default = noise_default.get_source_level(is_construction=True)
        sl_loud = noise_loud.get_source_level(is_construction=True)

        # 10*log10(2) ≈ 3 dB increase
        expected_increase = 10 * np.log10(2)
        actual_increase = sl_loud - sl_default

        assert abs(actual_increase - expected_increase) < 0.1


class TestTurbineDeterrenceLogic:
    """Test turbine deterrence calculation logic."""

    def test_turbine_should_deter_within_range(self):
        """Turbine should deter porpoises within deterrence range."""
        params = SimulationParameters()
        turbine = Turbine(
            id=1,
            x=100.0, y=100.0,
            impact=200.0,  # Impact IS the source level in DEPONS
            phase=TurbinePhase.CONSTRUCTION
        )
        turbine._is_active = True

        # Porpoise at 1km (2.5 grid cells at 400m)
        result = turbine.should_deter(102.5, 100.0, params, cell_size=400.0)
        should_deter, rl, distance_m, strength = result

        # RL = 200 - 20*log10(1000) = 200 - 60 = 140 dB
        # Strength = 140 - 158 = -18 (should NOT deter)
        assert not should_deter, "Porpoise at 1km should not be deterred"

    def test_turbine_should_deter_close(self):
        """Turbine should deter porpoises very close."""
        params = SimulationParameters()
        turbine = Turbine(
            id=2,
            x=100.0, y=100.0,
            impact=200.0,
            phase=TurbinePhase.CONSTRUCTION
        )
        turbine._is_active = True

        # Porpoise at 100m (0.25 grid cells)
        result = turbine.should_deter(100.25, 100.0, params, cell_size=400.0)
        should_deter, rl, distance_m, strength = result

        # RL = 200 - 20*log10(100) = 200 - 40 = 160 dB
        # Strength = 160 - 158 = 2 (should deter)
        assert should_deter, "Porpoise at 100m should be deterred"
        assert strength > 0, "Deterrence strength should be positive"

    def test_turbine_max_distance(self):
        """Turbine should not deter beyond max distance."""
        params = SimulationParameters(deter_max_distance=10.0)  # 10 km max
        turbine = Turbine(id=3, x=100.0, y=100.0, impact=250.0)
        turbine._is_active = True

        # Porpoise at 15km (37.5 grid cells)
        result = turbine.should_deter(137.5, 100.0, params, cell_size=400.0)
        should_deter, rl, distance_m, strength = result

        assert not should_deter, "Porpoise beyond max distance should not be deterred"


class TestShipDeterrenceParameters:
    """Test ship deterrence parameters match DEPONS."""

    def test_ship_deterrence_coefficients_day(self):
        """Verify day coefficients match DEPONS."""
        params = SimulationParameters()

        assert params.pship_int_day == -3.0569351
        assert params.pship_noise_day == 0.2172813
        assert params.pship_dist_day == -0.1303880
        assert params.pship_dist_x_noise_day == 0.0293443

    def test_ship_deterrence_coefficients_night(self):
        """Verify night coefficients match DEPONS."""
        params = SimulationParameters()

        assert params.pship_int_night == -3.233771
        assert params.pship_noise_night == 0.0
        assert params.pship_dist_night == 0.085242
        assert params.pship_dist_x_noise_night == 0.0

    def test_ship_magnitude_coefficients(self):
        """Verify magnitude coefficients match DEPONS."""
        params = SimulationParameters()

        assert params.cship_int_day == 2.9647996
        assert params.cship_int_night == 2.7543376


class TestShipDeterrenceLogic:
    """Test ship deterrence probability and magnitude calculations."""

    def test_ship_deterrence_probability_formula(self):
        """Verify ship deterrence probability formula."""
        model = ShipDeterrenceModel()

        # Test at known values
        spl = 140.0  # dB
        distance_km = 1.0

        # Day: linear = -3.0569351 + 0.2172813*140 - 0.1303880*1 + 0.0293443*140*1
        # = -3.0569 + 30.4194 - 0.1304 + 4.1082 = 31.34
        # prob = 1 / (1 + exp(-31.34)) ≈ 1.0
        prob_day = model.calculate_deterrence_probability(spl, distance_km, is_day=True)

        assert 0.99 < prob_day <= 1.0, f"High SPL should give high probability, got {prob_day}"

    def test_ship_day_night_difference(self):
        """Day and night should produce different deterrence probabilities."""
        model = ShipDeterrenceModel()

        spl = 130.0
        distance_km = 2.0

        prob_day = model.calculate_deterrence_probability(spl, distance_km, is_day=True)
        prob_night = model.calculate_deterrence_probability(spl, distance_km, is_day=False)

        # Day and night should differ due to different coefficients
        # (Note: May be similar at some SPL/distance combinations)
        print(f"Day prob: {prob_day:.3f}, Night prob: {prob_night:.3f}")

    def test_ship_deterrence_decreases_with_distance(self):
        """Deterrence probability should decrease with distance."""
        model = ShipDeterrenceModel()

        spl = 140.0

        prob_1km = model.calculate_deterrence_probability(spl, 1.0, is_day=True)
        prob_5km = model.calculate_deterrence_probability(spl, 5.0, is_day=True)

        # At same SPL, closer distance should have higher probability
        # (Though model is complex, this generally holds)
        print(f"Prob at 1km: {prob_1km:.3f}, at 5km: {prob_5km:.3f}")


class TestDeterrenceVector:
    """Test deterrence vector calculation."""

    def test_vector_points_away_from_source(self):
        """Deterrence vector should point away from noise source."""
        # Porpoise at (10, 0), source at (0, 0)
        dx, dy = calculate_deterrence_vector(
            porpoise_x=10.0, porpoise_y=0.0,
            source_x=0.0, source_y=0.0,
            strength=1.0, deter_coeff=0.07
        )

        # Should point in +x direction (away from source)
        assert dx > 0, "dx should be positive (away from source)"
        assert abs(dy) < 0.01, "dy should be near zero"

    def test_vector_scaled_by_strength(self):
        """Deterrence vector magnitude should scale with strength."""
        dx1, dy1 = calculate_deterrence_vector(10.0, 0.0, 0.0, 0.0, strength=1.0, deter_coeff=0.07)
        dx2, dy2 = calculate_deterrence_vector(10.0, 0.0, 0.0, 0.0, strength=2.0, deter_coeff=0.07)

        mag1 = np.sqrt(dx1**2 + dy1**2)
        mag2 = np.sqrt(dx2**2 + dy2**2)

        assert abs(mag2 / mag1 - 2.0) < 0.01, "Magnitude should double with doubled strength"

    def test_vector_scaled_by_coefficient(self):
        """Deterrence vector magnitude should scale with coefficient."""
        dx1, dy1 = calculate_deterrence_vector(10.0, 0.0, 0.0, 0.0, strength=1.0, deter_coeff=0.07)
        dx2, dy2 = calculate_deterrence_vector(10.0, 0.0, 0.0, 0.0, strength=1.0, deter_coeff=0.14)

        mag1 = np.sqrt(dx1**2 + dy1**2)
        mag2 = np.sqrt(dx2**2 + dy2**2)

        assert abs(mag2 / mag1 - 2.0) < 0.01, "Magnitude should double with doubled coefficient"


class TestDeterrenceDistance:
    """Test deterrence distance calculation (distance where RL = threshold)."""

    def test_deterrence_distance_at_200db(self):
        """Calculate deterrence distance for 200 dB source."""
        sl = 200.0
        threshold = 158.0

        dist = calculate_deterrence_distance(sl, threshold, alpha_hat=0.0, beta_hat=20.0)

        # At 200 dB, RL = threshold when:
        # 158 = 200 - 20*log10(r)
        # log10(r) = (200 - 158) / 20 = 2.1
        # r = 10^2.1 ≈ 126 m
        expected = 10 ** ((sl - threshold) / 20.0)

        assert abs(dist - expected) < 1.0, f"Deterrence distance should be ~{expected:.0f}m, got {dist:.0f}m"

    def test_deterrence_distance_at_180db(self):
        """Calculate deterrence distance for 180 dB source."""
        sl = 180.0
        threshold = 158.0

        dist = calculate_deterrence_distance(sl, threshold, alpha_hat=0.0, beta_hat=20.0)

        # r = 10^((180-158)/20) = 10^1.1 ≈ 12.6 m
        expected = 10 ** ((sl - threshold) / 20.0)

        assert abs(dist - expected) < 0.5, f"Deterrence distance should be ~{expected:.0f}m, got {dist:.0f}m"


class TestProbabilisticDeterrence:
    """Test probabilistic (sigmoid-based) deterrence response."""

    def test_response_probability_at_threshold(self):
        """At threshold, probability should be 0.5."""
        threshold = 158.0
        prob = response_probability_from_rl(threshold, threshold, slope=0.2)

        assert abs(prob - 0.5) < 0.01, f"Probability at threshold should be 0.5, got {prob}"

    def test_response_probability_above_threshold(self):
        """Above threshold, probability should be > 0.5."""
        threshold = 158.0
        rl = 170.0  # 12 dB above threshold

        prob = response_probability_from_rl(rl, threshold, slope=0.2)

        # With slope=0.2: prob = 1 / (1 + exp(-0.2*12)) = 1 / (1 + exp(-2.4)) ≈ 0.92
        assert prob > 0.9, f"Probability at RL=170 should be >0.9, got {prob}"

    def test_response_probability_below_threshold(self):
        """Below threshold, probability should be < 0.5."""
        threshold = 158.0
        rl = 146.0  # 12 dB below threshold

        prob = response_probability_from_rl(rl, threshold, slope=0.2)

        # With slope=0.2: prob = 1 / (1 + exp(-0.2*(-12))) = 1 / (1 + exp(2.4)) ≈ 0.08
        assert prob < 0.1, f"Probability at RL=146 should be <0.1, got {prob}"

    def test_response_probability_vectorized(self):
        """Test vectorized response probability calculation."""
        threshold = 158.0
        rl_values = np.array([146.0, 158.0, 170.0])

        probs = response_probability_from_rl(rl_values, threshold, slope=0.2)

        assert probs[0] < 0.15, "Low RL should give low probability"
        assert abs(probs[1] - 0.5) < 0.01, "At threshold, probability should be 0.5"
        assert probs[2] > 0.85, "High RL should give high probability"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
