"""Tests for JOMOPANS ship source level model."""

import math

import pytest
from cenop.agents.ship import VesselClass
from cenop.behavior.jomopans_spl import jomopans_spl


class TestJomopansSPL:
    """Test JOMOPANS calibrated source levels."""

    def test_cargo_ship_spl(self):
        """Cargo ship decidecade band SPL at ~16kHz should be reasonable."""
        spl = jomopans_spl(VesselClass.CARGO, speed_knots=12.0, length_m=150.0)
        # Single decidecade band at high frequency, not broadband
        assert 100 < spl < 200, f"Cargo SPL={spl} outside expected range"

    def test_spl_increases_with_speed(self):
        """Faster ships should be louder."""
        spl_slow = jomopans_spl(VesselClass.CARGO, speed_knots=5.0, length_m=150.0)
        spl_fast = jomopans_spl(VesselClass.CARGO, speed_knots=20.0, length_m=150.0)
        assert spl_fast > spl_slow

    def test_15_vessel_classes(self):
        """VesselClass enum should have all 15 JOMOPANS vessel classes (13 Java + CARGO alias + CHEMICAL_TANKER)."""
        assert len(VesselClass) == 15

    def test_all_classes_produce_valid_spl(self):
        """Every vessel class should produce a valid SPL."""
        for vc in VesselClass:
            spl = jomopans_spl(vc, speed_knots=10.0, length_m=100.0)
            assert 100 < spl < 220, f"{vc.name} SPL={spl} outside expected range"

    def test_zero_speed_returns_zero(self):
        """Zero speed should return 0 SPL."""
        spl = jomopans_spl(VesselClass.CARGO, speed_knots=0.0, length_m=150.0)
        assert spl == 0.0

    def test_spl_increases_with_length(self):
        """Longer ships should be louder."""
        spl_small = jomopans_spl(VesselClass.TANKER, speed_knots=10.0, length_m=50.0)
        spl_large = jomopans_spl(VesselClass.TANKER, speed_knots=10.0, length_m=300.0)
        assert spl_large > spl_small

    def test_non_positive_length_does_not_raise(self):
        """A non-positive length (malformed ship file / ships.json) must NOT crash
        the per-tick source-level call — it is clamped to a positive floor."""
        for bad_len in (0.0, -5.0, -100.0):
            spl = jomopans_spl(VesselClass.CARGO, speed_knots=10.0, length_m=bad_len)
            assert math.isfinite(spl), f"length_m={bad_len} produced non-finite SPL {spl}"

    def test_non_positive_length_clamps_to_minimum(self):
        """A non-positive length is clamped to the 1.0 m floor, so its SPL equals
        the SPL computed at length_m=1.0 (and positive lengths are untouched)."""
        clamped = jomopans_spl(VesselClass.CARGO, speed_knots=10.0, length_m=-5.0)
        floor = jomopans_spl(VesselClass.CARGO, speed_knots=10.0, length_m=1.0)
        assert clamped == floor
        # A valid positive length is NOT altered by the clamp.
        assert jomopans_spl(VesselClass.CARGO, 10.0, 150.0) != floor
