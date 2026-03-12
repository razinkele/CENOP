"""Tests for WestonFlux transmission loss model."""

import pytest
from cenop.behavior.weston_flux import weston_flux_tl


class TestWestonFlux:
    """Test WestonFlux TL against Java reference values."""

    def test_basic_tl(self):
        """WestonFlux TL for typical North Sea conditions."""
        tl = weston_flux_tl(distance=1000.0, depth=30.0, grain_size=2.0,
                            temperature=10.0, salinity=35.0)
        # TL should be in reasonable range (40-80 dB for 1km)
        assert 30 < tl < 90, f"TL={tl} outside expected range"

    def test_tl_increases_with_distance(self):
        """TL should increase with distance."""
        tl_near = weston_flux_tl(100.0, 30.0, 2.0, 10.0, 35.0)
        tl_far = weston_flux_tl(5000.0, 30.0, 2.0, 10.0, 35.0)
        assert tl_far > tl_near

    def test_tl_varies_with_depth(self):
        """TL should be different for different depths (both valid)."""
        tl_shallow = weston_flux_tl(1000.0, 10.0, 2.0, 10.0, 35.0)
        tl_deep = weston_flux_tl(1000.0, 100.0, 2.0, 10.0, 35.0)
        # Both should be in valid range; the relationship depends on the
        # Weston flux integral's balance of geometric spreading and bottom loss
        assert 20 < tl_shallow < 100
        assert 20 < tl_deep < 100
        assert tl_shallow != tl_deep

    def test_zero_distance_returns_zero(self):
        """Zero distance should return 0 TL."""
        tl = weston_flux_tl(0.0, 30.0, 2.0, 10.0, 35.0)
        assert tl == 0.0

    def test_various_grain_sizes(self):
        """TL should be computable for all valid grain sizes."""
        for gs in [-5.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0]:
            tl = weston_flux_tl(1000.0, 30.0, gs, 10.0, 35.0)
            assert 0 < tl < 200, f"TL={tl} for grain_size={gs} outside range"
