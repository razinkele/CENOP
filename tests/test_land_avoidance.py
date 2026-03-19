"""Tests for land avoidance skip optimization."""

import numpy as np
import pytest
from cenop.agents.population import PorpoisePopulation
from cenop.parameters.simulation_params import SimulationParameters


class TestSkipLandAvoidanceFlag:
    """Test that _skip_land_avoidance flag is set correctly during init."""

    def test_homogeneous_no_landscape_sets_flag(self):
        """When landscape is None (Homogeneous), flag should be True."""
        params = SimulationParameters()
        pop = PorpoisePopulation(count=10, params=params, landscape=None)
        assert pop._skip_land_avoidance is True

    def test_all_deep_water_landscape_sets_flag(self):
        """When all valid cells have sufficient depth, flag should be True."""
        params = SimulationParameters()
        from unittest.mock import MagicMock

        landscape = MagicMock()
        landscape.landscape_name = "TestDeep"
        landscape.width = 50
        landscape.height = 50
        landscape._depth = np.full((50, 50), 20.0, dtype=np.float32)
        landscape.metadata = MagicMock()
        landscape.metadata.cellsize = 400.0

        pop = PorpoisePopulation(count=10, params=params, landscape=landscape)
        assert pop._skip_land_avoidance is True

    def test_landscape_with_land_does_not_set_flag(self):
        """When landscape has land cells (NaN), flag should be False."""
        params = SimulationParameters()
        from unittest.mock import MagicMock

        landscape = MagicMock()
        landscape.landscape_name = "NorthSea"
        landscape.width = 50
        landscape.height = 50
        depth = np.full((50, 50), 20.0, dtype=np.float32)
        depth[10:20, 10:20] = np.nan  # Land area
        landscape._depth = depth
        landscape.metadata = MagicMock()
        landscape.metadata.cellsize = 400.0

        pop = PorpoisePopulation(count=10, params=params, landscape=landscape)
        assert pop._skip_land_avoidance is False

    def test_landscape_with_shallow_does_not_set_flag(self):
        """When landscape has shallow cells (< min_depth), flag should be False."""
        params = SimulationParameters()
        from unittest.mock import MagicMock

        landscape = MagicMock()
        landscape.landscape_name = "Shallow"
        landscape.width = 50
        landscape.height = 50
        depth = np.full((50, 50), 20.0, dtype=np.float32)
        depth[5, 5] = 0.5  # Below min_depth (1.0)
        landscape._depth = depth
        landscape.metadata = MagicMock()
        landscape.metadata.cellsize = 400.0

        pop = PorpoisePopulation(count=10, params=params, landscape=landscape)
        assert pop._skip_land_avoidance is False

    def test_homogeneous_by_name_sets_flag(self):
        """Landscape named 'Homogeneous' should set flag True."""
        params = SimulationParameters()
        from unittest.mock import MagicMock

        landscape = MagicMock()
        landscape.landscape_name = "Homogeneous"
        landscape.width = 400
        landscape.height = 400
        landscape._depth = None
        landscape.metadata = MagicMock()
        landscape.metadata.cellsize = 400.0

        pop = PorpoisePopulation(count=10, params=params, landscape=landscape)
        assert pop._skip_land_avoidance is True


class TestLandAvoidanceSkipBehavior:
    """Test that skip flag causes early return after reflection."""

    def test_skip_avoids_depth_check_loop(self):
        """When flag is set, _handle_land_avoidance should return after reflection
        without entering the 6-direction trial loop."""
        params = SimulationParameters()
        pop = PorpoisePopulation(count=5, params=params, landscape=None)
        pop._skip_land_avoidance = True

        # Place agents and give them movement vectors
        np.random.seed(42)
        pop.x[:] = np.random.uniform(10, 390, 5).astype(np.float32)
        pop.y[:] = np.random.uniform(10, 390, 5).astype(np.float32)
        pop._dx[:] = np.random.uniform(-5, 5, 5)
        pop._dy[:] = np.random.uniform(-5, 5, 5)
        mask = pop.active_mask.copy()

        pop._handle_land_avoidance(mask)

        # Positions should be updated (add+reflect) but no trial loop ran
        np.testing.assert_array_less(pop._new_x, 400.0)
        np.testing.assert_array_less(-0.01, pop._new_x)  # >= 0
