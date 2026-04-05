"""Tests for codebase review round 2 fixes."""
import logging

import numpy as np
import pytest


class TestParameterDefaults:
    """Verify DEPONS parameter defaults match Java parameters.xml."""

    def test_mean_disp_dist_matches_parameters_xml(self):
        """Java parameters.xml line 117: defaultValue='2'."""
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()
        assert params.mean_disp_dist == 2.0

    def test_psm_angle_matches_parameters_xml(self):
        """Java parameters.xml line 52: defaultValue='40'."""
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()
        assert params.psm_angle == 40.0

    def test_psm_dist_mean_matches_parameters_xml(self):
        """Java parameters.xml line 46: defaultValue='N(350;100)' -> mean=350."""
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()
        assert params.psm_dist_mean == 350.0
