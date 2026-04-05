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


class TestLandscapeLoaderValidation:
    """Verify landscape loader validates headers and logs missing files."""

    def test_missing_ncols_raises_value_error(self, tmp_path):
        """ASC file with missing ncols must raise, not silently default to 0."""
        from cenop.landscape.loader import LandscapeLoader
        asc_file = tmp_path / "bad.asc"
        asc_file.write_text(
            "nrows 10\nxllcorner 0\nyllcorner 0\ncellsize 400\n"
            "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n" * 10
        )
        loader = LandscapeLoader.__new__(LandscapeLoader)
        loader.landscape_path = tmp_path
        with pytest.raises(ValueError, match="ncols"):
            loader._load_asc("bad.asc")

    def test_missing_nrows_raises_value_error(self, tmp_path):
        """ASC file with missing nrows must raise, not silently default to 0."""
        from cenop.landscape.loader import LandscapeLoader
        asc_file = tmp_path / "bad.asc"
        asc_file.write_text(
            "ncols 10\nxllcorner 0\nyllcorner 0\ncellsize 400\n"
            "1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n" * 10
        )
        loader = LandscapeLoader.__new__(LandscapeLoader)
        loader.landscape_path = tmp_path
        with pytest.raises(ValueError, match="nrows"):
            loader._load_asc("bad.asc")

    def test_valid_header_parses_successfully(self, tmp_path):
        """Complete ASC header should parse without error."""
        from cenop.landscape.loader import LandscapeLoader
        asc_file = tmp_path / "good.asc"
        asc_file.write_text(
            "ncols 3\nnrows 2\nxllcorner 100\nyllcorner 200\ncellsize 400\n"
            "1.0 2.0 3.0\n4.0 5.0 6.0\n"
        )
        loader = LandscapeLoader.__new__(LandscapeLoader)
        loader.landscape_path = tmp_path
        data, metadata = loader._load_asc("good.asc")
        assert metadata.ncols == 3
        assert metadata.nrows == 2
        assert data.shape == (2, 3)

    def test_missing_monthly_file_logs_warning(self, tmp_path, caplog):
        """Missing monthly file should log a warning, not silently duplicate."""
        from cenop.landscape.loader import LandscapeLoader
        for month in [1]:
            fname = tmp_path / f"prey{month:02d}.asc"
            fname.write_text(
                "ncols 2\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 400\n"
                "nodata_value -9999\n1.0 1.0\n1.0 1.0\n"
            )
        loader = LandscapeLoader.__new__(LandscapeLoader)
        loader.landscape_path = tmp_path
        with caplog.at_level(logging.WARNING):
            data = loader._load_monthly("prey")
        assert data.shape[0] == 12
        assert "missing" in caplog.text.lower() or "duplicat" in caplog.text.lower()
