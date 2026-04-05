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


class TestShipLoaderWarnings:
    """Verify ship loader logs warnings for missing/invalid data."""

    def test_missing_route_file_logs_warning(self, tmp_path, caplog):
        """Missing route file should log warning, not silently return empty."""
        from cenop.agents.ship import ShipManager
        mgr = ShipManager.__new__(ShipManager)
        with caplog.at_level(logging.WARNING):
            routes = mgr._load_routes(
                str(tmp_path / "nonexistent_routes.txt"),
                0.0, 0.0, 400.0,
            )
        assert routes == {}
        assert "route" in caplog.text.lower() or "not found" in caplog.text.lower()

    def test_unknown_route_name_logs_warning(self, tmp_path, caplog):
        """Ship referencing an unknown route name should log warning."""
        from cenop.agents.ship import ShipManager
        ship_file = tmp_path / "ships.txt"
        ship_file.write_text(
            "name\ttype\tlength\troute\n"
            "TestShip\tother\t10.0\tnonexistent_route\n"
        )
        mgr = ShipManager.__new__(ShipManager)
        with caplog.at_level(logging.WARNING):
            ships = mgr._load_ships(str(ship_file), {})
        assert len(ships) == 1
        assert "route" in caplog.text.lower() or "unknown" in caplog.text.lower()

    def test_invalid_tick_timing_logs_warning(self, tmp_path, caplog):
        """Invalid tick timing values should log warning."""
        from cenop.agents.ship import ShipManager
        ship_file = tmp_path / "ships.txt"
        ship_file.write_text(
            "name\ttype\tlength\troute\ttick_start\ttick_end\n"
            "TestShip\tother\t10.0\troute1\tabc\txyz\n"
        )
        mgr = ShipManager.__new__(ShipManager)
        with caplog.at_level(logging.WARNING):
            ships = mgr._load_ships(str(ship_file), {"route1": type("Route", (), {"buoys": []})()})
        assert "tick" in caplog.text.lower() or "timing" in caplog.text.lower()

    def test_json_parse_error_disables_manager(self, tmp_path, caplog):
        """Invalid JSON should log error and disable the ship manager."""
        from cenop.agents.ship import ShipManager
        json_file = tmp_path / "ships.json"
        json_file.write_text("{invalid json")
        mgr = ShipManager.__new__(ShipManager)
        mgr.ships = []
        mgr.enabled = True
        with caplog.at_level(logging.ERROR):
            mgr.load_from_json(str(json_file), 0.0, 0.0, 400.0)
        assert not mgr.enabled, "Ship manager should be disabled after JSON parse failure"


class TestPathTraversalPrevention:
    """Verify path traversal attacks are blocked."""

    def test_get_data_file_strips_traversal(self):
        """get_data_file must resolve within DATA_DIR, not traverse out."""
        from cenop.config import get_data_file, DATA_DIR
        result = get_data_file("../../etc/passwd")
        assert str(result.resolve()).startswith(str(DATA_DIR.resolve())), (
            f"Path {result} escapes DATA_DIR {DATA_DIR}"
        )

    def test_get_wind_farm_file_strips_traversal(self):
        """get_wind_farm_file must resolve within WIND_FARMS_DIR."""
        from cenop.config import get_wind_farm_file, WIND_FARMS_DIR
        result = get_wind_farm_file("../../../etc/passwd")
        assert str(result.resolve()).startswith(str(WIND_FARMS_DIR.resolve())), (
            f"Path {result} escapes WIND_FARMS_DIR {WIND_FARMS_DIR}"
        )

    def test_get_data_file_rejects_absolute_path(self):
        """get_data_file must not allow absolute paths to escape."""
        from cenop.config import get_data_file, DATA_DIR
        result = get_data_file("/etc/passwd")
        assert str(result.resolve()).startswith(str(DATA_DIR.resolve()))

    def test_get_wind_farm_file_rejects_absolute_path(self):
        """get_wind_farm_file must not allow absolute paths."""
        from cenop.config import get_wind_farm_file, WIND_FARMS_DIR
        result = get_wind_farm_file("/etc/passwd")
        assert str(result.resolve()).startswith(str(WIND_FARMS_DIR.resolve()))


class TestErrorSanitization:
    """Verify internal details are not leaked and bounds are enforced."""

    def test_porpoise_count_rejects_above_50000(self):
        """Server must reject porpoise count above 50000."""
        from cenop.parameters.simulation_params import SimulationParameters
        with pytest.raises(ValueError, match="50,000"):
            SimulationParameters(porpoise_count=50001)

    def test_porpoise_count_accepts_50000(self):
        """Exactly 50000 should be accepted."""
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters(porpoise_count=50000)
        assert params.porpoise_count == 50000
