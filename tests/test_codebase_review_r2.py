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


class TestDtypeAndFoodBugs:
    """Verify dtype preservation and food calculation correctness."""

    def test_mating_day_stays_int16(self):
        """mating_day must remain int16 after initialization, not promote to int64."""
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.agents.population import PorpoisePopulation

        params = SimulationParameters(porpoise_count=10)
        pop = PorpoisePopulation(10, params, landscape=None)
        assert pop.mating_day.dtype == np.int16, (
            f"mating_day dtype is {pop.mating_day.dtype}, expected int16"
        )

    def test_eat_food_conserved_multi_agent_cell(self, monkeypatch):
        """Food conservation: sum(eaten) + remaining = original food.

        The NumPy fallback has a bug where actual_available is computed
        from the already-depleted grid. Numba kernels don't have this bug.
        We must force the NumPy path by making kernel imports fail.
        """
        import sys
        import types
        import cenop.optimizations as _opt_pkg
        import cenop.optimizations.kernels  # ensure loaded

        class _BlockedKernels(types.ModuleType):
            def __getattr__(self, name):
                raise ImportError(f"blocked for test: {name}")

        stub = _BlockedKernels("cenop.optimizations.kernels")
        monkeypatch.setitem(sys.modules, "cenop.optimizations.kernels", stub)
        monkeypatch.setattr(_opt_pkg, "kernels", stub)

        from cenop.landscape.cell_data import create_homogeneous_landscape

        landscape = create_homogeneous_landscape(width=10, height=10, depth=20.0)

        # Two agents at same position, each wanting 80% of the food
        x = np.array([5.0, 5.0], dtype=np.float64)
        y = np.array([5.0, 5.0], dtype=np.float64)
        fract = np.array([0.8, 0.8], dtype=np.float32)

        i, j = int(y[0]), int(x[0])
        original_food = float(landscape._food_value[i, j])

        eaten = landscape.eat_food_vectorized(x, y, fract)

        remaining_food = float(landscape._food_value[i, j])
        total = float(np.sum(eaten)) + remaining_food

        # Food conservation: what was eaten + what remains = original
        # (allowing for 0.01 artificial food floor)
        assert total >= original_food - 0.02, (
            f"Food not conserved: eaten={np.sum(eaten):.4f} + "
            f"remaining={remaining_food:.4f} = {total:.4f}, "
            f"original={original_food:.4f}"
        )
        # Each agent should get a fair share (roughly equal)
        if len(eaten) > 1 and eaten[0] > 0:
            assert abs(eaten[0] - eaten[1]) < 0.05, (
                f"Unequal shares: {eaten[0]:.4f} vs {eaten[1]:.4f}"
            )

    def test_eat_food_untouched_cells_unchanged(self):
        """Cells where no agent ate should retain original food value."""
        from cenop.landscape.cell_data import create_homogeneous_landscape

        landscape = create_homogeneous_landscape(width=10, height=10, depth=20.0)

        # One agent at (5, 5)
        x = np.array([5.0], dtype=np.float64)
        y = np.array([5.0], dtype=np.float64)
        fract = np.array([0.5], dtype=np.float32)

        # Record food at a cell that NO agent touches
        original_remote_food = float(landscape._food_value[0, 0])

        landscape.eat_food_vectorized(x, y, fract)

        # Remote cell should be unchanged
        assert landscape._food_value[0, 0] == pytest.approx(original_remote_food), (
            f"Untouched cell changed from {original_remote_food} to "
            f"{landscape._food_value[0, 0]}"
        )


class TestSocialBuffersAndTickCounter:
    """Verify dead social buffers are removed and tick_counter is guarded."""

    def test_no_dead_social_f64_buffers(self):
        """Population should not have unused _social_f64_* buffers."""
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=5)
        pop = PorpoisePopulation(5, params, landscape=None)
        assert not hasattr(pop, '_social_f64_dx'), (
            "_social_f64_dx still exists — dead buffer not removed"
        )
        assert not hasattr(pop, '_social_f64_dy'), (
            "_social_f64_dy still exists — dead buffer not removed"
        )

    def test_energy_history_skips_in_jax_mode(self):
        """_update_energy_history should be a no-op when _use_jax is True."""
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=5)
        pop = PorpoisePopulation(5, params, landscape=None)
        pop._use_jax = True
        mask = pop.active_mask.copy()
        old_counter = pop._tick_counter
        pop._update_energy_history(mask)
        assert pop._tick_counter == old_counter, (
            "_tick_counter changed in JAX mode — should have been skipped"
        )


class TestSilentFailureLogging:
    """Verify silent failures now produce log messages."""

    def test_safe_float_logs_on_invalid_input(self, caplog):
        """_safe_float should log when falling back to default."""
        from cenop.server.simulation_controller import _safe_float
        with caplog.at_level(logging.WARNING):
            result = _safe_float(lambda: "not_a_number", 42.0)
        assert result == 42.0
        assert caplog.text, "_safe_float should log on invalid input"

    def test_safe_input_logs_on_invalid_input(self, caplog):
        """_safe_input should log when falling back to default."""
        from cenop.server.simulation_controller import _safe_input
        mock_input = type("MockInput", (), {})()
        with caplog.at_level(logging.WARNING):
            result = _safe_input(mock_input, "nonexistent_field", "default_val")
        assert result == "default_val"
        assert caplog.text, "_safe_input should log on missing attribute"

    def test_initial_position_fallback_logs_warning(self, caplog):
        """Fallback to center position should log a warning."""
        from cenop.core.simulation import Simulation
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        landscape = create_homogeneous_landscape(width=100, height=100, depth=-1.0)
        params = SimulationParameters(porpoise_count=1, min_depth=5.0)
        sim = Simulation.__new__(Simulation)
        sim.params = params
        sim._cell_data = landscape
        with caplog.at_level(logging.WARNING):
            x, y = sim._get_valid_initial_position()
        assert x == pytest.approx(50.0)
        assert y == pytest.approx(50.0)
        assert "center" in caplog.text.lower() or "position" in caplog.text.lower()
