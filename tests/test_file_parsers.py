"""Tests for file parser validation and error handling."""

import pytest
import numpy as np
import tempfile
import os

from cenop.landscape.cell_data import load_bathymetry_from_asc
from cenop.agents.turbine import Turbine
from cenop.agents.ship import ShipManager, Route, Buoy


class TestASCParser:
    """Tests for ASC file parser validation."""

    def _write_asc(self, content: str) -> str:
        """Write content to a temp ASC file and return path."""
        fd, path = tempfile.mkstemp(suffix='.asc')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_valid_file(self):
        """Valid ASC file parses correctly."""
        content = (
            "ncols 3\n"
            "nrows 2\n"
            "xllcorner 0.0\n"
            "yllcorner 0.0\n"
            "cellsize 400.0\n"
            "NODATA_value -9999\n"
            "1.0 2.0 3.0\n"
            "4.0 5.0 6.0\n"
        )
        path = self._write_asc(content)
        try:
            depth, meta = load_bathymetry_from_asc(path)
            assert depth.shape == (2, 3)
            assert meta.ncols == 3
            assert meta.nrows == 2
        finally:
            os.unlink(path)

    def test_missing_header_line(self):
        """Missing header line raises ValueError."""
        content = (
            "ncols 3\n"
            "nrows 2\n"
            "1.0 2.0 3.0\n"
            "4.0 5.0 6.0\n"
        )
        path = self._write_asc(content)
        try:
            with pytest.raises(ValueError, match="header"):
                load_bathymetry_from_asc(path)
        finally:
            os.unlink(path)

    def test_wrong_data_count(self):
        """Data count not matching nrows*ncols raises ValueError."""
        content = (
            "ncols 3\n"
            "nrows 2\n"
            "xllcorner 0.0\n"
            "yllcorner 0.0\n"
            "cellsize 400.0\n"
            "NODATA_value -9999\n"
            "1.0 2.0\n"
            "4.0 5.0 6.0\n"
        )
        path = self._write_asc(content)
        try:
            with pytest.raises(ValueError, match="data values"):
                load_bathymetry_from_asc(path)
        finally:
            os.unlink(path)

    def test_empty_file(self):
        """Empty file raises ValueError."""
        path = self._write_asc("")
        try:
            with pytest.raises((ValueError, IndexError)):
                load_bathymetry_from_asc(path)
        finally:
            os.unlink(path)

    def test_non_numeric_data(self):
        """Non-numeric data values raise ValueError."""
        content = (
            "ncols 2\n"
            "nrows 1\n"
            "xllcorner 0.0\n"
            "yllcorner 0.0\n"
            "cellsize 400.0\n"
            "NODATA_value -9999\n"
            "abc def\n"
        )
        path = self._write_asc(content)
        try:
            with pytest.raises(ValueError):
                load_bathymetry_from_asc(path)
        finally:
            os.unlink(path)


class TestTurbineParser:
    """Tests for turbine file parser validation."""

    def _write_turbine(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.txt')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_valid_file(self):
        """Valid turbine file parses correctly."""
        content = (
            "name x y impact start end\n"
            "T1 500000.0 6000000.0 0.5 0 1000\n"
            "T2 500400.0 6000400.0 0.7 0 1000\n"
        )
        path = self._write_turbine(content)
        try:
            turbines = Turbine.load_from_file(path, 500000.0, 6000000.0, 400.0)
            assert len(turbines) == 2
        finally:
            os.unlink(path)

    def test_non_numeric_coordinates(self):
        """Non-numeric coordinates are skipped with warning."""
        content = (
            "name x y impact\n"
            "T1 abc def 0.5\n"
            "T2 500400.0 6000400.0 0.7\n"
        )
        path = self._write_turbine(content)
        try:
            turbines = Turbine.load_from_file(path, 500000.0, 6000000.0, 400.0)
            assert len(turbines) == 1  # Bad row skipped
        finally:
            os.unlink(path)

    def test_empty_file(self):
        """Empty file (header only) returns empty list."""
        path = self._write_turbine("name x y impact\n")
        try:
            turbines = Turbine.load_from_file(path, 500000.0, 6000000.0, 400.0)
            assert len(turbines) == 0
        finally:
            os.unlink(path)


class TestShipParser:
    """Tests for ship route/ship file parser validation."""

    def _write_file(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix='.txt')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_valid_route(self):
        """Valid route file parses correctly."""
        content = (
            "ROUTE test_route\n"
            "500000.0 6000000.0 10.0\n"
            "500400.0 6000400.0 10.0\n"
        )
        path = self._write_file(content)
        try:
            manager = ShipManager.__new__(ShipManager)
            routes = manager._load_routes(path, 500000.0, 6000000.0, 400.0)
            assert "test_route" in routes
            assert len(routes["test_route"].buoys) == 2
        finally:
            os.unlink(path)

    def test_non_numeric_route_coordinates(self):
        """Non-numeric route coordinates are skipped with warning."""
        content = (
            "ROUTE test_route\n"
            "abc def\n"
            "500400.0 6000400.0 10.0\n"
        )
        path = self._write_file(content)
        try:
            manager = ShipManager.__new__(ShipManager)
            routes = manager._load_routes(path, 500000.0, 6000000.0, 400.0)
            route = routes.get("test_route")
            assert route is not None
            assert len(route.buoys) == 1  # Bad row skipped
        finally:
            os.unlink(path)
