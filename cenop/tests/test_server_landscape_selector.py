"""Unit tests for server-side landscape data table and details modal helpers."""
import pytest
from unittest.mock import Mock, MagicMock


def test_simulation_state_has_last_refreshed():
    """Test that SimulationState has the last_refreshed reactive value."""
    from cenop.server.reactive_state import create_state
    state = create_state()
    assert hasattr(state, "last_refreshed")
    # Default should be None; access internal _value to avoid reactive context in unit tests
    assert getattr(state.last_refreshed, '_value', None) is None


def test_build_landscape_table_rows_with_valid_landscapes(monkeypatch):
    """Test _build_landscape_table_rows with mock LandscapeLoader returning valid data."""
    from cenop.server.main import _build_landscape_table_rows
    
    # Mock LandscapeLoader class
    class MockLoader:
        def __init__(self, name):
            self.name = name
        
        def list_files(self):
            # Return all core files as present, plus some monthly data
            return {
                'bathy.asc': True,
                'disttocoast.asc': True,
                'sediment.asc': True,
                'patches.asc': True,
                'blocks.asc': True,
                'prey_months': [1, 2, 3],
                'salinity_months': [1, 2]
            }
    
    # Create a mock module and inject it into sys.modules
    import sys
    import types
    mock_loader_module = types.ModuleType('cenop.landscape.loader')
    mock_loader_module.LandscapeLoader = MockLoader
    monkeypatch.setitem(sys.modules, 'cenop.landscape.loader', mock_loader_module)
    
    # Call helper with test landscapes
    landscapes = ['TestLandscape1', 'TestLandscape2']
    rows = _build_landscape_table_rows(landscapes)
    
    # Should return 2 rows (sorted)
    assert len(rows) == 2
    
    # Check first row
    assert rows[0]['name'] == 'TestLandscape1'
    assert rows[0]['core_icons'] == ['✅', '✅', '✅', '✅', '✅']
    assert rows[0]['prey_months'] == [1, 2, 3]
    assert rows[0]['salinity_months'] == [1, 2]
    assert rows[0]['error'] is None
    
    # Check second row
    assert rows[1]['name'] == 'TestLandscape2'
    assert rows[1]['error'] is None


def test_build_landscape_table_rows_with_missing_files(monkeypatch):
    """Test _build_landscape_table_rows when some core files are missing."""
    from cenop.server.main import _build_landscape_table_rows
    
    class MockLoader:
        def __init__(self, name):
            self.name = name
        
        def list_files(self):
            # Only bathy and disttocoast present
            return {
                'bathy.asc': True,
                'disttocoast.asc': True,
                'sediment.asc': False,
                'patches.asc': False,
                'blocks.asc': False,
                'prey_months': [],
                'salinity_months': []
            }
    
    import sys
    import types
    mock_loader_module = types.ModuleType('cenop.landscape.loader')
    mock_loader_module.LandscapeLoader = MockLoader
    monkeypatch.setitem(sys.modules, 'cenop.landscape.loader', mock_loader_module)
    
    landscapes = ['IncompleteLandscape']
    rows = _build_landscape_table_rows(landscapes)
    
    assert len(rows) == 1
    assert rows[0]['name'] == 'IncompleteLandscape'
    # First two files present, last three missing
    assert rows[0]['core_icons'] == ['✅', '✅', '❌', '❌', '❌']
    assert rows[0]['prey_months'] == []
    assert rows[0]['salinity_months'] == []


def test_build_landscape_table_rows_with_loader_error(monkeypatch):
    """Test _build_landscape_table_rows when LandscapeLoader raises an error."""
    from cenop.server.main import _build_landscape_table_rows
    
    class MockLoader:
        def __init__(self, name):
            raise ValueError("Invalid landscape configuration")
        
        def list_files(self):
            pass
    
    import sys
    import types
    mock_loader_module = types.ModuleType('cenop.landscape.loader')
    mock_loader_module.LandscapeLoader = MockLoader
    monkeypatch.setitem(sys.modules, 'cenop.landscape.loader', mock_loader_module)
    
    landscapes = ['BrokenLandscape']
    rows = _build_landscape_table_rows(landscapes)
    
    assert len(rows) == 1
    assert rows[0]['name'] == 'BrokenLandscape'
    assert rows[0]['error'] == 'Invalid landscape configuration'
    assert rows[0]['core_icons'] == []
    assert rows[0]['prey_months'] == []


def test_build_details_modal_content_with_complete_data():
    """Test _build_details_modal_content with all files and warnings."""
    from cenop.server.main import _build_details_modal_content
    
    info = {
        'bathy.asc': True,
        'disttocoast.asc': True,
        'sediment.asc': True,
        'patches.asc': True,
        'blocks.asc': True,
        'prey_months': [1, 2, 3, 4],
        'salinity_months': [1, 2, 3]
    }
    warnings = ['Warning: Low resolution detected', 'Warning: Missing metadata']
    
    result = _build_details_modal_content('TestLandscape', info, warnings)
    
    # Convert to string for checking content
    result_str = str(result)
    
    # Should contain landscape name
    assert 'TestLandscape' in result_str
    
    # Should contain all checkmarks for core files
    assert '✅ bathy.asc' in result_str
    assert '✅ disttocoast.asc' in result_str
    assert '✅ sediment.asc' in result_str
    assert '✅ patches.asc' in result_str
    assert '✅ blocks.asc' in result_str
    
    # Should contain prey and salinity months
    assert '[1, 2, 3, 4]' in result_str
    assert '[1, 2, 3]' in result_str
    
    # Should contain warnings
    assert 'Low resolution detected' in result_str
    assert 'Missing metadata' in result_str


def test_build_details_modal_content_with_missing_files():
    """Test _build_details_modal_content when some files are missing."""
    from cenop.server.main import _build_details_modal_content
    
    info = {
        'bathy.asc': True,
        'disttocoast.asc': False,
        'sediment.asc': True,
        'patches.asc': False,
        'blocks.asc': False,
        'prey_months': [],
        'salinity_months': []
    }
    warnings = []
    
    result = _build_details_modal_content('PartialLandscape', info, warnings)
    result_str = str(result)
    
    # Should have checkmarks for present files
    assert '✅ bathy.asc' in result_str
    assert '✅ sediment.asc' in result_str
    
    # Should have X marks for missing files
    assert '❌ disttocoast.asc' in result_str
    assert '❌ patches.asc' in result_str
    assert '❌ blocks.asc' in result_str
    
    # Should show '—' for empty monthly data
    assert '—' in result_str
    
    # Should show "No loader warnings reported" when warnings list is empty
    assert 'No loader warnings reported' in result_str


def test_build_details_modal_content_no_warnings():
    """Test _build_details_modal_content with no warnings."""
    from cenop.server.main import _build_details_modal_content
    
    info = {
        'bathy.asc': True,
        'disttocoast.asc': True,
        'sediment.asc': True,
        'patches.asc': True,
        'blocks.asc': True,
        'prey_months': [1],
        'salinity_months': [1]
    }
    
    result = _build_details_modal_content('CleanLandscape', info, [])
    result_str = str(result)
    
    # Should indicate no warnings
    assert 'No loader warnings reported' in result_str
    assert 'Loader warnings' not in result_str  # Not showing warning section header

