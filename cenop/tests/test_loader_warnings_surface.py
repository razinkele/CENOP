from pathlib import Path
from cenop.landscape.loader import LandscapeLoader


def test_loader_returns_warnings_when_monthly_missing(tmp_path, caplog):
    ls_dir = tmp_path / "UserDefined"
    ls_dir.mkdir()
    # no monthly files created
    loader = LandscapeLoader('UserDefined', data_dir=tmp_path)
    arr, warnings = loader._load_monthly('prey')
    assert isinstance(warnings, list)
    assert len(warnings) >= 1
    assert 'falling back to zeros' in warnings[0]
