import numpy as np
from pathlib import Path
from cenop.landscape.loader import LandscapeLoader


def write_small_asc(path: Path, ncols: int = 3, nrows: int = 2):
    # Simple ASC content
    header = f"ncols {ncols}\n nrows {nrows}\n xllcorner 0\n yllcorner 0\n cellsize 400\n NODATA_value -9999\n"
    rows = "\n".join([" ".join(["1"] * ncols) for _ in range(nrows)])
    path.write_text(header + rows)


def test_load_monthly_fallback_to_bathy(tmp_path):
    # Create landscape dir with only bathy.asc
    ls_dir = tmp_path / "UserDefined"
    ls_dir.mkdir()
    bathy = ls_dir / "bathy.asc"
    write_small_asc(bathy, ncols=4, nrows=3)

    loader = LandscapeLoader('UserDefined', data_dir=tmp_path)
    data = loader.load_all()

    assert 'entropy' in data
    entropy = data['entropy']
    assert entropy.shape == (12, 3, 4)
    assert np.allclose(entropy, 0.0)
