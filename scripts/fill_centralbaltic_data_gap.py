"""Fill the CentralBaltic data gap left when bathy.asc was replaced with EMODnet depths.

scripts/generate_central_baltic.py built every CentralBaltic layer on its own synthetic
bathymetry, whose only land is a border band (distance to the grid edge < 2% on linspace(0, 1)
coordinates; islands are made shallow, never land). scripts/download_emodnet_tiles.py later
replaced bathy.asc alone with EMODnet depths, which have water in much of that band. Those
59,776 water cells (3.25% of the water) kept the generator's land values: NODATA for salinity
and prey, 0 for patches, sediment, blocks and distance to coast. Verified cell for cell in
scripts/investigations/2026-11-11-centralbaltic-data-gap/gap_origin.py.

This script gives exactly those cells the values the generator gives water cells, using its own
rules (fixed seed), with the EMODnet depth where a rule depends on depth:
  - patches  = min(0.3 + U(0, 0.5) + max(0.3 * (1 - edge / 0.15), 0), 1)
  - salinity = clip((8 + 10 (1 - X)(1 - Y)) * (1 - 0.15 sin((m - 4) pi / 6)) + N(0, 0.5), 2, 25)
  - prey     = clip(patches * (0.7 + 0.3 sin((m - 3) pi / 6)) + N(0, 0.1), 0, 1)
  - sediment = 1 if depth > 50, 2 if depth > 20, else 3
  - disttocoast = 50 * depth
  - blocks   = (row // 50) * (ncols // 50) + col // 50 + 1
Every other value and the header lines are left byte-for-byte unchanged, values are written
like the generator writes them (%.6f), and running it again is a no-op.

Run from the repo root: python3 scripts/fill_centralbaltic_data_gap.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LANDSCAPE_DIR = Path(__file__).resolve().parents[1] / "data" / "CentralBaltic"
SEED = 20261111
HEADER_LINES = 6
NODATA = b"-9999"
ZERO = b"0.000000"
MONTHS = range(1, 13)


def read_grid(path: Path) -> tuple[list[bytes], list[list[bytes]]]:
    """Header lines (verbatim) and per-row value tokens of an LF-terminated ASC grid."""
    raw = path.read_bytes()
    if b"\r\n" in raw:
        raise ValueError(f"{path.name}: expected LF line endings")
    lines = raw.rstrip(b"\n").split(b"\n")
    return lines[:HEADER_LINES], [line.split(b" ") for line in lines[HEADER_LINES:]]


def write_grid(path: Path, header: list[bytes], rows: list[list[bytes]]) -> None:
    path.write_bytes(b"\n".join(header + [b" ".join(row) for row in rows]) + b"\n")


def generator_values(depth: np.ndarray, cells: tuple[np.ndarray, np.ndarray]) -> dict:
    """Generator-rule values for ``cells`` (row, col), drawn in a fixed order."""
    nrows, ncols = depth.shape
    r, c = cells
    x = np.linspace(0, 1, ncols)[c]
    y = np.linspace(0, 1, nrows)[r]
    edge = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    d = depth[r, c]
    rng = np.random.default_rng(SEED)

    patches = 0.3 + rng.uniform(0, 0.5, r.size) + np.maximum((1 - edge / 0.15) * 0.3, 0)
    patches = np.minimum(patches, 1.0)
    values = {
        "patches.asc": patches,
        "sediment.asc": np.select([d > 50, d > 20], [1.0, 2.0], 3.0),
        "disttocoast.asc": d * 50,
        "blocks.asc": ((r // 50) * (ncols // 50) + c // 50 + 1).astype(float),
    }
    base_salinity = 8 + 10 * (1 - x) * (1 - y)
    for m in MONTHS:
        season = 1.0 - 0.15 * np.sin((m - 4) * np.pi / 6)
        sal = base_salinity * season + rng.normal(0, 0.5, r.size)
        values[f"salinity{m:02d}.asc"] = np.clip(sal, 2, 25)
        prey_season = 0.7 + 0.3 * np.sin((m - 3) * np.pi / 6)
        prey = patches * prey_season + rng.normal(0, 0.1, r.size)
        values[f"prey{m:02d}.asc"] = np.clip(prey, 0, 1)
    return values


def main() -> None:
    _, bathy_rows = read_grid(LANDSCAPE_DIR / "bathy.asc")
    depth = np.array([[float(t) for t in row] for row in bathy_rows])
    nrows, ncols = depth.shape
    water = depth > 0
    x, y = np.meshgrid(np.linspace(0, 1, ncols), np.linspace(0, 1, nrows))
    band = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)) < 0.02
    cells = np.where(water & band)
    values = generator_values(depth, cells)

    for name, new in values.items():
        land_token = NODATA if name.startswith(("salinity", "prey")) else ZERO
        path = LANDSCAPE_DIR / name
        header, rows = read_grid(path)
        tokens = np.array(rows, dtype=object)
        if tokens.shape != depth.shape:
            raise ValueError(f"{name}: grid {tokens.shape} != bathy {depth.shape}")
        is_land_value = water & (tokens == land_token)
        if (is_land_value & ~band).any():
            raise ValueError(f"{name}: land values on water outside the generator band")
        todo = is_land_value[cells]
        for r, c, v in zip(cells[0][todo], cells[1][todo], new[todo]):
            rows[r][c] = f"{v:.6f}".encode()
        if todo.any():
            write_grid(path, header, rows)
        print(f"{name:16s} filled {int(todo.sum()):6d} water cells")

    for name in values:
        land_token = NODATA if name.startswith(("salinity", "prey")) else ZERO
        _, rows = read_grid(LANDSCAPE_DIR / name)
        left = int((water & (np.array(rows, dtype=object) == land_token)).sum())
        assert left == 0, f"{name}: {left} water cells still hold the land value"
    print("all layers: every water cell has a generator water value")


if __name__ == "__main__":
    main()
