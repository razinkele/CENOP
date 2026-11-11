"""Fill the Lithuania data gap left by commit 7d50cfb (2026-03-12).

7d50cfb replaced data/Lithuania/bathy.asc with a larger 329x381 grid (12,014 more water cells:
a 55 km open-sea strip in the west, a refined coastline and the lagoon) but only padded the other
28 layers with NODATA. Their data still covers exactly the old water mask, so 19,767 of the 73,464
water cells (27%) had no salinity, prey, food patches, sediment, blocks or distance to coast.

The existing layers are synthetic placeholders, not measured data (see
scripts/investigations/2026-11-11-lithuania-data-gap/):
  - patches, prey01-12, salinity01-12: independent per-cell noise (neighbour correlation ~0,
    no dependence on depth or position), 2-decimal values around a monthly base;
  - sediment = 1 and blocks = 0 everywhere;
  - disttocoast = a plane in row/col, not a real distance.

This script completes each layer with the same kind of values, on gap cells only:
  - noise and constant layers: each gap cell gets a value drawn (fixed seed) from that layer's
    own existing water values, month by month, so the distribution is unchanged;
  - disttocoast: the plane fitted to the existing values, extended over the gap and clipped
    below at the existing minimum.
Every existing value and every header line is left byte-for-byte unchanged. Running it again is a
no-op. Only water cells (bathy > 0) are filled; land stays NODATA.

Run from the repo root: python3 scripts/fill_lithuania_data_gap.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LANDSCAPE_DIR = Path(__file__).resolve().parents[1] / "data" / "Lithuania"
SEED = 20261111
NODATA = b"-9999"
HEADER_LINES = 6

NOISE_LAYERS = (
    ["patches.asc", "sediment.asc", "blocks.asc"]
    + [f"prey{m:02d}.asc" for m in range(1, 13)]
    + [f"salinity{m:02d}.asc" for m in range(1, 13)]
)
PLANE_LAYERS = ["disttocoast.asc"]


def read_grid(path: Path) -> tuple[list[bytes], list[list[bytes]]]:
    """Header lines and per-row value tokens of an LF-terminated ASC grid."""
    raw = path.read_bytes()
    if b"\r\n" in raw:
        raise ValueError(f"{path.name}: expected LF line endings")
    lines = raw.rstrip(b"\n").split(b"\n")
    return lines[:HEADER_LINES], [line.split(b" ") for line in lines[HEADER_LINES:]]


def write_grid(path: Path, header: list[bytes], rows: list[list[bytes]]) -> None:
    path.write_bytes(b"\n".join(header + [b" ".join(row) for row in rows]) + b"\n")


def water_mask() -> np.ndarray:
    raw = (LANDSCAPE_DIR / "bathy.asc").read_bytes().replace(b"\r\n", b"\n")
    rows = raw.rstrip(b"\n").split(b"\n")[HEADER_LINES:]
    return np.array([[float(t) for t in row.split()] for row in rows]) > 0


def fill_noise(tokens: np.ndarray, gap: np.ndarray, water: np.ndarray, rng) -> None:
    pool = tokens[water & ~gap]
    tokens[gap] = pool[rng.integers(0, len(pool), int(gap.sum()))]


def fill_plane(tokens: np.ndarray, gap: np.ndarray, water: np.ndarray) -> None:
    rows, cols = np.indices(tokens.shape)
    have = water & ~gap
    values = tokens[have].astype(float)
    design = np.column_stack([np.ones(values.size), rows[have], cols[have]])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    extended = coef[0] + coef[1] * rows[gap] + coef[2] * cols[gap]
    extended = np.maximum(extended, values.min())
    tokens[gap] = np.array([f"{v:.2f}".encode() for v in extended], dtype=object)


def main() -> None:
    water = water_mask()
    rng = np.random.default_rng(SEED)
    for name in NOISE_LAYERS + PLANE_LAYERS:
        path = LANDSCAPE_DIR / name
        header, rows = read_grid(path)
        tokens = np.array(rows, dtype=object)
        if tokens.shape != water.shape:
            raise ValueError(f"{name}: grid {tokens.shape} != bathy {water.shape}")
        gap = water & (tokens == NODATA)
        n = int(gap.sum())
        if n:
            original = tokens.copy()
            if name in PLANE_LAYERS:
                fill_plane(tokens, gap, water)
            else:
                fill_noise(tokens, gap, water, rng)
            assert np.array_equal(tokens[~gap], original[~gap]), name
            write_grid(path, header, tokens.tolist())
        print(f"{name:18s} filled {n:6d} water cells")

    for name in NOISE_LAYERS + PLANE_LAYERS:
        _, rows = read_grid(LANDSCAPE_DIR / name)
        left = int((water & (np.array(rows, dtype=object) == NODATA)).sum())
        assert left == 0, f"{name}: {left} water cells still NODATA"
    print("all layers: no NODATA left on water cells")


if __name__ == "__main__":
    main()
