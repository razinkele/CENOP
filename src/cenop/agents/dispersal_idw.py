"""DEPONS 3.2 InnerDanishWatersDispersal, transliterated one animal at a time.

Source: DEPONS-3.2/src/dk/au/bios/porpoise/behavior/InnerDanishWatersDispersal.java (tag
8f533f8). Line numbers in comments refer to that file unless another is named.

Everything here is in **Java coordinates**: the continuous space has its origin at 0.5
(PorpoiseSimBuilder.java:224-226), positions are rounded to cells (Agent.ndPointToGridPoint)
and x_java = x_cenop - 0.5. Callers convert at the boundary. Headings are degrees, 0 = north,
clockwise. Grids are float64 arrays indexed [row = y, col = x] with row 0 south; land reads
-9999 in all of them. The navigation grid holds block index + 1.

The DEPONS quirks are copied on purpose; see
docs/superpowers/specs/2026-11-18-dispersal-idw-design.md. The functions stay inside the
Numba subset and contain no random calls.
"""

import math

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without Numba
    HAS_NUMBA = False

    def njit(func):
        return func


N_DISP_TARGET = 12  # Java 105
MIN_DIST_TO_TARGET_CELLS = 100.0 / 0.4  # Java 50, 516

# Java 53-83. Blocks are indexed 0-59 from the north-west, six per row.
BLOCK_CENTRES_X = np.array([50.0, 150.0, 250.0, 350.0, 450.0, 550.0] * 10)
BLOCK_CENTRES_Y = np.repeat(
    np.array([950.0, 850.0, 750.0, 650.0, 550.0, 450.0, 350.0, 250.0, 150.0, 50.0]), 6
)
# fmt: off
BLOCK_VAL_KAT_1 = np.array([
    0, 0.007, 0.0042, 0.0022, 8e-04, 0, 0, 0.0454, 0.0522,
    0.0176, 0.0185, 0, 0, 0.0458, 0.0348, 0.0062, 0.0182, 0.4446, 0, 0.2898, 0.1013, 0.0769,
    0.126, 0.4615, 0.4439, 0.5104, 0.5741, 0.4593, 0.6447, 0.6498, 0.8685, 0.5063, 0.971,
    0.1697, 0.27, 0.1261, 0.9438, 0.7346, 1, 0.3994, 0.1576, 0.2355, 0.7718, 0.9082, 0.8524,
    0.2161, 0.4916, 0.2975, 0.8397, 0.7533, 0.7199, 0.885, 0.727, 0.1057, 0, 0, 0.8284,
    0.7116, 0.4866, 0,
], dtype=np.float64)
# fmt: on


@njit
def java_round(v):
    """Java Math.round: half-up (Python's round is half-to-even)."""
    return math.floor(v + 0.5)


@njit
def cell_of(xj, yj, w, h):
    """Agent.ndPointToGridPoint (Agent.java:85-98)."""
    x = int(math.floor(xj + 0.5))
    y = int(math.floor(yj + 0.5))
    if y == h:
        y -= 1
    if x == w:
        x -= 1
    return x, y


@njit
def grid_at(grid, xj, yj):
    x, y = cell_of(xj, yj, grid.shape[1], grid.shape[0])
    return grid[y, x]


@njit
def _bounce(loc, disp, lo, hi):
    """Repast BouncyBorders$Dimension.translate (read from the 2.10 class file).

    v = loc + disp; while v < min or v >= max: below min mirror about min, above max mirror
    about max, exactly max step just below it. min = -origin, max = dimension - origin.
    """
    v = loc + disp
    while v < lo or v >= hi:
        if v < lo:
            v = lo + (lo - v)
        elif v > hi:
            v = hi - (v - hi)
        else:
            v = np.nextafter(v, -np.inf)
    return v


@njit
def _displacement(heading, dist):
    """SpatialMath.getDisplacement with the angle Java passes: pi/2 - heading."""
    a = (math.pi / 2) - math.radians(heading % 360.0)
    return dist * math.cos(a), dist * math.sin(a)


@njit
def point_at(xj, yj, heading, dist, w, h):
    """Porpoise.getPointAtHeadingAndDist (Porpoise.java:843-855), border bounce included."""
    dx, dy = _displacement(heading, dist)
    return _bounce(xj, dx, -0.5, w - 0.5), _bounce(yj, dy, -0.5, h - 0.5)


@njit
def select_target(xj, yj, u, blocks):
    """dispTargetSelect (495-564): centre of one of the twelve best blocks. u is U[0, 1)."""
    n_blocks = BLOCK_VAL_KAT_1.shape[0]
    block_quality = np.zeros(n_blocks)
    for nbr in range(n_blocks):  # 505-520
        one_dx = abs(xj - BLOCK_CENTRES_X[nbr])
        one_dy = abs(yj - BLOCK_CENTRES_Y[nbr])
        block_dist = java_round(math.sqrt(one_dx * one_dx + one_dy * one_dy))
        block_quality[nbr] = BLOCK_VAL_KAT_1[nbr]
        if block_dist < MIN_DIST_TO_TARGET_CELLS:
            block_quality[nbr] = 0.0

    high_quality_blocks = np.zeros(N_DISP_TARGET, dtype=np.int64)  # 528-541
    block_quality_sorted = np.sort(block_quality)
    for i in range(N_DISP_TARGET):
        hi_qual_i = block_quality_sorted[n_blocks - i - 1]
        for j in range(n_blocks):
            if block_quality[j] == hi_qual_i:
                high_quality_blocks[i] = j

    the_nbr = int(u * N_DISP_TARGET)  # 546, GeneratedRandomSource.java:142
    sel_block = high_quality_blocks[the_nbr]

    # 550-553: block slightly more to the east if currently far north
    if grid_at(blocks, xj, yj) < 20 and (
        sel_block == 30 or sel_block == 31 or sel_block == 36 or sel_block == 37 or sel_block == 43
    ):
        sel_block += 2

    return BLOCK_CENTRES_X[sel_block], BLOCK_CENTRES_Y[sel_block]
