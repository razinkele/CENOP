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


@njit
def _norm(angle):
    """Agent.normHeading (Agent.java:192-203)."""
    while angle < 0:
        angle += 360.0
    while angle >= 360.0:
        angle -= 360.0
    return angle


@njit
def _angle_from_displacement(dx, dy):
    """repast.simphony.relogo.Utility.angleFromDisplacement."""
    if dy == 0:
        return 90.0 if dx >= 0 else 270.0
    ang_rad = math.atan(dx / dy)
    if dy > 0:
        if ang_rad < 0:
            return math.degrees(2 * math.pi + ang_rad)
        return math.degrees(ang_rad)
    return math.degrees(math.pi + ang_rad)


@njit
def _face(xj, yj, px, py, heading):
    """Agent.facePoint (Agent.java:128-134): no change when the point is the position."""
    if px == xj and py == yj:
        return heading
    return _norm(_angle_from_displacement(px - xj, py - yj))


@njit
def _at(grid, xj, yj, heading, dist):
    """Grid value at getPointAtHeadingAndDist(heading, dist)."""
    px, py = point_at(xj, yj, heading, dist, grid.shape[1], grid.shape[0])
    return grid_at(grid, px, py)


@njit
def _disp1(xj, yj, heading, tx, ty, d1x, d1y, d8x, d8y, enough_water, depth, dtc, blocks, mean_disp_dist, min_disp_depth,):  # fmt: skip
    """disp1 (192-379). Returns (phase, xj, yj, heading, paid_moves)."""
    w = depth.shape[1]
    h = depth.shape[0]
    min_dist_to_land = 2000.0  # 196

    # 200-203: south-east of Sealand, no directed dispersal
    if xj > 438 and yj < 478:
        return 2, xj, yj, heading, 0

    current_block = grid_at(blocks, xj, yj)  # 205
    # 207-210: north of Djursland or in Little Belt
    if current_block == 14 or current_block == 31:
        return 2, xj, yj, heading, 0

    # 213-218: go north if north of Funen. Agent.forward (model 4 < 5): a plain bounced move.
    if current_block == 32:
        heading = 0.0
        nx, ny = point_at(xj, yj, heading, 1.0, w, h)
        return 2, nx, ny, heading, 0

    # 221-225: distance to the target block centre; face it
    the_dx = tx - xj
    the_dy = ty - yj
    the_dist = java_round(math.sqrt(the_dx * the_dx + the_dy * the_dy))
    heading = _face(xj, yj, xj + the_dx / 2, yj + the_dy / 2, heading)

    m = mean_disp_dist
    # 229-235: depth just ahead (padding entries are -999)
    bathymetry_ahead = np.full(9, -999.0)
    for i in range(2, 7):
        bathymetry_ahead[i] = _at(depth, xj, yj, heading + (i - 4) * 10.0, m)
    # 237-245: depth far ahead
    bathymetry_far_ahead = np.full(9, -999.0)
    for i in range(1, 8):
        bathymetry_far_ahead[i] = _at(depth, xj, yj, heading + (i - 4) * 10.0, m * 8)

    # 250-253
    good_heading = np.zeros(9, dtype=np.bool_)
    for i in range(9):
        good_heading[i] = bathymetry_far_ahead[i] > 0

    # 255-270: turn towards the deepest water (last match wins)
    bathymetry_choice = -999.0
    sel_angle = 0.0
    for i in range(9):
        bathymetry_choice = max(bathymetry_ahead[i], bathymetry_choice)
    for i in range(9):
        if bathymetry_ahead[i] == bathymetry_choice:
            sel_angle = (i - 4) * 10.0
    heading = _norm(heading + sel_angle)

    # 274-282: distance to coast ahead, from the new heading
    dist_to_coast_ahead = np.full(9, -999.0)
    for i in range(1, 8):
        dist_to_coast_ahead[i] = _at(dtc, xj, yj, heading + (i - 4) * 10.0, m * 2)
    # 286-290: only where there was water far ahead (read before the turn above)
    for i in range(9):
        if not good_heading[i]:
            dist_to_coast_ahead[i] = -999.0

    disttocoast_choice = -999.0
    # 294-298: index 1 is a padding entry, so this is always true (DEPONS quirk)
    low_water_ahead = not (
        bathymetry_ahead[1] > min_disp_depth
        and bathymetry_ahead[2] > min_disp_depth
        and bathymetry_ahead[3] > min_disp_depth
        and bathymetry_ahead[4] > min_disp_depth
        and bathymetry_ahead[5] > min_disp_depth
    )
    dtc_here = grid_at(dtc, xj, yj)
    if low_water_ahead or dtc_here < min_dist_to_land:  # 300-310
        for i in range(9):
            disttocoast_choice = max(disttocoast_choice, dist_to_coast_ahead[i])
        for i in range(9):
            if dist_to_coast_ahead[i] == disttocoast_choice:
                sel_angle = (i - 4) * 10.0
    heading = _norm(heading + sel_angle)  # 312

    # --- when to stop dispersing ---
    phase = 1
    if math.sqrt((xj - d1x) ** 2 + (yj - d1y) ** 2) < 2:  # 317
        phase = 2
    if math.sqrt((xj - d8x) ** 2 + (yj - d8y) ** 2) < 6:  # 325
        phase = 2
    if dtc_here < min_dist_to_land:  # 333
        phase = 2
    if the_dist < 50:  # 341
        phase = 2
    if not (  # 348-351
        _at(depth, xj, yj, heading, m * 4) > 0
        and _at(depth, xj, yj, heading, m * 3) > 0
        and _at(depth, xj, yj, heading, m * 2) > 0
    ):
        phase = 2
    if not enough_water:  # 358
        phase = 0

    paid = 0
    if phase == 1:  # 366-376
        nx, ny = point_at(xj, yj, heading, m / 0.4, w, h)
        if not (grid_at(depth, nx, ny) >= 0):
            nx, ny = point_at(nx, ny, heading, -m / 0.4, w, h)
            phase = 0
        xj = nx
        yj = ny
        paid = 1
    return phase, xj, yj, heading, paid


@njit
def _disp2(xj, yj, heading, d1x, d1y, enough_water, depth, dtc, mean_disp_dist, min_depth):
    """disp2 (384-493). Returns (phase, xj, yj, heading, paid_moves)."""
    w = depth.shape[1]
    h = depth.shape[0]
    m = mean_disp_dist

    # 389-391: turn away from the place visited one day ago
    heading = _face(xj, yj, d1x, d1y, heading)
    heading = _norm(heading + 180.0)

    # 397-414
    disttocoast_ahead = np.empty(17)
    for i in range(17):
        disttocoast_ahead[i] = _at(dtc, xj, yj, heading + (i - 8) * 10.0, m)

    # 419-445
    dtc_diff = 9999999.0
    dtc_max = -9999.0
    dtc_min = 9999.0
    dtc_nbr_alike = 9
    dtc_nbr_max = 9
    dtc_nbr_min = 9
    dtc_here = grid_at(dtc, xj, yj)
    for i in range(17):
        tmp = abs(disttocoast_ahead[i] - dtc_here)
        if tmp <= dtc_diff:
            dtc_diff = tmp
            dtc_nbr_alike = i
        if tmp <= dtc_min:
            dtc_min = tmp
            dtc_nbr_min = i
        if tmp > dtc_max:
            dtc_max = tmp
            dtc_nbr_max = i

    # 447-464
    sel_angle = -9999.0
    if dtc_here > 4000:
        sel_angle = (dtc_nbr_min - 8) * 10.0
    if dtc_here < 1000:
        sel_angle = (dtc_nbr_max - 8) * 10.0
    if dtc_here <= 4000 and dtc_here >= 2000:
        sel_angle = (dtc_nbr_alike - 8) * 10.0
    heading = _norm(heading + sel_angle)

    # --- when to stop dispersing ---
    phase = 2
    if not enough_water:  # 469
        phase = 0
    if not (_at(depth, xj, yj, heading, m / 0.4) > min_depth):  # 477-479
        phase = 0

    paid = 0
    if phase == 2:  # 486-492
        xj, yj = point_at(xj, yj, heading, m / 0.4, w, h)
        paid = 1
    return phase, xj, yj, heading, paid


@njit
def disperse(phase, xj, yj, heading, tx, ty, d1x, d1y, d8x, d8y, enough_water, depth, dtc, blocks, mean_disp_dist, min_disp_depth, min_depth,):  # fmt: skip
    """disperse() (148-161) for one animal in one tick.

    ``d1*`` / ``d8*`` are getPosListDaily().get(1) and .get(8). Returns
    (phase, xj, yj, heading, paid_moves); paid_moves counts the moves that charge the
    dispersal step's energy cost. Phase 0 means dispersal stopped; the heading is not
    restored.
    """
    paid = 0
    if phase == 1:
        phase, xj, yj, heading, paid = _disp1(
            xj, yj, heading, tx, ty, d1x, d1y, d8x, d8y, enough_water,
            depth, dtc, blocks, mean_disp_dist, min_disp_depth,
        )  # fmt: skip
    # Must be if, not elif: disp1 can change the phase (Java 157)
    if phase == 2:
        phase, xj, yj, heading, paid2 = _disp2(
            xj, yj, heading, d1x, d1y, enough_water, depth, dtc, mean_disp_dist, min_depth
        )
        paid += paid2
    return phase, xj, yj, heading, paid


@njit
def disperse_many(idx, phase, x, y, heading, tx, ty, d1x, d1y, d8x, d8y, enough_water, depth, dtc, blocks, mean_disp_dist, min_disp_depth, min_depth,):  # fmt: skip
    """disperse() for the animals ``idx``, read from CENOP's per-animal arrays.

    Positions go in and come out in CENOP coordinates (Java + 0.5); nothing is written to the
    inputs. Returns (phase, x, y, heading, paid_moves), one entry per element of ``idx``.
    One call replaces a Python loop over disperse() and gives the same numbers.
    """
    n = idx.shape[0]
    out_phase = np.empty(n, dtype=np.int64)
    out_x = np.empty(n, dtype=np.float64)
    out_y = np.empty(n, dtype=np.float64)
    out_heading = np.empty(n, dtype=np.float64)
    out_paid = np.empty(n, dtype=np.int64)
    for k in range(n):
        i = idx[k]
        ph, xj, yj, hd, paid = disperse(
            int(phase[i]),
            float(x[i]) - 0.5,
            float(y[i]) - 0.5,
            float(heading[i]),
            float(tx[i]) - 0.5,
            float(ty[i]) - 0.5,
            float(d1x[i]) - 0.5,
            float(d1y[i]) - 0.5,
            float(d8x[i]) - 0.5,
            float(d8y[i]) - 0.5,
            bool(enough_water[i]),
            depth,
            dtc,
            blocks,
            mean_disp_dist,
            min_disp_depth,
            min_depth,
        )
        out_phase[k] = ph
        out_x[k] = xj + 0.5
        out_y[k] = yj + 0.5
        out_heading[k] = hd
        out_paid[k] = paid
    return out_phase, out_x, out_y, out_heading, out_paid
