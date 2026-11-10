"""DEPONS 3.2 land-avoidance sequence as a per-agent kernel.

Port of Porpoise.stdMove's land loop (467-533), checkDepth (795-833), avoidLandTurn (885-904)
and avoidLand (918-985), run after the CRW turn and before heading composition:

1. Up to 100 rounds: the path check (checkDepth) looks presMov = 10**log_mov cells ahead (4x the
   step). The point presMov ahead must have depth >= 0, and every point at 0.1-cell spacing,
   k = 1 .. ceil(presMov / 0.1), depth > 0. If it fails, avoidLand draws one U(0, 10) jitter r
   and tries +-(40 + r), +-(70 + r), +-(120 + r) at presMov ahead; a point is good if its depth
   >= wmin. Turn toward the good side (the deeper one if both, right on ties; right = +).
   If no tier is good (beh 4), walk back through the newest <= 20 remembered positions, facing
   and stepping onto each, until the path check passes. 20 steps end the rounds; a shorter
   history that runs out without a pass does not.
2. A final path check; its result goes to `out_enough`. If it fails (beh 5), face the centre of
   the deepest of the 8 neighbouring cells and set log_mov = 0 (presMov = 1). Depth ties go to
   the first maximum in a fixed neighbour order; DEPONS shuffles getNeighbors (Agent.java 118),
   a documented deviation that saves an RNG draw.

Cells use CENOP's lookup: floor of the position, depth indexed [y, x]; NaN reads as -9999
(DEPONS getDepth). Look-ahead points (the ahead point, every path sample, every escape point)
are first reflected into [0, W) x [0, H), as DEPONS's BouncyBorders translate does on every
non-homogeneous landscape (PorpoiseSimBuilder 214-220), so an open-water grid edge does not read
as land. Only beh-5 neighbour cells outside the grid read as -9999.
Heading 0 is north (+y), 90 east (+x).

The kernel loops over agents sequentially: the jitter comes from np.random.uniform inside it, so
the caller seeds Numba's RNG (or NumPy's global RNG for the Python path) once per call.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - Numba is a hard dependency in practice
    HAS_NUMBA = False

_NO_DEPTH = -9999.0
_DEG = math.pi / 180.0
# Moore neighbours, in the order beh 5 scans them (first maximum wins)
_NB_DX = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
_NB_DY = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)


def _depth_at(depth, x, y):
    """Depth of the cell holding (x, y); -9999 off the grid or for NaN."""
    if not (x >= 0.0 and y >= 0.0 and x < depth.shape[1] and y < depth.shape[0]):
        return _NO_DEPTH
    v = float(depth[int(math.floor(y)), int(math.floor(x))])
    if v != v:
        return _NO_DEPTH
    return v


def _reflect(v, m):
    """Bounce a coordinate into [0, m) (Repast BouncyBorders): v < 0 -> -v, v > m -> 2m - v,
    v == m -> the largest float below m. One reflection suffices: look-ahead distances
    (presMov <= 10**1.73 ~ 54 cells) are far below the grid size. NaN passes through."""
    if v < 0.0:
        v = -v
    elif v > m:
        v = 2.0 * m - v
    if v == m:
        v = math.nextafter(m, 0.0)
    return v


def _look_depth(depth, x, y):
    """Depth at a look-ahead point, reflected into the grid first."""
    return _depth_at(depth, _reflect(x, float(depth.shape[1])), _reflect(y, float(depth.shape[0])))


def _path_ok(depth, x, y, h, pm):
    """DEPONS checkDepth: True if there is enough water presMov cells ahead along heading h."""
    rad = h * _DEG
    s = math.sin(rad)
    c = math.cos(rad)
    if _look_depth(depth, x + s * pm, y + c * pm) < 0.0:
        return False
    vx = 0.1 * s
    vy = 0.1 * c
    n = int(math.ceil(pm / 0.1))
    for k in range(1, n + 1):
        if not (_look_depth(depth, x + vx * k, y + vy * k) > 0.0):
            return False
    return True


def _py(
    x,
    y,
    heading,
    log_mov,
    mask,
    depth,
    wmin,
    pos_hist_x,
    pos_hist_y,
    mem_ptr,
    mem_count,
    out_enough,
):
    mem_size = pos_hist_x.shape[1]
    for i in range(x.shape[0]):
        if not mask[i]:
            continue
        px = float(x[i])
        py = float(y[i])
        h = float(heading[i])
        pm = 10.0 ** float(log_mov[i])
        moved = False
        turned = False

        for _round in range(100):
            if _path_ok(depth, px, py, h, pm):
                break
            # avoidLand: one jitter per call, shared by the three tiers
            r = np.random.uniform(0.0, 10.0)
            found = False
            turn = 0.0
            for tier in range(3):
                if tier == 0:
                    a = 40.0 + r
                elif tier == 1:
                    a = 70.0 + r
                else:
                    a = 120.0 + r
                rr = (h + a) * _DEG
                rl = (h - a) * _DEG
                dr = _look_depth(depth, px + math.sin(rr) * pm, py + math.cos(rr) * pm)
                dl = _look_depth(depth, px + math.sin(rl) * pm, py + math.cos(rl) * pm)
                good_r = dr >= wmin
                good_l = dl >= wmin
                if good_r or good_l:
                    if good_r and good_l:
                        turn = a if dr >= dl else -a
                    elif good_r:
                        turn = a
                    else:
                        turn = -a
                    found = True
                    break
            if found:
                h += turn
                turned = True
                continue

            # beh 4: walk back from the newest remembered position (DEPONS posList.get(0))
            n_back = min(20, int(mem_count[i]))
            ptr = int(mem_ptr[i])
            passed = False
            for back in range(n_back):
                b = (ptr - 1 - back) % mem_size
                bx = float(pos_hist_x[i, b])
                by = float(pos_hist_y[i, b])
                if bx != px or by != py:  # facing one's own position is a no-op
                    h = math.atan2(bx - px, by - py) / _DEG
                    px = bx
                    py = by
                    moved = True
                    turned = True
                if _path_ok(depth, px, py, h, pm):
                    passed = True
                    break
            if passed or n_back == 20:  # DEPONS: j == 20 forces enoughWaterAhead
                break

        enough = _path_ok(depth, px, py, h, pm)
        out_enough[i] = enough
        if not enough:
            # beh 5: face the centre of the deepest Moore neighbour, presMov = 1
            cx = int(math.floor(px))
            cy = int(math.floor(py))
            best = 0
            best_d = _depth_at(depth, cx + _NB_DX[0] + 0.5, cy + _NB_DY[0] + 0.5)
            for k in range(1, 8):
                d = _depth_at(depth, cx + _NB_DX[k] + 0.5, cy + _NB_DY[k] + 0.5)
                if d > best_d:
                    best_d = d
                    best = k
            tx = cx + _NB_DX[best] + 0.5
            ty = cy + _NB_DY[best] + 0.5
            h = math.atan2(tx - px, ty - py) / _DEG
            turned = True
            log_mov[i] = 0.0

        if turned:
            heading[i] = h % 360.0
            if heading[i] >= 360.0:  # float32 rounding of a value just below 360
                heading[i] = 0.0
        if moved:
            x[i] = px
            y[i] = py


if HAS_NUMBA:
    _depth_at = njit(cache=True)(_depth_at)
    _reflect = njit(cache=True)(_reflect)
    _look_depth = njit(cache=True)(_look_depth)
    _path_ok = njit(cache=True)(_path_ok)
    _impl = njit(cache=True)(_py)
else:  # pragma: no cover
    _impl = _py


def depons_land_check(
    x,
    y,
    heading,
    log_mov,
    mask,
    depth,
    wmin,
    pos_hist_x,
    pos_hist_y,
    mem_ptr,
    mem_count,
    out_enough,
):
    """Run the DEPONS land-avoidance sequence for `mask` agents, in place.

    Edits `x`, `y`, `heading` and `log_mov` of masked agents and writes the final path-check
    result to `out_enough`. `pos_hist_x/_y` (n, mem_size) are the reference-memory ring buffer;
    `mem_ptr` is its next write index and `mem_count` the number of stored entries. The caller
    seeds the RNG (Numba's, or NumPy's global one without Numba) before the call.
    """
    _impl(
        x,
        y,
        heading,
        log_mov,
        mask,
        depth,
        wmin,
        pos_hist_x,
        pos_hist_y,
        mem_ptr,
        mem_count,
        out_enough,
    )
