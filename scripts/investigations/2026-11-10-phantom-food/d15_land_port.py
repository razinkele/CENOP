"""D15 decisive test: DEPONS 3.2's land-avoidance sequence ported into CENOP, in memory only.

DEPONS Porpoise.stdMove (320-666), for animals not dispersing:
1. after the CRW turn and BEFORE heading composition: up to 100 rounds of
   checkDepth (795-833): the point presMov cells ahead must have depth >= 0, and every point at
   0.1-cell spacing out to presMov cells (4x the step) must have depth > 0; if not, avoidLand
   (918-985): one U(0,10) jitter r per call; try +-(40+r), then +-(70+r), then +-(120+r) at
   presMov cells ahead, a point being good if depth >= wmin (1 m); turn toward the good side
   (the deeper one if both, avoidLandTurn 885-904); if none, beh 4: walk back along the position
   list, facing and stepping onto earlier positions until there is enough water ahead (max 20).
2. checkDepth once more; if still not enough water, beh 5: face the deepest Moore neighbour and
   use presMov = 1 (which also becomes prevLogMov, 637).
3. heading composition (vt, deterrence) -- unchanged here (CENOP's heading kernel).
4. forward() has no land check, only border reflection.
5. beh 6 (639-661): if the new cell has depth <= 0, face back and return to the previous position.

In CENOP: (1)-(2) run inside a wrapper on `_update_reference_memory` (called right after the CRW
turn in `_update_movement`); (4) replaces `_handle_land_avoidance` with border reflection only;
(5) is CENOP's existing `_apply_positions` rollback plus the heading flip. Cells use CENOP's lookup
(floor; out of grid = -9999, as DEPONS getDepth). Dispersers are untouched (their look-ahead stop
keeps them in water). The beh-4 walk uses CENOP's reference-memory position ring buffer, whose
newest entry is DEPONS's posList.get(1); this is an approximation of a rare branch.

HISTORICAL: this prototype applies only to code before the D15 plan (commit 46e2a39). The port is now
in the source (commits 75df4b9 ... 480c7b7); applying this on top would double-patch.

Usage: import d15_land_port; d15_land_port.apply(P) where P is cenop.agents.population.
"""

import numpy as np

WMIN = 1.0
STATS = {"checked": 0, "avoid": 0, "beh4": 0, "beh5": 0, "beh6": 0}


def _depth(land, x, y):
    h, w = land._depth.shape
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    ok = (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
    out = np.full(np.shape(x), -9999.0)
    out[ok] = land._depth[yi[ok], xi[ok]]
    return np.nan_to_num(out, nan=-9999.0)


def _ahead(x, y, h_deg, dist):
    r = np.radians(h_deg)
    return x + np.sin(r) * dist, y + np.cos(r) * dist


def _enough_water(land, x, y, h_deg, pres_mov):
    """checkDepth for a vector of agents. Returns bool array."""
    px, py = _ahead(x, y, h_deg, pres_mov)
    ok = _depth(land, px, py) >= 0.0
    n_pts = np.ceil(pres_mov / 0.1).astype(np.int64)
    kmax = int(n_pts.max()) if len(n_pts) else 0
    for k in range(1, kmax + 1):
        live = ok & (k <= n_pts)
        if not live.any():
            break
        qx, qy = _ahead(x[live], y[live], h_deg[live], 0.1 * k)
        bad = ~(_depth(land, qx, qy) > 0.0)
        idx = np.flatnonzero(live)
        ok[idx[bad]] = False
    return ok


def _land_loop(self, sel):
    land = self.landscape
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        return
    STATS["checked"] += len(idx)
    x = self.x[idx].astype(np.float64)
    y = self.y[idx].astype(np.float64)
    h = self.heading[idx].astype(np.float64)
    pm = 10.0 ** self._log_mov[idx].astype(np.float64)
    todo = np.ones(len(idx), dtype=bool)
    for _ in range(100):
        cur = np.flatnonzero(todo)
        if len(cur) == 0:
            break
        enough = _enough_water(land, x[cur], y[cur], h[cur], pm[cur])
        todo[cur[enough]] = False
        need = cur[~enough]
        if len(need) == 0:
            break
        STATS["avoid"] += len(need)
        r = self.rng.uniform(0.0, 10.0, len(need))
        turn = np.zeros(len(need))
        done = np.zeros(len(need), dtype=bool)
        for base in (40.0, 70.0, 120.0):
            a = base + r
            rx, ry = _ahead(x[need], y[need], h[need] + a, pm[need])
            lx, ly = _ahead(x[need], y[need], h[need] - a, pm[need])
            dr, dl = _depth(land, rx, ry), _depth(land, lx, ly)
            gr, gl = dr >= WMIN, dl >= WMIN
            hit = ~done & (gr | gl)
            right = (gr & gl & (dr >= dl)) | (gr & ~gl)
            turn[hit & right] = a[hit & right]
            turn[hit & ~right] = -a[hit & ~right]
            done |= hit
        h[need] += turn
        # beh 4: walk back along remembered positions
        for j in np.flatnonzero(~done):
            k = need[j]
            i = idx[k]
            STATS["beh4"] += 1
            size = self._stored_util.shape[1]
            for back in range(min(20, int(self._mem_count[i]))):
                b = (int(self._mem_ptr[i]) - 1 - back) % size
                bx, by = float(self._pos_history_x[i, b]), float(self._pos_history_y[i, b])
                h[k] = np.degrees(np.arctan2(bx - x[k], by - y[k])) % 360.0
                x[k], y[k] = bx, by
                if _enough_water(land, x[k : k + 1], y[k : k + 1], h[k : k + 1], pm[k : k + 1])[0]:
                    break
            todo[k] = False
    final = _enough_water(land, x, y, h, pm)
    for k in np.flatnonzero(~final):  # beh 5: face the deepest neighbour, presMov = 1
        STATS["beh5"] += 1
        cx, cy = np.floor(x[k]), np.floor(y[k])
        nx = cx + np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.float64)
        ny = cy + np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.float64)
        d = _depth(land, nx + 0.5, ny + 0.5)
        m = int(np.argmax(d))
        h[k] = np.degrees(np.arctan2(nx[m] + 0.5 - x[k], ny[m] + 0.5 - y[k])) % 360.0
        pm[k] = 1.0
    self.heading[idx] = (h % 360.0).astype(self.heading.dtype)
    self.x[idx] = x.astype(self.x.dtype)
    self.y[idx] = y.astype(self.y.dtype)
    self._log_mov[idx] = np.log10(pm)


def apply(P):
    orig_refmem = P.PorpoisePopulation._update_reference_memory
    orig_apply = P.PorpoisePopulation._apply_positions

    def refmem(self, mask):
        if self.landscape is not None and getattr(self.landscape, "_depth", None) is not None:
            _land_loop(self, mask & ~self.is_dispersing)
        return orig_refmem(self, mask)

    def land_avoid_reflect_only(self, mask):
        np.add(self.x, self._dx, out=self._new_x)
        np.add(self.y, self._dy, out=self._new_y)
        w = self.landscape.width if self.landscape else self.params.world_width
        h = self.landscape.height if self.landscape else self.params.world_height
        np.copyto(self._orig_dx, self._dx)
        np.copyto(self._orig_dy, self._dy)
        self._reflect_boundaries(self._new_x, self._new_y, self._dx, self._dy, w, h, mask)
        refl = mask & ((self._dx != self._orig_dx) | (self._dy != self._orig_dy))
        if refl.any():
            self.heading[refl] = np.degrees(np.arctan2(self._dx[refl], self._dy[refl])) % 360.0

    def apply_positions(self, mask):
        tried_x, tried_y = self._new_x.copy(), self._new_y.copy()
        orig_apply(self, mask)
        rolled = (
            mask
            & (self.x == self._pre_move_x)
            & (self.y == self._pre_move_y)
            & ((tried_x != self._pre_move_x) | (tried_y != self._pre_move_y))
        )
        if rolled.any():  # beh 6: face back toward the previous position
            STATS["beh6"] += int(rolled.sum())
            self.heading[rolled] = (
                np.degrees(
                    np.arctan2(
                        self._pre_move_x[rolled] - tried_x[rolled],
                        self._pre_move_y[rolled] - tried_y[rolled],
                    )
                )
                % 360.0
            )

    P.PorpoisePopulation._update_reference_memory = refmem
    P.PorpoisePopulation._handle_land_avoidance = land_avoid_reflect_only
    P.PorpoisePopulation._apply_positions = apply_positions
