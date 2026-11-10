"""Finding 3, like-for-like measurement (CENOP side). Same definitions as depons_disp_rates.py,
over days [0, max_day), all animals:

- predicate rate: animal-days whose 3-dp-rounded daily mean energy fell strictly three times
  running, regardless of dispersal state (DEPONS condition, independent of CENOP's t_disp);
- activation rate: visible episode starts (>= 1 dispersal move) per eligible predicate-day,
  split into raw trigger activations and same-tick kills (energy stop / water stop);
- tick share dispersing, averaged over every tick;
- episode length in dispersal moves, and the end reason as attributed in code.

Modes (combinable with '+'), all applied in memory, no source edits:
  stock     CENOP as is (communication off)
  tdisp4    t_disp=4, i.e. DEPONS's three drops (D1)
  water     DEPONS per-step stops on the dispersal heading: point 5 cells ahead off-grid or
            depth <= wdisp (D9; the not-enough-water state is not modelled)
  target    DEPONS target selection (D3-D5): no 50-cell trigger gate; memory target only with
            >= 50 visited cells, plain food/ticks fitness over [pref +- tol]/0.4; otherwise up to
            1000 random tries needing an in-grid cell with depth > 0 and prey > 0 this month;
            failure = no dispersal
  heading   start from the target heading: previousStepHeading = targetHeading (D6)
  angle40   +-40 deg PSM turn instead of the 20 deg fallback (D7)
  pathlen   path length drives both the logistic and completion (D8, D10)
  nodisp    no dispersal at all (controlled baseline for the on/off ratio)

Usage: cenop_disp_rates.py <data_dir with Kattegat/> <mode> [max_day] [rate_days] [seed]
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
import cenop.agents.population as P  # noqa: E402
from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

data_dir, mode = sys.argv[1], sys.argv[2]
max_day = int(sys.argv[3]) if len(sys.argv) > 3 else 120
rate_days = int(sys.argv[4]) if len(sys.argv) > 4 else max_day  # rates over [0, rate_days)
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 42
WDISP = 4.0
REASONS = {1: "water/off-grid ahead", 2: "day-boundary (energy stop)", 3: "complete/deterrence"}
S = Counter()

_orig_trig = P.PorpoisePopulation._check_dispersal_trigger
_orig_upd = P.PorpoisePopulation._update_dispersal


def _depons_activate(self, i):
    """AbstractPSMDispersal.activate + findMostAttractiveMemCell (PSM-Type2). Returns success."""
    tol = self.params.psm_tol
    pref = self._psm_instances[i].preferred_distance
    sx, sy = float(self.x[i]), float(self.y[i])
    ticks = self.psm_buffer[i, :, :, 0]
    vy, vx = np.nonzero(ticks)
    tgt = None
    if len(vx) >= 50:
        cs = self.psm_cell_size
        cx = vx * cs + cs // 2
        cy = vy * cs + cs // 2
        d = np.hypot(cx - sx, cy - sy)
        cand = (d >= (pref - tol) / 0.4) & (d <= (pref + tol) / 0.4)
        if cand.any():
            fit = self.psm_buffer[i, vy, vx, 1] / ticks[vy, vx]
            k = np.flatnonzero(cand)[np.argmax(fit[cand])]
            tgt = (float(cx[k]), float(cy[k]))
    if tgt is None:
        # 1000 tries drawn at once; the first valid one wins (same semantics as the Java loop).
        w, h = self.landscape.width, self.landscape.height
        ent = self.landscape._entropy[self._get_current_month() - 1]
        hd = np.radians(self.rng.random(1000) * 360.0)
        dist = (pref + self.rng.random(1000) * 2 * tol - tol) / 0.4
        tx = sx + dist * np.sin(hd)
        ty = sy + dist * np.cos(hd)
        xi = np.rint(tx).astype(np.int64)
        yi = np.rint(ty).astype(np.int64)
        ok = (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
        xc, yc = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        ok &= (self.landscape._depth[yc, xc] > 0) & (ent[yc, xc] > 0)
        if ok.any():
            k = int(np.argmax(ok))
            tgt = (float(tx[k]), float(ty[k]))
    if tgt is None:
        S["target_fail"] += 1
        return False
    th = np.degrees(np.arctan2(tgt[0] - sx, tgt[1] - sy)) % 360.0
    self.is_dispersing[i] = True
    self.dispersal_start_x[i], self.dispersal_start_y[i] = sx, sy
    self.dispersal_target_x[i], self.dispersal_target_y[i] = tgt
    self.dispersal_target_distance[i] = float(np.hypot(tgt[0] - sx, tgt[1] - sy))
    self.dispersal_distance_traveled[i] = 0.0
    self._prev_step_heading[i] = th if "heading" in mode else self.heading[i]
    self.heading[i] = th
    return True


def _trig(self, mask):
    before = self.is_dispersing.copy()
    if self._global_tick // 48 >= 4:
        r = np.round(self._energy_history[:, :4].astype(np.float64), 3)
        pred = mask & (r[:, 0] < r[:, 1]) & (r[:, 1] < r[:, 2]) & (r[:, 2] < r[:, 3])
        S["pred"] += int(pred.sum())
        S["elig"] += int((pred & ~before).sum())
    if "nodisp" in mode:
        pass
    elif "target" in mode:
        t_disp = max(2, min(self.params.t_disp, self._energy_history.shape[1]))
        dec = np.ones(self.count, dtype=bool)
        for k in range(t_disp - 1):
            dec &= self._energy_history[:, k] < self._energy_history[:, k + 1]
        for i in np.flatnonzero(mask & dec & ~self.is_dispersing):
            _depons_activate(self, i)
    else:
        _orig_trig(self, mask)
        if "heading" in mode:
            new = self.is_dispersing & ~before
            dx = self.dispersal_target_x[new] - self.x[new]
            dy = self.dispersal_target_y[new] - self.y[new]
            self._prev_step_heading[new] = np.degrees(np.arctan2(dx, dy)) % 360.0
    self._new_act = self.is_dispersing & ~before
    S["raw_act"] += int(self._new_act.sum())


_orig_head = P.PorpoisePopulation._apply_dispersal_heading


def _head_pathlen(self, mask):
    """DispersalPSMType2.calculateNewHeading with distPerc = path / (0.95 x start-to-target)."""
    disp = mask & self.is_dispersing
    if not disp.any():
        return
    ang = getattr(self.params, "psm_type2_random_angle", 20.0)
    delta = self.rng.uniform(-ang, ang, int(disp.sum()))
    tgt = 0.95 * self.dispersal_target_distance[disp]
    perc = np.where(tgt > 0, self.dispersal_distance_traveled[disp] / np.maximum(tgt, 1e-9), 0.0)
    logistic = 1.0 / (1.0 + np.exp(-(3 * perc - 1.5) / self.params.psm_log))
    self.heading[disp] = (self._prev_step_heading[disp] + delta * logistic) % 360.0
    self._prev_step_heading[disp] = self.heading[disp]


def _upd(self, mask):
    before = self.is_dispersing & mask
    water = np.zeros_like(before)
    if "water" in mode and before.any() and self.landscape is not None:
        idx = np.flatnonzero(before)
        step = self.params.mean_disp_dist / 0.4
        rad = np.radians(self._prev_step_heading[idx])
        px = self.x[idx] + np.sin(rad) * step
        py = self.y[idx] + np.cos(rad) * step
        w, h = self.landscape.width, self.landscape.height
        outside = (px < 0) | (py < 0) | (px >= w - 0.5) | (py >= h - 0.5)
        xi = np.clip(px.astype(np.int32), 0, w - 1)
        yi = np.clip(py.astype(np.int32), 0, h - 1)
        stop = idx[outside | (self.landscape._depth[yi, xi] <= WDISP)]
        water[stop] = True
        self.is_dispersing[stop] = False
        self.dispersal_distance_traveled[stop] = 0.0
        self.days_declining_energy[stop] = 0
    if "pathlen" in mode:
        done = (
            mask
            & self.is_dispersing
            & (self.dispersal_distance_traveled >= 0.95 * self.dispersal_target_distance)
        )
        self.is_dispersing[done] = False
        self.dispersal_distance_traveled[done] = 0.0
        self.days_declining_energy[done] = 0
    _orig_upd(self, mask)
    ended = before & ~self.is_dispersing
    boundary = self._tick_counter == 0 and self._global_tick > 0
    reason = np.where(water, 1, 2 if boundary else 3)
    self._end_reason[ended] = reason[ended]
    new = getattr(self, "_new_act", None)
    if new is not None and boundary:
        S["kill_water0"] += int((new & water).sum())
        S["kill_energy0"] += int((new & ended & ~water).sum())
        self._new_act = None


P.PorpoisePopulation._check_dispersal_trigger = _trig
P.PorpoisePopulation._update_dispersal = _upd
if "pathlen" in mode:
    P.PorpoisePopulation._apply_dispersal_heading = _head_pathlen

kw = {"t_disp": 4} if "tdisp4" in mode else {}
params = SimulationParameters(
    porpoise_count=2000,
    sim_years=1,
    random_seed=SEED,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
    communication_enabled=False,
    **kw,
)
if "angle40" in mode:
    params.psm_type2_random_angle = 40.0
cd = CellData("Kattegat", data_dir)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=SEED)
sim.initialize()
pm = sim.population_manager
N = pm.count
pm._end_reason = np.zeros(N, dtype=np.int8)
in_ep = np.zeros(N, dtype=bool)
moves = np.zeros(N, dtype=np.int64)
ep_id = np.zeros(N, dtype=np.int64)
lengths, reasons = [], Counter()
starts = disp_ticks = alive_ticks = 0
S_win = None
for tick in range(1, max_day * 48 + 1):
    window = tick <= rate_days * 48
    if window:
        a0 = pm.active_mask.copy()
        d0 = pm.is_dispersing & a0
        new = d0 & ~in_ep
        starts += int(new.sum())
        in_ep |= new
        moves[new] = 0
        ep_id[new] = pm.ids[new]
        moves[d0] += 1
        disp_ticks += int(d0.sum())
        alive_ticks += int(a0.sum())
    sim.step()
    a1 = pm.active_mask
    if window:
        gone = in_ep & (~a1 | (pm.ids != ep_id))  # death or slot recycled: censored
        ended = in_ep & ~gone & ~pm.is_dispersing
        for e in np.flatnonzero(ended):
            lengths.append(int(moves[e]))
            reasons[REASONS.get(int(pm._end_reason[e]), "?")] += 1
        in_ep[gone | ended] = False
    if tick == rate_days * 48:
        S_win = Counter(S)
    if tick % (30 * 48) == 0:
        print(
            f"[{mode}] day {tick // 48:3d}: N={int(a1.sum()):4d} E={pm.energy[a1].mean():5.2f}",
            flush=True,
        )

S = S_win if S_win is not None else S
ad = alive_ticks / 48.0
print(f"CENOP [{mode}] rates over days 0-{rate_days}: {ad:.0f} animal-days")
print(f"  predicate rate      {S['pred'] / ad:.4f} per animal-day")
print(f"  eligible pred-days  {S['elig'] / ad:.4f} per animal-day")
print(
    f"  raw activations     {S['raw_act'] / ad:.4f} per animal-day "
    f"(killed same tick: energy {S['kill_energy0']}, water {S['kill_water0']}; "
    f"target failures {S['target_fail']})"
)
print(f"  starts (visible)    {starts / ad:.4f} per animal-day")
print(f"  activation rate     {starts / max(S['elig'], 1):.3f} starts per eligible predicate-day")
print(f"  tick share disp.    {disp_ticks / alive_ticks:.4f}")
if lengths:
    L = np.array(lengths)
    print(
        f"  episode moves       median {np.median(L):.0f}, mean {L.mean():.1f}, "
        f"p90 {np.percentile(L, 90):.0f}, n={len(L)}"
    )
    for k, v in reasons.most_common():
        print(f"    end: {k:28s} {v / len(L):.0%}")
