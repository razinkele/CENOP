"""Finding 3, like-for-like measurement (CENOP side). Same definitions as depons_disp_rates.py,
over days [0, max_day), all animals:

- predicate rate: animal-days whose 3-dp-rounded daily mean energy fell strictly three times
  running, regardless of dispersal state (DEPONS condition, independent of CENOP's t_disp);
- activation rate: visible episode starts (>= 1 dispersal move) per eligible predicate-day,
  split into raw trigger activations and same-tick kills (energy stop / water stop);
- tick share dispersing, averaged over every tick;
- episode length in dispersal moves, and the end reason as attributed in code.

Modes: only these two are valid on the fixed code (PSM-Type2 parity, D1-D13 in the source):
  stock     CENOP as is (communication off)
  nodisp    no dispersal at all (controlled baseline for the on/off ratio)
The earlier modes (tdisp4, water, target, heading, angle40, pathlen) re-applied fixes that are
now in the source; their code paths were removed. Starts, moves and tick share count animals that
make a dispersal move in a tick: PorpoisePopulation._disperse_step is wrapped and the mask of
animals still dispersing after it (completion and look-ahead stops applied) is recorded.

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
REASONS = {
    1: "off-grid/shallow ahead",
    2: "day-boundary (energy stop)",
    3: "path complete/deterrence",
}
S = Counter()

_orig_trig = P.PorpoisePopulation._check_dispersal_trigger
_orig_upd = P.PorpoisePopulation._update_dispersal


def _trig(self, mask):
    before = self.is_dispersing.copy()
    if self._global_tick // 48 >= 4:
        r = np.round(self._energy_history[:, :4].astype(np.float64), 3)
        pred = mask & (r[:, 0] < r[:, 1]) & (r[:, 1] < r[:, 2]) & (r[:, 2] < r[:, 3])
        S["pred"] += int(pred.sum())
        S["elig"] += int((pred & ~before).sum())
    if "nodisp" not in mode:
        _orig_trig(self, mask)
    self._new_act = self.is_dispersing & ~before
    S["raw_act"] += int(self._new_act.sum())


_orig_step = P.PorpoisePopulation._disperse_step


def _step(self, mask):
    """Record who makes a dispersal move this tick; attribute look-ahead / path-length stops."""
    before = mask & self.is_dispersing
    complete = before & (self.dispersal_distance_traveled >= 0.95 * self.dispersal_target_distance)
    _orig_step(self, mask)
    self._moving = mask & self.is_dispersing
    ended = before & ~self.is_dispersing
    self._end_reason[ended] = np.where(complete, 3, 1)[ended]


def _upd(self, mask):
    before = self.is_dispersing & mask
    _orig_upd(self, mask)
    ended = before & ~self.is_dispersing
    boundary = self._tick_counter == 0 and self._global_tick > 0
    self._end_reason[ended] = 2 if boundary else 3
    new = getattr(self, "_new_act", None)
    if new is not None and boundary:
        S["kill_energy0"] += int((new & ended).sum())
        self._new_act = None


P.PorpoisePopulation._check_dispersal_trigger = _trig
P.PorpoisePopulation._update_dispersal = _upd
P.PorpoisePopulation._disperse_step = _step
if "--d15-port" in sys.argv:  # D15 investigation: DEPONS land-avoidance port, in memory
    sys.argv.remove("--d15-port")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import d15_land_port  # noqa: E402

    d15_land_port.apply(P)

params = SimulationParameters(
    porpoise_count=2000,
    sim_years=1,
    random_seed=SEED,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
    communication_enabled=False,
)
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
    a0 = pm.active_mask.copy()
    pm._moving = np.zeros(N, dtype=bool)
    sim.step()
    a1 = pm.active_mask
    if window:
        d0 = pm._moving & a0  # animals that made a dispersal move this tick
        new = d0 & ~in_ep
        starts += int(new.sum())
        in_ep |= new
        moves[new] = 0
        ep_id[new] = pm.ids[new]
        moves[d0] += 1
        disp_ticks += int(d0.sum())
        alive_ticks += int(a0.sum())
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
    f"(killed same tick: energy {S['kill_energy0']})"
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
