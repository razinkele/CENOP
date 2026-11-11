"""Lithuania sea-vs-lagoon diagnostic (from the 2026-11-10/11 lagoon investigation).

Usage: python3 lith_diag.py TAG SEED DAYS N [norestrict]
`norestrict` turns off the Lithuania main-water-body placement restriction (da3c5de), so the
with/without comparison (CHANGELOG: N 9991 vs 9856 at day 60, seed 42, N=10000) can be re-run.
"""

import sys
import time

import numpy as np
from scipy import ndimage

import cenop
import cenop.agents.population as population_module
from cenop.core.simulation import Simulation
from cenop.parameters.simulation_params import SimulationParameters

tag = sys.argv[1]
seed = int(sys.argv[2])
days = int(sys.argv[3])
n = int(sys.argv[4])
if len(sys.argv) > 5 and sys.argv[5] == "norestrict":
    population_module.MAIN_WATER_BODY_LANDSCAPES = frozenset()
print("cenop from", cenop.__file__, flush=True)
params = SimulationParameters(random_seed=seed, porpoise_count=n)
sim = Simulation(params=params, seed=seed)
sim.initialize()
pm = sim.population_manager
cd = sim._cell_data
depth = cd._depth
H, W = depth.shape
lab, ncomp = ndimage.label(depth > 0)
sizes = np.bincount(lab.ravel())
sizes[0] = 0
sea = int(np.argmax(sizes))
sizes2 = sizes.copy()
sizes2[sea] = 0
lagoon = int(np.argmax(sizes2))
coast = cd._dist_to_coast
water = depth > 0
print("water cells with disttocoast<1:", float((coast[water] < 1.0).mean()), flush=True)


def comp_of(x, y):
    xi = np.clip(x.astype(int), 0, W - 1)
    yi = np.clip(y.astype(int), 0, H - 1)
    return lab[yi, xi], depth[yi, xi], coast[yi, xi]


a0 = pm.active_mask.copy()
c0, _, _ = comp_of(pm.x[a0], pm.y[a0])
print(
    "start: N",
    int(a0.sum()),
    "in lagoon",
    int((c0 == lagoon).sum()),
    "in sea",
    int((c0 == sea).sum()),
    "other",
    int(((c0 != lagoon) & (c0 != sea)).sum()),
    flush=True,
)

intake = []
stuck = crossings = ticks_agents = 0
deaths_lagoon = deaths_sea = deaths_other = 0
d_lt1 = d_1_5 = d_land = coast_lt1 = 0
t0 = time.time()
for t in range(days * 48):
    a_before = pm.active_mask.copy()
    xb, yb = pm.x.copy(), pm.y.copy()
    cb, _, _ = comp_of(xb, yb)
    sim.step()
    a = pm.active_mask
    both = a_before & a
    stuck += int(((pm.x == xb) & (pm.y == yb) & both).sum())
    ca, da, dc = comp_of(pm.x, pm.y)
    crossings += int(((ca != cb) & both).sum())
    ticks_agents += int(both.sum())
    died = a_before & ~a
    deaths_lagoon += int((died & (cb == lagoon)).sum())
    deaths_sea += int((died & (cb == sea)).sum())
    deaths_other += int((died & (cb != lagoon) & (cb != sea)).sum())
    d_lt1 += int((a & (da > 0) & (da < 1)).sum())
    d_1_5 += int((a & (da >= 1) & (da < 5)).sum())
    d_land += int((a & (da <= 0)).sum())
    coast_lt1 += int((a & (dc < 1.0)).sum())
    intake.append(pm.avg_food_gained)
    if (t + 1) % (5 * 48) == 0:
        aa = pm.active_mask
        cc, _, _ = comp_of(pm.x[aa], pm.y[aa])
        e = pm.energy[aa]
        el = e[cc == lagoon]
        es = e[cc == sea]
        print(
            f"[{tag}] day {(t + 1) // 48}: N={int(aa.sum())} meanE={e.mean():.3f} "
            f"lagoonN={el.size} lagoonE={el.mean() if el.size else float('nan'):.3f} "
            f"seaE={es.mean():.3f} intake/at(last5d)={np.mean(intake[-240:]):.5f} "
            f"wall={time.time() - t0:.0f}s",
            flush=True,
        )

ta = max(ticks_agents, 1)
print(
    f"[{tag}] SUMMARY seed {seed} {days}d n={n}: intake/at={np.mean(intake):.5f} "
    f"stuck_share={stuck / ta:.4f} crossings/at={crossings / ta:.6f} crossings={crossings} "
    f"deaths lagoon/sea/other={deaths_lagoon}/{deaths_sea}/{deaths_other} "
    f"share depth(0,1)={d_lt1 / ta:.4f} depth[1,5)={d_1_5 / ta:.4f} on_land={d_land / ta:.5f} "
    f"coast<1={coast_lt1 / ta:.4f}",
    flush=True,
)
