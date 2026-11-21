"""Recovery of porpoise numbers near a test wind farm after piling: DEPONS vs JASMINE memory.

python3 scripts/investigations/2026-11-21-learned-avoidance/recovery_study.py VARIANT SEED
VARIANT: none (no turbines, the control), depons, default (JASMINE mode as coded), old
(decay 0.001, radius 20, raw strengths), new<km> (capped, decay 0.016, radius <km>).
Then recovery_summary.py prints ratios to the control. Results of 2026-11-21 (seeds 1-3) are
in the changelog (Changed: learned avoidance calibrated).

Kattegat, 2000 animals. A test farm of up to 11 turbines in the densest 10 km area (cell 462,
462); after 30 days one pile every 2 days, 3 hours each. Counts animals within 0-5, 5-10 and
10-20 km of the farm, relative to days 20-30, by hours since the last pile ended and by days
after the last pile.
"""

import sys
from pathlib import Path

import numpy as np

import cenop.config as cc
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters
from cenop.server.simulation_controller import build_simulation

variant, seed = sys.argv[1], int(sys.argv[2])
SCR = Path("output/memory-recovery")  # counts and the test-farm file go here
SCR.mkdir(parents=True, exist_ok=True)
cd = CellData("Kattegat", "data")
cd.load()
m = cd.metadata

# Test farm in the densest area, turbines on water only
cx, cy = 462, 462
offsets = [
    (0, 0),
    (5, 0),
    (-5, 0),
    (0, 5),
    (0, -5),
    (5, 5),
    (-5, -5),
    (5, -5),
    (-5, 5),
    (10, 0),
    (-10, 0),
]
START, EVERY, DUR = 30 * 48, 96, 6
rows, piles, gx, gy = [], [], [], []
k = 0
for ox, oy in offsets:
    x, y = cx + ox, cy + oy
    if cd.get_depth(x, y) <= 0:
        continue
    s = START + k * EVERY
    rows.append(
        f"T{k}\t{m.xllcorner + x * 400:.0f}\t{m.yllcorner + y * 400:.0f}\t234\t{s}\t{s + DUR}"
    )
    piles.append((s, s + DUR))
    gx.append(x)
    gy.append(y)
    k += 1
farm_file = SCR / "Test-dense.txt"
farm_file.write_text("id\tx\ty\timpact\ttick.start\ttick.end\n" + "\n".join(rows) + "\n")
orig_get = cc.get_wind_farm_file
cc.get_wind_farm_file = lambda fn: farm_file if fn == "Test-dense.txt" else orig_get(fn)
last_end = piles[-1][1]
days = int(last_end / 48) + 31

kw = dict(simulation_mode="JASMINE")
cap = False
turb = "Test-dense"
if variant == "depons":
    kw = dict(simulation_mode="DEPONS")
elif variant == "none":
    kw = dict(simulation_mode="DEPONS")
    turb = "off"
elif variant == "default":
    pass  # JASMINE mode with the code's own memory defaults
elif variant == "old":
    kw.update(memory_decay_rate=0.001, avoidance_radius=20.0)
else:  # new<km>: capped at 1, fading below 0.1 within ~3 days, radius in km
    km = float(variant[3:])
    kw.update(memory_decay_rate=0.016, avoidance_radius=km / 2.0)
    cap = True
params = SimulationParameters(
    porpoise_count=2000,
    sim_years=1,
    random_seed=seed,
    landscape="Kattegat",
    ships_enabled=False,
    turbines=turb,
    **kw,
)
sim = build_simulation(params, cd)
sim.initialize()
pm = sim.population_manager
if cap and pm._memory_module is not None:
    mm = pm._memory_module
    orig_rec = mm.record_disturbance

    def rec(state, context, mask):
        context.disturbance_intensity = np.minimum(context.disturbance_intensity, 1.0)
        return orig_rec(state, context, mask)

    mm.record_disturbance = rec

fx, fy = float(np.mean(gx)) + 0.5, float(np.mean(gy)) + 0.5
np.save(SCR / f"rec_counts_{variant}_{seed}.npy", np.zeros(1))
edges = [0, 5, 10, 20]
counts = np.zeros((days * 48, 3))
for t in range(days * 48):
    sim.step()
    a = pm.active_mask
    d = np.hypot(pm.x[a] - fx, pm.y[a] - fy) * 0.4
    counts[t] = [((d >= lo) & (d < hi)).sum() for lo, hi in zip(edges[:-1], edges[1:])]

np.save(SCR / f"rec_counts_{variant}_{seed}.npy", counts)
tick = np.arange(1, days * 48 + 1)
base = counts[(tick >= 20 * 48) & (tick < 30 * 48)].mean(axis=0)
ends = np.array([e for _, e in piles])
since = np.full(tick.size, np.inf)
for i, t in enumerate(tick):
    past = ends[ends <= t]
    if past.size:
        since[i] = (t - past.max()) / 2.0  # hours
during = (tick > piles[0][0]) & (tick <= last_end)
out = [f"{variant:7s} seed {seed}: baseline {base.round(1).tolist()} (0-5, 5-10, 10-20 km)"]
for lo, hi in [(0, 3), (3, 12), (12, 24), (24, 48), (48, 96)]:
    sel = during & (since >= lo) & (since < hi)
    out.append(
        f"  construction, {lo}-{hi} h after a pile: "
        + " ".join(f"{v:.2f}" for v in counts[sel].mean(axis=0) / base)
    )
after = tick > last_end
for lo, hi in [(0, 1), (1, 3), (3, 7), (7, 14), (14, 30)]:
    sel = after & ((tick - last_end) / 48 >= lo) & ((tick - last_end) / 48 < hi)
    out.append(
        f"  after the last pile, day {lo}-{hi}: "
        + " ".join(f"{v:.2f}" for v in counts[sel].mean(axis=0) / base)
    )
a = pm.active_mask
out.append(f"  alive {int(a.sum())}, mean energy {pm.energy[a].mean():.2f}, turbines {len(piles)}")
print("\n".join(out))
