"""Effect of the JASMINE social weight on movement and grouping (Kattegat, 2000 animals).

python3 scripts/investigations/2026-11-20-social-vector/curve.py <weight|off> [days]

Seed 42, InnerDanishWaters, no ships; measured over days 10 to 10 + days. Prints the median
net movement per day, the share of ticks with a heading reversal (> 150 degrees), the mean
distance to the nearest other animal and the share of animals with another within 1 km
(both at the end of each day, averaged over days).
"""

import sys

import numpy as np
from scipy.spatial import cKDTree

import cenop
from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

arg = sys.argv[1]
days = int(sys.argv[2]) if len(sys.argv) > 2 else 10
on = arg != "off"
cd = CellData("Kattegat", "data")
cd.load()
params = SimulationParameters(
    porpoise_count=2000, sim_years=1, random_seed=42, landscape="Kattegat",
    ships_enabled=False, turbines="off", dispersal="InnerDanishWaters",
    communication_enabled=on, social_weight=float(arg) if on else 0.3,
)  # fmt: skip
sim = Simulation(params=params, cell_data=cd, seed=42)
sim.initialize()
pm = sim.population_manager
for _ in range(10 * 48):
    sim.step()
n0 = pm.x.size
alive0 = pm.active_mask.copy()
x0, y0 = pm.x.copy(), pm.y.copy()
prev_h = pm.heading.copy()
reversals = ticks = 0
net, nn_mean, within_1km = [], [], []
for t in range(days * 48):
    sim.step()
    a = pm.active_mask[:n0] & alive0
    d = (pm.heading[:n0] - prev_h + 180.0) % 360.0 - 180.0
    reversals += int((np.abs(d[a]) > 150).sum())
    ticks += int(a.sum())
    prev_h = pm.heading[:n0].copy()
    if (t + 1) % 48 == 0:
        net.append(np.hypot(pm.x[:n0] - x0, pm.y[:n0] - y0)[a] * 0.4)
        x0, y0 = pm.x[:n0].copy(), pm.y[:n0].copy()
        live = pm.active_mask
        pts = np.column_stack((pm.x[live], pm.y[live])).astype(np.float64)
        dist, _ = cKDTree(pts).query(pts, k=2)
        nn = dist[:, 1] * 0.4
        nn_mean.append(nn.mean())
        within_1km.append((nn <= 1.0).mean())
label = "off" if not on else f"w={arg}"
tree = "before" if "scratchpad" in cenop.__file__ else "after"
print(
    f"{tree:6s} {label:7s} median net movement {np.median(np.concatenate(net)):5.2f} km/day | "
    f"reversals {reversals / ticks:.3f} | nearest neighbour {np.mean(nn_mean):5.2f} km | "
    f"with a neighbour within 1 km {np.mean(within_1km):.3f}"
)
