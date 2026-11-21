"""Learned avoidance (JASMINE disturbance memory) on Kattegat with ships, memory on or off.

JASMINE mode with its defaults (DEPONS movement and energy), 500 animals, seed 7, 60 days.
Prints survival, energy, the share of animal-ticks deterred and, with the memory on, the
share of animals avoiding, the mean avoidance pull, and how far the remembered disturbance
cell lies from the nearest deterring ship compared with the animal itself.

python3 scripts/investigations/2026-11-21-learned-avoidance/avoid_study.py JASMINE|DEPONS

2026-11-21 (CENOP feat/jasmine-depons-defaults): memory off 499 alive, energy 10.50, deterred
0.152; memory on 438 alive, energy 7.47, deterred 0.116, avoiding 0.76, pull 0.026 (ship
deterrence 0.062), remembered cell 5.5 km from the ship against 5.1 km for the animal.
"""

import sys

import numpy as np

from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters
from cenop.server.simulation_controller import build_simulation

mem = sys.argv[1]
cd = CellData("Kattegat", "data")
cd.load()
params = SimulationParameters(
    porpoise_count=500,
    sim_years=1,
    random_seed=7,
    landscape="Kattegat",
    ships_enabled=True,
    turbines="off",
    simulation_mode="JASMINE",
    memory_mode=mem,
)
sim = build_simulation(params, cd)
sim.initialize()
pm = sim.population_manager
sm = sim._ship_manager
avoid_share, avoid_str, det_mag, deterred, src_err = [], [], [], [], []
for t in range(60 * 48):
    sim.step()
    a = pm.active_mask
    d = a & (pm.deter_strength > 0.01)
    deterred.append(d.sum() / a.sum())
    if d.any():
        det_mag.append(pm.deter_strength[d].mean())
        if t % 12 == 0:
            ships = [s for s in sm.ships if s.is_deterring]
            if ships:
                sx = np.array([s.x for s in ships])
                sy = np.array([s.y for s in ships])
                idx = np.flatnonzero(d)[:50]
                srcx = pm.x[idx] - pm._dx[idx] * 5.0
                srcy = pm.y[idx] - pm._dy[idx] * 5.0
                near = np.hypot(
                    sx[None, :] - pm.x[idx, None], sy[None, :] - pm.y[idx, None]
                ).argmin(axis=1)
                anim_to_ship = np.hypot(sx[near] - pm.x[idx], sy[near] - pm.y[idx])
                src_to_ship = np.hypot(sx[near] - srcx, sy[near] - srcy)
                src_err.append(np.c_[anim_to_ship, src_to_ship])
    av = pm._avoidance_result
    if av is not None:
        s = av.avoidance_strength[a]
        avoid_share.append((s > 0).mean())
        if (s > 0).any():
            avoid_str.append(s[s > 0].mean())
e = np.concatenate(src_err) if src_err else np.zeros((1, 2))
print(
    f"memory_mode={mem}: alive d60 {int(pm.active_mask.sum())}, mean E {pm.energy[pm.active_mask].mean():.2f}, deterred {np.mean(deterred):.3f} of animal-ticks, ship deterrence when deterred {np.mean(det_mag):.3f}"
)
if mem == "JASMINE":
    print(
        f"  avoiding {np.mean(avoid_share):.2f} of animals; avoidance pull when avoiding {np.mean(avoid_str):.3f} (cap 0.8); remembered source vs nearest deterring ship: animal-ship median {np.median(e[:,0])*0.4:.1f} km, remembered-cell-ship median {np.median(e[:,1])*0.4:.1f} km"
    )
