"""Which lever of learned avoidance drives its energy cost: 60-day variants on Kattegat.

JASMINE mode (DEPONS movement and energy), 500 animals, ships, seed 7. Variants: off (no
memory), on, own_cell (remember the animal's own cell), r5 / r2 (avoidance radius in 2 km
memory cells), scaled (pull times crwContrib; patched here, built in since 2026-11-21).

python3 scripts/investigations/2026-11-21-learned-avoidance/avoid_levers.py VARIANT

2026-11-21, before the fix (alive d60, mean energy, deterred animal-ticks): off 499 10.50 0.152;
on 438 7.47 0.116; own_cell 448 7.79 0.126; r5 465 9.36 0.138; r2 482 10.06 0.152;
scaled 497 10.58 0.146. With the fix built in, "on" gives 498 10.68 0.147.
"""

import sys
import types

import numpy as np

from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters
from cenop.server.simulation_controller import build_simulation

variant = sys.argv[1]
kw = {}
if variant == "off":
    kw["memory_mode"] = "DEPONS"
elif variant.startswith("r"):
    kw["avoidance_radius"] = float(variant[1:])
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
    **kw,
)
sim = build_simulation(params, cd)
sim.initialize()
pm = sim.population_manager
orig = pm._update_disturbance_memory

if variant == "own_cell":
    saved = {}

    def patched(self, mask):
        dx, dy = self._dx.copy(), self._dy.copy()
        self._dx[:] = 0.0
        self._dy[:] = 0.0
        try:
            orig(mask)
        finally:
            self._dx[:] = dx
            self._dy[:] = dy

    pm._update_disturbance_memory = types.MethodType(patched, pm)
elif variant == "scaled":

    def patched(self, mask):
        orig(mask)
        av = self._avoidance_result
        if av is not None:
            crw = self.params.inertia_const + np.power(10.0, self._log_mov) * self._ve_total
            av.avoidance_strength = (av.avoidance_strength * crw).astype(np.float32)

    pm._update_disturbance_memory = types.MethodType(patched, pm)

crw_vals, vt_vals, pull, avoiding, deterred = [], [], [], [], []
for t in range(60 * 48):
    sim.step()
    a = pm.active_mask
    deterred.append((pm.deter_strength[a] > 0.01).mean())
    if t % 6 == 0:
        crw_vals.append(
            np.median(params.inertia_const + np.power(10.0, pm._log_mov[a]) * pm._ve_total[a])
        )
        vt_vals.append(np.median(np.hypot(pm._vt_x[a], pm._vt_y[a])))
        av = pm._avoidance_result
        if av is not None:
            s = av.avoidance_strength[a]
            avoiding.append((s > 0).mean())
            if (s > 0).any():
                pull.append(np.median(s[s > 0]))
a = pm.active_mask
print(
    f"{variant:9s} alive d60 {int(a.sum())}, mean E {pm.energy[a].mean():.2f}, deterred {np.mean(deterred):.3f}, avoiding {np.mean(avoiding) if avoiding else 0:.2f}, median pull {np.median(pull) if pull else 0:.4f}, median crwContrib {np.median(crw_vals):.4f}, median |food-memory pull| {np.median(vt_vals):.4f}"
)
