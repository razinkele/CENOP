"""CENOP on the DEPONS v3.2 Kattegat landscape: log every death (tick, age, energy) and births.

Like-for-like with the DEPONS-Java runs under output/depons_java/kattegat-*-undisturbed-v1
(Reproduction.csv has one row per death with TickAtDeath and AgeAtDeath).

python3 scripts/investigations/2026-11-18-population-gap/cenop_deaths.py <mode> <seed> <days> \
    <data_root> [java-init]

``java-init`` replaces CENOP's starting sex and pregnancy state with DEPONS's (see below).
Writes output/population_gap/<mode>-seed-<seed>-deaths.csv (tick, age, last_energy) and
...-daily.csv (day, n, mean_energy, births_today, energy quantiles 5/25/50/75/95).
"""

import sys
from pathlib import Path

import numpy as np

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

ROOT = Path(__file__).resolve().parents[3]
mode, seed, days, data_root = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
params = SimulationParameters(
    porpoise_count=2000,
    sim_years=max(1, -(-days // 360)),
    random_seed=seed,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
    dispersal=mode,
)
cd = CellData("Kattegat", data_root)
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=seed)
sim.initialize()
pm = sim.population_manager
variant = sys.argv[5] if len(sys.argv) > 5 else ""
if variant == "java-init":
    # DEPONS start (Porpoise.java:165-178): every agent is female; every agent with age > 0
    # is pregnant with probability h, whatever its age, with daysSinceMating =
    # 360 - round(N(225, 20)); the rest have status 0. A separate generator is used so the
    # simulation's own random stream is not shifted.
    g = np.random.default_rng(seed + 1000)
    n0 = pm.count
    pm.is_female[:] = True
    pm.mating_day[:] = np.clip(g.normal(225, 20, n0), 0, 359).astype(np.int16)
    pm.pregnancy_status[:] = 0
    pm.days_since_mating[:] = -99
    preg = pm.active_mask & (pm.age > 0) & (g.random(n0) < params.conceive_prob)
    pm.pregnancy_status[preg] = 1
    dsm = 360 - np.round(g.normal(225, 20, int(preg.sum())))
    pm.days_since_mating[preg] = np.clip(dsm, 0, 360).astype(np.int16)
    mode_tag = mode + "-javainit"
else:
    mode_tag = mode
deaths, daily = [], []
births_today = 0
prev_active = pm.active_mask.copy()
prev_age = pm.age.copy()
prev_energy = pm.energy.copy()
for t in range(1, days * 48 + 1):
    sim.step()
    a = pm.active_mask
    n = min(len(a), len(prev_active))
    died = prev_active[:n] & ~a[:n]
    born = a.copy()
    born[:n] &= ~prev_active[:n]
    # a slot that died and was refilled in the same tick shows as neither; count via age reset
    reused = prev_active[:n] & a[:n] & (pm.age[:n] < prev_age[:n] - 0.5)
    for i in np.flatnonzero(died | reused):
        deaths.append((t, float(prev_age[i]), float(prev_energy[i])))
    births_today += int(born.sum()) + int(reused.sum())
    prev_active = a.copy()
    prev_age = pm.age.copy()
    prev_energy = pm.energy.copy()
    if t % 48 == 0:
        e = pm.energy[a]
        q = np.percentile(e, [5, 25, 50, 75, 95]) if e.size else np.zeros(5)
        daily.append([t // 48, int(a.sum()), float(e.mean()) if e.size else 0.0, births_today, *q])
        births_today = 0
out = ROOT / "output" / "population_gap"
out.mkdir(parents=True, exist_ok=True)
np.savetxt(out / f"{mode_tag}-seed-{seed}-deaths.csv", np.array(deaths), delimiter=",",
           header="tick,age,last_energy", comments="")  # fmt: skip
np.savetxt(out / f"{mode_tag}-seed-{seed}-daily.csv", np.array(daily), delimiter=",",
           header="day,n,mean_energy,births,q5,q25,q50,q75,q95", comments="")  # fmt: skip
print(mode_tag, seed, "deaths", len(deaths), "final n", daily[-1][1])
