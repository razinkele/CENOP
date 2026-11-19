"""InnerDanishWaters: dispersal statistics and energy, DEPONS-Java (200 tracked animals) vs CENOP.

Java: RandomPorpoise.csv of scenarios/idw_tracked200.json (one row per tracked animal per tick;
an animal's rows stop changing when it dies, which is detected as its last tick with a change
in position or energy). CENOP: all animals of one run, counted every tick.

python3 scripts/investigations/2026-11-18-dispersal-idw/tracked_compare.py <RandomPorpoise.csv> \
    [cenop_seed] [days]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

ROOT = Path(__file__).resolve().parents[3]
rp = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
DAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 180
BIN = 30


def report(label, ticks, p1, p2, starts, energy_sum, lengths):
    print(label)
    for b in range(DAYS // BIN):
        t = ticks[b]
        if t == 0:
            continue
        ln = np.array(lengths[b]) if lengths[b] else np.array([np.nan])
        print(
            f"  days {b * BIN:3d}-{(b + 1) * BIN:3d}: phase1 {p1[b] / t:.4f}  phase2 {p2[b] / t:.4f}  "
            f"starts/animal-day {starts[b] / (t / 48):.3f}  episode mean {np.nanmean(ln):5.1f}  "
            f"mean energy {energy_sum[b] / t:.2f}  animal-days {t / 48:.0f}"
        )


# --- Java ---
df = pd.read_csv(rp, usecols=["tick", "Id", "UtmX", "UtmY", "EnergyLevel", "DispersalMode"])
df = df[df["tick"] <= DAYS * 48].sort_values(["Id", "tick"])
nb = DAYS // BIN
ticks = np.zeros(nb)
p1 = np.zeros(nb)
p2 = np.zeros(nb)
starts = np.zeros(nb)
esum = np.zeros(nb)
lengths = [[] for _ in range(nb)]
for _, g in df.groupby("Id"):
    x = g["UtmX"].to_numpy()
    e = g["EnergyLevel"].to_numpy()
    mode = g["DispersalMode"].to_numpy()
    t = g["tick"].to_numpy()
    changed = np.flatnonzero((np.diff(x) != 0) | (np.diff(e) != 0))
    last = changed[-1] + 1 if changed.size else 0  # index of the last tick alive
    alive = np.arange(len(t)) <= last
    b = np.minimum((np.maximum(t, 1) - 1) // (BIN * 48), nb - 1).astype(int)
    np.add.at(ticks, b[alive], 1)
    np.add.at(p1, b[alive & (mode == 1)], 1)
    np.add.at(p2, b[alive & (mode == 2)], 1)
    np.add.at(esum, b[alive], e[alive])
    on = (mode > 0) & alive
    began = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    ended = np.flatnonzero(~on[1:] & on[:-1]) + 1
    np.add.at(starts, b[began], 1)
    for s_i in began:
        nxt = ended[ended > s_i]
        if nxt.size:
            lengths[b[s_i]].append(int(nxt[0] - s_i))
print("DispersalMode values in Java:", sorted(df["DispersalMode"].unique().tolist()))
report("Java (200 tracked animals, seed 42)", ticks, p1, p2, starts, esum, lengths)

# --- CENOP ---
params = SimulationParameters(
    porpoise_count=2000,
    sim_years=1,
    random_seed=seed,
    landscape="Kattegat",
    ships_enabled=False,
    turbines="off",
    dispersal="InnerDanishWaters",
)
cd = CellData("Kattegat", str(ROOT / "data"))
cd.load()
sim = Simulation(params=params, cell_data=cd, seed=seed)
sim.initialize()
pm = sim.population_manager
ticks = np.zeros(nb)
p1 = np.zeros(nb)
p2 = np.zeros(nb)
starts = np.zeros(nb)
esum = np.zeros(nb)
lengths = [[] for _ in range(nb)]
began_at = np.full(pm.count, -1)
began_bin = np.zeros(pm.count, dtype=int)
prev = np.zeros(pm.count, dtype=bool)
for t in range(1, DAYS * 48 + 1):
    sim.step()
    b = min((t - 1) // (BIN * 48), nb - 1)
    a = pm.active_mask[: prev.size]
    ph = pm.dispersal_phase[: prev.size]
    ticks[b] += int(a.sum())
    p1[b] += int((ph[a] == 1).sum())
    p2[b] += int((ph[a] == 2).sum())
    esum[b] += float(pm.energy[: prev.size][a].sum())
    cur = (ph > 0) & a
    began = cur & ~prev
    began_at[began] = t
    began_bin[began] = b
    starts[b] += int(began.sum())
    for i in np.flatnonzero(prev & ~cur & a):
        lengths[began_bin[i]].append(int(t - began_at[i]))
    prev = cur.copy()
report(f"CENOP (all animals, seed {seed})", ticks, p1, p2, starts, esum, lengths)
