"""DEPONS side: the 200 tracked animals of each seed, from RandomPorpoise.csv.

python3 scripts/investigations/2026-11-20-ship-energy/java_side.py <dir with seed-*/> [out.npz]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import common as c

root = Path(sys.argv[1])
acc = c.empty()
cols = ["tick", "Id", "UtmX", "UtmY", "EnergyLevel", "DeterShipStrength", "DispersalMode"]
last_tick = c.BINS[-1][1] * 48
for seed_dir in sorted(root.glob("seed-*")):
    df = pd.read_csv(seed_dir / "RandomPorpoise.csv", usecols=cols)
    df = df[df["tick"] <= last_tick + 48 * 30].sort_values(["Id", "tick"])
    for _, g in df.groupby("Id"):
        t = g["tick"].to_numpy().astype(int)
        x, y, e = g["UtmX"].to_numpy(), g["UtmY"].to_numpy(), g["EnergyLevel"].to_numpy()
        changed = np.flatnonzero((np.diff(x) != 0) | (np.diff(e) != 0))
        last = changed[-1] + 1 if changed.size else 0
        alive = np.arange(len(t)) <= last
        step = np.full(len(t), np.nan)
        step[1:] = np.hypot(np.diff(x), np.diff(y)) / 400.0
        strength = g["DeterShipStrength"].to_numpy()
        disp = g["DispersalMode"].to_numpy() > 0
        day = (np.maximum(t, 1) - 1) // 48 + 1
        for b, (lo, hi) in enumerate(c.BINS):
            m = (day > lo) & (day <= hi)
            if m.any():
                c.add(acc, b, alive[m], strength[m], disp[m], step[m], e[m])
    print(seed_dir.name, "done", flush=True)
c.table("DEPONS (200 tracked animals per seed)", acc)
if len(sys.argv) > 2:
    np.savez(sys.argv[2], **acc)
