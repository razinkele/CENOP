"""DEPONS population-level on-patch rate from a run with many tracked porpoises
(trackedPorpoiseCount=200), in the same day windows as onpatch_rate.py prints for CENOP.

Usage: depons_onpatch_tracked.py <RandomPorpoise.csv> <patches.asc (v3.2)> [Statistics.csv of
the untracked reference run, to check that tracking did not perturb the trajectory]
"""

import csv
import sys

import numpy as np

XLL, YLL, CS, NROWS, NCOLS = 529473.0, 5972242.0, 400.0, 1000, 600
WINDOWS = ((0, 1), (1, 10), (10, 20), (20, 30), (30, 60), (60, 120), (120, 360))

with open(sys.argv[2]) as fh:
    patches = np.loadtxt(fh.readlines()[6:]) > 0

t, x, y, e, ids = [], [], [], [], []
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        ux, uy = float(r["UtmX"]), float(r["UtmY"])
        if XLL <= ux < XLL + NCOLS * CS and YLL <= uy < YLL + NROWS * CS:
            t.append(float(r["tick"]))
            x.append(ux)
            y.append(uy)
            e.append(float(r["EnergyLevel"]))
            ids.append(r["Id"])
t, x, y, e = map(np.array, (t, x, y, e))
col = np.floor((x - XLL) / CS).astype(int)
row = NROWS - 1 - np.floor((y - YLL) / CS).astype(int)
on = patches[row, col]
print(f"tracked animals: {len(set(ids))}, in-grid rows: {len(t)}; patch share of water 0.0159")
for lo, hi in WINDOWS:
    m = (t >= lo * 48) & (t < hi * 48)
    if m.any():
        print(f"DEPONS days {lo:3d}-{hi:3d}: on-patch {on[m].mean():.4f} over {m.sum():7d} "
              f"animal-ticks, mean E {e[m].mean():.2f}")

if len(sys.argv) > 3:
    with open(sys.argv[3]) as fh:
        ref = {int(float(r["tick"])): int(float(r["PorpoiseCount"])) for r in csv.DictReader(fh)}
    with open(sys.argv[1].replace("RandomPorpoise.csv", "Statistics.csv")) as fh:
        new = {int(float(r["tick"])): int(float(r["PorpoiseCount"])) for r in csv.DictReader(fh)}
    for d in (30, 90, 180, 359):
        k = d * 48
        print(f"PorpoiseCount day {d}: untracked run {ref.get(k)}  tracked-200 run {new.get(k)}")
