"""Where do DEPONS PSM-Type2 dispersals start, where are their targets, how long do they last?

Tests hypothesis D6: in Kattegat (600x1000 cells) a target ~875 cells away usually falls outside
the grid, so DEPONS's random-target fallback fails and the animal does not disperse at all.
Prediction: dispersal starts cluster at the N/S ends of the grid, targets are short relative to
N(350,100) km, and only a minority of animals ever disperse.

Usage: depons_disp_targets.py <RandomPorpoise.csv> [max_day]
"""

import csv
import sys
from collections import defaultdict

import numpy as np

X0, Y0, CS = 529473.0, 5972242.0, 400.0
path = sys.argv[1]
max_day = int(sys.argv[2]) if len(sys.argv) > 2 else 120

prev_mode = {}
start = {}
starts = []
episodes = []
alive_ids = set()
disp_ids = set()
ticks_by_id = defaultdict(int)
with open(path) as fh:
    for r in csv.DictReader(fh):
        t = int(float(r["tick"]))
        if t >= max_day * 48:
            break
        ux = float(r["UtmX"])
        if ux <= 0:
            continue
        pid = r["Id"]
        alive_ids.add(pid)
        ticks_by_id[pid] += 1
        gx = (ux - X0) / CS
        gy = (float(r["UtmY"]) - Y0) / CS
        mode = r["DispersalMode"] != "0"
        was = prev_mode.get(pid, False)
        if mode and not was:
            tx = (float(r["PSMTargetUtmX"]) - X0) / CS
            ty = (float(r["PSMTargetUtmY"]) - Y0) / CS
            start[pid] = (t, gx, gy)
            starts.append((t, gx, gy, tx, ty, float(np.hypot(tx - gx, ty - gy))))
            disp_ids.add(pid)
        if was and not mode and pid in start:
            t0, sx, sy = start.pop(pid)
            episodes.append((t - t0, float(np.hypot(gx - sx, gy - sy))))
        prev_mode[pid] = mode

s = np.array(starts)
print(
    f"days 0-{max_day}: {len(alive_ids)} tracked animals, {len(disp_ids)} ever dispersed "
    f"({len(disp_ids) / max(len(alive_ids), 1):.0%}), {len(starts)} starts"
)
if len(s):
    gy = s[:, 2]
    print(
        "start grid-y (0=south,1000=north) deciles:",
        np.percentile(gy, np.arange(0, 101, 10)).round(0),
    )
    print("share of starts with y<150 or y>850:", f"{np.mean((gy < 150) | (gy > 850)):.0%}")
    d = s[:, 5]
    print(
        "target distance [cells] p10/50/90:",
        np.percentile(d, [10, 50, 90]).round(0),
        f" (= {np.median(d) * 0.4:.0f} km median; N(350,100) km would be ~875 cells)",
    )
    print("target grid-y deciles:", np.percentile(s[:, 4], np.arange(0, 101, 10)).round(0))
if episodes:
    e = np.array(episodes)
    print(
        "episode length [ticks] p10/50/90:",
        np.percentile(e[:, 0], [10, 50, 90]),
        " displacement [cells] p50/90:",
        np.percentile(e[:, 1], [50, 90]).round(1),
    )
