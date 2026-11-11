"""Finding 3, like-for-like measurement (DEPONS side). From a tracked-porpoise RandomPorpoise.csv
(per-tick energy, position, dispersal mode, PSM target) compute, over days [0, max_day):

- predicate rate: animal-days whose 3-dp-rounded daily mean energy fell strictly three times
  running (E[d] < E[d-1] < E[d-2] < E[d-3]), regardless of dispersal state;
- activation rate: episode starts per eligible predicate-day (predicate true, not dispersing);
- tick share dispersing, averaged over every tick (not a day-boundary snapshot);
- episode length (rows with mode != 0 = dispersal moves) and a classified end reason.

Scheduling (DEPONS 3.2): DailyTask (priority 1500) runs before Porpoise.move (1000), so a start
logged at tick 48(d+1) comes from day d's predicate. Rows are end-of-tick state (verified: every
step into a first dispersing row is exactly 5 cells; every step out of an episode is a CRW step).

End reasons, in DEPONS evaluation order: day boundary (daily energy stop) > path complete
(5 cells x moves >= 0.95 x start-to-target) > the point 5 cells ahead along the last step is
off-grid > its depth <= wdisp > other (classifier error: heading jitter, grid rounding;
the not-enough-water state is frozen while dispersing and cannot end a visible episode).

Usage: depons_disp_rates.py <RandomPorpoise.csv> <bathy.asc DEPONS ran on> [max_day]
"""

import csv
import sys
from collections import Counter, defaultdict

import numpy as np

X0, Y0, CS = 529473.0, 5972242.0, 400.0
STEP, WDISP = 2.0 / 0.4, 4.0
rp_path, bathy_path = sys.argv[1], sys.argv[2]
max_day = int(sys.argv[3]) if len(sys.argv) > 3 else 120

bathy = np.loadtxt(bathy_path, skiprows=6)  # row 0 = north
H, W = bathy.shape


def depth(gx, gy):
    xi, yi = int(round(gx)), int(round(gy))
    if xi < 0 or yi < 0 or xi >= W or yi >= H:
        return -9999.0
    return bathy[H - 1 - yi, xi]


rows = defaultdict(list)  # id -> [(tick, gx, gy, E, mode, tx, ty)]
with open(rp_path) as fh:
    for r in csv.DictReader(fh):
        t = int(float(r["tick"]))
        if t >= max_day * 48:
            break
        ux = float(r["UtmX"])
        if ux <= 0:
            continue
        rows[r["Id"]].append(
            (
                t,
                (ux - X0) / CS,
                (float(r["UtmY"]) - Y0) / CS,
                float(r["EnergyLevel"]),
                r["DispersalMode"] != "0",
                (float(r["PSMTargetUtmX"]) - X0) / CS,
                (float(r["PSMTargetUtmY"]) - Y0) / CS,
            )
        )

pred_days = elig_days = starts = starts_on_pred = survive_estop = 0
disp_ticks = alive_ticks = 0
lengths, reasons = [], Counter()
for rs in rows.values():
    by_tick = {r[0]: r for r in rs}
    alive_ticks += len(rs)
    disp_ticks += sum(r[4] for r in rs)
    # daily means (only full days)
    day_e = defaultdict(list)
    for r in rs:
        day_e[r[0] // 48].append(r[3])
    mean = {d: round(sum(v) / 48.0, 3) for d, v in day_e.items() if len(v) == 48}
    pred = {}
    for d in mean:
        if all(k in mean for k in (d - 1, d - 2, d - 3)):
            pred[d] = mean[d] < mean[d - 1] < mean[d - 2] < mean[d - 3]
    for d, p in pred.items():
        if d + 1 >= max_day:
            continue
        pred_days += p
        last = by_tick.get(48 * (d + 1) - 1)
        if p and last is not None and not last[4]:
            elig_days += 1
            # Same-day energy stop: E[d] > min(E[d-1..d-7]); the buffer starts as zeros.
            past = [mean.get(d - k, 0.0) for k in range(1, 8)]
            survive_estop += not (mean[d] > min(past))
    # episodes
    i = 0
    while i < len(rs):
        if not rs[i][4] or (i > 0 and rs[i - 1][4]):
            i += 1
            continue
        a = i
        while i + 1 < len(rs) and rs[i + 1][4] and rs[i + 1][0] == rs[i][0] + 1:
            i += 1
        b = i
        t_a, t_b = rs[a][0], rs[b][0]
        starts += 1
        if t_a % 48 == 0 and pred.get(t_a // 48 - 1):
            starts_on_pred += 1
        n = b - a + 1
        if b + 1 >= len(rs) or rs[b + 1][0] != t_b + 1:
            i += 1
            continue  # censored (end of window or death)
        lengths.append(n)
        sx, sy = (rs[a - 1][1], rs[a - 1][2]) if a > 0 else (rs[a][1], rs[a][2])
        tx, ty = rs[a][5], rs[a][6]
        if n >= 2:
            hx, hy = rs[b][1] - rs[b - 1][1], rs[b][2] - rs[b - 1][2]
        else:
            hx, hy = tx - rs[b][1], ty - rs[b][2]
        norm = float(np.hypot(hx, hy)) or 1.0
        ax, ay = rs[b][1] + STEP * hx / norm, rs[b][2] + STEP * hy / norm
        if (t_b + 1) % 48 == 0:
            reasons["day-boundary (energy stop)"] += 1
        elif STEP * n >= 0.95 * float(np.hypot(tx - sx, ty - sy)):
            reasons["path complete"] += 1
        elif ax < 0 or ay < 0 or ax >= W - 0.5 or ay >= H - 0.5:
            reasons["off-grid ahead"] += 1
        elif depth(ax, ay) <= WDISP:
            reasons["depth <= wdisp ahead"] += 1
        else:
            reasons["other"] += 1
        i += 1

animal_days = alive_ticks / 48.0
print(f"DEPONS days 0-{max_day}: {len(rows)} animals, {animal_days:.0f} animal-days")
print(f"  predicate rate      {pred_days / animal_days:.4f} per animal-day")
print(f"  eligible pred-days  {elig_days / animal_days:.4f} per animal-day")
print(
    f"  starts              {starts / animal_days:.4f} per animal-day "
    f"({starts_on_pred / max(starts, 1):.0%} on a predicate day)"
)
print(f"  activation rate     {starts / max(elig_days, 1):.3f} starts per eligible predicate-day")
print(f"    survive same-day energy stop  {survive_estop / max(elig_days, 1):.3f}")
print(f"    then visible (target found, first step ok)  " f"{starts / max(survive_estop, 1):.3f}")
print(f"  tick share disp.    {disp_ticks / alive_ticks:.4f}")
if lengths:
    L = np.array(lengths)
    print(
        f"  episode moves       median {np.median(L):.0f}, mean {L.mean():.1f}, "
        f"p90 {np.percentile(L, 90):.0f}, n={len(L)}"
    )
    for k, v in reasons.most_common():
        print(f"    end: {k:28s} {v / len(L):.0%}")
