"""Per-capita food intake per tick in DEPONS, inferred from FoodEnergyLevel drops within a day."""

import csv
import sys

p = sys.argv[1]
rows = list(csv.DictReader(open(p)))
t = [int(float(r["tick"])) for r in rows]
n = [int(float(r["PorpoiseCount"])) for r in rows]
f = [float(r["FoodEnergyLevel"]) for r in rows]
e = [float(r["PorpoiseEnergyLevel"]) for r in rows]
print("first ticks:", [(t[i], n[i], round(f[i], 3), round(e[i] / n[i], 3)) for i in range(4)])
# find regrowth ticks: food increases
ups = [t[i] for i in range(1, len(t)) if f[i] > f[i - 1]]
print("first food-increase ticks:", ups[:6], "... spacing", [b - a for a, b in zip(ups, ups[1:6])])


def window(day0, ndays):
    eaten, pt = 0.0, 0
    for i in range(1, len(t)):
        d = t[i] // 48
        if day0 <= d < day0 + ndays and f[i] <= f[i - 1]:
            eaten += f[i - 1] - f[i]
            pt += n[i]
    return eaten / pt if pt else float("nan")


for d0 in (0, 1, 5, 30, 60, 120, 400, 800, 1500):
    i = min(range(len(t)), key=lambda k: abs(t[k] - d0 * 48))
    print(f"day {d0:5d}: N={n[i]:5d} food={f[i]:8.2f} meanE={e[i]/n[i]:6.2f}  "
          f"intake/porp/tick (next 3 d)={window(d0, 3):.5f}")
