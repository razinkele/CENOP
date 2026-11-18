"""Compare CENOP runs of kattegat-idw-undisturbed-v1 with the DEPONS-Java reference runs.

Inputs:
- CENOP: output/cenop_idw/<mode>/seed-*.csv from run_cenop.py (modes InnerDanishWaters,
  PSM-Type2).
- Java: output/depons_java/kattegat-{idw,psm2}-undisturbed-v1/N2000-Y5/seed-*/ with
  Statistics.csv (tick, PorpoiseCount), PorpoisePerBlock.csv (tick, Block, PorpoiseCount)
  (RandomPorpoise.csv, the one tracked animal, is used by episodes.py's write-up).

Prints: population at the end of each year; the share of animals per block averaged over a
window of days (ten most populated Java-IDW blocks; block index 0-59
from the north-west on both sides), with the IDW-minus-PSM2 shift on each
side. Time spent dispersing is compared by episodes.py, per animal-tick.

Run from the repo root:
python3 scripts/investigations/2026-11-18-dispersal-idw/compare_with_java.py [tag]

``tag`` selects CENOP runs under output/cenop_idw/<tag>/ (e.g. v32: CENOP on the DEPONS v3.2
landscape, the one the Java runs used). Default: CENOP's own data/Kattegat (v3.0).
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
JAVA = {
    "InnerDanishWaters": ROOT / "output/depons_java/kattegat-idw-undisturbed-v1/N2000-Y5",
    "PSM-Type2": ROOT / "output/depons_java/kattegat-psm2-undisturbed-v1/N2000-Y5",
}
CENOP = ROOT / "output/cenop_idw" / (sys.argv[1] if len(sys.argv) > 1 else "")
YEAR_DAYS = 360
WINDOW = (30, 360)  # days averaged for the block distribution: after spin-up, before few are left


def java_run(d):
    """Daily population and per-block counts from one Java seed directory."""
    stats = np.genfromtxt(d / "Statistics.csv", delimiter=",", skip_header=1, usecols=(1, 2))
    pop = {int(round(t)) // 48: int(n) for t, n in stats if (int(round(t)) + 1) % 48 == 0}
    per_block = np.zeros((5 * YEAR_DAYS + 1, 60))
    with open(d / "PorpoisePerBlock.csv") as fh:
        rd = csv.reader(fh)
        next(rd)
        for _, tick, block, count in rd:
            t = int(float(tick))
            if (t + 1) % 48 == 0:
                # Java reports the raw blocks.asc value (1-60, Block.java:76); index = value - 1
                if int(block) >= 1:
                    per_block[(t + 1) // 48, int(block) - 1] = float(count)
    return pop, per_block


def cenop_run(f):
    a = np.genfromtxt(f, delimiter=",", skip_header=1)
    return a[:, 1], a[:, 5:65], a[:, 3], a[:, 4]


def shares(per_block):
    lo, hi = WINDOW
    w = per_block[lo : hi + 1]
    tot = w.sum(axis=1, keepdims=True)
    return (w / np.where(tot == 0, 1, tot)).mean(axis=0)


def main():
    java, cen = {}, {}
    for mode, base in JAVA.items():
        java[mode] = {d.name: java_run(d) for d in sorted(base.glob("seed-*"))}
        cen[mode] = {f.stem: cenop_run(f) for f in sorted((CENOP / mode).glob("seed-*.csv"))}

    print("== population at the end of each year ==")
    for mode in JAVA:
        for seed, (pop, _) in java[mode].items():
            ys = [pop.get(y * YEAR_DAYS - 1, -1) for y in range(1, 6)]
            print(f"Java  {mode:18s} {seed}: {ys}")
        for seed, (n, _, _, _) in cen[mode].items():
            ys = [int(n[y * YEAR_DAYS]) for y in range(1, 6) if y * YEAR_DAYS < len(n)]
            print(f"CENOP {mode:18s} {seed}: {ys}")

    print(f"\n== mean share of animals per block, days {WINDOW[0]}-{WINDOW[1]} ==")
    sh = {}
    for side, runs in (("Java", java), ("CENOP", cen)):
        for mode in JAVA:
            per_seed = np.array([shares(r[1]) for r in runs[mode].values()])
            sh[side, mode] = per_seed
    top = np.argsort(sh["Java", "InnerDanishWaters"].mean(axis=0))[::-1][:10]
    print("block  Java-IDW (min-max)      CENOP-IDW (min-max)     Java-PSM2  CENOP-PSM2")
    for b in top:
        ji, ci = sh["Java", "InnerDanishWaters"][:, b], sh["CENOP", "InnerDanishWaters"][:, b]
        jp, cp = sh["Java", "PSM-Type2"][:, b].mean(), sh["CENOP", "PSM-Type2"][:, b].mean()
        print(
            f"{b:5d}  {ji.mean():.3f} ({ji.min():.3f}-{ji.max():.3f})   "
            f"{ci.mean():.3f} ({ci.min():.3f}-{ci.max():.3f})   {jp:.3f}      {cp:.3f}"
        )
    m = {k: v.mean(axis=0) for k, v in sh.items()}
    for a, b, label in [
        (("Java", "InnerDanishWaters"), ("CENOP", "InnerDanishWaters"), "Java-IDW vs CENOP-IDW"),
        (("Java", "PSM-Type2"), ("CENOP", "PSM-Type2"), "Java-PSM2 vs CENOP-PSM2"),
        (("Java", "InnerDanishWaters"), ("Java", "PSM-Type2"), "Java-IDW vs Java-PSM2"),
        (("CENOP", "InnerDanishWaters"), ("CENOP", "PSM-Type2"), "CENOP-IDW vs CENOP-PSM2"),
    ]:
        tvd = 0.5 * np.abs(m[a] - m[b]).sum()
        r = np.corrcoef(m[a], m[b])[0, 1]
        print(f"{label:26s} total variation distance {tvd:.3f}   correlation {r:.3f}")
    sj = m["Java", "InnerDanishWaters"] - m["Java", "PSM-Type2"]
    sc = m["CENOP", "InnerDanishWaters"] - m["CENOP", "PSM-Type2"]
    print(f"IDW-minus-PSM2 shift, Java vs CENOP: correlation {np.corrcoef(sj, sc)[0, 1]:.3f}")
    for side, s in (("Java", sj), ("CENOP", sc)):
        between = 0.5 * np.abs(np.diff(sh[side, "InnerDanishWaters"][:2], axis=0)).sum()
        print(
            f"{side}: IDW-PSM2 distance {0.5 * np.abs(s).sum():.3f}; between two IDW seeds {between:.3f}"
        )

    print("\nTime spent dispersing: see episodes.py (the daily snapshots here are taken right")
    print("after the daily trigger and overstate it; Java tracks one animal per run).")


if __name__ == "__main__":
    main()
