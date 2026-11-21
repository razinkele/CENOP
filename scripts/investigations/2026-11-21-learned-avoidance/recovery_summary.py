import numpy as np

"""Ratios of the recovery_study.py counts to the no-turbine control, mean of seeds 1-3."""
S = "output/memory-recovery"
START, EVERY, DUR, N = 30 * 48, 96, 6, 11
piles = [(START + k * EVERY, START + k * EVERY + DUR) for k in range(N)]
ends = np.array([e for _, e in piles])
last_end = ends[-1]
variants = ["depons", "old", "default"]
res = {v: [] for v in variants}
for sd in (1, 2, 3):
    ctrl = np.load(f"{S}/rec_counts_none_{sd}.npy")
    tick = np.arange(1, ctrl.shape[0] + 1)
    since = np.array(
        [(t - ends[ends <= t].max()) / 2.0 if (ends <= t).any() else np.inf for t in tick]
    )
    during = (tick > piles[0][0]) & (tick <= last_end)
    after_d = (tick - last_end) / 48
    windows = [
        ("pile +0-3 h", during & (since < 3)),
        ("pile +3-12 h", during & (since >= 3) & (since < 12)),
        ("pile +12-24 h", during & (since >= 12) & (since < 24)),
        ("pile +24-48 h", during & (since >= 24) & (since < 48)),
        ("after d0-1", (after_d > 0) & (after_d < 1)),
        ("after d1-3", (after_d >= 1) & (after_d < 3)),
        ("after d3-7", (after_d >= 3) & (after_d < 7)),
        ("after d7-30", (after_d >= 7) & (after_d < 30)),
    ]
    for v in variants:
        c = np.load(f"{S}/rec_counts_{v}_{sd}.npy")
        res[v].append([c[w].sum(axis=0) / np.maximum(ctrl[w].sum(axis=0), 1) for _, w in windows])
names = [n for n, _ in windows]
print("ratio to the no-turbine control (same seed), mean of 3 seeds; columns 0-5 / 5-10 / 10-20 km")
print(f"{'window':14s} " + " | ".join(f"{v:^17s}" for v in variants))
for i, n in enumerate(names):
    cells = []
    for v in variants:
        m = np.mean([r[i] for r in res[v]], axis=0)
        cells.append(" ".join(f"{x:.2f}" for x in m))
    print(f"{n:14s} " + " | ".join(f"{c:^17s}" for c in cells))
