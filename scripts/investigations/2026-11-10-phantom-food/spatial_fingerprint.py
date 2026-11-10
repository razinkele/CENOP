"""D15 step 2: where do animals sit, and where do they eat, in DEPONS vs CENOP (dispersal off)?

Per 30-day bin (days 0-90, Jan-Mar, energy-use scaling 1, no lactation), by depth band and by
distance-to-coast band of the cell where the animal ate (the pre-move cell in both models):
share of animal-ticks, and mean intake per animal-tick.

- DEPONS: tracked animals in RandomPorpoise.csv. Intake per tick = E(t) - E(t-1) + 0.0045
  (use = 0.001 * 1 * EUse 4.5; rows are end-of-tick state), attributed to the position in row
  t-1 (DEPONS eats at posList.get(1)). Cells by DEPONS's own lookup (Math.round), arrays from
  the landscape DEPONS ran on (row 0 = north).
- CENOP: all animals, dispersal off, communication off; exact food passed to `_update_psm`,
  attributed to the pre-step cell by CENOP's own lookup (int truncation) and arrays.

Usage: spatial_fingerprint.py depons <RandomPorpoise.csv> <landscape dir with bathy.asc>
       spatial_fingerprint.py cenop <data_dir with Kattegat/> [seed]
"""

import csv
import sys
from pathlib import Path

import numpy as np

DEPTH_EDGES = [-1e9, 1.0, 5.0, 10.0, 20.0, 30.0, 1e9]
DEPTH_LABELS = ["<1 m", "1-5", "5-10", "10-20", "20-30", ">30"]
COAST_EDGES = [-1e9, 1000.0, 2000.0, 5000.0, 10000.0, 1e12]
COAST_LABELS = ["<1 km", "1-2", "2-5", "5-10", ">10"]
BINS = [(0, 30), (30, 60), (60, 90)]
USE = 0.0045


def report(name, depth, coast, intake, day):
    print(f"== {name}: {len(intake)} animal-ticks")
    for lo, hi in BINS:
        sel = (day >= lo) & (day < hi)
        if not sel.any():
            continue
        print(
            f"-- days {lo}-{hi}: mean intake/at {intake[sel].mean():.5f}, "
            f"share at depth<1 m {np.mean(depth[sel] < 1.0):.4f}"
        )
        for label, edges, values in (("depth", DEPTH_EDGES, depth), ("coast", COAST_EDGES, coast)):
            labels = DEPTH_LABELS if label == "depth" else COAST_LABELS
            k = np.digitize(values[sel], edges[1:-1])
            parts = []
            for j, lab in enumerate(labels):
                m = k == j
                share = m.mean()
                mi = intake[sel][m].mean() if m.any() else float("nan")
                parts.append(f"{lab}: {share:.3f}/{mi:.5f}")
            print(f"   {label:5s} share/intake  " + "  ".join(parts))


def depons(rp_path, land_dir):
    land = Path(land_dir)
    bathy = np.loadtxt(land / "bathy.asc", skiprows=6)
    coast = np.loadtxt(land / "disttocoast.asc", skiprows=6)
    H, W = bathy.shape
    x0, y0, cs = 529473.0, 5972242.0, 400.0
    last = {}
    out_d, out_c, out_i, out_t = [], [], [], []
    with open(rp_path) as fh:
        for r in csv.DictReader(fh):
            t = int(float(r["tick"]))
            if t > 90 * 48:
                break
            ux = float(r["UtmX"])
            pid = r["Id"]
            if ux <= 0:
                last.pop(pid, None)
                continue
            e = float(r["EnergyLevel"])
            gx, gy = (ux - x0) / cs, (float(r["UtmY"]) - y0) / cs
            prev = last.get(pid)
            if prev is not None and prev[0] == t - 1:
                xi, yi = int(round(prev[1])), int(round(prev[2]))
                xi, yi = min(max(xi, 0), W - 1), min(max(yi, 0), H - 1)
                out_d.append(bathy[H - 1 - yi, xi])
                out_c.append(coast[H - 1 - yi, xi])
                out_i.append(e - prev[3] + USE)
                out_t.append(t // 48)
            last[pid] = (t, gx, gy, e)
    report("DEPONS", np.array(out_d), np.array(out_c), np.array(out_i), np.array(out_t))


def cenop(data_dir, seed):
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "src"))
    import cenop.agents.population as P
    from cenop.core.simulation import Simulation
    from cenop.landscape.cell_data import CellData
    from cenop.parameters.simulation_params import SimulationParameters

    P.PorpoisePopulation._check_dispersal_trigger = lambda self, mask: None
    fed = {}
    orig = P.PorpoisePopulation._update_psm

    def psm(self, mask, food_gained):
        if food_gained is not None:
            fed["v"] = np.where(mask, np.asarray(food_gained), np.nan)
        return orig(self, mask, food_gained)

    P.PorpoisePopulation._update_psm = psm
    params = SimulationParameters(
        porpoise_count=2000,
        sim_years=1,
        random_seed=seed,
        landscape="Kattegat",
        ships_enabled=False,
        turbines="off",
        communication_enabled=False,
    )
    cd = CellData("Kattegat", data_dir)
    cd.load()
    sim = Simulation(params=params, cell_data=cd, seed=seed)
    sim.initialize()
    pm = sim.population_manager
    H, W = cd._depth.shape
    out_d, out_c, out_i, out_t = [], [], [], []
    for tick in range(1, 90 * 48 + 1):
        a = pm.active_mask.copy()
        xi = np.clip(pm.x.astype(np.int64), 0, W - 1)
        yi = np.clip(pm.y.astype(np.int64), 0, H - 1)
        fed.pop("v", None)
        sim.step()
        v = fed.get("v")
        if v is None:
            continue
        ok = a & np.isfinite(v)
        out_d.append(cd._depth[yi[ok], xi[ok]])
        out_c.append(cd._dist_to_coast[yi[ok], xi[ok]])
        out_i.append(v[ok])
        out_t.append(np.full(int(ok.sum()), (tick - 1) // 48))
    report(
        f"CENOP seed {seed}",
        np.concatenate(out_d),
        np.concatenate(out_c),
        np.concatenate(out_i),
        np.concatenate(out_t),
    )


if sys.argv[1] == "depons":
    depons(sys.argv[2], sys.argv[3])
else:
    cenop(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 42)
