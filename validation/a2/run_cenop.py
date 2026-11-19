"""One CENOP run of an A2 scenario, logged once per simulated day.

python3 -m validation.a2.run_cenop <scenario> <seed> [smoke_days]

With ``smoke_days`` the run stops after that many days and writes under output/a2/smoke/
(a check of the runner itself, not a result).

``scenario`` is the name of a file in validation/a2/scenarios (u-idw, u-psm, u-off, d-ships);
the same file drives the DEPONS-Java run through the Track A1 harness. Writes

  output/a2/cenop/<scenario>/seed-<seed>.csv         one row per day:
      day, n, mean_energy, ship_ticks, alive_ticks, b0..b59
  output/a2/cenop/<scenario>/seed-<seed>-deaths.csv  one row per death: day, age (whole years)

``ship_ticks`` / ``alive_ticks`` are the animal-ticks of that day with ship deterrence > 0 and
the animal-ticks alive. b0..b59 are animals per 40 km block (blocks.asc value - 1) at the end
of the day. Design: docs/superpowers/specs/2026-11-19-a2-cross-validation-design.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

ROOT = Path(__file__).resolve().parents[2]


def main(scenario: str, seed: int, smoke_days: int | None = None) -> None:
    spec = json.loads((ROOT / "validation/a2/scenarios" / f"{scenario}.json").read_text())
    ov = spec["overrides"]
    assert ov["landscape"] == "Kattegat" and ov["turbines"] == "off"
    years = int(ov["simYears"])
    ships = ov["ships"] == "true"
    params = SimulationParameters(
        porpoise_count=int(ov["porpoiseCount"]),
        sim_years=years,
        random_seed=seed,
        landscape="Kattegat",
        ships_enabled=ships,
        turbines="off",
        dispersal=ov["dispersal"],
    )
    cd = CellData("Kattegat", str(ROOT / "data"))
    cd.load()
    sim = Simulation(params=params, cell_data=cd, seed=seed)
    sim.initialize()
    pm = sim.population_manager
    if ships:
        assert sim._ship_manager.count > 1000, "the Kattegat ship file was not loaded"
    blocks = cd._blocks

    rows = []
    ship_ticks = alive_ticks = 0
    n_days = smoke_days if smoke_days else years * 360
    for tick in range(1, n_days * 48 + 1):
        sim.step()
        a = pm.active_mask
        alive_ticks += int(a.sum())
        if ships:
            ship_ticks += int((pm.deter_strength[a] > 0).sum())
        if tick % 48 == 0:
            xi = np.clip(pm.x[a].astype(np.intp), 0, blocks.shape[1] - 1)
            yi = np.clip(pm.y[a].astype(np.intp), 0, blocks.shape[0] - 1)
            b = blocks[yi, xi]
            per_block = np.bincount(b[b >= 1] - 1, minlength=60)[:60]
            n = int(a.sum())
            rows.append(
                [tick // 48, n, float(pm.energy[a].mean()) if n else 0.0, ship_ticks, alive_ticks]
                + per_block.tolist()
            )
            ship_ticks = alive_ticks = 0

    out = ROOT / "output" / "a2" / ("smoke" if smoke_days else "cenop") / scenario
    out.mkdir(parents=True, exist_ok=True)
    header = "day,n,mean_energy,ship_ticks,alive_ticks," + ",".join(f"b{i}" for i in range(60))
    np.savetxt(out / f"seed-{seed}.csv", np.array(rows), delimiter=",", header=header, comments="")
    deaths = np.column_stack([pm.death_days, pm.death_ages]) if pm.death_days else np.empty((0, 2))
    np.savetxt(
        out / f"seed-{seed}-deaths.csv",
        deaths,
        delimiter=",",
        header="day,age",
        comments="",
        fmt="%d",
    )
    print(scenario, seed, "final n", rows[-1][1], "deaths", len(deaths))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
