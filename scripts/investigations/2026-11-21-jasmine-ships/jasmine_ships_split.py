"""JASMINE mode with ships: survival on Kattegat, with single subsystems switched back to DEPONS.

Builds the simulation the way the app does (simulation_controller: time, movement, behaviour
state machine, energy and memory modules from the effective modes), runs 60 days with 500
animals (seed 7) and prints alive, mean energy, the share of animals in DISTURBED and the
share deterred at that moment, at days 10, 30 and 60.

python3 scripts/investigations/2026-11-21-jasmine-ships/jasmine_ships_split.py LABEL [key=value ...]

key=value pairs go to SimulationParameters (ships=yes switches ships on). JASMINE movement
and energy are selected unless overridden (they are opt-in since 2026-11-21). Alive at day 60,
re-measured 2026-11-21 at b7cf39c (after the learned-avoidance and default changes; just after
the Kattegat salinity fix: 499, 455, 493, 459, 376; before it, with steps 3.3 times too long:
497, 72, 498, 134, 376):
    (none)                                                         499
    ships=yes                                                      441
    ships=yes energy_mode=DEPONS                                   497
    ships=yes disturbance_recovery_ticks=0                         455
    ships=yes disturbance_recovery_ticks=0 movement_mode=DEPONS    444
"""

import sys

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters
from cenop.server import simulation_controller as sc

DISTURBED = 5  # BehaviorState.DISTURBED.value

label = sys.argv[1]
over = {k: (int(v) if v.isdigit() else v) for k, v in (kv.split("=") for kv in sys.argv[2:])}
ships = over.pop("ships", "no") == "yes"

cd = CellData("Kattegat", "data")
cd.load()
params = SimulationParameters(
    porpoise_count=500,
    sim_years=1,
    random_seed=7,
    landscape="Kattegat",
    ships_enabled=ships,
    turbines="off",
    simulation_mode="JASMINE",
    # The JASMINE movement and energy are opt-in since 2026-11-21; select them unless overridden
    **{"movement_mode": "JASMINE", "energy_mode": "JASMINE", **over},
)
tm = sc._get_time_mode(params.get_effective_time_mode())
sim = Simulation(
    params,
    cell_data=cd,
    seed=7,
    time_mode=tm,
    movement_module=sc.create_movement_module(
        params, tm, sc._get_movement_mode(params.get_effective_movement_mode())
    ),
    behavior_fsm=sc.create_behavior_fsm(params, tm),
    energy_module=sc.create_energy_module(
        params, sc._get_energy_mode(params.get_effective_energy_mode())
    ),
    memory_module=sc.create_memory_module(
        params, sc._get_memory_mode(params.get_effective_memory_mode())
    ),
)
sim.initialize()
pm = sim.population_manager
out = []
for d in range(60):
    for _ in range(48):
        sim.step()
    if d + 1 in (10, 30, 60):
        a = pm.active_mask
        out.append(
            f"d{d + 1} alive {int(a.sum())} E {pm.energy[a].mean():.1f} "
            f"disturbed {(pm._behavior_state.state[a] == DISTURBED).mean():.0%} "
            f"deterred-now {(pm.deter_strength[a] > 0.01).mean():.0%}"
        )
print(
    f"{label:28s} ships={ships} time={params.get_effective_time_mode()} "
    f"move={params.get_effective_movement_mode()} energy={params.get_effective_energy_mode()} | "
    + " | ".join(out)
)
