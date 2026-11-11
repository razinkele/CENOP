"""Reproduce the live-app overflow: Lithuania + CuronianNord_35_15MW, 100 porpoises, DEPONS modes,
built exactly like simulation_controller.create_simulation_from_inputs (UI defaults).

Instruments crw_core.compose_movement: on any overflow/invalid, records which agents had
non-finite step_dist, whether they were in the mask / dispersing / alive, their salinity,
and the output dtype (float32 = module-internal compose, float64 = population compose).
"""

import sys
import warnings
from collections import Counter

import numpy as np

import cenop.movement.crw_core as crw_core
from cenop import Simulation, SimulationParameters
from cenop.behavior import create_behavior_fsm, create_memory_module
from cenop.landscape import CellData
from cenop.movement import create_movement_module
from cenop.physiology import create_energy_module
from cenop.server.simulation_controller import (
    _get_energy_mode,
    _get_memory_mode,
    _get_movement_mode,
    _get_time_mode,
)

TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 48 * 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1

params = SimulationParameters(
    porpoise_count=100,
    sim_years=5,
    landscape="Lithuania",
    turbines="CuronianNord_35_15MW",
    random_seed=SEED,
)
landscape = CellData(params.landscape)
tm = _get_time_mode(params.get_effective_time_mode())
mm = _get_movement_mode(params.get_effective_movement_mode())
sim = Simulation(
    params,
    cell_data=landscape,
    time_mode=tm,
    movement_module=create_movement_module(params, tm, mm),
    behavior_fsm=create_behavior_fsm(params, tm),
    energy_module=create_energy_module(
        params, _get_energy_mode(params.get_effective_energy_mode())
    ),
    memory_module=create_memory_module(
        params, _get_memory_mode(params.get_effective_memory_mode())
    ),
)
print("movement mode", mm.name, "module", type(sim.population_manager._movement_module).__name__)

events = Counter()
examples = []
_orig = crw_core.compose_movement


def traced(
    heading,
    pres_angle,
    log_mov,
    ve_total,
    vt_x,
    vt_y,
    d_dx,
    d_dy,
    is_dispersing,
    mask,
    inertia_const,
    disp_step,
    rads,
    dx,
    dy,
    step_dist,
):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _orig(
            heading,
            pres_angle,
            log_mov,
            ve_total,
            vt_x,
            vt_y,
            d_dx,
            d_dy,
            is_dispersing,
            mask,
            inertia_const,
            disp_step,
            rads,
            dx,
            dy,
            step_dist,
        )
    if w:
        pop = sim.population_manager
        bad = ~np.isfinite(np.power(10.0, log_mov.astype(np.float64)).astype(step_dist.dtype))
        sal = pop._salinity_vals
        for i in np.where(bad)[0]:
            key = (
                str(step_dist.dtype),
                bool(mask[i]),
                bool(is_dispersing[i]),
                bool(pop.active_mask[i]),
                bool(sal[i] <= -9000),
            )
            events[key] += 1
            if len(examples) < 5:
                examples.append(
                    (
                        sim._global_tick if hasattr(sim, "_global_tick") else None,
                        int(i),
                        float(log_mov[i]),
                        float(sal[i]),
                        key,
                    )
                )
        # Did anything non-finite leak into population state?
        for name in ("x", "y", "heading", "prev_log_mov", "prev_angle"):
            arr = getattr(pop, name)
            act = pop.active_mask
            if not np.all(np.isfinite(arr[act])):
                events[("LEAK", name)] += 1
    return out


crw_core.compose_movement = traced

for t in range(TICKS):
    sim.step()

pop = sim.population_manager
print("ticks", TICKS, "seed", SEED, "alive", int(pop.active_mask.sum()))
print("overflow events by (dtype, in_mask, dispersing, alive, salinity_nodata):")
for k, v in sorted(events.items(), key=lambda kv: -kv[1]):
    print("  ", k, v)
print("examples (tick, idx, log_mov, salinity, key):")
for e in examples:
    print("  ", e)
act = pop.active_mask
sal_now = landscape.get_salinities_vectorized(None, xi=pop._cell_xi, yi=pop._cell_yi)
print(
    "alive agents currently on salinity-nodata cells:",
    int((sal_now[act] <= -9000).sum()),
    "of",
    int(act.sum()),
)
