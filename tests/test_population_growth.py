"""A weaned calf always joins the population: the per-animal arrays grow when no slot is free.

DEPONS adds the calf to the context unconditionally (Porpoise.java:1210-1215); CENOP's
fixed-size arrays used to drop it when every slot was occupied.
"""

import numpy as np
import pytest

import cenop.agents.population as pop_mod
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters


class _HighRoll:
    """The population's generator with every uniform draw at 0.99 (each weaned calf is kept)."""

    def __init__(self, rng):
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._rng, name)

    def random(self, *args, **kwargs):
        out = self._rng.random(*args, **kwargs)
        if isinstance(out, np.ndarray):
            out[:] = 0.99
        return out


def _jasmine_modules(params):
    from cenop.behavior import MemoryMode, create_behavior_fsm, create_memory_module
    from cenop.core.time_manager import TimeMode
    from cenop.movement import MovementMode, create_movement_module
    from cenop.physiology import EnergyMode, create_energy_module

    return dict(
        movement_module=create_movement_module(params, TimeMode.DEPONS, MovementMode.HYBRID),
        behavior_fsm=create_behavior_fsm(params, TimeMode.DEPONS),
        energy_module=create_energy_module(params, EnergyMode.DEPONS),
        memory_module=create_memory_module(params, MemoryMode.JASMINE),
    )


def _build(count, jasmine=False, seed=7):
    params = SimulationParameters(random_seed=seed, porpoise_count=count)
    land = create_homogeneous_landscape(width=100, height=100, food_prob=0.5)
    modules = _jasmine_modules(params) if jasmine else {}
    return PorpoisePopulation(count=count, params=params, landscape=land, **modules)


def _shapes(obj, prefix="", seen=None, depth=0):
    """Shapes of every ndarray reachable from obj through cenop objects and lists."""
    seen = set() if seen is None else seen
    out = {}
    for name, v in vars(obj).items():
        if name in ("landscape", "params", "rng", "_active_idx"):
            continue
        if isinstance(v, np.ndarray):
            out[prefix + name] = v.shape
        elif isinstance(v, list):
            out[prefix + name] = ("list", len(v))
        elif (
            hasattr(v, "__dict__")
            and type(v).__module__.startswith("cenop")
            and id(v) not in seen
            and depth < 3
        ):
            seen.add(id(v))
            out.update(_shapes(v, prefix + name + ".", seen, depth + 1))
    return out


def _set_weaning(pop, mothers):
    pop.pregnancy_status[:] = 0
    pop.days_since_mating[:] = -99
    pop.mating_day[:] = -99
    pop.with_calf[:] = False
    pop.with_calf[mothers] = True
    pop.days_since_birth[mothers] = pop.params.nursing_time
    pop.pregnancy_status[mothers] = 2
    pop.rng = _HighRoll(pop.rng)


class TestGrow:
    @pytest.mark.parametrize("jasmine", [False, True])
    def test_grown_population_has_the_layout_of_a_larger_one(self, jasmine):
        pop = _build(7, jasmine)
        pop._grow(5)
        assert pop.count == 12
        assert _shapes(pop) == _shapes(_build(12, jasmine))

    def test_existing_animals_are_untouched_and_new_slots_are_inactive(self):
        pop = _build(7)
        before = {
            k: getattr(pop, k).copy()
            for k in ("x", "y", "age", "energy", "active_mask", "mating_day", "_stored_util")
        }
        pop._grow(5)
        for k, v in before.items():
            np.testing.assert_array_equal(getattr(pop, k)[:7], v)
        assert not pop.active_mask[7:].any()
        assert pop.ids.tolist() == list(range(12))
        assert pop._all_mask.all()

    def test_jasmine_energy_state_still_shares_the_energy_array(self):
        pop = _build(7, jasmine=True)
        pop._grow(5)
        assert pop._energy_state.energy is pop.energy

    def test_growing_draws_nothing_from_the_population_generator(self):
        pop = _build(7)
        state = pop.rng.bit_generator.state
        pop._grow(5)
        assert pop.rng.bit_generator.state == state

    @pytest.mark.parametrize("jasmine", [False, True])
    def test_growing_leaves_the_landscape_modules_and_global_random_state_alone(self, jasmine):
        pop = _build(7, jasmine)
        for _ in range(5):
            pop.step()
        food = pop.landscape._food_value.copy()
        depth = pop.landscape._depth.copy()
        modules = (
            pop._movement_module,
            pop._behavior_fsm,
            pop._energy_module,
            pop._memory_module,
        )
        global_state = np.random.get_state()
        tick, day = pop._global_tick, pop._day_of_year

        pop._grow(5)

        np.testing.assert_array_equal(pop.landscape._food_value, food)
        np.testing.assert_array_equal(pop.landscape._depth, depth)
        after = np.random.get_state()
        assert after[0] == global_state[0] and after[2:] == global_state[2:]
        np.testing.assert_array_equal(after[1], global_state[1])
        assert (pop._global_tick, pop._day_of_year) == (tick, day)
        for before, now in zip(
            modules,
            (pop._movement_module, pop._behavior_fsm, pop._energy_module, pop._memory_module),
        ):
            assert now is before

    @pytest.mark.parametrize("backend", ["numpy", "cython", "jasmine"])
    def test_growth_late_in_a_run_matches_a_larger_population_run_as_long(
        self, monkeypatch, backend
    ):
        # Arrays created lazily during stepping must grow too.
        if backend == "cython" and not pop_mod._HAS_CYTHON:
            pytest.skip("Cython not built")
        monkeypatch.setattr(pop_mod, "_HAS_CYTHON", backend == "cython")
        jasmine = backend == "jasmine"
        pop = _build(7, jasmine)
        big = _build(12, jasmine)
        for _ in range(60):
            pop.step()
            big.step()
        pop._grow(5)
        for _ in range(60):
            pop.step()
            big.step()
        assert _shapes(pop) == _shapes(big)


class TestWeaningWhenFull:
    def test_calf_joins_a_full_population(self):
        pop = _build(6)
        assert pop.active_mask.all()
        _set_weaning(pop, [0, 3])
        pop.last_step_births = 0

        pop._update_pregnancy_status(pop.active_mask)

        assert pop.population_size == 8
        assert pop.last_step_births == 2
        assert pop.count >= 8
        calves = np.flatnonzero(pop.active_mask)[6:]
        assert sorted(zip(pop.x[calves].tolist(), pop.y[calves].tolist())) == sorted(
            zip(pop.x[[0, 3]].tolist(), pop.y[[0, 3]].tolist())
        )
        assert (pop.age[calves] == 0).all()
        assert (pop.pregnancy_status[calves] == 0).all() and not pop.with_calf[calves].any()
        assert not pop.with_calf[[0, 3]].any()

    def test_free_slots_are_used_before_growing(self):
        pop = _build(6)
        pop.active_mask[5] = False
        _set_weaning(pop, [0])
        pop._update_pregnancy_status(pop.active_mask)
        assert pop.count == 6 and pop.active_mask[5]

    @pytest.mark.parametrize("backend", ["numpy", "cython", "jasmine"])
    def test_run_continues_after_growth(self, monkeypatch, backend):
        if backend == "cython" and not pop_mod._HAS_CYTHON:
            pytest.skip("Cython not built")
        monkeypatch.setattr(pop_mod, "_HAS_CYTHON", backend == "cython")
        pop = _build(6, jasmine=backend == "jasmine")
        _set_weaning(pop, [0, 1, 2])
        pop._day_of_year = 47
        pop._global_tick = 47
        pop.step()  # tick 48: daily update, three calves, no free slot
        assert pop.population_size >= 8  # 9 unless something starved in this tick
        grown = pop.count
        assert grown > 6
        for _ in range(100):
            pop.step()
        a = pop.active_mask
        assert pop.count == grown
        for arr in (pop.x, pop.y, pop.heading, pop.energy):
            assert arr.shape == (grown,) and np.isfinite(arr[a]).all()

    def test_simulation_continues_after_growth(self):
        from cenop.core.simulation import Simulation

        params = SimulationParameters(
            porpoise_count=6, sim_years=1, random_seed=7, landscape="Homogeneous"
        )
        sim = Simulation(params=params)
        pop = sim.population_manager
        _set_weaning(pop, [0, 1])
        pop._day_of_year = 47
        pop._global_tick = 47
        for _ in range(60):
            sim.step()
        assert pop.count > 6
        assert sim.state.population == pop.population_size >= 7
