"""The daily mortality check and the abandonment-tick charge on every tick backend,
the JASMINE energy module in a run, and the population cap."""

import numpy as np
import pytest

import cenop.agents.population as pop_mod
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters
from cenop.physiology import EnergyMode, create_energy_module


def _pop(count=8, energy_mode=None, **kw):
    params = SimulationParameters(random_seed=5, porpoise_count=count, **kw)
    land = create_homogeneous_landscape(width=100, height=100, food_prob=0.5)
    modules = {}
    if energy_mode is not None:
        modules["energy_module"] = create_energy_module(params, energy_mode)
    return PorpoisePopulation(count=count, params=params, landscape=land, **modules)


class _ForcedRoll:
    """The population's generator with element 0 of every uniform draw above 1."""

    def __init__(self, rng):
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._rng, name)

    def random(self, *args, **kwargs):
        out = self._rng.random(*args, **kwargs)
        if isinstance(out, np.ndarray) and out.size:
            out[0] = 2.0
        return out


needs_cython = pytest.mark.skipif(not pop_mod._HAS_CYTHON, reason="Cython not built")


class TestDailyMortalityOnTheCythonPath:
    @staticmethod
    def _counting(monkeypatch):
        calls = []
        real = pop_mod._cython_post_crw
        monkeypatch.setattr(pop_mod, "_HAS_CYTHON", True)
        monkeypatch.setattr(pop_mod, "_cython_post_crw", lambda *a: calls.append(1) or real(*a))
        return calls

    @needs_cython
    def test_max_age_at_tick_1_and_at_a_day_boundary(self, monkeypatch):
        calls = self._counting(monkeypatch)
        pop = _pop()
        pop.age[0] = pop.params.max_age
        pop.step()
        assert calls and not pop.active_mask[0] and pop.active_mask[1:].all()
        assert pop.death_causes == ["old_age"]
        assert pop.last_step_deaths == 1 and pop.last_step_deaths_old_age == 1

        for _ in range(46):
            pop.step()
        pop.age[1] = pop.params.max_age
        pop.step()  # tick 48
        assert not pop.active_mask[1]
        assert pop.death_causes.count("old_age") == 2

    @needs_cython
    def test_bycatch_is_drawn_on_the_cython_path(self, monkeypatch):
        calls = self._counting(monkeypatch)
        pop = _pop(count=200, bycatch_prob=0.999999)
        pop.step()
        assert calls
        # daily survival is (1 - 0.999999) ** (1 / 360) = 0.962: about 8 of 200 die
        assert 1 <= pop.death_causes.count("bycatch") <= 25
        assert set(pop.death_causes) == {"bycatch"}
        assert pop.last_step_deaths_bycatch == pop.death_causes.count("bycatch")


class TestEnergyModulePath:
    def test_abandonment_tick_is_still_charged_for_lactation(self, monkeypatch):
        monkeypatch.setattr(pop_mod, "_HAS_CYTHON", False)
        pop = _pop(energy_mode=EnergyMode.DEPONS)
        pop.energy[:] = 20.0
        pop.with_calf[:] = False
        pop.with_calf[:2] = True
        pop.rng = _ForcedRoll(pop.rng)
        pop.step()
        assert pop.active_mask[:3].all()
        assert not pop.with_calf[0] and pop.with_calf[1]
        base = 20.0 - pop.energy[2]
        assert base > 0
        assert 20.0 - pop.energy[1] == pytest.approx(base * pop.params.e_lact, rel=1e-3)
        assert 20.0 - pop.energy[0] == pytest.approx(base * pop.params.e_lact, rel=1e-3)

    @pytest.mark.parametrize("mode", [EnergyMode.JASMINE, EnergyMode.HYBRID])
    def test_jasmine_energy_module_runs(self, mode):
        pop = _pop(count=30, energy_mode=mode)
        for _ in range(100):
            pop.step()
        a = pop.active_mask
        assert np.isfinite(pop.energy[a]).all()
        assert ((pop.energy[a] >= 0) & (pop.energy[a] <= 20)).all()

    def test_jasmine_split_phases_add_up_to_the_combined_update(self):
        from cenop.physiology.energy_budget import EnergyContext, EnergyState

        params = SimulationParameters(random_seed=5)
        module = create_energy_module(params, EnergyMode.JASMINE)
        state = EnergyState.create(6, initial_energy=10.0)
        context = EnergyContext.create_default(6, month=6)
        context.is_lactating[:2] = True
        mask = np.array([True, True, True, True, False, True])
        combined = module.compute_energy_update(state, context, mask)
        intake = module.compute_food_intake(state, context, mask)
        cost = module.compute_bmr_cost(state, context, mask)
        np.testing.assert_allclose(intake - cost, combined.net_energy_change, rtol=1e-6)
        assert cost[0] > cost[2] > 0 and cost[4] == 0


class TestPopulationCap:
    @staticmethod
    def _weaning(pop, mothers):
        pop.pregnancy_status[:] = 0
        pop.days_since_mating[:] = -99
        pop.mating_day[:] = -99
        pop.with_calf[:] = False
        pop.with_calf[mothers] = True
        pop.days_since_birth[mothers] = pop.params.nursing_time
        pop.pregnancy_status[mothers] = 2

        class _High:
            def __init__(self, rng):
                self._rng = rng

            def __getattr__(self, name):
                return getattr(self._rng, name)

            def random(self, *a, **k):
                out = self._rng.random(*a, **k)
                out[:] = 0.99
                return out

        pop.rng = _High(pop.rng)

    def test_no_cap_by_default(self):
        assert SimulationParameters().max_population == 0

    def test_cap_limits_growth_and_counts_the_calves_it_drops(self):
        pop = _pop(count=6, max_population=8)
        self._weaning(pop, [0, 1, 2, 3])
        pop.last_step_births = 0
        pop._update_pregnancy_status(pop.active_mask)
        assert pop.count == 8 and pop.population_size == 8
        assert pop.last_step_births == 2
        assert pop.calves_dropped == 2
        assert not pop.with_calf.any()

    def test_cap_below_the_starting_size_is_rejected(self):
        with pytest.raises(ValueError):
            SimulationParameters(porpoise_count=10, max_population=5)

    def test_growth_is_geometric(self):
        pop = _pop(count=400)
        self._weaning(pop, [0])
        pop._update_pregnancy_status(pop.active_mask)
        assert pop.count == 500  # a quarter more, so repeated growth copies little
