"""Births and deaths over a run are counted once each.

The simulation keeps month-to-date counts in its daily history records, and
Simulation.total_births/total_deaths summed those records, so an event was counted once for
every remaining day of its month (a batch run showed 11 births for a population that grew
from 40 to 41). The app's runner counted net daily population changes instead, so births and
deaths on the same day cancelled. Births are weaned calves added to the population; deaths
are all removals (starvation, old age, bycatch).
"""

import pytest

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters
from cenop.server.simulation_controller import SimulationRunner


def _sim(n=10, **kw):
    params = SimulationParameters(porpoise_count=n, sim_years=2, random_seed=8, **kw)
    return Simulation(params, cell_data=create_homogeneous_landscape())


def test_daily_history_records_do_not_multiply_the_counts():
    sim = _sim()
    pm = sim.population_manager
    pm.last_step_births, pm.last_step_deaths = 2, 1
    sim._update_population_statistics()
    for _ in range(5):  # five days of the same month: the records hold month-to-date counts
        sim._record_history()
    assert (sim.total_births, sim.total_deaths) == (2, 1)  # was (10, 5)


def test_runner_reports_the_simulation_totals():
    sim = _sim()
    runner = SimulationRunner(sim)
    runner.set_ticks_per_update(1)
    sim._births_total, sim._deaths_total = 5, 3  # births and deaths on the same day
    runner.step_ticks()
    assert (runner.total_births, runner.total_deaths) == (5, 3)  # net change counted (0, 0)
    assert (runner.total_births, runner.total_deaths) == (sim.total_births, sim.total_deaths)


@pytest.mark.slow
def test_births_minus_deaths_is_the_population_change():
    """40 animals for 420 days through the app's runner (the first calves are weaned in year 2)."""
    sim = _sim(40, bycatch_prob=0.3)
    start = sim.population_size
    runner = SimulationRunner(sim)
    for _ in range(420):
        runner.step_ticks()
    assert sim.total_deaths > 0 and sim.total_births > 0
    assert sim.total_births - sim.total_deaths == sim.population_size - start
    assert (runner.total_births, runner.total_deaths) == (sim.total_births, sim.total_deaths)
