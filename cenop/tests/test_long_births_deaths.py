import os
import pytest

from cenop import Simulation, SimulationParameters


@pytest.mark.skipif(os.getenv('RUN_SLOW', '0') != '1', reason='Long tests skipped by default')
def test_births_deaths_long_optional():
    """Long optional test: run 2000 ticks on 1000 porpoises and assert totals consistency."""
    params = SimulationParameters(
        porpoise_count=1000,
        sim_years=1,
        landscape="Homogeneous",
        random_seed=42,
    )

    sim = Simulation(params)
    sim.initialize()

    initial_pop = sim.state.population

    ticks = 2000
    for _ in range(ticks):
        sim.step()

    total_births = sim.total_births
    total_deaths = sim.total_deaths
    net = total_births - total_deaths
    net_pop = sim.state.population - initial_pop

    assert (net == net_pop), (
        f"Inconsistent totals after {ticks} ticks: births({total_births}) - deaths({total_deaths}) != net_pop({net_pop})"
    )

    assert total_births >= 0 and total_deaths >= 0
    # generous sanity bounds
    assert total_births <= ticks * params.porpoise_count
    assert total_deaths <= ticks * params.porpoise_count
