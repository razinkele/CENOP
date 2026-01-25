import pytest

from cenop import Simulation, SimulationParameters


def run_simulation_and_assert_consistency(seed=99, ticks=200, porpoise_count=500):
    params = SimulationParameters(
        porpoise_count=porpoise_count,
        sim_years=1,
        landscape="Homogeneous",
        random_seed=seed,
        turbines="off",
        ships_enabled=False,
    )
    sim = Simulation(params)
    sim.initialize()

    initial_pop = sim.state.population
    for _ in range(ticks):
        sim.step()

    total_births = sim.total_births
    total_deaths = sim.total_deaths

    # Net population change must equal births - deaths
    net_change = sim.state.population - initial_pop
    assert (total_births - total_deaths) == net_change, (
        f"Inconsistent totals after {ticks} ticks: births({total_births}) - deaths({total_deaths}) != net_change({net_change})"
    )

    # Additional sanity: totals non-negative and not absurdly large
    assert total_births >= 0 and total_deaths >= 0
    assert total_births <= ticks * porpoise_count * 2  # generous bound


def test_births_deaths_consistency_long_run():
    run_simulation_and_assert_consistency(seed=42, ticks=500, porpoise_count=200)