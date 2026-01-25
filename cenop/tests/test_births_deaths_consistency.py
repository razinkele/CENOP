import pytest

from cenop import Simulation, SimulationParameters


def run_short_simulation_and_get_stats(seed=42, ticks=500, porpoise_count=100):
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

    # Ensure history recorded at least once
    assert sim._history, "No history recorded"

    total_births = sim.total_births
    total_deaths = sim.total_deaths

    # Sanity checks
    # 1) Net population change equals births - deaths
    net_change = sim.state.population - initial_pop
    assert (total_births - total_deaths) == net_change, (
        f"Inconsistent totals: births({total_births}) - deaths({total_deaths}) != net_change({net_change})"
    )

    # 2) Totals should be non-negative and plausibly bounded
    assert total_births >= 0 and total_deaths >= 0
    # Not expecting more births than (ticks * porpoise_count)
    assert total_births <= ticks * porpoise_count


def test_births_deaths_consistency_short_run():
    run_short_simulation_and_get_stats(seed=123, ticks=500, porpoise_count=100)
