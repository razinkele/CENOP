import pytest

from cenop import Simulation, SimulationParameters


def test_births_deaths_smoke_short():
    """Smoke test: run a short simulation and assert births - deaths == net population change."""
    params = SimulationParameters(
        porpoise_count=50,
        sim_years=1,
        landscape="Homogeneous",
        random_seed=42,
    )

    sim = Simulation(params)
    sim.initialize()

    initial_pop = sim.state.population

    ticks = 96  # 2 days (48 ticks/day)
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
    # sanity bound (generous)
    assert total_births <= ticks * params.porpoise_count
    assert total_deaths <= ticks * params.porpoise_count
