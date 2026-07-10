from cenop.agents.population import PorpoisePopulation
from cenop.parameters.simulation_params import SimulationParameters


def test_porpoise_dataframe_contains_new_fields():
    params = SimulationParameters(porpoise_count=10)
    pop = PorpoisePopulation(count=10, params=params)
    df = pop.to_dataframe()
    assert "heading" in df.columns
    assert "is_disturbed" in df.columns
    assert "behavioral_state" in df.columns
    assert "energy" in df.columns
    assert "age" in df.columns
