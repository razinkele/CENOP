from cenop import Simulation, SimulationParameters

params = SimulationParameters(porpoise_count=1000, sim_years=1, landscape='Homogeneous', random_seed=42)
sim = Simulation(params)
sim.initialize()
print('Initialized: population', sim.state.population)
for t in range(1, 13):
    sim.step()
    history_sum = sum(h['deaths'] for h in sim._history)
    print(f"tick={t:2d} pop={sim.state.population} state_deaths={sim.state.deaths} total_deaths={sim.total_deaths} history_sum={history_sum} history_last={sim._history[-1] if sim._history else None}")
