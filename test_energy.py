"""Quick test to verify energy balance and population dynamics work."""
import sys
sys.path.insert(0, 'src')

from cenop import Simulation, SimulationParameters
from cenop.landscape import create_homogeneous_landscape

params = SimulationParameters(porpoise_count=200, sim_years=1)
landscape = create_homogeneous_landscape()
sim = Simulation(params, cell_data=landscape)

print('Initial pop:', sim.population_size)
pm = sim.population_manager
print(f'Initial energy avg: {pm.energy[pm.active_mask].mean():.2f}')

initial_pop = sim.population_size
for day in range(30):  # Run for 30 days
    for _ in range(48):  # 48 steps per day
        sim.step()
    if pm.count > 0:
        active = pm.active_mask
        avg_energy = pm.energy[active].mean()
        current_pop = sim.population_size
        pop_change = current_pop - initial_pop
        print(f'Day {day+1}: pop={current_pop} ({pop_change:+d}), energy_avg={avg_energy:.2f}')
    else:
        print(f'Day {day+1}: pop=0 - EXTINCTION')
        break

final_pop = sim.population_size
print(f'\nFinal: {final_pop} porpoises (started with {initial_pop}, change: {final_pop - initial_pop:+d})')
