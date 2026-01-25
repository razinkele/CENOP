"""
Lightweight stress script to reproduce and instrument births/deaths totals.
Usage:
    python scripts/stress_test_births_deaths.py --ticks 5000 --porpoise 1000 --seed 42 --log stress_log.txt
"""
import argparse
import traceback
import time

from cenop import Simulation, SimulationParameters


def run(ticks=5000, porpoise=1000, seed=42, log_file=None, warn_threshold=500):
    params = SimulationParameters(
        porpoise_count=porpoise,
        sim_years=1,
        landscape="Homogeneous",
        random_seed=seed,
        turbines="off",
        ships_enabled=False,
    )

    sim = Simulation(params)
    sim.initialize()

    initial_pop = sim.state.population
    prev_births = sim.total_births
    prev_deaths = sim.total_deaths

    start = time.time()
    out = []
    try:
        for t in range(1, ticks + 1):
            sim.step()

            births = sim.total_births
            deaths = sim.total_deaths
            pop = sim.state.population
            net = births - deaths
            net_pop = pop - initial_pop
            births_delta = births - prev_births
            deaths_delta = deaths - prev_deaths

            if t % 100 == 0:
                print(f"tick={t:5d} pop={pop:4d} births={births:7d} deaths={deaths:7d} net={net:7d} net_pop={net_pop:7d} bΔ={births_delta} dΔ={deaths_delta}")

            # Check for discrepancy
            if net != net_pop:
                msg = (
                    f"DISCREPANCY at tick {t}: pop={pop}, births={births}, deaths={deaths}, net={net}, net_pop={net_pop}\n"
                    f"births_delta={births_delta}, deaths_delta={deaths_delta}\n"
                    f"last_history={sim._history[-10:]}\n"
                )
                print(msg)
                out.append(msg)
                # Continue collecting but also log it

            # Check for large spikes
            if births_delta > warn_threshold or deaths_delta > warn_threshold:
                msg = f"[WARNING] Large increment at tick {t}: births_delta={births_delta}, deaths_delta={deaths_delta}, pop={pop}"
                print(msg)
                out.append(msg)

            prev_births = births
            prev_deaths = deaths

    except Exception as e:
        tb = traceback.format_exc()
        msg = f"EXCEPTION at tick {t}: {e}\n{tb}"
        print(msg)
        out.append(msg)
    elapsed = time.time() - start
    summary = f"Finished {t} ticks in {elapsed:.2f}s - final pop={sim.state.population}, births={sim.total_births}, deaths={sim.total_deaths}"
    print(summary)
    out.append(summary)

    if log_file:
        with open(log_file, 'w', encoding='utf-8') as f:
            for line in out:
                f.write(line + "\n")

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticks', type=int, default=5000)
    parser.add_argument('--porpoise', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default='stress_log.txt')
    parser.add_argument('--warn', type=int, default=500)
    args = parser.parse_args()
    run(ticks=args.ticks, porpoise=args.porpoise, seed=args.seed, log_file=args.log, warn_threshold=args.warn)
