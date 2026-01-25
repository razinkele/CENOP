"""
Run a short instrumented simulation that enables PorpoisePopulation instrumentation and writes detailed logs.
Usage:
    python scripts/instrumented_run.py --ticks 2000 --porpoise 1000 --seed 42 --log instrumented_log.txt
"""
import argparse
import logging
import os
import time

from cenop import Simulation, SimulationParameters


def run(ticks=2000, porpoise=1000, seed=42, log_file='instrumented_log.txt', warn_threshold=200):
    # Ensure environment or params flag enables instrumentation in population manager
    os.environ['CENOP_INSTRUMENT'] = '1'

    # Configure logging to both file and console; detailed debug goes to file
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logging.getLogger().addHandler(file_handler)

    params = SimulationParameters(
        porpoise_count=porpoise,
        sim_years=1,
        landscape='Homogeneous',
        random_seed=seed,
        turbines='off',
        ships_enabled=False,
    )
    # Also set attribute for explicit access by population manager
    params.debug_instrumentation = True

    sim = Simulation(params)
    sim.initialize()

    initial_pop = sim.state.population
    prev_births = sim.total_births
    prev_deaths = sim.total_deaths

    logger = logging.getLogger('instrumented_run')
    start = time.time()
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

            if t % 50 == 0:
                logger.info(f"tick={t:5d} pop={pop:4d} births={births:7d} deaths={deaths:7d} net={net:7d} net_pop={net_pop:7d} bΔ={births_delta} dΔ={deaths_delta} state_b={sim.state.births} state_d={sim.state.deaths}")
            
            # Include state counters when logging discrepancies
            if net != net_pop:
                logger.error(f"DISCREPANCY at tick {t}: pop={pop}, births={births}, deaths={deaths}, net={net}, net_pop={net_pop} state_b={sim.state.births} state_d={sim.state.deaths} last_history={sim._history[-6:]}")
                

            # Write warnings
            if births_delta > warn_threshold or deaths_delta > warn_threshold:
                logger.warning(f"Large increment at tick {t}: births_delta={births_delta}, deaths_delta={deaths_delta}, pop={pop}")

            # Check for consistency
            if net != net_pop:
                logger.error(f"DISCREPANCY at tick {t}: pop={pop}, births={births}, deaths={deaths}, net={net}, net_pop={net_pop}")

            prev_births = births
            prev_deaths = deaths

    except Exception as e:
        logger.exception("Exception during instrumented run")
    elapsed = time.time() - start

    summary = f"Finished {t} ticks in {elapsed:.2f}s - final pop={sim.state.population}, births={sim.total_births}, deaths={sim.total_deaths}"
    logger.info(summary)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticks', type=int, default=2000)
    parser.add_argument('--porpoise', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log', type=str, default='instrumented_log.txt')
    parser.add_argument('--warn', type=int, default=200)
    args = parser.parse_args()
    run(ticks=args.ticks, porpoise=args.porpoise, seed=args.seed, log_file=args.log, warn_threshold=args.warn)