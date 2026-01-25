"""
Lightweight profiling harness for CENOP simulation.

Usage:
    python -m cenop.tools.profile_simulation --ticks 1000 --pop 2000

Produces a cProfile report printed to stdout and a saved stats file `cprofile.prof`.
"""
import argparse
import cProfile
import pstats
import io
import sys
import pathlib

# Ensure the project's src directory is on sys.path so the `cenop` package can be imported
ROOT = pathlib.Path(__file__).resolve().parents[2]  # points to top-level 'cenop' dir
SRC = ROOT.parent / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from cenop.core.simulation import Simulation
    from cenop.parameters import SimulationParameters
except Exception as e:
    raise ImportError(f"Failed to import CENOP modules: {e}. "
                      f"Make sure you run this script from the repository root or install the package "
                      f"('pip install -e .'). Added {SRC} to sys.path for convenience.") from e


def run_simulation(params: SimulationParameters, ticks: int = 1000):
    sim = Simulation(params)
    sim.initialize()
    for i in range(ticks):
        sim.step()


def main():
    parser = argparse.ArgumentParser(description='Profile a short CENOP simulation run')
    parser.add_argument('--ticks', type=int, default=500, help='Number of ticks to simulate')
    parser.add_argument('--pop', type=int, default=2000, help='Porpoise population size')
    args = parser.parse_args()

    params = SimulationParameters(porpoise_count=args.pop, sim_years=1, landscape='Homogeneous')

    pr = cProfile.Profile()
    pr.enable()
    run_simulation(params, ticks=args.ticks)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(40)
    print(s.getvalue())

    pr.dump_stats('cprofile.prof')

if __name__ == '__main__':
    main()