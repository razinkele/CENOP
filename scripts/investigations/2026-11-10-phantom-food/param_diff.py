"""Diff the effective DEPONS parameters (A1 manifest) against CENOP SimulationParameters defaults
for the parameters that drive energetics, food and mortality."""

import dataclasses
import json
import sys

sys.path.insert(0, "/home/razinka/cenjas/CENOP/src")
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402

man = json.load(open(
    "/home/razinka/cenjas/CENOP/output/depons_java/kattegat-psm2-undisturbed-v1/N2000-Y5/seed-42/manifest.json"))
dp = man["parameters"]
print("DEPONS params:", len(dp))
for k in sorted(dp):
    print(f"  {k} = {dp[k]}")

cp = {f.name: getattr(SimulationParameters(), f.name) for f in dataclasses.fields(SimulationParameters)}
print("\nCENOP params:", len(cp))
for k in sorted(cp):
    print(f"  {k} = {cp[k]}")
