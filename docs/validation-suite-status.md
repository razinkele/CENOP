# Validation Suite Status

This document records the status of CENOP's slow demographic-validation tier — the
multi-year population-dynamics tests that confirm the model reproduces DEPONS-documented
demographic behavior (population stability, mortality, reproduction, energy, age structure).

## ⚠️ No CI — the slow validation tier is MANUAL

> ⚠️ **No CI — the slow validation tier is MANUAL.** Run `pytest tests/ -m slow` before every release/merge of model-behavior changes. If CI is ever added, it MUST include a slow-tier job, or these demographic-validation tests silently never run.

**Why this matters.** The 11 slow tests are marked `@pytest.mark.slow` and are
**deselected by default** so day-to-day iteration on the fast test suite stays quick.
That fast-iteration design is deliberate — but it means these demographic checks never
run unless explicitly requested. The repository has **no `.github/` directory and no CI
pipeline**, so there is no automated safety net. Running `pytest tests/ -m slow` by hand
before each model-behavior release/merge is the **compensating control** for that gap.
If CI is added later, a dedicated slow-tier job is mandatory.

## The 11 slow tests

**`tests/test_validation.py`** (6 tests):
- `TestPopulationDynamicsValidation::test_population_stability_one_year`
- `TestPopulationDynamicsValidation::test_birth_rate_realistic`
- `TestPopulationDynamicsValidation::test_mortality_rate_age_dependent`
- `TestPopulationDynamicsValidation::test_age_distribution_remains_valid`
- `TestPopulationDynamicsValidation::test_energy_distribution_stable`
- `TestDEPONSComparisonValidation::test_annual_mortality_rate_realistic`

**`tests/test_depons_physiology.py`** (5 tests):
- `TestPopulationTrajectory::test_population_stability_short_term`
- `TestPopulationTrajectory::test_age_distribution_evolves`
- `TestDEPONSTrajectoryComparison::test_annual_mortality_rate`
- `TestDEPONSTrajectoryComparison::test_energy_dynamics_over_year`
- `TestDEPONSTrajectoryComparison::test_female_reproduction_rate`

## How to run

```bash
micromamba run -n shiny python3 -m pytest tests/ -m slow -q --durations=0
```

## Last recorded run (2026-06-12)

- **Outcome:** 11 passed, 724 deselected
- **Total runtime:** 261.14 s (~4 min 21 s)
- **Slowest tests:**
  - 41.27 s — `test_validation.py::TestPopulationDynamicsValidation::test_age_distribution_remains_valid`
  - 38.14 s — `test_depons_physiology.py::TestPopulationTrajectory::test_population_stability_short_term`
  - 37.29 s — `test_depons_physiology.py::TestDEPONSTrajectoryComparison::test_annual_mortality_rate`
- **Expectation changes:** none — all 11 tests passed clean against their existing
  assertions. No triage or assertion adjustments were required.
