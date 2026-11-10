# Validation Suite Status

This document records the status of CENOP's slow demographic-validation tier — the
multi-year population-dynamics tests that confirm the model reproduces DEPONS-documented
demographic behavior (population stability, mortality, reproduction, energy, age structure).

## CI coverage — nightly job, plus a local run before releases

The 11 slow tests are marked `@pytest.mark.slow` and are **deselected by default** so
day-to-day iteration on the fast suite stays quick. They run automatically only in the
dedicated GitHub Actions workflow `.github/workflows/slow.yml`: nightly at 03:00 UTC and
on manual dispatch (Actions → "Slow validation tier" → Run workflow). The fast-suite
workflow (`ci.yml`) does **not** run them.

> ⚠️ **Caveats.** GitHub runs `schedule` and `workflow_dispatch` workflows only from the
> repository's **default branch**. Until 2026-11-10 the default branch was a stale `main`,
> so this workflow was never registered and never ran; the default is now `CENOP-JASMINE`.
> GitHub also **disables scheduled workflows after ~60 days without repository activity**.
> Check the Actions tab and re-enable the workflow if it has been paused. Run
> `pytest tests/ -m slow` locally before every release/merge of model-behavior changes.

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

## Last recorded run (2026-11-10, local, at v2.2.0 `2db80ea`)

- **Outcome:** 11 passed, 810 deselected
- **Total runtime:** 289.04 s (~4 min 49 s); per-test durations not captured

## Previous recorded run (2026-06-12)

- **Outcome:** 11 passed, 724 deselected
- **Total runtime:** 261.14 s (~4 min 21 s)
- **Slowest tests:**
  - 41.27 s — `test_validation.py::TestPopulationDynamicsValidation::test_age_distribution_remains_valid`
  - 38.14 s — `test_depons_physiology.py::TestPopulationTrajectory::test_population_stability_short_term`
  - 37.29 s — `test_depons_physiology.py::TestDEPONSTrajectoryComparison::test_annual_mortality_rate`
- **Expectation changes:** none — all 11 tests passed clean against their existing
  assertions. No triage or assertion adjustments were required.
