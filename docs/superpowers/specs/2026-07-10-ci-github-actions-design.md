# CI (GitHub Actions) — Design Spec (2026-07-10)

## Goal

Add continuous integration for CENOP (`razinkele/CENOP`), which currently has **no CI**
(no `.github/`, no pre-commit). The project memory records a hard requirement: any CI
**must include a slow-tier job** (`pytest -m slow`), because that tier is otherwise manual
and is the pre-release validation gate.

## Decisions (confirmed with user)

- **Platform:** GitHub Actions (CENOP has a GitHub remote; only real option).
- **Environment:** micromamba / conda-forge, reproducing the local `shiny` scientific
  stack from a newly committed `environment.yml`. (Chosen over pip for faithful
  reproduction of the numba/rasterio/geopandas/shapely stack.)
- **Slow tier cadence:** nightly `schedule` + `workflow_dispatch` (keeps PRs fast; the
  ~5-min tier runs off the critical path).
- **Python:** single version `3.13` (matches local dev).
- **Lint: DEFERRED.** The codebase is not lint-clean repo-wide (93/111 files unformatted
  under `black --line-length 100`; 3189 `ruff` errors under the repo's own
  `select = [E,F,W,I,UP]`) because the auto-format hook only runs on edited files. A
  strict lint gate would red on pre-existing debt, so CI ships test + slow only; a lint
  job is added later after a dedicated repo-wide format pass.

## Context that shapes the design (verified in-repo)

- Deps are conda/pip installable; **JAX and Cython are optional** — their tests skip when
  absent (`_HAS_JAX`, `_HAS_CYTHON`/`CYTHON_OK`). To make CI actually exercise the
  Phase 5 (JAX) and Phase 6 (Cython) parity work — including the now-green
  `test_cython_postcrw_matches_reference` — CI **installs `jax[cpu]` and builds the
  Cython extension**.
- The Cython `.so`/`.c` are **gitignored** (`*.so`), so CI must build the extension
  (`python setup_cython.py build_ext --inplace`) or the Cython tests silently skip.
- Slow tests (`test_validation.py`, `test_depons_physiology.py`) build **synthetic**
  landscapes (`create_homogeneous_landscape`) — self-contained, no external data fetch.
- The only committed-data dependency in tests is `data/Kattegat/ships.json` (present).
- `data/` is **2.1 GB committed** → every checkout pulls it (functional but slow; noted
  as a future `sparse-checkout` optimization, out of scope here).
- Only remote branch is `CENOP-JASMINE` (no `main`) → triggers target `CENOP-JASMINE`.
- pytest must run from the repo root so relative data paths resolve (checkout root is the
  repo root; the ship-loader test is now CWD-robust regardless).

## Deliverables

### 1. `environment.yml` (repo root, new)

`conda-forge` channel; env name `cenop-ci` (distinct from the local `shiny` env so it
never clobbers it). conda-forge provides the compiled/scientific + dev-tool deps
(python 3.13, numpy, scipy, pandas, numba, rasterio, geopandas, shapely,
`matplotlib-base`, cython, `c-compiler`, setuptools, pytest, pytest-asyncio, black,
ruff); a `pip:` subsection provides the Shiny UI stack (shiny, shinyswatch, shinywidgets,
htmltools), plotly, pydantic, tqdm, and `jax[cpu]`.

### 2. `.github/workflows/ci.yml` — fast gate

Triggers: `push` to `CENOP-JASMINE`, `pull_request`. One **`test`** job:
setup-micromamba (cached env) → `pip install -e . --no-deps` (makes `cenop` importable
without re-resolving deps) → build Cython extension →
`JAX_PLATFORMS=cpu pytest tests/ -q` (fast suite; slow auto-deselected by the
`addopts = -m "not slow"` in `pyproject.toml`). No lint job (deferred; see Decisions).

### 3. `.github/workflows/slow.yml` — nightly validation

Triggers: `schedule` (`0 3 * * *` UTC) + `workflow_dispatch`. One **`slow`** job: same
env + editable install + Cython build → `JAX_PLATFORMS=cpu pytest tests/ -m slow -q`
(the 11 multi-year tests, ~5 min).

## Shared setup pattern

Each job: `actions/checkout` → `mamba-org/setup-micromamba@v2` with
`environment-file: environment.yml`, `cache-environment: true` → steps run in the
activated env via `shell: bash -el {0}`.

## Non-goals (YAGNI)

- **Lint job** (deferred — pre-existing black/ruff debt; needs a repo-wide format first),
  multi-Python matrix, coverage reporting, GPU/JAX-GPU testing, deploy/publish steps,
  sparse-checkout of `data/`, caching the built Cython `.so` across runs, pre-commit.
  Each can be added later without reworking this baseline.

## Verification

Before committing: validate the workflow YAML parses; smoke-test the exact env-setup,
editable-install, Cython-build, and pytest commands in the local `shiny` env from the repo
root so the CI steps are known-good.
