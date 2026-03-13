# CENOP - CETacean Noise-Population Model

## Environment
- Python env: `micromamba run -n shiny <command>` (or activate with `eval "$(micromamba shell hook --shell bash)" && micromamba activate shiny`)
- Use `python3`, not `python`
- Run from `/home/razinka/cenjas/CENOP/` for all commands

## Testing
- `cd /home/razinka/cenjas/CENOP && python3 -m pytest tests/ -x -q`
- 502+ tests across 22 test files; naming convention: `test_<module>.py` matches `src/cenop/<module>.py`
- Numba/coverage compatibility: `tests/conftest.py` patches `coverage.types`

## Git
- CENOP has its own nested git repo — commit from within `CENOP/`, not from parent `cenjas/`
- Main development branch: `CENOP-JASMINE`
- Git worktrees don't work well here due to nested repo structure

## Code Patterns
- SoA (struct-of-arrays) in `population.py` for vectorized simulation
- Numba `@njit` kernels in `src/cenop/optimizations/kernels.py` — use `prange` for parallelism
- `deterrence_vectors` is a tuple `(dx_array, dy_array)`, not `(N, 2)` array
- `create_energy_module()` always returns a module (never None)
- `_global_tick` starts at 0, incremented to 1 before first step; day boundaries at `tick % 48 == 0`

## Style
- Line length: 100 (black + ruff auto-applied via hooks)
- Don't add type annotations or docstrings to unchanged code
