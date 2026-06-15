# CENOP - CETacean Noise-Population Model

## Environment
- Python env: `micromamba run -n shiny <command>` (or activate with `eval "$(micromamba shell hook --shell bash)" && micromamba activate shiny`)
- Use `python3`, not `python`
- Run from `/home/razinka/cenjas/CENOP/` for all commands

## Testing
- Fast suite (default; slow sim tests deselected): `cd /home/razinka/cenjas/CENOP && python3 -m pytest tests/ -x -q`
- Slow tier (multi-year validation/physiology): `python3 -m pytest tests/ -m slow -q`
- Everything: `python3 -m pytest tests/ -m "slow or not slow" -q`
- NOTE: a default `addopts = -m "not slow"` is active. To run a single SLOW test you must add a selector, e.g. `pytest tests/test_validation.py::Cls::test_x -m "slow or not slow"` — a bare nodeid (or bare `-k`) silently reports it as `deselected`, not run.
- The slow tier is MANUAL (this repo has no CI). Run `pytest tests/ -m slow` before releasing/merging model-behavior changes. Any older `--ignore=...test_validation.py --ignore=...test_depons_physiology.py` alias is now obsolete — it re-hides the fast tests this change recovered; drop it.
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

# Shell Command Rules

When writing bash commands, strictly follow these rules to avoid triggering security permission prompts:

## No backslash-escaped whitespace
- Never escape spaces with backslashes (e.g. My\ File.txt)
- Always use double quotes around paths containing spaces: "My File.txt"
- Quote all variable expansions and file paths as a default habit

## No $() command substitution
- Never use $() or backtick substitution in inline bash commands
- Instead, run the inner command first as a separate bash call, read its output, then use the literal value in the next command
- When possible, rewrite using pipes and xargs (e.g. pgrep python | xargs kill instead of kill $(pgrep python))

## No multi-line commands with comments
- Never include # comments inside inline multi-line bash commands
- If a command requires multiple lines with comments, write it to a temporary script file in /tmp/ and execute that file instead
- Prefer collapsing short multi-line commands into a single line joined with &&

## No quoted newlines followed by # lines
- Never produce a bash command where a quoted string contains a newline followed by a line starting with #
- This pattern triggers the "quoted newline followed by a #-prefixed line" security block
- If you need multi-line strings that include comment-like lines, write them to a file first using a separate bash call, or use printf instead of echo with literal newlines
- Avoid heredocs (cat << EOF) that contain # lines inside inline bash commands; write the content to a file in a prior step instead

## No cd && git compound commands
- Never chain cd <path> && git <cmd>
- Use git -C <path> <cmd> instead

## General principles
- Prefer multiple simple, single-line bash calls over one complex compound command
- For anything that cannot be expressed as a clean single-line command, write a temporary .sh script and run it
- Always use double quotes around paths, variables, and arguments containing spaces or special characters
