# Dispersal Selector (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `params.dispersal` take effect: `off` and `Undirected` work, `PSM-Type2` stays bit-identical, `InnerDanishWaters` and unknown values are rejected.

**Architecture:** `SimulationParameters._validate` is the single gate for valid mode strings. `PorpoisePopulation` reads the mode once into `self._dispersal_mode` and branches on it in the four existing dispersal methods. The JAX backend refuses `Undirected`; no new modules.

**Tech Stack:** Python 3.13, NumPy, pytest, Shiny for Python. Env: `micromamba run -n shiny python3 ...` from `/home/razinka/cenop/CENOP`.

**Spec:** `docs/superpowers/specs/2026-11-18-dispersal-selector-design.md`

## Global Constraints

- Valid modes are exactly `"off"`, `"PSM-Type2"`, `"Undirected"` (case-sensitive, as in DEPONS `parameters.xml`).
- No silent fallback to PSM-Type2 anywhere. A mode a backend cannot run raises `ValueError`.
- No change to random-number draw order on the PSM-Type2 path. No baseline files are edited.
- Default stays `dispersal="PSM-Type2"`.
- Do not touch `behavior/dispersal.py`, `agents/porpoise.py`, `optimizations/` (the Cython kernel only moves animals; it has no dispersal turn or stop).
- Branch `feat/dispersal-selector`. Git: `git -C /home/razinka/cenop/CENOP ...`, commit messages via `-F <file>`, no `$()`, no `cd && git`. Line length 100 (hooks run black + ruff).
- Run pytest from the repo root. A slow test needs `-m slow`.

## Review Focus

1. **Wrong case or stray spaces** (`"Off"`, `"undirected"`, `" off"`): rejected with a message that lists the three valid values, not silently accepted or silently PSM-Type2. Test in Task 1.
2. **`Undirected` with `use_jax=True` on a machine without JAX:** the population falls back to NumPy, so it must run, not raise. The guard keys on the backend actually in use. Test in Task 3.
3. **`Undirected` when no random target is found** (1000 tries all on land): the animal does not start dispersing and its state is untouched. Test in Task 3.
4. **`Undirected` with target distance 0:** completes on the first step without NaN headings. Test in Task 3.
5. **`off` set while a population already has dispersers** (state built by hand or restored): existing dispersers finish normally; only new starts are blocked. Test in Task 2.

---

### Task 1: Validate the mode and record the PSM-Type2 reference hash

**Files:**
- Modify: `src/cenop/parameters/simulation_params.py` (constant near the top; check in `_validate`)
- Create: `scripts/investigations/2026-11-18-dispersal-selector/type2_state_hash.py`
- Test: `tests/test_dispersal_modes.py` (new)

**Interfaces:**
- Produces: `DISPERSAL_MODES: tuple[str, ...] = ("off", "PSM-Type2", "Undirected")` exported from `cenop.parameters.simulation_params`.

- [ ] **Step 1: Write the reference-hash script.** Model it on `scripts/investigations/2026-11-11-lithuania-data-gap/katt_init_hash.py`: Kattegat, `porpoise_count=2000`, `random_seed=42`, `ships_enabled=False`, `turbines="off"`, default dispersal. Run 1440 ticks (30 days). Track `ever = ever | pm.is_dispersing` each tick. Print `state hash` = first 16 hex of sha256 over `np.concatenate([pm.x, pm.y, pm.heading, pm.energy, pm.is_dispersing]).astype(np.float32).tobytes()`, and `ever dispersed` = `int(ever.sum())`.

- [ ] **Step 2: Run it and record the output**

Run: `micromamba run -n shiny python3 scripts/investigations/2026-11-18-dispersal-selector/type2_state_hash.py`
Expected: a hash and `ever dispersed` > 0. If it is 0 the check is vacuous: lengthen the run until it is not. Write both values into the script's docstring as the expected pre-change result.

- [ ] **Step 3: Write the failing tests**

```python
import pytest
from cenop.parameters.simulation_params import DISPERSAL_MODES, SimulationParameters


def test_modes_constant():
    assert DISPERSAL_MODES == ("off", "PSM-Type2", "Undirected")


@pytest.mark.parametrize("mode", ["off", "PSM-Type2", "Undirected"])
def test_valid_modes_accepted(mode):
    assert SimulationParameters(dispersal=mode).dispersal == mode


def test_inner_danish_waters_not_implemented():
    with pytest.raises(ValueError, match="InnerDanishWaters.*not implemented"):
        SimulationParameters(dispersal="InnerDanishWaters")


@pytest.mark.parametrize("bad", ["Off", "undirected", " off", "PSM-Typ2", ""])
def test_unknown_mode_lists_valid_values(bad):
    with pytest.raises(ValueError, match="off.*PSM-Type2.*Undirected"):
        SimulationParameters(dispersal=bad)
```

- [ ] **Step 4: Run to verify they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_modes.py -q`
Expected: FAIL (ImportError on `DISPERSAL_MODES`).

- [ ] **Step 5: Implement.** Add `DISPERSAL_MODES`. In `_validate`: `InnerDanishWaters` raises `ValueError("dispersal 'InnerDanishWaters' is not implemented yet in CENOP; use one of: off, PSM-Type2, Undirected")`; any other value outside `DISPERSAL_MODES` raises `ValueError` naming the bad value and the three valid ones in that order.

- [ ] **Step 6: Run the new tests, then the fast suite**

Run: `micromamba run -n shiny python3 -m pytest tests/ -q`
Expected: all pass. If an existing test or fixture builds parameters with a now-invalid mode, fix the fixture, not the validation, and say so in the commit.

- [ ] **Step 7: Commit** (`feat(params): validate dispersal mode; reject InnerDanishWaters until ported`).

---

### Task 2: `off` blocks new dispersal

**Files:**
- Modify: `src/cenop/agents/population.py` (`__init__`; `_check_dispersal_trigger`)
- Test: `tests/test_dispersal_modes.py`

**Interfaces:**
- Produces: `PorpoisePopulation._dispersal_mode: str`, set once in `__init__` from `params.dispersal`.

- [ ] **Step 1: Write the failing tests.** Reuse the helper style of `tests/test_dispersal_parity.py` (`_pop`, `_disperser`; copy them, do not import across test files).

```python
def _declining(pop):
    # newest day at index 0; strictly dropping for t_disp=3 days
    pop._energy_history[:, :4] = np.array([7.0, 8.0, 9.0, 10.0])


def test_type2_control_starts_dispersal():
    pop = _pop(count=20, dispersal="PSM-Type2")
    _declining(pop)
    pop._check_dispersal_trigger(pop.active_mask)
    assert pop.is_dispersing.any()


def test_off_starts_none():
    pop = _pop(count=20, dispersal="off")
    _declining(pop)
    pop._check_dispersal_trigger(pop.active_mask)
    assert not pop.is_dispersing.any()


def test_off_lets_existing_dispersers_finish():
    pop = _pop(dispersal="off", psm_angle=0.0)
    _disperser(pop, 0, 100.0, 100.0, 90.0, target_dist=500.0, traveled=0.0)
    pop._disperse_step(pop.active_mask)
    assert pop.is_dispersing[0]
    pop.dispersal_distance_traveled[0] = 0.95 * 500.0
    pop._disperse_step(pop.active_mask)
    assert not pop.is_dispersing[0]
```

If `_pop` without a landscape cannot find a random target, `test_type2_control_starts_dispersal` will fail for the wrong reason; `test_random_target_without_landscape` in the parity file shows the working setup.

- [ ] **Step 2: Run to verify `test_off_starts_none` fails** (the other two pass or fail only on the missing attribute).

- [ ] **Step 3: Implement.** Set `self._dispersal_mode = params.dispersal` in `__init__`. First line of `_check_dispersal_trigger`: return if the mode is `"off"`. This one method is the only place dispersal starts, on the NumPy and the JAX day-boundary paths alike.

- [ ] **Step 4: Run** `micromamba run -n shiny python3 -m pytest tests/test_dispersal_modes.py tests/test_dispersal_parity.py tests/test_settings_defaults.py -q`. Expected: PASS.

- [ ] **Step 5: Commit** (`feat(dispersal): dispersal=off starts no dispersal`).

---

### Task 3: `Undirected`

**Files:**
- Modify: `src/cenop/agents/population.py` (`__init__` JAX guard; `_select_dispersal_target`; `_disperse_step` step 1; `_apply_dispersal_heading`; `_update_dispersal` generic-path completion)
- Test: `tests/test_dispersal_modes.py`

**Interfaces:**
- Consumes: `self._dispersal_mode` (Task 2).

Behaviour, from `UndirectedDispersal extends DispersalPSMType3` (see the spec table):

| Method | PSM-Type2 (unchanged) | Undirected |
|---|---|---|
| `_select_dispersal_target` | memory target, else random | random only |
| `_apply_dispersal_heading` | `prev + U(-a, a) * logistic` | `prev + U(-a, a)`; same `rng.uniform` call |
| `_disperse_step` completion | `traveled >= 0.95 * target` | `hypot(x - start_x, y - start_y) >= target` |
| `_update_dispersal` generic path | displacement `>= 0.95 * target` | displacement `>= target` |

- [ ] **Step 1: Write the failing tests**

```python
T = 500.0


def test_undirected_ignores_memory(monkeypatch):
    pop = _pop(dispersal="Undirected")
    monkeypatch.setattr(pop, "_memory_dispersal_target", lambda idx: (1.0, 2.0))
    monkeypatch.setattr(pop, "_random_dispersal_target", lambda idx: (30.0, 40.0))
    assert pop._select_dispersal_target(0) == (30.0, 40.0)


def test_type2_still_prefers_memory(monkeypatch):
    pop = _pop(dispersal="PSM-Type2")
    monkeypatch.setattr(pop, "_memory_dispersal_target", lambda idx: (1.0, 2.0))
    assert pop._select_dispersal_target(0) == (1.0, 2.0)


def _turns(mode):
    pop = _pop(count=2000, dispersal=mode)
    for i in range(2000):
        _disperser(pop, i, 100.0, 100.0, 90.0, target_dist=T, traveled=0.0)
    before = pop._prev_step_heading.copy()
    pop._apply_dispersal_heading(np.ones(2000, dtype=bool))
    return (pop.heading - before + 180) % 360 - 180


def test_undirected_turn_is_undamped():
    # At distance 0 the Type2 logistic is 1/(1+exp(1.5/0.6)) = 0.076
    t2, und = _turns("PSM-Type2"), _turns("Undirected")
    assert np.abs(t2).max() <= 40 * 0.08
    assert 35 < np.abs(und).max() <= 40
    np.testing.assert_allclose(t2, und * 0.0759, atol=0.05)  # same draws, same seed


def test_undirected_stops_on_displacement_not_path():
    pop = _pop(dispersal="Undirected", psm_angle=0.0)
    _disperser(pop, 0, 100.0, 100.0, 90.0, target_dist=T, traveled=0.99 * T)  # long path, no displacement
    _disperser(pop, 1, 100.0, 100.0, 90.0, target_dist=T, traveled=0.0)
    pop.x[1] = 100.0 + T  # displacement reached
    _disperser(pop, 2, 100.0, 100.0, 90.0, target_dist=T, traveled=0.0)
    pop.x[2] = 100.0 + 0.97 * T  # past Type2's 0.95, short of full
    pop._disperse_step(pop.active_mask)
    assert pop.is_dispersing[0] and not pop.is_dispersing[1] and pop.is_dispersing[2]


def test_undirected_generic_path_uses_full_distance():
    params = SimulationParameters(random_seed=1, communication_enabled=False, dispersal="Undirected")
    pop = PorpoisePopulation(4, params, movement_module=JASMINEPhysicsMovement(params))
    _disperser(pop, 0, 100.0, 100.0, 90.0, target_dist=T)
    pop.x[0] = 100.0 + 0.97 * T
    _disperser(pop, 1, 100.0, 300.0, 90.0, target_dist=T)
    pop.x[1] = 100.0 + T
    pop._update_dispersal(pop.active_mask)
    assert pop.is_dispersing[0] and not pop.is_dispersing[1]


def test_undirected_zero_target_completes_without_nan():
    pop = _pop(dispersal="Undirected")
    _disperser(pop, 0, 100.0, 100.0, 90.0, target_dist=0.0)
    pop._disperse_step(pop.active_mask)
    assert not pop.is_dispersing[0] and np.isfinite(pop.heading).all()


def test_undirected_failed_target_leaves_animal_untouched(monkeypatch):
    pop = _pop(dispersal="Undirected")
    monkeypatch.setattr(pop, "_random_dispersal_target", lambda idx: None)
    h = pop.heading[0]
    assert pop._start_dispersal(0) is False
    assert not pop.is_dispersing[0] and pop.heading[0] == h


def test_undirected_refused_on_jax():
    pytest.importorskip("jax")
    with pytest.raises(ValueError, match="Undirected.*JAX"):
        _pop(dispersal="Undirected", use_jax=True)


def test_undirected_runs_when_jax_missing(monkeypatch):
    monkeypatch.setattr(popmod, "_HAS_JAX", False)
    pop = _pop(dispersal="Undirected", use_jax=True)
    assert pop._use_jax is False


def test_off_allowed_on_jax():
    pytest.importorskip("jax")
    _pop(dispersal="off", use_jax=True)
```

The `0.0759` cross-check assumes the two populations draw the same uniform sequence from the same seed; if population construction consumes the RNG differently by mode, that is a bug in the implementation, not the test.

- [ ] **Step 2: Run to verify the Undirected tests fail** and `test_type2_still_prefers_memory` passes.

- [ ] **Step 3: Implement** the four branches in the table. Constraints:
  - In `_apply_dispersal_heading` keep the single `self.rng.uniform(-psm_angle, psm_angle, n)` call for both modes; skip only the logistic multiply.
  - In `__init__`, after `self._use_jax` is decided: if it is true and the mode is `"Undirected"`, raise `ValueError("dispersal 'Undirected' is not supported on the JAX backend; set use_jax=False")`.
  - Update the docstrings of `_disperse_step`, `_apply_dispersal_heading` and `_update_dispersal` to name both modes.

- [ ] **Step 4: Run** `micromamba run -n shiny python3 -m pytest tests/test_dispersal_modes.py tests/test_dispersal_parity.py tests/test_jax_tick.py -q`. Expected: PASS.

- [ ] **Step 5: Commit** (`feat(dispersal): Undirected mode (DEPONS UndirectedDispersal)`).

---

### Task 4: UI, output file, changelog

**Files:**
- Modify: `src/cenop/ui/tabs/settings.py` (`TOOLTIPS["dispersal"]`; `_dispersal_settings_panel` choices)
- Modify: `src/cenop/ui/layout.py:569,575` (help text that says dispersal is always PSM-Type2)
- Modify: `src/cenop/core/output_writer.py:433`
- Modify: `CHANGELOG.md` ([Unreleased])
- Test: `tests/test_settings_defaults.py`, `tests/test_dispersal_modes.py`

**Interfaces:**
- Consumes: `DISPERSAL_MODES` (Task 1).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_settings_defaults.py
def test_dispersal_dropdown_offers_only_working_modes():
    from cenop.parameters.simulation_params import DISPERSAL_MODES
    from cenop.ui.tabs.settings import TOOLTIPS, _dispersal_settings_panel

    html = str(_dispersal_settings_panel())
    for mode in DISPERSAL_MODES:
        assert f'value="{mode}"' in html
    assert 'value="InnerDanishWaters"' not in html
    assert "not available yet" in TOOLTIPS["dispersal"]
```

For the output file, add a test that `OutputWriter._detect_dispersal` records the mode name: build a small `Simulation` with `dispersal="Undirected"`, set `writer._prev_dispersing` to all-false, flip `is_dispersing[0]` to true, call `_detect_dispersal(sim)`, and assert the recorded event's `dispersal_type == "Undirected"`. `tests/test_phase5.py` (around line 202) shows how an `OutputWriter` with `dispersal=True` is built.

- [ ] **Step 2: Run to verify both fail.**

- [ ] **Step 3: Implement.**
  - Dropdown: `choices=list(DISPERSAL_MODES)`, `selected="PSM-Type2"`.
  - Tooltip, exact text: `"Dispersal behavior type: 'off' = no dispersal, 'PSM-Type2' = memory-based with heading dampening, 'Undirected' = random target, undamped turns. 'InnerDanishWaters' (DEPONS Kattegat mode) is not available yet."`
  - `layout.py`: reword the two lines so they say PSM-Type2 is the default mode and name `off` and `Undirected` as the alternatives.
  - `output_writer.py`: `dispersal_type=sim.params.dispersal`.
  - CHANGELOG [Unreleased]: under Fixed, "the dispersal setting was ignored (every run used PSM-Type2)"; under Added, `off` and `Undirected`; under Changed, `InnerDanishWaters` removed from the UI and rejected by parameters until ported, and the dispersal events file now records the mode name instead of `PSM`. Under Known issues, the PSM-Type2 calf-memory difference from the spec's "Found on the way".

- [ ] **Step 4: Run** `micromamba run -n shiny python3 -m pytest tests/test_settings_defaults.py tests/test_dispersal_modes.py tests/test_phase5.py -q`. Expected: PASS.

- [ ] **Step 5: Commit** (`feat(ui): dispersal selector offers only working modes; events file records the mode`).

---

### Task 5: Acceptance

- [ ] **Step 1: PSM-Type2 unchanged.** Run `micromamba run -n shiny python3 scripts/investigations/2026-11-18-dispersal-selector/type2_state_hash.py`. Expected: the `state hash` and `ever dispersed` recorded in Task 1, exactly.
- [ ] **Step 2: Kattegat init hash.** Run `micromamba run -n shiny python3 scripts/investigations/2026-11-11-lithuania-data-gap/katt_init_hash.py`. Expected: `init hash 9ab880c9644d8fc4`.
- [ ] **Step 3: Fast suite.** `micromamba run -n shiny python3 -m pytest tests/ -q`. Expected: 1064 + the new tests pass, 11 deselected.
- [ ] **Step 4: Slow tier.** `micromamba run -n shiny python3 -m pytest tests/ -q -m slow`. Expected: 11 passed, no baseline edits.
- [ ] **Step 5: Lint.** `ruff check src tests validation && black --check --line-length 100 src tests validation`. Expected: clean.
- [ ] **Step 6: Live check.** Run a 30-day Kattegat simulation (N=2000, seed 42) in each of `off` and `Undirected` and report: animals that ever dispersed (`off` must be 0, `Undirected` > 0) and final population. This is the only end-to-end run of the two new modes.
- [ ] **Step 7:** Commit the script docstring update if any; do not merge or push. Report results to the user.
