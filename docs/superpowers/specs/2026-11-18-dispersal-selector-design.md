# Dispersal selector, stage 1: make the setting real

Date: 2026-11-18. Branch: `feat/dispersal-selector`. Status: approved 2026-11-18, implemented.

## Intent

`params.dispersal` is set by the UI and by scripts but nothing in the simulation reads it.
Every run uses PSM-Type2 whatever the setting says. Stage 1 makes the setting take effect
for the modes that are cheap to support, and makes the one that is not (InnerDanishWaters)
fail loudly instead of silently running PSM-Type2. Stage 2, a separate design, ports
InnerDanishWaters.

## Approaches

1. **Two stages (chosen by the user, 2026-11-18).** Wire `off`, `PSM-Type2` and
   `Undirected` now; reject `InnerDanishWaters` until its own port. The UI stops lying
   quickly and the 593-line InnerDanishWaters port gets its own design and validation.
2. **One piece.** All four modes together. One baseline regeneration, but nothing ships
   until InnerDanishWaters is validated against the Java harness.

## Goals

- `dispersal="off"`: no animal ever starts dispersing.
- `dispersal="PSM-Type2"`: bit-identical to today.
- `dispersal="Undirected"`: DEPONS `UndirectedDispersal` behaviour (below).
- `dispersal="InnerDanishWaters"` or any unknown string: rejected when parameters are built.
- The UI offers only what works.

## Non-goals

- Porting InnerDanishWaters (stage 2).
- PSM-Type1, PSM-Type3 and its variants (not in DEPONS `parameters.xml`, not in the UI).
- Regenerating baselines: no default changes, PSM-Type2 results do not move.
- Fixing the calf-memory gap noted under "Found on the way".
- Reworking or deleting the unused per-animal classes in `behavior/dispersal.py`.

## Anti-goals

- Any silent fallback to PSM-Type2. A mode a backend cannot run must raise.
- Any change to the random-number draw order on the PSM-Type2 path.

## How each mode behaves

Source: `DEPONS-3.2/src/dk/au/bios/porpoise/behavior/` (`DispersalFactory`, `DispersalOff`,
`AbstractPSMDispersal`, `DispersalPSMType2`, `DispersalPSMType3`, `UndirectedDispersal`).

**off.** DEPONS skips the daily dispersal check (`Porpoise.java:1089`). In CENOP the only
place dispersal starts is `Population._check_dispersal_trigger`, called from both the NumPy
and the JAX day-boundary code. Returning early there when the mode is `off` covers every
backend.

**Undirected.** `UndirectedDispersal extends DispersalPSMType3`, so it differs from
PSM-Type2 in four ways:

| | PSM-Type2 (today) | Undirected |
|---|---|---|
| Target | best remembered cell in the preferred-distance band, else random | always random; memory never consulted |
| Turn per step | `prev + U(-angle, +angle) * logistic(path fraction)` | `prev + U(-angle, +angle)`, no damping |
| Stop | path length >= 0.95 x target distance | straight-line distance from the start point >= full target distance |
| Calves | calf has its own memory, inherits preferred distance | no calf memory, preferred distance redrawn |

Everything else in `AbstractPSMDispersal.disperse()` is shared and stays as it is: the
not-enough-water, off-grid and too-shallow stops, the fixed step, the energy cost, the
deterrence and energy-recovery stops.

The calf row needs no code: CENOP already redraws the preferred distance and clears the
memory for every newborn (`population.py`, newborn slot reset).

## Design

- **One place decides the mode.** `SimulationParameters._validate` accepts exactly `off`,
  `PSM-Type2`, `Undirected`. `InnerDanishWaters` raises `ValueError` saying it is not
  implemented yet; anything else raises `ValueError` listing the valid values. This covers
  the UI, `batch_runner` and scripts alike.
- **Population (NumPy path).** The mode is read once at construction.
  - `_check_dispersal_trigger`: return early for `off`.
  - `_select_dispersal_target`: skip the memory lookup for `Undirected`.
  - `_disperse_step`: completion test by mode (path length x 0.95, or displacement from
    `dispersal_start_x/y` at full distance).
  - `_apply_dispersal_heading`: skip the logistic factor for `Undirected`. The uniform
    draw is the same call in both modes, so PSM-Type2 draw order is untouched.
  - `_update_dispersal` (non-CRW module path): completion factor 1.0 instead of 0.95 for
    `Undirected`. That path has no per-step dispersal turn in either mode; unchanged.
- **JAX path** (`use_jax=True`, off by default). `jax_disperse_step` embeds the PSM-Type2
  turn and stop. `off` works there through the shared trigger. `Undirected` with
  `use_jax=True` raises at population construction. Not ported, because the future of the
  JAX backend is undecided. *(My call.)*
- **Cython path.** Disabled at construction today (`_use_cython = False`). The plan must
  confirm whether `step()` can still switch it on and, if so, apply the same rule as JAX.
- **UI.** Remove `InnerDanishWaters` from the Dispersal Type dropdown and say in the
  tooltip that it is not available yet. Fix the two help texts in `ui/layout.py` that say
  dispersal is always PSM-Type2. *(My call: removing beats a choice that errors.)*
- **Error display.** A bad value already surfaces as `Error: <message>` in the progress
  line (`server/main.py`, start handler); no change needed.
- **Output.** `output_writer` writes `dispersal_type="PSM"` for every event; write the
  active mode name instead. *(My call; small, and otherwise the file misreports Undirected.)*

## Tests (acceptance)

- **PSM-Type2 unchanged.** Kattegat initial-state hash `9ab880c9644d8fc4` unchanged; a
  short seeded run gives the same state hash before and after the change; fast suite and
  slow tier pass with no baseline edits.
- **off.** In a setup where PSM-Type2 does start dispersal (forced declining energy
  history), `off` starts none. Without that control the test proves nothing.
- **Undirected target.** An animal with 50+ remembered cells and a clear best cell gets a
  random target, not that cell.
- **Undirected turn.** At distance travelled 0 the PSM-Type2 turn is scaled to about 8 %
  of the draw (logistic at -1.5 with `psm_log=0.6`); Undirected gets the full draw. Same
  seed, compare the two.
- **Undirected stop.** Path length past 0.95 x target but displacement short: keeps
  dispersing. Displacement >= target: stops.
- **Validation.** `InnerDanishWaters` and a misspelt value raise; `Undirected` with
  `use_jax=True` raises.
- **UI.** Dropdown choices are exactly `off`, `PSM-Type2`, `Undirected`.

## Left to the builder

- How the mode is stored on the population (string or small enum).
- Whether `agents/porpoise.py` (per-animal class, still imported by `core/simulation.py`)
  is on any live path. If it is, it needs the `off` gate too; if not, leave it.
- Existing files that set a non-default mode and now behave differently:
  `tests/test_settings_defaults.py` (`dispersal="off"`, Homogeneous; should still pass) and
  the phantom-food investigation scenarios (`dispoff*.json`, which are DEPONS-side inputs).
- CHANGELOG entry under [Unreleased].

## Found on the way (not fixed here)

For PSM-Type2, DEPONS gives a nursing calf its own memory, filled as the mother forages,
and the calf inherits the mother's preferred distance (`calfHasPSM` and
`calfInheritsPsmDist` both true). CENOP clears the memory and redraws the distance for
every newborn. That is an existing difference from DEPONS in the default mode. Fixing it
would change PSM-Type2 results, so it needs its own decision and a baseline regeneration.
