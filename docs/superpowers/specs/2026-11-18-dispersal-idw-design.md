# Dispersal selector, stage 2: InnerDanishWaters

Date: 2026-11-18. Branch: `feat/dispersal-idw`. Status: approved 2026-11-18, implemented.
Follows `2026-11-18-dispersal-selector-design.md` (stage 1, merged `0087b8f`).

## Intent

The DEPONS 3.2 TRACE document (section 3.2.4) says Kattegat simulations use the
`InnerDanishWaters` dispersal. CENOP rejects that mode today, so every CENOP Kattegat run
is PSM-Type2 and cannot be compared like for like with canonical DEPONS Kattegat output.
Stage 2 ports the mode so that comparison (the canonical half of roadmap item A2) becomes
possible.

## Approaches

1. **Scalar per-animal port in its own module (chosen by the user, 2026-11-18).** A near
   line-for-line transliteration of the Java, one animal at a time, compiled with Numba
   when available. Each Java line has a visible Python counterpart; cost is a per-animal
   loop over the dispersing minority.
2. **Vectorised NumPy branches in `population.py`.** One style throughout, but the tie
   rules, the repeated heading adjustment and the phase change within a step must be
   re-expressed as array operations, which is where a port drifts unnoticed.
3. **Wire up the per-animal class in `behavior/dispersal.py`.** It covers target selection
   only and keeps state per object; nothing on the live path works that way.

## Goals

- `dispersal="InnerDanishWaters"` runs on the Kattegat landscape and reproduces
  `InnerDanishWatersDispersal.java` (DEPONS 3.2, tag `8f533f8`) step for step.
- The mode is opt-in. PSM-Type2 stays the default on every landscape *(user decision,
  2026-11-18)*.
- `off`, `PSM-Type2` and `Undirected` runs are bit-identical to today.
- A recorded comparison against DEPONS-Java InnerDanishWaters runs, with its limits stated.

## Non-goals

- Changing any default, or regenerating any baseline.
- The Homogeneous landscape. Java allows it, but the navigation grid is 600 x 1000 cells
  and CENOP's homogeneous world is 1000 x 1000 by default. *(My call: Kattegat only for
  now; say so in the error message.)*
- The JAX backend: the mode raises there, as `Undirected` does. *(My call.)*
- Fixing the PSM-Type2 calf-memory gap or the `Dispersal.txt` km bug (both recorded in
  stage 1).
- Deleting the unused classes in `behavior/dispersal.py`.

## Anti-goals

- "Improving" DEPONS. The quirks listed below are copied, not fixed.
- Any silent fallback: a landscape or backend that cannot run the mode raises.
- Any change to the random-number draw order of the other three modes.

## What the mode does

Source: `DEPONS-3.2/src/dk/au/bios/porpoise/behavior/InnerDanishWatersDispersal.java`,
plus `Porpoise.java` for the hooks. All distances are in grid cells (400 m) unless said.

**Shared with the other modes (already in CENOP, unchanged):** the daily trigger (energy
dropping `t_disp` days in a row), the daily energy-recovery stop, the deterrence stop, and
the rule that a dispersing animal skips the normal move.

**Two numberings.** Target blocks are indexed 0-59 from the north-west (the centre and
quality tables). The navigation grid holds index + 1, so its values run 1-60: value 32 is
the block with index 31. Every "navigation block" test below (14, 31, 32, "below 20") uses
the grid value; every target pick uses the index.

**Start (`activate` -> `dispTargetSelect`).** Sixty 40 km blocks with fixed centres and
fixed quality values. Distances to block centres are rounded. Blocks whose centre is nearer than
250 cells (100 km) get quality 0. The twelve highest-quality blocks are found with Java's
sort-and-match loop, and one is drawn uniformly. If the animal's navigation block (grid value) is below
20 and the pick is index 30, 31, 36, 37 or 43, the pick moves two blocks east. The target
is that block's centre. The animal enters phase 1. Unlike PSM, a start cannot fail.

**Look-ahead distances.** The Java uses `mean_disp_dist` (2) directly as a number of
cells for its look-ahead points, and `mean_disp_dist / 0.4` (5 cells) only for the move. So
"2 cells ahead" below is `mean_disp_dist`, 4 is twice that, 16 is eight times that.

**Phase 1 (`disp1`), per tick, in this order:**
1. South-east of Sealand (x > 438 and y < 478): switch to phase 2 and leave phase 1.
2. Navigation block 14 or 31: same.
3. Navigation block 32: heading 0, switch to phase 2, move 1 cell north (no energy cost),
   leave phase 1.
4. Distance to the target, rounded. Face the target.
5. Read depth at five points 2 cells ahead (-20 to +20 degrees) and at seven points 16
   cells ahead (-30 to +30). Turn to the deepest of the five (last on ties).
6. From the new heading, read distance to coast at seven points 4 cells ahead (-30 to
   +30). Discard directions whose 16-cell depth was not above 0; that depth was read
   before the turn in step 5 and is matched by list position, not by direction. Choose the
   largest remaining (last on ties) and turn by that angle. If every direction is
   discarded the turn is +40 degrees. This step always runs: see the quirks.
7. Switch to phase 2 if any holds: less than 2 cells from the daily position one before
   the latest (`posListDaily.get(1)`); less than 6 cells from the one at index 8; coast
   nearer than 2000 m at the current position; target nearer than 50 cells (distance from
   step 4); depth not above 0 at any of 4, 6 and 8 cells straight ahead.
8. Not enough water ahead: stop dispersing (this overrides a switch made in step 7).
9. If still in phase 1: move 5 cells. If the depth there is not >= 0, step back and stop
   dispersing. Pay the step's energy cost (`0.001 * E_USE_PER_KM * 5`) in both cases.

**Phase 2 (`disp2`), per tick:**
1. Face the daily position at index 1, then turn 180 degrees.
2. Read distance to coast at 17 points 2 cells ahead (-80 to +80 degrees) and at the
   current position. For each point take the absolute difference from the current value.
3. Turn by: the angle with the smallest difference (last on ties) if the coast is farther
   than 4000 m or between 2000 and 4000 m inclusive; the angle with the largest difference
   (first on ties) if nearer than 1000 m; otherwise (1000 m up to but not including
   2000 m) by the sentinel, -9999 degrees.
4. Stop dispersing if there is not enough water ahead, or if the depth 5 cells ahead is
   not above `min_depth` (wmin).
5. Otherwise move 5 cells and pay the step's energy cost.

**Order within a tick.** `disperse()` runs phase 1 if the animal is in phase 1, then
phase 2 if the animal is in phase 2 by then. Every switch to phase 2 made in phase 1,
including the early exits in steps 1-3, is followed by a phase-2 move in the same tick.
The port copies `disperse()` exactly rather than this summary.

**Daily step order.** The daily position is pushed before the trigger and the
energy-recovery stop are evaluated (`Porpoise.java:1087-1118`).

**After a stop.** The mode never restores the heading: an animal that stops keeps the
turned heading (in phase 2 that includes the 180-degree flip), unlike PSM. Because it is
no longer dispersing, it makes its normal move in the same tick (`Porpoise.java:246, 295`).
A stop in step 8 of phase 1 means no phase-2 move that tick.

**No other stops.** There is no completion test and no off-grid stop. Dispersal ends only
through the stops above, the daily energy recovery, or deterrence.

**Energy.** `E_USE_PER_KM` is 0.0 in DEPONS (`SimulationConstants.java:54`) and in CENOP,
so the step cost is zero. The port still computes it through the same constant.

**Land values.** The tie and padding behaviour depends on land reading as -9999 for depth
and coast distance, as in the Java. The port must read the grids so that this holds.

**Calves.** No calf memory, preferred distance not inherited; CENOP's newborn reset
already does this.

### DEPONS quirks that are copied

- The quarter-1 block-quality table is used all year; the other three tables are dead code.
- The top-twelve loop matches blocks by quality value, so equal values (the zeros in
  particular) can put the same block in the list more than once.
- Phase 1's "shallow water ahead" test reads a padding entry (-999) of the depth list, so
  it is always true and the coast-distance turn in step 6 runs every tick.
- Step 6 filters directions with depths read before the step 5 turn.
- Ties go to the last candidate, except the "largest difference" in phase 2 (first).
- Phase 2 has two names for the same choice (smallest difference), and no rule for a coast
  distance from 1000 m up to 2000 m, where the heading changes by -9999 degrees.
- The daily position buffer starts filled with (0, 0), so the index-1 and index-8 reads
  (the two "moved too little" tests and the phase-2 "face away" direction) see the origin
  during an animal's first days.
- Look-ahead points use kilometres as if they were cells (2 instead of 5).
- When every real candidate in a phase-1 list is -999 or lower (on land, or no water far
  ahead), the trailing padding entry wins and the turn is +40 degrees.
- `forward` reflects at the grid border, so the step back in phase 1 is not always an
  exact undo.
- The target draw is `(int)(u * 12)`.
- Stops inside the mode set the phase to 0 without clearing the target.

## Design

- **New module `src/cenop/agents/dispersal_idw.py`.** Constants (block centres, the
  quarter-1 quality table), `select_target`, and one `disperse` function that takes one
  animal's state and the landscape arrays and returns its new phase, heading, position
  change and energy cost. Plain Python that Numba compiles when present; the two must
  give identical results.
- **Navigation grid.** The Java resource `InnerDanishWatersDispersalNavigationBlocks.asc`
  is byte-identical to `data/Kattegat/blocks.asc`, which CENOP already loads as the
  landscape's blocks. The mode reads that; nothing is copied. *(My call.)* With CENOP's
  row order every grid value k lies inside the square of index k - 1, so the orientation
  is settled.
- **Position to cell.** Java rounds a position to a cell with the space origin at 0.5;
  CENOP truncates. The mode uses CENOP's existing rule for every lookup, like the rest of
  the model, and the builder establishes from the earlier parity work whether CENOP
  coordinates carry a half-cell offset against Java's. If they do, the hard-coded
  thresholds (x > 438, y < 478) and block centres are shifted to match and the reason is
  recorded. *(My call.)*
- **Random stream.** The target draw comes from the population's generator inside the
  daily trigger loop. The other modes never reach it, so their draw order is unchanged.
- **New population state.**
  - Dispersal phase per animal (0 none, 1, 2). `is_dispersing` stays the public flag and
    is true for both phases.
  - The last 10 daily positions per animal, pushed at the day boundary where the energy
    history is pushed, initialised to (0, 0), reset for newborns.
- **Population wiring.** In `_disperse_step`, the `InnerDanishWaters` mode loops over the
  dispersing animals and calls the module; the PSM code is untouched. `_start_dispersal`
  calls `select_target` for this mode. `_end_dispersal` clears the phase.
- **Validation.** `DISPERSAL_MODES` gains `InnerDanishWaters`. `_validate` raises unless
  `landscape == "Kattegat"`. Population construction raises for `use_jax=True` and for a
  non-CRW movement module *(my call: stage 1 let `Undirected` run half-applied on those
  modules with a caveat; for this mode a half-applied version would be meaningless)*.
- **UI.** The dropdown offers the mode again; the tooltip says Kattegat only. A wrong
  landscape surfaces through the existing `Error: <message>` line.
- **Output.** `Dispersal.txt` already writes the mode name.

## Tests (acceptance)

- **Other modes unchanged.** `type2_state_hash.py` gives `aec15387d1425dd9` for
  PSM-Type2; the `off` and `Undirected` hashes recorded in stage 1 are unchanged; fast
  suite and slow tier pass with no baseline edits.
- **Target selection.** Hand-worked cases from the Java: a position where the 100 km
  exclusion changes the top twelve; the east shift (navigation grid value < 20, pick
  index 30 -> 32); a repeated block in the list; the draw index. One real Kattegat
  position pinned to its navigation grid value.
- **Phase 1.** One test per step in the list above, on small synthetic depth and
  coast-distance grids, including each quirk (step 6 always runs, stale depth filter,
  +40 when all directions are discarded, last tie wins, stop overriding a switch).
- **Phase 2.** Each coast-distance band, the sentinel band, both stops.
- **Same-tick phase change.** A phase-1 animal that switches in step 7 moves by the
  phase-2 rule in that tick.
- **Numba = Python.** The compiled and uncompiled functions agree on a few thousand
  random states.
- **Guards.** Non-Kattegat landscape, JAX, and a non-CRW movement module each raise.
- **End to end.** A 30-day Kattegat run: animals disperse, none ends on land, no NaN.
- **Stops.** A stopped animal keeps its turned heading and makes a normal move in the
  same tick.
- **Against Java.** Harness runs of the existing `kattegat-idw-undisturbed-v1` scenario
  (2000 animals, 5 years, seeds 42 and 43; the only run on disk is a 200-animal, 1-year
  trial). The plan sets how many seeds. Compare population trajectory and the per-block
  distribution with CENOP runs of the same setup. Random streams differ, so this is a
  distribution comparison across seeds, not a match. The result is written up whichever
  way it comes out; a clear mismatch is a finding to report, not something to tune away.

## Left to the builder

- How the disperser's move is applied. Today the heading kernel moves every dispersing
  animal by the fixed step and adds to its path length; this mode must bypass that, and
  can move twice in a tick (1 cell north of Funen, then a phase-2 step).
- The CENOP equivalent of `isEnoughWaterAhead` (PSM uses `_enough_water_ahead`, frozen
  while dispersing) and of `getPointAtHeadingAndDist` at the grid border.
- Depth and coast-distance conventions (sign, units, NODATA) against the Java getters.
- Whether the loop over dispersers costs enough to matter; measure on Kattegat N=2000.
- CHANGELOG entry; `docs/USER_GUIDE.md` line on dispersal modes.

## Open risks

- Earlier parity work found CENOP's Kattegat data is the DEPONS v3.0 landscape. The Java
  comparison must use the same landscape, or differences cannot be attributed to the port.
- The half-cell question under "Position to cell" moves every hard-coded threshold if it
  is answered wrongly; the comparison with Java is the check on it.
