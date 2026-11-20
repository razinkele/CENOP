# Tick profile and the top two costs (roadmap B2)

Date: 2026-11-20. Scripts: `scripts/investigations/2026-11-20-tick-profile/`.

## Result

The default (NumPy/Numba) tick on Kattegat is 22-42% faster, with identical results: the four
30-day state hashes are unchanged and the fast suite passes.

| Run (Kattegat, seed 42, days 10-20) | Before | After | Change |
|---|---|---|---|
| 2000 animals, InnerDanishWaters | 8.50 ms | 4.94 ms | -42% |
| 2000 animals, PSM-Type2 | 9.59 ms | 7.41 ms | -23% |
| 500 animals, InnerDanishWaters | 2.58 ms | 1.88 ms | -27% |
| 500 animals, PSM-Type2 | 2.89 ms | 2.26 ms | -22% |

Time per tick, best of three blocks of 480 ticks, no ships or turbines. "Before" is commit
`d1923cf`. The machine is shared and was busy; the 2000-animal InnerDanishWaters pair was
repeated three times and agreed within 0.02 ms, the other rows were measured once.

## Where the time went (2000 animals, InnerDanishWaters, cProfile over 960 ticks)

| Part of the tick | Before | After |
|---|---|---|
| InnerDanishWaters dispersal step | 33% | 19% |
| Reference memory: attraction vector | 18% | 11% |
| Reference memory: expected food value | 15% | 8% |
| Land check | 12% | 20% |
| Movement glue (`_update_movement` itself) | 5% | 10% |
| Daily food regrowth | 2% | 4% |

Shares of profiled time. The "before" profile was taken while the machine was heavily
loaded, so its shares are rougher than the "after" ones.

## What was changed

- **InnerDanishWaters dispersal step.** The step called the compiled per-animal function
  from a Python loop, once per dispersing animal (about half the population in the first
  weeks). The loop is now inside a compiled function, `dispersal_idw.disperse_many`. A test
  checks it returns exactly what the per-animal calls return.
- **Reference memory.** Both kernels computed the ring-buffer index with a modulo for each
  of the 120 remembered positions of each animal. The index is now stepped down and wrapped,
  which reads the same entries in the same order.

## What was tried and rejected

- **Running the reference-memory kernels in parallel.** Results stayed identical, but the
  tick was slower (16 ms against 9 ms on the busy machine). The work per call is too small
  to repay waking the threads. The kernels stay serial.

## What is left

- **Land check (20%).** It draws random numbers inside the kernel, in animal order, so it
  cannot be run in parallel without changing seeded results.
- **Ships.** Not profiled here. A two-year ship run takes about 45 minutes against a few
  minutes without ships; the ship loop is the dominant cost of any run with ships, and a
  batched form would not be bit-identical.
- **Against DEPONS.** The roadmap's figure for DEPONS-Java (0.84 ms per tick at 500 animals)
  dates from June 2026 and was not re-measured, so the current ratio to Java is not known.
  CENOP's own June figure (2.22 ms at 500 animals) predates the parity fixes of November,
  which added work to the tick (land check, daily step).
