# Ship-Deterrence Vectorized-Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CENOP's production (vectorized) ship deterrence implement the DEPONS 3.2 ship *response* model (Tships gate, probabilistic reaction + magnitude, 10 km cap, unit-vector × magnitude, loudest-ship-wins), replacing the turbine model currently misapplied to ships.

**Architecture:** A single shared kernel `ShipDeterrenceModel.deterrence_components` computes prob/magnitude/react/vector from arrays; both the scalar oracle and the vectorized production path call it. Correctness is asserted against hand-computed DEPONS values, not against the (previously buggy) scalar output.

**Tech Stack:** Python 3, NumPy. Tests: pytest. Env: `micromamba run -n shiny`. DEPONS Java reference: `DEPONS-3.2/src/dk/au/bios/porpoise/Ship.java`, `ships/ShipDeterrence.java`.

**Spec:** `docs/superpowers/specs/2026-06-03-ship-deterrence-vectorized-port-design.md` (v2)

**Revised v2 after a four-angle plan review (verified against code):**
- Added a `_force_u` test seam — the DEPONS ship-response probability caps ~0.2, so reaction-dependent semantics (loudest-ship-wins, order invariance, no-deter_coeff) were **vacuous** (every seeded draw gave `react=False`); they now force a reaction and assert non-zero first.
- Task 6 fixed: there are **three** L1 `deter_strength` sites in `population.py` (1033, 1236, 2192) plus the JAX kernel (`jax_kernels.py:371`) and its pinning test (`test_jax_tick.py`) — all switched to L2.
- Moved the superseded-test rewrite into Task 3 (was Task 7) so every commit stays green; removed orphaned imports in the same commits to avoid ruff F401 failures.
- Task 4 red step now genuinely fails-first (tick-varies-draws) + a sequential-run reproducibility guard (no global-RNG cross-talk).
- NODATA→RL=0 now also tests salinity and `TL≤0`; documented the intentional RNG-stream divergence; added u==prob / day-night / on-ship / NODATA edge tests and a pinned characterization snapshot; added baseline provenance.

---

## Conventions

- **Run dir:** all commands assume `cd /home/razinka/cenjas/CENOP`.
- **Test prefix:** `eval "$(micromamba shell hook --shell bash)" && micromamba activate shiny && cd /home/razinka/cenjas/CENOP && python3 -m pytest <args>`.
- **CENOP is a nested git repo** — commit from inside `CENOP/`, branch `CENOP-JASMINE`.
- New tests go in `tests/test_ship_deterrence_port.py` unless stated otherwise.

## File Structure

- **Modify** `src/cenop/behavior/sound.py`
  - `ShipDeterrenceModel.calculate_deterrence_probability` / `_magnitude`: make array-capable; add the missing `exp` link to magnitude (DEPONS `predictMag`).
  - Add method `ShipDeterrenceModel.deterrence_components(...)` — the shared kernel.
- **Modify** `src/cenop/agents/ship.py`
  - Add module constant `MAX_DETER_DIST_M = 10_000.0`.
  - Rewrite `ShipManager.calculate_aggregate_deterrence_vectorized` onto the kernel (max-RL-ship, 10 km cap + 100 m floor, NODATA→RL=0, identity-seeded RNG). Delete the turbine-model block (`str_val`, `deter_probabilistic`, `prob_response`, `response_probability_from_rl`).
  - Refactor scalar `Ship.calculate_deterrence` + `ShipManager.calculate_aggregate_deterrence` onto the kernel; fix the min-distance boundary to DEPONS strict `>`.
- **Modify** `src/cenop/core/simulation.py`
  - Thread `base_seed` + `tick` into the vectorized ship call.
- **Modify** `src/cenop/agents/population.py`
  - `deter_strength`: L1 (`|dx|+|dy|`) → L2 (`hypot`), at both sites (~line 1033 and ~line 1236).
- **Modify** `tests/test_deterrence.py`
  - Rewrite `test_no_normalization_vectorized_ship` (it pins the removed turbine-on-ships formula). Keep the turbine version.
- **Create** `tests/test_ship_deterrence_port.py` — kernel, gate, vector, RNG, integration, characterization tests.
- **Regenerate** `output/kattegat_ref_ships/` and update provenance + investigation-doc status.

---

## Task 1: Add `exp` link + array support to `ShipDeterrenceModel` (oracle fix O1)

**Files:**
- Modify: `src/cenop/behavior/sound.py` (`calculate_deterrence_probability` ~line 296, `calculate_deterrence_magnitude` ~line 329)
- Test: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ship_deterrence_port.py
import numpy as np
import pytest
from cenop.behavior.sound import ShipDeterrenceModel


class TestModelExpAndArrays:
    def test_magnitude_uses_exp_link_day(self):
        """DEPONS predictMag returns exp(Mag) (Ship.java:395), not the raw linear term."""
        m = ShipDeterrenceModel()
        # Hand-computed DEPONS day magnitude at RL=100 dB, dist=5 km:
        std = m.STD_MAG_DAY
        ns = (100.0 - std['noise_mean']) / std['noise_sd']
        ds = (5.0 - std['dist_mean']) / std['dist_sd']
        linear = (m.cship_int_day + m.cship_noise_day * ns
                  + m.cship_dist_day * ds + m.cship_dist_x_noise_day * ns * ds)
        expected = np.exp(linear)
        got = m.calculate_deterrence_magnitude(100.0, 5.0, is_day=True)
        assert got == pytest.approx(expected), f"magnitude should be exp(Mag)={expected}, got {got}"

    def test_probability_and_magnitude_accept_arrays(self):
        """Both model functions must vectorize over arrays for the kernel."""
        m = ShipDeterrenceModel()
        rl = np.array([90.0, 110.0, 130.0])
        dist_km = np.array([1.0, 3.0, 8.0])
        p = m.calculate_deterrence_probability(rl, dist_km, is_day=True)
        mag = m.calculate_deterrence_magnitude(rl, dist_km, is_day=True)
        assert p.shape == (3,) and mag.shape == (3,)
        assert np.all((p >= 0.0) & (p <= 1.0))
        assert np.all(mag > 0.0)  # exp(x) > 0 always
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "ExpAndArrays" -v`
Expected: FAIL — `test_magnitude_uses_exp_link_day` (current returns raw linear), and array test fails because `float(...)` wraps a scalar.

- [ ] **Step 3: Make the implementation array-capable + add exp**

In `src/cenop/behavior/sound.py`, change the **return** of `calculate_deterrence_probability` from:

```python
        linear_clipped = np.clip(linear, -500, 500)
        prob = 1.0 / (1.0 + np.exp(-linear_clipped))
        return float(np.clip(prob, 0.0, 1.0))
```

to (drop `float()` so arrays pass through; scalars become `np.float64`):

```python
        linear_clipped = np.clip(linear, -500, 500)
        prob = 1.0 / (1.0 + np.exp(-linear_clipped))
        return np.clip(prob, 0.0, 1.0)
```

And change the **return** of `calculate_deterrence_magnitude` from:

```python
        return max(0.0, magnitude)
```

to (DEPONS `MagTrans = Math.exp(Mag)`; clip exponent for overflow safety):

```python
        return np.exp(np.clip(magnitude, -50.0, 50.0))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "ExpAndArrays" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Confirm no regression in existing ship tests**

Run: `... python3 -m pytest tests/test_deterrence.py tests/test_depons_deterrence.py -q`
Expected: PASS (these assert prob ordering/gates, not magnitude value).

- [ ] **Step 6: Commit**

```bash
cd /home/razinka/cenjas/CENOP
git add src/cenop/behavior/sound.py tests/test_ship_deterrence_port.py
git commit -m "fix: ShipDeterrenceModel magnitude uses exp link + array support (DEPONS predictMag)"
```

---

## Task 2: Add the shared kernel `deterrence_components`

**Files:**
- Modify: `src/cenop/behavior/sound.py` (add method to `ShipDeterrenceModel`, after `calculate_deterrence_magnitude`)
- Test: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Write the failing test**

```python
class TestDeterrenceComponents:
    def _model(self):
        return ShipDeterrenceModel()

    def test_gate_excludes_rl_at_or_below_tships(self):
        m = self._model()
        rl = np.array([80.0, 80.0001])
        dist_m = np.array([2000.0, 2000.0])
        gdx = np.array([5.0, 5.0]); gdy = np.array([0.0, 0.0])
        u = np.array([0.0, 0.0])  # always reacts if gated & prob>0
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, is_day=True, u_draw=u, tships=80.0)
        assert react[0] == False and react[1] == True  # strict > 80

    def test_reacts_iff_u_below_prob(self):
        m = self._model()
        rl = np.array([130.0, 130.0]); dist_m = np.array([1500.0, 1500.0])
        gdx = np.array([3.0, 3.0]); gdy = np.array([0.0, 0.0])
        p = m.calculate_deterrence_probability(130.0, 1.5, True)
        u = np.array([float(p) - 1e-6, float(p) + 1e-6])
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, u, tships=80.0)
        assert react[0] == True and react[1] == False  # strict u < prob

    def test_vector_is_grid_disp_over_metre_distance_times_mag(self):
        """DEPONS Ship.java:231-235: unit vector = grid displacement / metre distance."""
        m = self._model()
        rl = np.array([130.0]); dist_m = np.array([2000.0])
        gdx = np.array([5.0]); gdy = np.array([0.0])  # 5 cells east
        u = np.array([0.0])
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, u, tships=80.0)
        assert react[0]
        assert vx[0] == pytest.approx((5.0 / 2000.0) * mag[0])
        assert vy[0] == pytest.approx(0.0)
        assert vx[0] > 0  # pushed east, away from ship

    def test_zero_vector_when_not_reacting(self):
        m = self._model()
        rl = np.array([130.0]); dist_m = np.array([2000.0])
        gdx = np.array([5.0]); gdy = np.array([0.0])
        u = np.array([1.0])  # never reacts
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, u, tships=80.0)
        assert react[0] == False
        assert vx[0] == 0.0 and vy[0] == 0.0
        assert mag[0] > 0.0  # magnitude still computed, just not applied
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "DeterrenceComponents" -v`
Expected: FAIL — `AttributeError: 'ShipDeterrenceModel' object has no attribute 'deterrence_components'`.

- [ ] **Step 3: Implement the kernel**

Add to `ShipDeterrenceModel` in `src/cenop/behavior/sound.py`:

```python
    def deterrence_components(
        self,
        rl: np.ndarray,
        dist_m: np.ndarray,
        grid_dx: np.ndarray,
        grid_dy: np.ndarray,
        is_day: bool,
        u_draw: np.ndarray,
        tships: float,
    ):
        """DEPONS ship deterrence per porpoise for ONE ship (vectorized).

        Args (all arrays shape (N,) except is_day/tships):
            rl       received level (dB), already clamped >= 0
            dist_m   porpoise<->ship distance (m), already clamped >= 1
            grid_dx  (porpoise_x - ship_x) in GRID/cell units
            grid_dy  (porpoise_y - ship_y) in GRID/cell units
            u_draw   uniform(0,1) draws for the Bernoulli reaction
            tships   minimum RL (dB) to react (deter_ships_min_db)

        Returns (vx, vy, prob, mag, react) arrays. Vector is
        DEPONS unit-vector (grid displacement / metre distance) x magnitude,
        zeroed where the porpoise does not react. No deter_coeff (turbine-only).
        """
        rl = np.asarray(rl, dtype=np.float64)
        dist_m = np.asarray(dist_m, dtype=np.float64)
        dist_km = dist_m / 1000.0
        prob = np.asarray(self.calculate_deterrence_probability(rl, dist_km, is_day), dtype=np.float64)
        mag = np.asarray(self.calculate_deterrence_magnitude(rl, dist_km, is_day), dtype=np.float64)
        gate = rl > tships
        react = gate & (np.asarray(u_draw, dtype=np.float64) < prob)
        eff_mag = np.where(react, mag, 0.0)
        vx = (np.asarray(grid_dx, dtype=np.float64) / dist_m) * eff_mag
        vy = (np.asarray(grid_dy, dtype=np.float64) / dist_m) * eff_mag
        return vx, vy, prob, mag, react
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "DeterrenceComponents" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/cenop/behavior/sound.py tests/test_ship_deterrence_port.py
git commit -m "feat: shared ship-deterrence kernel (DEPONS response model, unit-vector x mag)"
```

---

## Task 3: Rewrite the vectorized production path onto the kernel

**Files:**
- Modify: `src/cenop/agents/ship.py` (add `MAX_DETER_DIST_M` constant near top ~line 36; rewrite `calculate_aggregate_deterrence_vectorized` ~lines 466-602)
- Test: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestVectorizedPath:
    def _mgr_with_ship(self, sx=50.0, sy=50.0, sl=170.0):
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        s = Ship(id=1, x=sx, y=sy, vessel_type=VesselClass.CARGO)
        s._is_active = True
        s.noise.base_source_level = sl
        mgr = ShipManager([s]); mgr.enabled = True
        return mgr, s

    def _params(self):
        from cenop.parameters.simulation_params import SimulationParameters
        return SimulationParameters()

    def test_10km_cap_boundary(self):
        """Default deter_max_distance=1000 km, but ships are capped at 10 km."""
        mgr, s = self._mgr_with_ship()
        p = self._params()
        # 9.9 km east = 24.75 cells (400 m/cell); 10.1 km = 25.25 cells
        near_x = np.array([50.0 + 9900.0 / 400.0]); y = np.array([50.0])
        far_x = np.array([50.0 + 10100.0 / 400.0])
        dxn, _ = mgr.calculate_aggregate_deterrence_vectorized(near_x, y, p, base_seed=1, tick=1)
        dxf, _ = mgr.calculate_aggregate_deterrence_vectorized(far_x, y, p, base_seed=1, tick=1)
        # near may or may not deter (RL/prob), but far MUST be exactly zero (out of cap)
        assert dxf[0] == 0.0

    def test_deter_max_distance_tightens_cap(self):
        mgr, s = self._mgr_with_ship()
        p = self._params(); p.deter_max_distance = 5.0  # km -> cap = 5 km
        x = np.array([50.0 + 6000.0 / 400.0]); y = np.array([50.0])  # 6 km
        dx, _ = mgr.calculate_aggregate_deterrence_vectorized(x, y, p, base_seed=1, tick=1)
        assert dx[0] == 0.0  # beyond 5 km cap

    def test_min_distance_floor(self):
        mgr, s = self._mgr_with_ship()
        p = self._params()  # deter_min_distance_ships = 0.1 km = 100 m
        x99 = np.array([50.0 + 99.0 / 400.0]); y = np.array([50.0])   # 99 m
        dx99, _ = mgr.calculate_aggregate_deterrence_vectorized(x99, y, p, base_seed=1, tick=1)
        assert dx99[0] == 0.0  # inside the 100 m floor -> excluded

    def test_deter_coeff_does_not_affect_ship_vector(self):
        """Ships must NOT use deter_coeff (turbine-only). _force_u=0 guarantees a reaction
        (the DEPONS ship prob caps ~0.2, so seeded draws can't be relied on to react)."""
        mgr, s = self._mgr_with_ship(sl=200.0)
        x = np.array([50.0 + 2000.0 / 400.0]); y = np.array([50.0])  # 2 km
        p1 = self._params(); p1.deter_coeff = 0.012
        p2 = self._params(); p2.deter_coeff = 0.5
        dx1, _ = mgr.calculate_aggregate_deterrence_vectorized(x, y, p1, _force_u=0.0)
        dx2, _ = mgr.calculate_aggregate_deterrence_vectorized(x, y, p2, _force_u=0.0)
        assert dx1[0] != 0.0                       # precondition: actually deterred (non-vacuous)
        assert dx1[0] == pytest.approx(dx2[0])     # deter_coeff has no effect on ships

    def test_loudest_ship_wins_not_sum(self):
        """Two ships near one porpoise -> only the higher-RL ship contributes (DEPONS recordStep)."""
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        p = self._params()
        # Loud ship close (west), quiet ship far (east). Porpoise between, nearer the loud one.
        loud = Ship(id=1, x=48.0, y=50.0, vessel_type=VesselClass.CARGO); loud._is_active = True
        loud.noise.base_source_level = 195.0
        quiet = Ship(id=2, x=60.0, y=50.0, vessel_type=VesselClass.CARGO); quiet._is_active = True
        quiet.noise.base_source_level = 175.0
        px = np.array([50.0]); py = np.array([50.0])
        mgr = ShipManager([loud, quiet]); mgr.enabled = True
        mgr_loud = ShipManager([loud]); mgr_loud.enabled = True
        mgr_quiet = ShipManager([quiet]); mgr_quiet.enabled = True
        dx_both, _ = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        dx_loud, _ = mgr_loud.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        dx_quiet, _ = mgr_quiet.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        assert dx_loud[0] != 0.0 and dx_quiet[0] != 0.0     # both would deter alone
        assert dx_both[0] == pytest.approx(dx_loud[0])      # loudest wins...
        assert dx_both[0] != pytest.approx(dx_loud[0] + dx_quiet[0])  # ...NOT a sum

    def test_order_and_membership_invariance(self):
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        p = self._params()
        a = Ship(id=1, x=49.0, y=50.0, vessel_type=VesselClass.CARGO); a._is_active = True; a.noise.base_source_level = 195.0
        b = Ship(id=2, x=51.0, y=50.0, vessel_type=VesselClass.CARGO); b._is_active = True; b.noise.base_source_level = 190.0
        far = Ship(id=3, x=500.0, y=900.0, vessel_type=VesselClass.CARGO); far._is_active = True; far.noise.base_source_level = 195.0
        px = np.array([50.0]); py = np.array([50.0])
        ab = ShipManager([a, b]); ab.enabled = True
        ba = ShipManager([b, a]); ba.enabled = True
        abfar = ShipManager([a, b, far]); abfar.enabled = True
        r_ab = ab.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        r_ba = ba.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        r_abfar = abfar.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        assert r_ab[0][0] != 0.0                            # precondition: non-vacuous
        assert r_ab[0][0] == pytest.approx(r_ba[0][0])      # order invariant
        assert r_ab[0][0] == pytest.approx(r_abfar[0][0])   # far out-of-range ship has no effect
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "VectorizedPath" -v`
Expected: FAIL — current signature lacks `base_seed`/`tick`; far ship still in 1000 km range; `deter_coeff` affects vector; sums ships.

- [ ] **Step 3: Add the module constant**

In `src/cenop/agents/ship.py`, after the imports block (~line 36, before `_compute_tl_percell`), add:

```python
# DEPONS Ship.java:51 — ship deterrence is hard-capped at 10 km regardless of dmax_deter.
MAX_DETER_DIST_M = 10_000.0
```

- [ ] **Step 4: Rewrite `calculate_aggregate_deterrence_vectorized`**

Replace the whole method body (the loop using `str_val`/`deter_probabilistic`/`prob_response`) with:

```python
    def calculate_aggregate_deterrence_vectorized(
        self,
        porpoise_x: np.ndarray,
        porpoise_y: np.ndarray,
        params: "SimulationParameters",
        is_day: bool = True,
        cell_size: float = 400.0,
        cell_data=None,
        month: int = 1,
        base_seed: int = 0,
        tick: int = 0,
        _force_u: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate DEPONS ship deterrence over active ships (loudest ship wins).

        Per porpoise, keeps the contribution of the highest-received-level ship
        (DEPONS ShipDeterrence.recordStep), not a sum. Reaction draws are seeded
        per (base_seed, tick, ship.id) so results are invariant to ship order/count.

        NOTE: per-ship SeedSequence draws preserve the marginal Bernoulli
        probability but DELIBERATELY do not reproduce DEPONS' global-RNG draw
        order (impossible under SoA). Only the reaction *rate* matches DEPONS.

        _force_u (test-only): if set, every porpoise's reaction draw for every
        ship is this constant instead of the seeded draw. The DEPONS ship-response
        probability never approaches 1 (it caps ~0.2), so reaction-dependent
        semantics (loudest-ship-wins, order invariance, no-deter_coeff) can only be
        tested deterministically by forcing u (e.g. 0.0 = always react).
        """
        n = porpoise_x.shape[0]
        total_dx = np.zeros(n, dtype=np.float64)
        total_dy = np.zeros(n, dtype=np.float64)
        if not self.enabled:
            return (total_dx, total_dy)
        active_ships = self.get_active_ships()
        if not active_ships:
            return (total_dx, total_dy)

        best_rl = np.full(n, -np.inf, dtype=np.float64)
        min_dist_m = params.deter_min_distance_ships * 1000.0
        max_dist_m = min(MAX_DETER_DIST_M, params.deter_max_distance * 1000.0)
        tships = getattr(params, "deter_ships_min_db", 80.0)

        for ship in active_ships:
            grid_dx = porpoise_x - ship.x          # cell units
            grid_dy = porpoise_y - ship.y
            dist_m = np.hypot(grid_dx * cell_size, grid_dy * cell_size)
            np.maximum(dist_m, 1.0, out=dist_m)

            in_range = (dist_m > min_dist_m) & (dist_m <= max_dist_m)
            if not np.any(in_range):
                continue

            # Received level (clamped >= 0; NODATA -> 0)
            rl = np.zeros(n, dtype=np.float64)
            d_masked = dist_m[in_range]
            source_level = ship.noise.get_source_level()
            if (params.weston_flux_percell and cell_data is not None
                    and getattr(cell_data, "_sediment", None) is not None):
                pos = np.column_stack((porpoise_x[in_range], porpoise_y[in_range]))
                depths = cell_data.get_depths_vectorized(pos)
                grains = cell_data.get_sediments_vectorized(pos)
                sal = cell_data.get_salinities_vectorized(pos, month)
                tl = _compute_tl_percell(
                    d_masked, depths, grains, sal,
                    params.weston_flux_default_temperature,
                    params.beta_hat, params.alpha_hat,
                )
                rl_masked = source_level - tl
                # DEPONS Ship.java:296-307: NODATA (depth/grain/salinity <= -9999 or depth<=0)
                # OR TL<=0 -> received level 0.
                nodata = (depths <= 0.0) | (grains <= -9999.0) | (sal <= -9999.0)
                rl_masked = np.where(nodata | (tl <= 0.0), 0.0, rl_masked)
            else:
                tl = params.beta_hat * np.log10(d_masked) + params.alpha_hat * d_masked
                rl_masked = source_level - tl
            rl_masked = np.maximum(rl_masked, 0.0)
            rl[in_range] = rl_masked

            # Identity-seeded reaction draws (order/count invariant), or forced (tests)
            if _force_u is not None:
                u = np.full(n, float(_force_u), dtype=np.float64)
            else:
                rng = np.random.default_rng(np.random.SeedSequence([base_seed, tick, int(ship.id)]))
                u = rng.random(n)

            # Each Ship owns a ShipDeterrenceModel (ship.py:172); ShipManager has none.
            vx, vy, prob, mag, react = ship.deterrence_model.deterrence_components(
                rl, dist_m, grid_dx, grid_dy, is_day, u, tships)

            # Loudest gated ship wins this porpoise's slot (vector is 0 if it didn't react)
            gated = in_range & (rl > tships)
            wins = gated & (rl > best_rl)
            best_rl = np.where(wins, rl, best_rl)
            total_dx = np.where(wins, vx, total_dx)
            total_dy = np.where(wins, vy, total_dy)

        return (total_dx, total_dy)
```

- [ ] **Step 5: Remove the now-orphaned import (avoid ruff F401 commit failure)**

After the rewrite, `response_probability_from_rl` is unused in `ship.py` (its only uses were in the deleted block). The auto-format hook runs ruff (F401), so it MUST be removed now or the commit fails / leaves a dirty tree. Remove **only** `response_probability_from_rl` from the `from cenop.behavior.sound import (...)` block (lines ~22-28). KEEP `calculate_deterrence_vector` — it is still used by `Ship.get_deterrence_vector` until Task 5. Verify:

Run: `... && ruff check src/cenop/agents/ship.py`
Expected: no F401 for `response_probability_from_rl`.

- [ ] **Step 5b: Move the superseded test rewrite into THIS commit (keep the suite green)**

`tests/test_deterrence.py::test_no_normalization_vectorized_ship` (line ~77) asserts the OLD raw-displacement ship formula (`dx[0] > 1.0`). The Task 3 rewrite makes it fail, and it must not stay red across Tasks 4-6. Replace its body now with the new-model assertion (forces a reaction via `_force_u=0.0`):

```python
    def test_ship_vectorized_uses_unit_vector_times_magnitude(self):
        """Ships use DEPONS unit-vector x magnitude (NOT raw displacement x deter_coeff)."""
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = 200.0
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([50.0 + 2000.0 / 400.0]); py = np.array([50.0])  # 2 km east
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, params, _force_u=0.0)
        assert dx[0] != 0.0          # forced reaction -> non-zero
        assert 0.0 < abs(dx[0]) < 5.0  # unit-vector x mag, NOT raw 5-cell displacement
        assert dx[0] > 0.0           # pushed east, away from ship
```

(Delete the old `params.deter_threshold = 0.0` / `assert dx[0] > 1.0` version. Note: Task 7 no longer needs to rewrite this test.)

- [ ] **Step 6: Run tests to verify they pass (incl. the rewritten superseded test — suite stays green)**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "VectorizedPath" tests/test_deterrence.py -q`
Expected: PASS — the 6 VectorizedPath tests AND the rewritten `test_ship_vectorized_uses_unit_vector_times_magnitude` (no lingering `dx>1.0` assertion). Confirms this commit leaves the suite green.

- [ ] **Step 7: Commit**

```bash
git add src/cenop/agents/ship.py tests/test_deterrence.py tests/test_ship_deterrence_port.py
git commit -m "feat: vectorized ship deterrence uses DEPONS model (Tships, 10km cap, loudest-ship, seeded draws)"
```

---

## Task 4: Thread RNG seed + tick from the simulation into the ship call

**Files:**
- Modify: `src/cenop/core/simulation.py` (the ship-deterrence call ~lines 488-494)
- Test: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Write the failing test**

```python
class TestSimulationDeterminism:
    def _sim_with_ship(self):
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.core.simulation import Simulation
        from cenop.agents.ship import Ship, VesselClass
        params = SimulationParameters(porpoise_count=50, sim_years=1, random_seed=42, ships_enabled=True)
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        sim = Simulation(params=params, cell_data=land, seed=42)
        sim.initialize()
        # Replace any sample ship with one guaranteed near the porpoises
        loud = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        loud._is_active = True; loud.noise.base_source_level = 195.0
        sim._ship_manager.ships = [loud]; sim._ship_manager.enabled = True
        return sim

    def test_tick_varies_ship_draws_after_wiring(self):
        """Before wiring, every tick uses base_seed=0/tick=0, so the per-ship reaction
        draw is IDENTICAL every tick; after wiring (tick threaded) the draw varies.
        Hold porpoise positions fixed so the only source of variation is the seed/tick."""
        import numpy as np
        sim = self._sim_with_ship()
        pm = sim.population_manager
        snaps = []
        for _ in range(6):
            # freeze porpoises so deter_strength changes ONLY if the ship draw changes
            pre_x, pre_y = pm.x.copy(), pm.y.copy()
            sim.step()
            pm.x[:] = pre_x; pm.y[:] = pre_y
            pm._recompute_cell_indices()
            snaps.append(pm.deter_strength.copy())
        # With tick threaded, at least two ticks must differ (draws vary by tick).
        assert any(not np.array_equal(snaps[0], s) for s in snaps[1:])

    def test_reproducible_across_sequential_runs(self):
        """Green guard: two identically-seeded sims (run SEQUENTIALLY to avoid global
        np.random cross-talk) produce identical ship deterrence."""
        import numpy as np
        s1 = self._sim_with_ship()
        for _ in range(5):
            s1.step()
        d1 = s1.population_manager.deter_strength.copy()
        s2 = self._sim_with_ship()   # constructed AFTER s1 finishes
        for _ in range(5):
            s2.step()
        d2 = s2.population_manager.deter_strength.copy()
        np.testing.assert_array_equal(d1, d2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "SimulationDeterminism" -v`
Expected: FAIL — before wiring, `base_seed`/`tick` default to 0 every tick, so the frozen-porpoise `deter_strength` is identical across all ticks and `any(...)` is False.

- [ ] **Step 3: Wire seed + tick**

In `src/cenop/core/simulation.py`, change the ship call from:

```python
            ship_dx, ship_dy = (
                self._ship_manager.calculate_aggregate_deterrence_vectorized(
                    px, py, self.params, is_day=self.time_manager.is_daytime,
                    cell_size=400.0, cell_data=self._cell_data,
                    month=self.state.month,
                )
            )
```

to:

```python
            ship_dx, ship_dy = (
                self._ship_manager.calculate_aggregate_deterrence_vectorized(
                    px, py, self.params, is_day=self.time_manager.is_daytime,
                    cell_size=400.0, cell_data=self._cell_data,
                    month=self.state.month,
                    base_seed=self._seed, tick=int(self.time_manager.tick),
                )
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "SimulationDeterminism" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cenop/core/simulation.py tests/test_ship_deterrence_port.py
git commit -m "feat: thread base_seed+tick into vectorized ship deterrence for reproducible draws"
```

---

## Task 5: Refactor the scalar oracle + aggregator onto the kernel; fix min-distance boundary

**Files:**
- Modify: `src/cenop/agents/ship.py` (`Ship.calculate_deterrence` ~lines 285-373; `ShipManager.calculate_aggregate_deterrence` ~lines 418-460)
- Test: `tests/test_ship_deterrence_port.py`, plus existing `tests/test_deterrence.py`

- [ ] **Step 1: Write the failing test**

```python
class TestScalarOracleConsistency:
    def test_scalar_matches_kernel(self):
        """Ship.calculate_deterrence must agree with the shared kernel for a fixed u."""
        import numpy as np
        from cenop.agents.ship import Ship, VesselClass
        from cenop.behavior.sound import ShipDeterrenceModel
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = 180.0
        px, py = 50.0 + 2000.0 / 400.0, 50.0  # 2 km east
        # Force u=0 by monkeypatching np.random in the scalar path
        import cenop.agents.ship as shipmod
        orig = np.random.random
        np.random.random = lambda *a, **k: 0.0
        try:
            should, prob, mag, dkm = s.calculate_deterrence(px, py, p, is_day=True)
        finally:
            np.random.random = orig
        # Independently compute via kernel
        m = ShipDeterrenceModel()
        gdx = np.array([px - 50.0]); gdy = np.array([0.0])
        dist_m = np.array([2000.0]); rl_simple = 180.0 - (p.beta_hat * np.log10(2000.0) + p.alpha_hat * 2000.0)
        rl = np.array([max(0.0, rl_simple)])
        _, _, kprob, kmag, kreact = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, np.array([0.0]), tships=p.deter_ships_min_db)
        assert should == bool(kreact[0])
        assert prob == pytest.approx(float(kprob[0]))
        assert mag == pytest.approx(float(kmag[0]))

    def test_min_distance_boundary_strict(self):
        """DEPONS uses strict '>' at the min-distance floor (Ship.java:220)."""
        from cenop.agents.ship import Ship, VesselClass
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()  # min = 0.1 km = 100 m
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = 200.0
        at_floor = 50.0 + 100.0 / 400.0   # exactly 100 m
        should, *_ = s.calculate_deterrence(at_floor, 50.0, p, is_day=True)
        assert should == False  # 100 m is excluded (strict >)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "ScalarOracleConsistency" -v`
Expected: FAIL — scalar magnitude differs (until it uses the kernel) and/or the `< min_dist_km` boundary is inclusive at the floor.

- [ ] **Step 3: Refactor `Ship.calculate_deterrence` onto the kernel**

In `src/cenop/agents/ship.py`, replace the gate + probability/magnitude tail of `calculate_deterrence` (from the `max_dist_km = min(10.0, ...)` line through the final `return (True, prob, magnitude, distance_km)`) with a kernel call. Keep the RL computation (`spl`) you already have; then:

```python
        # Distance gates — DEPONS Ship.java:220-222 (strict > at floor, <= at cap)
        max_dist_m = min(MAX_DETER_DIST_M, params.deter_max_distance * 1000.0)
        min_dist_m = params.deter_min_distance_ships * 1000.0
        if not (distance_m > min_dist_m and distance_m <= max_dist_m):
            return (False, 0.0, 0.0, distance_km)

        rl = np.array([max(0.0, float(spl))], dtype=np.float64)
        gdx = np.array([porpoise_x - self.x], dtype=np.float64)
        gdy = np.array([porpoise_y - self.y], dtype=np.float64)
        dm = np.array([max(distance_m, 1.0)], dtype=np.float64)
        u = np.array([np.random.random()], dtype=np.float64)
        _, _, prob, mag, react = self.deterrence_model.deterrence_components(
            rl, dm, gdx, gdy, is_day, u, getattr(params, "deter_ships_min_db", 80.0))
        return (bool(react[0]), float(prob[0]), float(mag[0]) if react[0] else 0.0, distance_km)
```

(`distance_m`, `distance_km`, `spl` are already computed earlier in the method.)

- [ ] **Step 4: Refactor `ShipManager.calculate_aggregate_deterrence` (scalar aggregator)**

Replace its `get_deterrence_vector(..., deter_coeff)` body with a per-ship loop reusing `Ship.calculate_deterrence` for the gate/mag and building the DEPONS unit vector (no `deter_coeff`), keeping loudest-ship semantics:

```python
        if not self.enabled:
            return (0.0, 0.0, 0.0)
        best_rl = -np.inf
        best_dx = best_dy = best_mag = 0.0
        max_dist_m = min(MAX_DETER_DIST_M, params.deter_max_distance * 1000.0)
        min_dist_m = params.deter_min_distance_ships * 1000.0
        tships = getattr(params, "deter_ships_min_db", 80.0)
        for ship in self.get_active_ships():
            gdx = porpoise_x - ship.x
            gdy = porpoise_y - ship.y
            dist_m = max(float(np.hypot(gdx * cell_size, gdy * cell_size)), 1.0)
            if not (dist_m > min_dist_m and dist_m <= max_dist_m):
                continue
            # Compute RL ONCE; use the same value for selection and the kernel (no double-compute).
            rl = max(0.0, ship.get_received_level(
                porpoise_x, porpoise_y, params.alpha_hat, params.beta_hat, cell_size))
            if rl <= best_rl:
                continue
            best_rl = rl
            vx, vy, _, mag, react = ship.deterrence_model.deterrence_components(
                np.array([rl]), np.array([dist_m]), np.array([gdx]), np.array([gdy]),
                is_day, np.array([np.random.random()]), tships)
            best_dx, best_dy = float(vx[0]), float(vy[0])
            best_mag = float(mag[0]) if bool(react[0]) else 0.0
        return (best_mag, best_dx, best_dy)
```

- [ ] **Step 4b: Delete the now-dead `Ship.get_deterrence_vector` and remove its import**

After Step 4, nothing calls `Ship.get_deterrence_vector` (grep confirms no caller outside `ship.py`). Delete the method (`ship.py` ~lines 375-387) and remove `calculate_deterrence_vector` from the `from cenop.behavior.sound import (...)` block (it is now unused). Verify:

Run: `... && ruff check src/cenop/agents/ship.py`
Expected: no F401 unused-import errors.

- [ ] **Step 5: Run tests to verify they pass**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "ScalarOracleConsistency" tests/test_deterrence.py -q`
Expected: PASS — kernel consistency + boundary, and existing gate/prob tests still green.

- [ ] **Step 6: Commit**

```bash
git add src/cenop/agents/ship.py tests/test_ship_deterrence_port.py
git commit -m "refactor: scalar ship deterrence + aggregator share the kernel; strict min-dist boundary"
```

---

## Task 6: Switch `deter_strength` from L1 to L2 (decision S1)

**Files:**
- Modify: `src/cenop/agents/population.py` (two sites: ~line 1033 and ~line 1236)
- Test: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Write the failing test**

```python
class TestDeterStrengthL2:
    def test_deter_strength_is_euclidean(self):
        """DEPONS ShipDeterrence.java:75 uses sqrt(dx^2+dy^2), not |dx|+|dy|."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=1)
        land = create_homogeneous_landscape(width=50, height=50, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=1, params=params, landscape=land)
        d_dx = np.array([3.0], dtype=np.float64)
        d_dy = np.array([4.0], dtype=np.float64)
        pop.step(deterrence_vectors=(d_dx, d_dy))
        assert pop.deter_strength[0] == pytest.approx(5.0)  # hypot(3,4), not 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "DeterStrengthL2" -v`
Expected: FAIL — gets 7.0 (L1) instead of 5.0 (L2).

- [ ] **Step 3: Change ALL THREE NumPy sites + the JAX kernel**

In `src/cenop/agents/population.py` there are **three** L1 sites (lines ~1033, ~1236, and ~2191-2192 in the JAX-step path). Replace every occurrence of the expression:

```python
np.abs(d_dx[mask]) + np.abs(d_dy[mask])
```

with:

```python
np.hypot(d_dx[mask], d_dy[mask])
```

(Use a single `replace_all` on that substring, then confirm with `grep -n "np.abs(d_dx" src/cenop/agents/population.py` returns nothing.)

Also change the JAX kernel `src/cenop/optimizations/jax_kernels.py:371` from:

```python
    deter_strength = jnp.abs(deter_dx) + jnp.abs(deter_dy)
```

to:

```python
    deter_strength = jnp.hypot(deter_dx, deter_dy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py -k "DeterStrengthL2" -v`
Expected: PASS.

- [ ] **Step 5: Update the JAX test that pins L1, then run turbine + JAX tests**

`tests/test_jax_tick.py::test_deter_strength_computed` (~line 689) computes `expected = np.abs(deter_dx) + np.abs(deter_dy)`. Change it to:

```python
        expected = np.hypot(deter_dx, deter_dy)
```

Then run: `... python3 -m pytest tests/test_deterrence.py tests/test_depons_deterrence.py tests/test_jax_tick.py -q`
Expected: PASS (NumPy and JAX backends now both L2; the JAX test asserts hypot).

- [ ] **Step 6: Commit**

```bash
git add src/cenop/agents/population.py src/cenop/optimizations/jax_kernels.py tests/test_jax_tick.py tests/test_ship_deterrence_port.py
git commit -m "fix: deter_strength uses Euclidean (L2) magnitude to match DEPONS (NumPy + JAX)"
```

---

## Task 7: Rewrite the superseded test + add integration & characterization tests

**Files:**
- Modify: `tests/test_deterrence.py` (`test_no_normalization_vectorized_ship` ~line 77)
- Create/extend: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Add the missing edge-case tests** (the superseded `test_no_normalization_vectorized_ship` was already rewritten in Task 3 Step 5b — do NOT redo it)

```python
class TestEdgeCases:
    def _model(self):
        return ShipDeterrenceModel()

    def test_u_equals_prob_does_not_react(self):
        """Strict '<': u == prob must NOT react."""
        m = self._model()
        rl = np.array([130.0]); dist_m = np.array([1500.0])
        p = float(m.calculate_deterrence_probability(130.0, 1.5, True))
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, np.array([3.0]), np.array([0.0]), True, np.array([p]), tships=80.0)
        assert react[0] == False

    def test_day_night_select_different_coefficients(self):
        """Night uses different std/coeffs; pship_noise_night=0 makes prob noise-independent."""
        m = self._model()
        p_lo = m.calculate_deterrence_probability(90.0, 2.0, is_day=False)
        p_hi = m.calculate_deterrence_probability(150.0, 2.0, is_day=False)
        assert float(p_lo) == pytest.approx(float(p_hi))  # night prob independent of RL
        # Day prob DOES depend on RL:
        d_lo = m.calculate_deterrence_probability(90.0, 2.0, is_day=True)
        d_hi = m.calculate_deterrence_probability(150.0, 2.0, is_day=True)
        assert float(d_hi) > float(d_lo)

    def test_porpoise_on_ship_gives_finite_zero_direction(self):
        """dist clamps to 1 m; grid disp 0 -> zero (defined, finite) vector."""
        m = self._model()
        vx, vy, *_ = m.deterrence_components(
            np.array([200.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]),
            True, np.array([0.0]), tships=80.0)
        assert np.isfinite(vx[0]) and vx[0] == 0.0 and vy[0] == 0.0

    def test_nodata_cell_gives_zero_rl_weston(self):
        """WestonFlux path: a NODATA cell -> RL 0 -> no deterrence (DEPONS Ship.java:300)."""
        import numpy as np
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        params = SimulationParameters(); params.weston_flux_percell = True
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        land._depth[:] = -9999.0  # all NODATA depth
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = 210.0
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([50.5]); py = np.array([50.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(
            px, py, params, cell_data=land, _force_u=0.0)
        assert dx[0] == 0.0 and dy[0] == 0.0  # NODATA -> RL 0 -> gate fails even with forced react
```

- [ ] **Step 2: Add integration + characterization tests**

```python
class TestIntegration:
    def _sim(self, source_level):
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.core.simulation import Simulation
        from cenop.agents.ship import Ship, VesselClass
        params = SimulationParameters(porpoise_count=200, sim_years=1, random_seed=42, ships_enabled=True)
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        sim = Simulation(params=params, cell_data=land, seed=42); sim.initialize()
        ship = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        ship._is_active = True; ship.noise.base_source_level = source_level
        sim._ship_manager.ships = [ship]; sim._ship_manager.enabled = True
        return sim

    def test_loud_ship_deters_some_porpoises(self):
        import numpy as np
        sim = self._sim(source_level=210.0)
        for _ in range(10):
            sim.step()
        assert float(np.max(sim.population_manager.deter_strength)) > 0.0

    def test_quiet_ship_below_tships_does_not_deter(self):
        import numpy as np
        sim = self._sim(source_level=70.0)  # RL stays below Tships=80 everywhere
        for _ in range(10):
            sim.step()
        assert float(np.max(sim.population_manager.deter_strength)) == 0.0


class TestCharacterizationSnapshot:
    """Pinned reference values for the kernel — locks the new behavior against drift.
    Recompute these expected numbers ONCE during implementation (print them), verify
    they are sane (vx>0 east, exp-magnitude), then hard-code them here."""

    def test_kernel_snapshot_day(self):
        import numpy as np
        from cenop.behavior.sound import ShipDeterrenceModel
        m = ShipDeterrenceModel()
        rl = np.array([160.0, 100.0, 79.0])          # high / mid / below-Tships
        dist_m = np.array([800.0, 3000.0, 500.0])
        gdx = np.array([2.0, -7.5, 1.25]); gdy = np.array([0.0, 0.0, 0.0])
        u = np.array([0.0, 0.0, 0.0])                # force reaction where gated
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, is_day=True, u_draw=u, tships=80.0)
        # Sanity (assert before pinning): 3rd porpoise gated out (RL 79 < 80).
        assert react.tolist() == [True, True, False]
        assert vx[0] > 0.0 and vx[1] < 0.0 and vx[2] == 0.0
        # PIN exact values (fill in from the implementation's printed output):
        # np.testing.assert_allclose(vx, [<v0>, <v1>, 0.0], rtol=1e-9)
        # np.testing.assert_allclose(mag, [<m0>, <m1>, <m2>], rtol=1e-9)
```

- [ ] **Step 3: Run the full file + the rewritten test**

Run: `... python3 -m pytest tests/test_ship_deterrence_port.py tests/test_deterrence.py -q`
Expected: PASS.

- [ ] **Step 4: Run the whole suite (with the standard hanging-file exclusions)**

Run: `... python3 -m pytest tests/ -q --ignore=tests/test_depons_physiology.py --ignore=tests/test_validation.py`
Expected: PASS. Fix any test that pinned the old ship-deterrence behavior (search for `deter_threshold = 0` near ship setups).

- [ ] **Step 5: Commit**

```bash
git add tests/test_deterrence.py tests/test_ship_deterrence_port.py
git commit -m "test: ship deterrence integration + characterization; supersede raw-displacement ship test"
```

---

## Task 8: Regenerate the ship baseline + update provenance & investigation status

**Files:**
- Regenerate: `output/kattegat_ref_ships/` (via `scripts/run_kattegat_reference.py`)
- Modify: `docs/superpowers/specs/2026-06-03-ship-deterrence-parity-investigation.md` (status note)

- [ ] **Step 1: Perf sanity-check the 10 km cull (N=2000, ships)**

Run: `eval "$(micromamba shell hook --shell bash)" && micromamba activate shiny && cd /home/razinka/cenjas/CENOP && python3 scripts/run_kattegat_reference.py --count 2000 --years 1 --seed 42 --ships --out /tmp/kattegat_ships_perf`
Expected: completes; note wall-clock vs the pre-fix run. The `dist_m > min & <= 10 km` mask now culls most porpoise-ship pairs from RL/prob work.

- [ ] **Step 2: Verify ship deterrence is now live**

Run:
```bash
eval "$(micromamba shell hook --shell bash)" && micromamba activate shiny && cd /home/razinka/cenjas/CENOP && python3 -c "import csv,glob; mx=0.0
for f in glob.glob('/tmp/kattegat_ships_perf/PorpoiseStatistics.txt'):
 r=csv.reader(open(f),delimiter='\t'); h=next(r); j=h.index('deter_strength')
 for row in r:
  if len(row)>j:
   try: mx=max(mx,abs(float(row[j])))
   except: pass
print('max deter_strength:', mx)"
```
Expected: `max deter_strength` > 0 (was 0 before the fix).

- [ ] **Step 3: Regenerate the committed 2-yr baseline**

Run: `eval "$(micromamba shell hook --shell bash)" && micromamba activate shiny && cd /home/razinka/cenjas/CENOP && python3 scripts/run_kattegat_reference.py --count 2000 --years 2 --seed 42 --ships --out output/kattegat_ref_ships`
Expected: completes; overwrites the compact baseline files.

- [ ] **Step 4: Update the investigation doc status + write baseline provenance**

In `docs/superpowers/specs/2026-06-03-ship-deterrence-parity-investigation.md`, change **Status** to note the fix landed and cite the implementing commits; mark the "Recommended fix" section as implemented.

Write `output/kattegat_ref_ships/PROVENANCE.txt` recording: that this baseline SUPERSEDES the prior near-zero-deterrence baseline, the producing commit hash (`git rev-parse HEAD`), the command used (`scripts/run_kattegat_reference.py --count 2000 --years 2 --seed 42 --ships`), and the date. (PROVENANCE.txt is a compact file — not matched by the gitignore patterns for `PorpoiseStatistics.txt`/`Dispersal.txt`.)

- [ ] **Step 5: Commit (compact baseline files only; heavy files are gitignored)**

```bash
cd /home/razinka/cenjas/CENOP
git add output/kattegat_ref_ships/Population.txt output/kattegat_ref_ships/Energy.txt output/kattegat_ref_ships/Mortality.txt output/kattegat_ref_ships/PROVENANCE.txt docs/superpowers/specs/2026-06-03-ship-deterrence-parity-investigation.md
git commit -m "test: regenerate Kattegat ship baseline with live DEPONS ship deterrence; close investigation"
```

---

## Notes / known non-goals (carried from the spec)

- **Source level** stays CENOP's class-based value (no JOMOPANS rewiring) → DEPONS *response* model, not absolute SL parity.
- **Sub-tick interpolation** is not implemented; with one step per tick the loudest-ship-wins rule is the DEPONS `recordStep` collapse. Documented parity gap.
- **`deter_strength`** is a single combined turbine+ship magnitude (no separate ship/turbine strengths, no `max()` stuck-detection, no `deterTime`/`deterDecay` persistence) — accepted divergence unless that persistence is later required. NOTE: with CENOP default `deter_time=0` this matches DEPONS; if `deter_time>0` is ever set, CENOP will diverge (no per-source decay/halving) — out of scope here.
- **RNG-stream divergence (documented, intentional):** reaction draws are seeded per `(base_seed, tick, ship.id)` so they are invariant to ship order/count. This preserves the marginal Bernoulli probability but does NOT reproduce DEPONS' global-RNG draw order (impossible under SoA). Only the reaction *rate* matches DEPONS, not the exact per-agent draw sequence.
- **Numba** kernel deferred; revisit only if Task 8 perf is inadequate.
