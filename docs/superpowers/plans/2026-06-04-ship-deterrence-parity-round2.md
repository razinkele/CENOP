# Ship-Deterrence Parity Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close three DEPONS-parity gaps in ship deterrence: (#1) the mis-scaled `is_disturbed` reporting threshold, (#2) ships wrongly deactivating dispersal, and (#3) the unwired JOMOPANS ship source level (incl. the JSON-loader fixes that make it fire on real vessel data).

**Architecture:** #1 is a one-line threshold change. #3 makes `ShipNoise.get_source_level` default to the calibrated `jomopans_spl(vessel_class, speed, length, band=12)` with `base_source_level` as an optional explicit override, and fixes the JSON loader to feed JOMOPANS real `type`/`length`/per-buoy-speed. #2 threads a turbine-only deterrence-strength signal so dispersal deactivation gates on turbines only (matching DEPONS), in both the NumPy and JAX paths.

**Tech Stack:** Python 3, NumPy, JAX. Tests: pytest. Env: `micromamba run -n shiny`. DEPONS reference: `DEPONS-3.2/.../Porpoise.java`, `Ship.java`, `ships/JomopansEchoSPL.java`.

**Spec:** `docs/superpowers/specs/2026-06-04-ship-deterrence-parity-round2-design.md`

**Revised after a four-angle plan review (verified against code; DEPONS parity confirmed
faithful — band 12, units, turbine-only, all 12 type mappings correct):** added the
route-parsing per-buoy-speed fix (`ship.py:817` hardcoded `speed=10.0` would feed JOMOPANS a
flat 10 kn — the real bug behind the per-buoy-speed goal) + a test; fixed the Task 4 test
fixture (`dispersal_target_distance=1e9`, else distance-completion deactivates dispersal
regardless of the gate → guaranteed red); made `import re` an explicit step; added an
end-to-end JAX dispersal-separation test; corrected two drifted line refs.

---

## Conventions

- Run dir: `cd /home/razinka/cenjas/CENOP`. Test prefix: `eval "$(micromamba shell hook --shell bash)" && micromamba activate shiny && cd /home/razinka/cenjas/CENOP && python3 -m pytest <args>`.
- CENOP is a nested git repo; commit from inside `CENOP/`. **Start by creating a feature branch:** `git checkout -b ship-deterrence-parity-r2` (base `CENOP-JASMINE`).
- New tests append to `tests/test_ship_deterrence_port.py` unless noted.

## File Structure

- `src/cenop/behavior/sound.py` — `ShipNoise`: `base_source_level` → optional override (default `None`), add `vessel_class`, rewrite `get_source_level` to JOMOPANS-default (lazy import), drop `vhf_weighting`/simplified formula. (#3)
- `src/cenop/agents/ship.py` — `Ship.__post_init__` stops seeding `base_source_level`; JSON loader (`load_from_json`) reads `type`→`VesselClass` (normalized), `length`, preserves per-buoy speed, `impact` only when explicit; add `_vessel_class_from_type` helper. (#3)
- `src/cenop/agents/population.py` — `_turbine_deter_strength` buffer; `step`/`_step_jax` gain `turbine_deterrence_vectors`; NumPy dispersal gate (`:3062`) + `is_disturbed` threshold (`:2718`). (#1, #2)
- `src/cenop/core/simulation.py` — pass `turbine_deterrence_vectors=(turb_dx,turb_dy)` to `population.step`. (#2)
- `src/cenop/optimizations/tick_jax.py` + `jax_kernels.py` — thread turbine-only strength to the JAX dispersal gate. (#2)
- Tests: `tests/test_ship_deterrence_port.py` (new), and updates to `tests/test_jax_tick.py`, `tests/test_dispersal.py`, `tests/test_integration.py`.

---

## Task 1: #1 — `is_disturbed` reporting threshold 0.1 → 0.01

**Files:** Modify `src/cenop/agents/population.py:2718`. Test: `tests/test_ship_deterrence_port.py`.

- [ ] **Step 1: Write the failing test**
```python
class TestIsDisturbedThreshold:
    def test_ship_magnitude_reports_disturbed(self):
        """is_disturbed must fire for ship-scale deterrence (~0.04), matching the
        disturbance-memory threshold (>0.01), not the old >0.1."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=1)
        land = create_homogeneous_landscape(width=50, height=50, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=1, params=params, landscape=land)
        pop.deter_strength[0] = 0.04
        df = pop.to_dataframe()
        assert bool(df["is_disturbed"].iloc[0]) is True
```

- [ ] **Step 2: Run, verify FAIL**
`... python3 -m pytest tests/test_ship_deterrence_port.py -k IsDisturbedThreshold -v` → FAIL (0.04 > 0.1 is False). (If `to_dataframe`'s column name differs, read `population.py:2718`+nearby to confirm the column key and adjust the test's column name accordingly — but the threshold assertion stands.)

- [ ] **Step 3: Change the threshold**
In `src/cenop/agents/population.py:2718`, change:
```python
        is_disturbed = self.deter_strength > 0.1
```
to:
```python
        is_disturbed = self.deter_strength > 0.01
```

- [ ] **Step 4: Run, verify PASS** — same command → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/cenop/agents/population.py tests/test_ship_deterrence_port.py
git commit -m "fix: is_disturbed reporting threshold 0.1->0.01 (captures ship deterrence)"
```

---

## Task 2: #3a — JOMOPANS-default ship source level (`ShipNoise`)

**Files:** Modify `src/cenop/behavior/sound.py` (`ShipNoise` ~195-230); `src/cenop/agents/ship.py` (`Ship.__post_init__` ~176-184). Test: `tests/test_ship_deterrence_port.py`.

- [ ] **Step 1: Pre-check — confirm no test/consumer sets `vhf_weighting`**
Run: `grep -rn "vhf_weighting" src/ tests/` — expect only `sound.py`. (If any other reader exists, keep the field; do NOT delete it. The grep result decides Step 3's field removal.)

- [ ] **Step 2: Write the failing tests**
```python
class TestJomopansSourceLevel:
    def test_default_uses_jomopans(self):
        """With no base_source_level override, get_source_level == jomopans_spl band 12."""
        from cenop.behavior.sound import ShipNoise
        from cenop.behavior.jomopans_spl import jomopans_spl
        from cenop.agents.ship import VesselClass
        n = ShipNoise(vessel_class=VesselClass.CARGO, length=200.0, speed=12.0)
        expected = jomopans_spl(VesselClass.CARGO, 12.0, 200.0, band=12)
        assert n.get_source_level() == expected

    def test_explicit_override_wins(self):
        """An explicit base_source_level overrides JOMOPANS (ships.json impact / tests)."""
        from cenop.behavior.sound import ShipNoise
        from cenop.agents.ship import VesselClass
        n = ShipNoise(vessel_class=VesselClass.CARGO, length=200.0, speed=12.0,
                      base_source_level=170.0)
        assert n.get_source_level() == 170.0

    def test_speed_zero_silent(self):
        """JOMOPANS returns 0.0 for a stationary ship."""
        from cenop.behavior.sound import ShipNoise
        from cenop.agents.ship import VesselClass
        n = ShipNoise(vessel_class=VesselClass.CARGO, length=200.0, speed=0.0)
        assert n.get_source_level() == 0.0

    def test_class_dependence(self):
        """Different vessel classes give different SL (JOMOPANS, not a flat default)."""
        from cenop.behavior.sound import ShipNoise
        from cenop.agents.ship import VesselClass
        a = ShipNoise(vessel_class=VesselClass.CARGO, length=200.0, speed=12.0)
        b = ShipNoise(vessel_class=VesselClass.FISHING, length=200.0, speed=12.0)
        assert a.get_source_level() != b.get_source_level()

    def test_post_init_leaves_override_none(self):
        """Ship.__post_init__ must NOT seed base_source_level (so JOMOPANS is the default)."""
        from cenop.agents.ship import Ship, VesselClass
        s = Ship(id=0, x=0.0, y=0.0, vessel_type=VesselClass.CARGO, vessel_length=200.0)
        assert s.noise.base_source_level is None
        assert s.noise.vessel_class == VesselClass.CARGO
```

- [ ] **Step 3: Run, verify FAIL**
`... python3 -m pytest tests/test_ship_deterrence_port.py -k JomopansSourceLevel -v` → FAIL (`ShipNoise` has no `vessel_class`; default uses the simplified formula; `__post_init__` seeds base_source_level).

- [ ] **Step 4: Rewrite `ShipNoise` in `src/cenop/behavior/sound.py`**
Replace the `ShipNoise` field block + `get_source_level` (lines ~204-230) with:
```python
    # Explicit source-level override (dB re 1 µPa @ 1m). When None, SL is computed
    # from the calibrated JOMOPANS model. Set by ships.json `impact` or by tests.
    base_source_level: float = None

    # Vessel class — drives the JOMOPANS source-level model (set from Ship.vessel_type).
    vessel_class: object = None

    # Vessel length (m) and speed (knots) — JOMOPANS inputs.
    length: float = 100.0
    speed: float = 12.0

    def get_source_level(self) -> float:
        """Source level (dB re 1 µPa @ 1m).

        Returns the explicit override if set, else the calibrated JOMOPANS
        decidecade band-12 SL (DEPONS Ship.java:286 / JOMOPANS_BAND=12).
        """
        if self.base_source_level is not None:
            return self.base_source_level
        # Lazy import breaks the sound -> jomopans -> ship -> sound module cycle.
        from cenop.behavior.jomopans_spl import jomopans_spl
        return jomopans_spl(self.vessel_class, self.speed, self.length, band=12)
```
(Delete the `vhf_weighting` field and the old simplified-formula body. If Step 1 found an external `vhf_weighting` reader, keep the field but still rewrite `get_source_level` as above.)

- [ ] **Step 5: Update `Ship.__post_init__` in `src/cenop/agents/ship.py` (~176-184)**
Replace with:
```python
    def __post_init__(self):
        """Initialize the noise model (JOMOPANS source level by default)."""
        self.noise = ShipNoise(
            vessel_class=self.vessel_type,
            length=self.vessel_length,
            speed=self.current_speed,
        )
```
(`base_source_level` is left at its `None` default → JOMOPANS. `VESSEL_BASE_LEVELS` is now unused for SL; leave it in place — removing it is out of scope.)

- [ ] **Step 6: Run, verify PASS + no regression**
`... python3 -m pytest tests/test_ship_deterrence_port.py -k JomopansSourceLevel tests/test_deterrence.py tests/test_depons_deterrence.py tests/test_weston_flux.py -q` → PASS. (Existing ship tests set `base_source_level` post-construction → override path → unaffected.)

- [ ] **Step 7: Commit**
```bash
git add src/cenop/behavior/sound.py src/cenop/agents/ship.py tests/test_ship_deterrence_port.py
git commit -m "feat: ship source level defaults to JOMOPANS (base_source_level optional override)"
```

---

## Task 3: #3b — JSON loader feeds JOMOPANS real type/length/speed

**Files:** Modify `src/cenop/agents/ship.py` (`load_from_json` ~810-876; add `_vessel_class_from_type` helper near `VesselClass`). Test: `tests/test_ship_deterrence_port.py`.

- [ ] **Step 1: Write the failing tests**
```python
class TestShipJsonLoader:
    def test_type_string_mapping_all_kattegat_types(self):
        """All 12 Kattegat ships.json `type` strings map to a valid VesselClass."""
        from cenop.agents.ship import _vessel_class_from_type, VesselClass
        cases = {
            "Bulker": VesselClass.BULKER, "Containership": VesselClass.CONTAINER,
            "Tanker": VesselClass.TANKER, "Government/Research": VesselClass.GOVERNMENT,
            "Cruise": VesselClass.CRUISE, "Dredger": VesselClass.DREDGER,
            "Passenger": VesselClass.PASSENGER, "Tug": VesselClass.TUG,
            "Recreational": VesselClass.RECREATIONAL, "Fishing": VesselClass.FISHING,
            "Naval": VesselClass.NAVAL, "Other": VesselClass.OTHER,
        }
        for s, vc in cases.items():
            assert _vessel_class_from_type(s) == vc, s

    def test_unknown_type_raises(self):
        from cenop.agents.ship import _vessel_class_from_type
        import pytest
        with pytest.raises(ValueError):
            _vessel_class_from_type("Submarine")

    def test_loader_reads_type_length_and_no_forced_impact(self):
        """Loader maps real type/length and does NOT force a 170 dB override when impact absent."""
        from cenop.agents.ship import ShipManager
        mgr = ShipManager()
        mgr.load_from_json("data/Kattegat/ships.json",
                           utm_origin_x=529473.0, utm_origin_y=5972242.0, cell_size=400.0)
        assert mgr.count > 0
        # No ship has an explicit impact in Kattegat -> base_source_level stays None -> JOMOPANS
        assert all(s.noise.base_source_level is None for s in mgr.ships)
        # Lengths and classes vary (not all OTHER/100m)
        assert len({s.vessel_length for s in mgr.ships}) > 1
        assert len({s.vessel_type for s in mgr.ships}) > 1

    def test_loader_preserves_real_per_buoy_speed(self):
        """Route buoys keep the JSON per-waypoint speeds (not a hardcoded 10.0), so
        JOMOPANS sees real speeds."""
        from cenop.agents.ship import ShipManager
        mgr = ShipManager()
        mgr.load_from_json("data/Kattegat/ships.json",
                           utm_origin_x=529473.0, utm_origin_y=5972242.0, cell_size=400.0)
        speeds = {round(b.speed, 3) for s in mgr.ships for b in s.route.buoys}
        assert speeds != {10.0}          # not all the hardcoded default
        assert len(speeds) > 1           # real per-waypoint variation preserved
```

- [ ] **Step 2: Run, verify FAIL**
`... python3 -m pytest tests/test_ship_deterrence_port.py -k ShipJsonLoader -v` → FAIL (`_vessel_class_from_type` undefined; loader forces base_source_level=170 and uses constant length/OTHER).

- [ ] **Step 3a: Add `import re`** to the top imports of `src/cenop/agents/ship.py` (it is currently absent and is needed by the helper below). Run `grep -n "^import re" src/cenop/agents/ship.py` after to confirm.

- [ ] **Step 3b: Fix the route-parsing loop to read per-waypoint speed (the real per-buoy-speed fix).**
The route-parsing loop hardcodes the buoy speed at `src/cenop/agents/ship.py:817`:
```python
                buoy = Buoy(x=grid_x, y=grid_y, speed=10.0, pause_ticks=0)
```
This **discards the real per-waypoint speeds** the Kattegat `ships.json` provides (e.g. 34.29) — so JOMOPANS (speed-dependent) would see a flat 10 kn. Change it to read the waypoint fields:
```python
                buoy = Buoy(x=grid_x, y=grid_y,
                            speed=waypoint.get("speed", 10.0),
                            pause_ticks=waypoint.get("pause", 0))
```
(`Ship.update` already syncs `noise.speed` from the current buoy each tick, so JOMOPANS then sees the real per-segment speed — matching DEPONS `Ship.getSpeed()`.)

- [ ] **Step 3c: Add the type-mapping helper**
In `src/cenop/agents/ship.py`, after the `VesselClass` enum, add:
```python
def _vessel_class_from_type(type_str: str) -> VesselClass:
    """Map a ships.json `type` string to a VesselClass (DEPONS VesselClass.forValue
    normalization: strip [-/ _], uppercase, match enum name). Raises on unknown type
    (fail-fast, matching DEPONS JomopansEchoSPL)."""
    norm = re.sub(r"[-/ _]", "", (type_str or "")).upper()
    aliases = {
        "CONTAINERSHIP": VesselClass.CONTAINER,
        "GOVERNMENTRESEARCH": VesselClass.GOVERNMENT,
    }
    if norm in aliases:
        return aliases[norm]
    for vc in VesselClass:
        if vc.name.replace("_", "") == norm:
            return vc
    raise ValueError(f"Unknown ship type: {type_str!r}")
```

- [ ] **Step 4: Fix the JSON loader (`load_from_json`, the `for i, ship_data ...` body ~824-876)**
Replace the per-ship parsing block with:
```python
        self.ships = []
        for i, ship_data in enumerate(data.get("ships", [])):
            name = ship_data.get("name", f"ship_{i}")
            speed = ship_data.get("speed")          # ship-level override; None -> keep buoy speeds
            impact = ship_data.get("impact")        # explicit SL override; None -> JOMOPANS
            start_tick = ship_data.get("start", 0)
            route_name = ship_data.get("route", "")
            length_m = ship_data.get("length", 100.0)

            route = routes_dict.get(route_name, Route())

            # Only overwrite buoy speeds when the ship record gives an explicit speed;
            # otherwise preserve the per-waypoint speeds the JSON route provides (JOMOPANS
            # is speed-dependent, so clobbering them with a default would corrupt SL).
            if speed is not None:
                for buoy in route.buoys:
                    buoy.speed = speed

            x, y = 0.0, 0.0
            if route.buoys:
                x = route.buoys[0].x
                y = route.buoys[0].y

            vessel_type = _vessel_class_from_type(ship_data.get("type", "Other"))

            ship = Ship(
                id=i, x=x, y=y, heading=0.0, name=name,
                vessel_type=vessel_type, vessel_length=length_m,
                route=route, tick_start=start_tick, tick_end=2147483647,
            )

            # Explicit dB override only when impact is present and positive (CENOP extension;
            # DEPONS always uses JOMOPANS). Absent impact -> base_source_level stays None -> JOMOPANS.
            if impact is not None and impact > 0:
                ship.noise.base_source_level = impact

            self.ships.append(ship)
```

- [ ] **Step 5: Run, verify PASS + no regression**
`... python3 -m pytest tests/test_ship_deterrence_port.py -k ShipJsonLoader tests/test_integration.py -q` → PASS. (If a sample-ship / `_load_ships` text-loader test breaks, it shouldn't — those paths construct via `__post_init__` and are unaffected; investigate only if red.)

- [ ] **Step 6: Commit**
```bash
git add src/cenop/agents/ship.py tests/test_ship_deterrence_port.py
git commit -m "fix: ship JSON loader maps real type/length, preserves per-buoy speed, JOMOPANS by default"
```

---

## Task 4: #2a — Turbine-only dispersal deactivation (NumPy path)

**Files:** Modify `src/cenop/agents/population.py` (buffer ~145; `step` signature ~2518; dispersal gate ~3062); `src/cenop/core/simulation.py` (population.step call ~535). Test: `tests/test_ship_deterrence_port.py`.

- [ ] **Step 1: Write the failing test**
```python
class TestTurbineOnlyDispersal:
    def _pop(self):
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=1)
        land = create_homogeneous_landscape(width=50, height=50, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=1, params=params, landscape=land)
        pop.is_dispersing[0] = True
        pop.dispersal_start_x[0] = pop.x[0]; pop.dispersal_start_y[0] = pop.y[0]
        # CRITICAL: dispersal_target_distance defaults to 0.0, so the distance-completion
        # check (distances >= 0.95*target) would deactivate dispersal regardless of the
        # deterrence gate, masking what we're testing. Set it huge so only the gate can fire.
        pop.dispersal_target_distance[0] = 1e9
        return pop

    def test_ship_only_deterrence_does_not_deactivate_dispersal(self):
        import numpy as np
        pop = self._pop()
        # Combined deterrence present (ship), but turbine component zero.
        d = (np.array([0.05], dtype=np.float64), np.array([0.0], dtype=np.float64))
        t = (np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64))
        pop.step(deterrence_vectors=d, turbine_deterrence_vectors=t)
        assert bool(pop.is_dispersing[0]) is True   # ship deterrence must NOT stop dispersal

    def test_turbine_deterrence_deactivates_dispersal(self):
        import numpy as np
        pop = self._pop()
        d = (np.array([0.05], dtype=np.float64), np.array([0.0], dtype=np.float64))
        t = (np.array([0.05], dtype=np.float64), np.array([0.0], dtype=np.float64))
        pop.step(deterrence_vectors=d, turbine_deterrence_vectors=t)
        assert bool(pop.is_dispersing[0]) is False  # turbine deterrence DOES stop dispersal
```

- [ ] **Step 2: Run, verify FAIL**
`... python3 -m pytest tests/test_ship_deterrence_port.py -k TurbineOnlyDispersal -v` → FAIL (`step` has no `turbine_deterrence_vectors`; gate uses combined `deter_strength` so ship-only also deactivates).

- [ ] **Step 3: Add the buffer** in `src/cenop/agents/population.py` right after the `deter_strength` buffer (~line 145):
```python
        # Turbine-only deterrence strength — DEPONS deactivates dispersal for turbine/
        # sound-source deterrence only, NOT ships (Porpoise.java:1277).
        self._turbine_deter_strength = np.zeros(count, dtype=np.float32)
```

- [ ] **Step 4: Add the param + populate the buffer.** Change the `step` signature (~2518) to:
```python
    def step(self, deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             ambient_rl: Optional[np.ndarray] = None,
             turbine_deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None):
```
Then at the **very top** of `step` (before the `if self._use_jax: self._step_jax(...)` dispatch at ~line 2547, after the `if not mask.any(): return` guard, so BOTH backends see it), populate the instance buffer:
```python
        if turbine_deterrence_vectors is not None:
            _t_dx, _t_dy = turbine_deterrence_vectors
            np.hypot(_t_dx, _t_dy, out=self._turbine_deter_strength)
        else:
            self._turbine_deter_strength.fill(0.0)
```
Do NOT change the `_step_jax` dispatch signature — `_step_jax` reads `self._turbine_deter_strength` directly (set here), so the buffer is already populated before the JAX path runs.

- [ ] **Step 5: Change the NumPy dispersal gate** at `src/cenop/agents/population.py:3062` from:
```python
        deterred = dispersing & (self.deter_strength > 0)
```
to:
```python
        deterred = dispersing & (self._turbine_deter_strength > 0)
```

- [ ] **Step 6: Pass turbine vectors from `simulation.py`.** In `src/cenop/core/simulation.py`, the `self.population_manager.step(deterrence_vectors=(total_dx, total_dy), ambient_rl=ambient_rl)` call (~535) → add the turbine-only pair (`turb_dx`/`turb_dy` already exist in that scope):
```python
        self.population_manager.step(
            deterrence_vectors=(total_dx, total_dy),
            ambient_rl=ambient_rl,
            turbine_deterrence_vectors=(turb_dx, turb_dy),
        )
```

- [ ] **Step 7: Run, verify PASS + no regression**
`... python3 -m pytest tests/test_ship_deterrence_port.py -k TurbineOnlyDispersal tests/test_dispersal.py tests/test_integration.py -q` → PASS. (Note: `tests/test_dispersal.py::test_deterrence_deactivates_dispersal` re-implements the gate inline on combined `deter_strength`; update it to use a turbine-only signal so it documents the new contract — change its inline `deter_strength` to a turbine-only array and assert turbine deters / ship doesn't.)

- [ ] **Step 8: Commit**
```bash
git add src/cenop/agents/population.py src/cenop/core/simulation.py tests/test_ship_deterrence_port.py tests/test_dispersal.py
git commit -m "feat: dispersal deactivation gates on turbine-only deterrence (NumPy); ships no longer stop dispersal"
```

---

## Task 5: #2b — Turbine-only dispersal deactivation (JAX path)

**Files:** Modify `src/cenop/optimizations/jax_kernels.py` (`jax_dispersal_update` ~812-845); `src/cenop/optimizations/tick_jax.py` (`jax_tick_energy` ~255-300, dispersal call ~358-372); `src/cenop/agents/population.py` (`_step_jax` signature ~2164 + the `jax_tick_energy` call ~2408-2430). Tests: `tests/test_jax_tick.py`, `tests/test_ship_deterrence_port.py`.

- [ ] **Step 1: Write the failing test**
```python
class TestTurbineOnlyDispersalJax:
    def test_jax_dispersal_uses_turbine_strength(self):
        import numpy as np, jax.numpy as jnp
        from cenop.optimizations.jax_kernels import jax_dispersal_update
        n = 2
        is_dispersing = jnp.array([True, True])
        zeros = jnp.zeros(n); ddt = jnp.zeros(n); dde = jnp.zeros(n, dtype=jnp.int32)
        x = jnp.zeros(n); y = jnp.zeros(n)
        eh = jnp.zeros((n, 8)); active = jnp.array([True, True])
        # turbine strength nonzero only for agent 0
        turb = jnp.array([0.05, 0.0])
        new_disp, _, _ = jax_dispersal_update(
            is_dispersing, zeros, zeros, jnp.full(n, 1e9), ddt, dde, x, y,
            turbine_deter_strength=turb, energy_history=eh, active_mask=active,
            is_day_boundary=False)
        assert bool(new_disp[0]) is False   # turbine-deterred -> dispersal off
        assert bool(new_disp[1]) is True    # not turbine-deterred -> still dispersing

    def test_step_jax_ship_only_keeps_dispersing(self):
        """End-to-end JAX: ship-only deterrence (turbine zero) must NOT deactivate dispersal
        — guards that _step_jax wires the TURBINE-only array (not the combined one) to the
        dispersal gate (a combined/turbine arg-swap would fail this)."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.population import PorpoisePopulation
        params = SimulationParameters(porpoise_count=1, use_jax=True)
        land = create_homogeneous_landscape(width=50, height=50, depth=20.0, food_prob=0.5)
        pop = PorpoisePopulation(count=1, params=params, landscape=land)
        if not getattr(pop, "_use_jax", False):
            import pytest; pytest.skip("JAX not active")
        pop.is_dispersing[0] = True
        pop.dispersal_start_x[0] = pop.x[0]; pop.dispersal_start_y[0] = pop.y[0]
        pop.dispersal_target_distance[0] = 1e9
        d = (np.array([0.05], dtype=np.float64), np.array([0.0], dtype=np.float64))
        t = (np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64))
        pop.step(deterrence_vectors=d, turbine_deterrence_vectors=t)
        assert bool(pop.is_dispersing[0]) is True
```
(Match the real positional/kwarg order of `jax_dispersal_update` as you edit it — read the current signature at `jax_kernels.py:812`.)

- [ ] **Step 2: Run, verify FAIL**
`... python3 -m pytest tests/test_ship_deterrence_port.py -k TurbineOnlyDispersalJax -v` → FAIL (`jax_dispersal_update` has no `turbine_deter_strength` param).

- [ ] **Step 3: Rename the param + gate in `jax_kernels.py`.** In `jax_dispersal_update` (~812): rename its `deter_strength` parameter to `turbine_deter_strength`, and change the gate (~843) from:
```python
    deterred = dispersing & (deter_strength > 0)
```
to:
```python
    deterred = dispersing & (turbine_deter_strength > 0)
```

- [ ] **Step 4: Thread it through `tick_jax.py::jax_tick_energy`.** In `src/cenop/optimizations/tick_jax.py`, rename the `deter_strength` parameter (the one in the "Dispersal state" group, ~line 280, immediately after `days_declining_energy`) to `turbine_deter_strength`. Then at the `jax_dispersal_update(...)` call (~360), change the argument it passes from `deter_strength` to `turbine_deter_strength`:
```python
            days_declining_energy,
            x,
            y,
            turbine_deter_strength,
            new_energy_history,
            new_active_mask,
            is_day_boundary,
        )
```
(Leave the separate `is_disturbed` and `deter_magnitude` params — those drive BMR/reporting and stay combined.)

- [ ] **Step 5: Point the JAX dispersal arg at the turbine buffer in `population.py::_step_jax`.** No signature change needed — `self._turbine_deter_strength` is already populated by `step()` (Task 4 Step 4) before the JAX dispatch. In the `jax_tick_energy(...)` call (~2408-2430), change the dispersal-strength argument — the `jnp.asarray(self.deter_strength)` positioned **immediately after `days_declining_energy`** (~line 2431, which feeds `jax_dispersal_update` for the dispersal gate — distinct from the earlier `self.deter_strength > 0` at ~2415 and `self.deter_strength` at ~2416 that feed `is_disturbed`/`deter_magnitude`) — to:
```python
            jnp.asarray(self._turbine_deter_strength),
```
(Leave the earlier `jnp.asarray(self.deter_strength > 0)` and `jnp.asarray(self.deter_strength)` that feed `is_disturbed`/`deter_magnitude` unchanged — those stay combined.)

- [ ] **Step 6: Update the two existing JAX tests that call `jax_dispersal_update`.** In `tests/test_jax_tick.py`, both `test_deterrence_cancels_dispersal` (~1265) and `test_distance_completion` (~1295) pass `deter_strength=...`; rename that kwarg to `turbine_deter_strength=...`. Update `test_deterrence_cancels_dispersal` to assert turbine deterrence cancels dispersal (its intent is preserved — it just feeds the turbine-only input now).

- [ ] **Step 7: Run, verify PASS (GPU free — no concurrent sim)**
`... python3 -m pytest tests/test_ship_deterrence_port.py -k TurbineOnlyDispersalJax tests/test_jax_tick.py -q` → PASS.

- [ ] **Step 8: Commit**
```bash
git add src/cenop/optimizations/jax_kernels.py src/cenop/optimizations/tick_jax.py src/cenop/agents/population.py tests/test_jax_tick.py tests/test_ship_deterrence_port.py
git commit -m "feat: dispersal deactivation gates on turbine-only deterrence (JAX path)"
```

---

## Task 6: Regenerate ship baseline + provenance + whole-suite verification

**Files:** Regenerate `output/kattegat_ref_ships/`; update `output/kattegat_ref_ships/PROVENANCE.txt`; update the stale comment in `tests/test_integration.py::test_ship_manager_creates_deterrence_vectors`.

- [ ] **Step 1: Pin the integration test's SL + fix its comment.** In `tests/test_integration.py::test_ship_manager_creates_deterrence_vectors` (~163), the CARGO ship is constructed without an explicit SL → it now uses JOMOPANS (~128–134 dB vs old 175), narrowing the margin. Set an explicit override so the assertion is guaranteed, e.g. after the `Ship(...)` construction add `ship.noise.base_source_level = 200.0`, and fix the stale comment (the "~175 dB / threshold 158 / 7 m" text — actual Tships=80). Run `... python3 -m pytest tests/test_integration.py -q` → PASS.

- [ ] **Step 2: Whole suite green**
`... python3 -m pytest tests/ -q --ignore=tests/test_depons_physiology.py --ignore=tests/test_validation.py` → PASS (JAX runs with GPU free). Fix any straggler that pinned old ship-SL/dispersal behavior.

- [ ] **Step 3: Commit the test/comment fix**
```bash
git add tests/test_integration.py
git commit -m "test: pin SL + fix stale comment in ship-deterrence integration test"
```

- [ ] **Step 4: Regenerate the baseline**
`eval "$(micromamba shell hook --shell bash)" && micromamba activate shiny && cd /home/razinka/cenjas/CENOP && python3 scripts/run_kattegat_reference.py --count 2000 --years 2 --seed 42 --ships --out output/kattegat_ref_ships` (≈1 hr; run in background). On completion confirm: population stable; `deter_strength` still nonzero (ship deterrence live); SLs now vary by class/length (sanity-check a few `PorpoiseStatistics` rows or the run log — not a constant 170).

- [ ] **Step 5: Update PROVENANCE + commit compact files**
Update `output/kattegat_ref_ships/PROVENANCE.txt`: new producing commit (`git rev-parse HEAD`), date 2026-06-04, and note this baseline reflects JOMOPANS per-vessel source levels + turbine-only dispersal deactivation (ships no longer stop dispersal) vs the prior baseline.
```bash
git add output/kattegat_ref_ships/Population.txt output/kattegat_ref_ships/Energy.txt output/kattegat_ref_ships/Mortality.txt output/kattegat_ref_ships/PROVENANCE.txt
git commit -m "test: regenerate Kattegat ship baseline (JOMOPANS SL + turbine-only dispersal)"
```

---

## Notes / non-goals (from the spec)
- **Non-goals:** sub-tick interpolation; the off-production scalar `Porpoise.deter()` (deactivates dispersal on any deterrence — legacy, not on the production path); the off-production scalar `ShipManager.calculate_aggregate_deterrence` simple-TL. Cython tick path does no deterrence/dispersal-deactivation → no change.
- **DEPONS parity confirmed:** band 12 (`Ship.java:53 JOMOPANS_BAND`), speed-0→0, turbine-only deactivation (`Porpoise.java:1277`), JOMOPANS tables match `JomopansEchoSPL.java`.
- **`base_source_level` override is a CENOP extension** (DEPONS always computes JOMOPANS); DEPONS-faithful runs leave it `None`.
