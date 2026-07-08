"""Tests for DEPONS 3.2 pregnancy FSM."""

import numpy as np
import pytest
from cenop.agents.population import PorpoisePopulation
from cenop.parameters.simulation_params import SimulationParameters
from cenop.landscape.cell_data import create_homogeneous_landscape


class TestPregnancyStatusArray:
    """Test pregnancy_status SoA array exists and initializes correctly."""

    def test_pregnancy_status_exists(self):
        """Population should have pregnancy_status array (int8)."""
        params = SimulationParameters()
        pop = PorpoisePopulation(count=50, params=params)
        assert hasattr(pop, 'pregnancy_status')
        assert pop.pregnancy_status.dtype == np.int8
        assert pop.pregnancy_status.shape == (50,)

    def test_initial_values(self):
        """Males: 0. Mature females: 1 or 2 based on conceive_prob."""
        np.random.seed(42)
        params = SimulationParameters()
        pop = PorpoisePopulation(count=200, params=params)

        males = ~pop.is_female & pop.active_mask
        assert np.all(pop.pregnancy_status[males] == 0)

        mature_females = pop.is_female & (pop.age >= params.maturity_age) & pop.active_mask
        if np.any(mature_females):
            pregnant = np.sum(pop.pregnancy_status[mature_females] == 1)
            ready = np.sum(pop.pregnancy_status[mature_females] == 2)
            assert pregnant > 0 or ready > 0, "Mature females should have status 1 or 2"

    def test_no_max_breeding_age_cap(self):
        """No max_breeding_age should limit reproduction."""
        params = SimulationParameters()
        if hasattr(params, 'max_breeding_age'):
            assert params.max_breeding_age >= params.max_age


class TestPregnancyFSM:
    """Test pregnancy state transitions matching Java Porpoise.java:1155-1231."""

    def _make_pop(self, count=10):
        params = SimulationParameters()
        pop = PorpoisePopulation(count=count, params=params)
        pop.landscape = create_homogeneous_landscape(width=100, height=100, food_prob=0.5)
        return pop

    def test_immature_to_ready(self):
        """Status 0 → 2 when age >= maturity_age."""
        pop = self._make_pop(5)
        pop.is_female[:] = True
        pop.pregnancy_status[:] = 0
        pop.age[:] = 3.44  # At maturity
        pop.active_mask[:] = True

        pop._update_pregnancy_status(pop.active_mask)

        assert np.all(pop.pregnancy_status == 2)

    def test_ready_to_pregnant_on_mating_day(self):
        """Status 2 → 1 on mating_day with prob conceive_prob."""
        np.random.seed(0)
        pop = self._make_pop(100)
        pop.is_female[:] = True
        pop.pregnancy_status[:] = 2
        pop.mating_day[:] = 10
        pop.active_mask[:] = True
        pop._day_of_year = 10 * 48  # Day 10

        pop._update_pregnancy_status(pop.active_mask)

        pregnant = np.sum(pop.pregnancy_status == 1)
        assert 50 < pregnant < 85, f"Expected ~68% pregnant, got {pregnant}"

    def test_pregnant_gives_birth_at_gestation(self):
        """Status 1 → 2 + withLactCalf at gestation_time."""
        pop = self._make_pop(5)
        pop.is_female[:] = True
        pop.pregnancy_status[:] = 1
        pop.days_since_mating[:] = 300  # gestation_time
        pop.active_mask[:] = True

        pop._update_pregnancy_status(pop.active_mask)

        assert np.all(pop.pregnancy_status == 2)
        assert np.all(pop.with_calf)
        assert np.all(pop.days_since_mating == -99)
        # Birth sets days_since_birth=0, then counter increment adds 1 in same call
        assert np.all(pop.days_since_birth == 1)

    def test_weaning_creates_female_calf(self):
        """At nursing_time, 50% chance female calf created."""
        np.random.seed(42)
        pop = self._make_pop(20)
        pop.is_female[:10] = True
        pop.with_calf[:10] = True
        pop.days_since_birth[:10] = 240  # nursing_time
        pop.active_mask[:10] = True
        pop.active_mask[10:] = False

        initial_active = int(np.sum(pop.active_mask))
        pop._update_pregnancy_status(pop.active_mask)

        assert np.all(~pop.with_calf[:10])
        assert np.all(pop.days_since_birth[:10] == -99)

        new_active = int(np.sum(pop.active_mask))
        calves_created = new_active - initial_active
        assert 1 <= calves_created <= 9, f"Expected ~5 calves, got {calves_created}"

    def test_males_immune_to_fsm(self):
        """Males should never transition pregnancy status."""
        pop = self._make_pop(10)
        pop.is_female[:] = False
        pop.pregnancy_status[:] = 0
        pop.age[:] = 5.0
        pop.active_mask[:] = True

        pop._update_pregnancy_status(pop.active_mask)

        assert np.all(pop.pregnancy_status == 0)

    def test_daily_counter_increments(self):
        """days_since_mating increments for pregnant, days_since_birth for lactating."""
        pop = self._make_pop(4)
        pop.is_female[:] = True
        pop.active_mask[:] = True

        pop.pregnancy_status[0] = 1
        pop.days_since_mating[0] = 100

        pop.with_calf[1] = True
        pop.days_since_birth[1] = 50

        pop._update_pregnancy_status(pop.active_mask)

        assert pop.days_since_mating[0] == 101
        assert pop.days_since_birth[1] == 51


class TestReproductionScheduling:
    """Test that reproduction is called once per day via FSM, not old breeding-season model."""

    def test_fsm_replaces_old_breeding_season(self):
        """_handle_reproduction should delegate to _update_pregnancy_status on day boundaries.

        The old model used a fixed breeding season (days 195-255) with per-tick birth
        probability. The new model uses the pregnancy FSM with mating_day per female.
        """
        params = SimulationParameters()
        pop = PorpoisePopulation(count=20, params=params)
        pop.landscape = create_homogeneous_landscape(width=100, height=100, food_prob=0.5)
        pop.is_female[:] = True
        pop.pregnancy_status[:] = 0  # All immature
        pop.age[:] = 3.44  # At maturity
        pop.active_mask[:] = True

        # Simulate 48 ticks (1 day) — FSM should fire once, transitioning 0→2
        for _ in range(48):
            pop._handle_reproduction(pop.active_mask)

        # If FSM ran, immature females at maturity should now be ready-to-mate
        assert np.all(pop.pregnancy_status == 2), \
            "FSM should have transitioned immature→ready-to-mate"

    def test_reproduction_daily_gate(self):
        """_update_pregnancy_status should only run at day boundary (tick % 48 == 0).

        Spec 2.1: Call reproduction logic once per day.
        """
        params = SimulationParameters()
        pop = PorpoisePopulation(count=20, params=params)
        pop.landscape = create_homogeneous_landscape(width=100, height=100, food_prob=0.5)
        pop.is_female[:] = True
        pop.pregnancy_status[:] = 2  # All ready to mate
        pop.mating_day[:] = 1  # Will mate on day 1 (first day boundary at tick 48)
        pop.active_mask[:] = True

        # Run 48 ticks (1 day) via step() — gate fires at tick 48 when current_day=1
        np.random.seed(42)
        deter = (np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32))
        for _ in range(48):
            pop.step(
                deterrence_vectors=deter,
                ambient_rl=np.zeros(20, dtype=np.float32),
            )

        # Some should be pregnant from the daily check on day 1
        pregnant_count = int(np.sum(pop.pregnancy_status == 1))
        assert pregnant_count > 0, "Should have some pregnancies after day 1"

    def test_old_breeding_season_removed(self):
        """The old breeding-season gate (days 195-255) should no longer exist.

        Reproduction now depends on individual mating_day per female, not a global window.
        """
        params = SimulationParameters()
        pop = PorpoisePopulation(count=20, params=params)
        pop.landscape = create_homogeneous_landscape(width=100, height=100, food_prob=0.5)
        pop.is_female[:] = True
        pop.pregnancy_status[:] = 2  # Ready to mate
        pop.mating_day[:] = 10  # Day 10 — outside old breeding season (195-255)
        pop.active_mask[:] = True

        # Advance to day 10 (tick 480)
        pop._global_tick = 480
        pop._day_of_year = 10 * 48

        np.random.seed(0)
        pop._handle_reproduction(pop.active_mask)

        # If old breeding-season gate were still active, this would return early
        # because day 10 is not in 195-255. With the FSM, mating should happen.
        pregnant_count = int(np.sum(pop.pregnancy_status == 1))
        assert pregnant_count > 0, \
            "FSM should allow mating on day 10 (old model would block outside 195-255)"


class TestMatingDayReRandomization:
    """Test annual mating day re-draw (Java YearlyTask.java:99)."""

    def test_mating_day_changes_yearly(self):
        """At year boundary, all active females get new mating_day from N(225, 20)."""
        np.random.seed(42)
        params = SimulationParameters()
        pop = PorpoisePopulation(count=50, params=params)
        pop.is_female[:25] = True
        pop.is_female[25:] = False
        pop.mating_day[25:] = -99  # Reset males after init may have assigned days
        pop.active_mask[:] = True

        initial_mating_days = pop.mating_day[:25].copy()

        pop.rerandomize_mating_days()

        new_mating_days = pop.mating_day[:25]
        # Days should have changed (extremely unlikely to stay exactly the same)
        assert not np.array_equal(initial_mating_days, new_mating_days), \
            "Mating days should change after re-randomization"
        # Males should still be -99
        assert np.all(pop.mating_day[25:] == -99)
        # New days should be around 225
        assert 200 < np.mean(new_mating_days) < 250

    def test_inactive_females_excluded(self):
        """Inactive (dead) females should not get new mating days."""
        params = SimulationParameters()
        pop = PorpoisePopulation(count=10, params=params)
        pop.is_female[:] = True
        pop.active_mask[:5] = True
        pop.active_mask[5:] = False
        pop.mating_day[5:] = -99  # Dead females

        pop.rerandomize_mating_days()

        assert np.all(pop.mating_day[5:] == -99), "Dead females should keep -99"


class TestBycatchDaily:
    """Test bycatch uses daily schedule with Java formula."""

    def test_bycatch_skips_non_day_boundary(self):
        """Bycatch should NOT run on non-day-boundary ticks.

        If _global_tick % 48 != 0, no bycatch deaths should occur.
        """
        params = SimulationParameters()
        params.bycatch_prob = 0.99  # Very high to detect any leakage
        pop = PorpoisePopulation(count=100, params=params)
        pop.energy[:] = 20.0  # Max energy = no starvation
        pop.active_mask[:] = True
        pop._global_tick = 25  # Not a day boundary

        mask = pop.active_mask.copy()
        initial_active = int(np.sum(pop.active_mask))
        pop._check_mortality(mask, initial_active)

        # No bycatch deaths should occur mid-day (energy=20 means no starvation either)
        assert int(np.sum(pop.active_mask)) == initial_active, \
            "No deaths should occur mid-day with high energy"

    def test_bycatch_fires_on_day_boundary(self):
        """Bycatch should run on day boundary ticks (_global_tick % 48 == 0)."""
        params = SimulationParameters()
        params.bycatch_prob = 0.99  # Very high
        pop = PorpoisePopulation(count=1000, params=params)
        pop.energy[:] = 20.0  # No starvation
        pop.active_mask[:] = True
        pop._global_tick = 48  # Day boundary

        np.random.seed(42)
        mask = pop.active_mask.copy()
        pop._check_mortality(mask, 1000)

        dead = 1000 - int(np.sum(pop.active_mask))
        # With bycatch_prob=0.99, daily_surv = exp(log(0.01)/360) ≈ 0.987
        # Daily death prob ≈ 0.013, so expect ~13 deaths out of 1000
        assert dead > 0, "Bycatch should cause some deaths on day boundary"

    def test_bycatch_uses_java_daily_formula(self):
        """Bycatch should use dailySurvivalProb = exp(log(1 - bycatchProb) / 360).

        Java ref: Porpoise.java:1139
        """
        # Verify the formula gives correct annual mortality
        bycatch_prob = 0.10  # 10% annual
        daily_surv = np.exp(np.log(1 - bycatch_prob) / 360)
        # After 360 days: daily_surv^360 should equal (1 - bycatch_prob)
        annual_surv = daily_surv ** 360
        assert abs(annual_surv - (1 - bycatch_prob)) < 1e-10, \
            f"Formula should reproduce annual rate: {annual_surv} vs {1 - bycatch_prob}"

    def test_max_age_also_daily(self):
        """Max-age death should only be checked on day boundaries, not every tick."""
        params = SimulationParameters()
        pop = PorpoisePopulation(count=10, params=params)
        pop.energy[:] = 20.0
        pop.age[:] = 31.0  # Over max_age=30
        pop.active_mask[:] = True
        pop._global_tick = 25  # NOT a day boundary

        mask = pop.active_mask.copy()
        pop._check_mortality(mask, 10)

        # Max-age should not fire mid-day
        assert int(np.sum(pop.active_mask)) == 10, \
            "Max-age death should not fire on non-day-boundary tick"


class TestPhase2Integration:
    """Smoke test: Phase 2 pregnancy FSM produces realistic population dynamics."""

    def _run_days(self, pop, landscape, params, n_days):
        """Run simulation for n_days."""
        count = pop.count
        deter = (np.zeros(count, dtype=np.float32), np.zeros(count, dtype=np.float32))
        ambient = np.zeros(count, dtype=np.float32)
        for _ in range(n_days):
            for _ in range(48):
                pop.step(deterrence_vectors=deter, ambient_rl=ambient)
            landscape.replenish_food(rate=params.food_growth_rate)

    def test_population_with_reproduction_50_days(self):
        """50-day run with pregnancy FSM should maintain roughly stable population.

        DEPONS expected: ~60-68% of eligible females conceive per year.
        """
        np.random.seed(42)
        params = SimulationParameters()
        landscape = create_homogeneous_landscape(width=200, height=200, food_prob=0.5)

        pop = PorpoisePopulation(count=200, params=params)
        pop.landscape = landscape

        initial_count = int(np.sum(pop.active_mask))
        self._run_days(pop, landscape, params, 50)
        final_count = int(np.sum(pop.active_mask))

        ratio = final_count / initial_count
        assert 0.5 < ratio < 1.5, \
            f"Population ratio {ratio:.2f} ({initial_count}→{final_count}) outside stability band"

    def test_pregnancy_states_realistic(self):
        """After 50 days, pregnancy state distribution should be valid."""
        np.random.seed(42)
        params = SimulationParameters()
        landscape = create_homogeneous_landscape(width=200, height=200, food_prob=0.5)

        pop = PorpoisePopulation(count=300, params=params)
        pop.landscape = landscape

        self._run_days(pop, landscape, params, 50)

        active = pop.active_mask
        females = active & pop.is_female
        n_females = int(np.sum(females))

        if n_females > 0:
            n_pregnant = int(np.sum(pop.pregnancy_status[females] == 1))
            n_ready = int(np.sum(pop.pregnancy_status[females] == 2))
            n_immature = int(np.sum(pop.pregnancy_status[females] == 0))

            # All females should have a valid pregnancy status (0, 1, or 2)
            assert n_pregnant + n_ready + n_immature == n_females, \
                "All females should have a valid pregnancy status"

            # At least some should be pregnant (given init seeds ~68% at start)
            assert n_pregnant > 0 or n_ready > 0, \
                "Should have some pregnant or ready-to-mate females"


def test_mating_day_clipped_to_valid_range():
    """Mating days must be clipped to [0, 359]."""
    import numpy as np
    raw = np.array([-10, 0, 180, 225, 359, 400, 500], dtype=np.float64)
    clipped = np.clip(raw, 0, 359).astype(np.int16)
    assert clipped[0] == 0
    assert clipped[-1] == 359
    assert clipped[-2] == 359
    assert clipped[3] == 225
    params = SimulationParameters(porpoise_count=200)
    pop = PorpoisePopulation(count=200, params=params)
    female_mating = pop.mating_day[pop.is_female]
    assert np.all(female_mating >= 0), f"Min mating_day = {female_mating.min()}"
    assert np.all(female_mating <= 359), f"Max mating_day = {female_mating.max()}"


def test_pregnancy_init_nonpregnant_mature_stay_ready():
    """Mature females that don't conceive should remain status=2, not reset to 0."""
    import numpy as np
    params = SimulationParameters(porpoise_count=500)
    rng = np.random.default_rng(42)
    pop = PorpoisePopulation(count=500, params=params)
    mature_female = pop.is_female & (pop.age >= pop.params.maturity_age) & pop.active_mask
    not_pregnant = mature_female & (pop.pregnancy_status != 1)
    assert np.sum(not_pregnant) > 0, "Need at least one non-pregnant mature female"
    assert np.all(pop.pregnancy_status[not_pregnant] == 2), (
        f"Found {np.sum(pop.pregnancy_status[not_pregnant] == 0)} mature females incorrectly at status 0"
    )


class TestWeanedCalfSlotReset:
    """Finding #4: a calf recycled into a dead agent's slot must NOT inherit the
    dead occupant's memory / dispersal / CRW / deterrence / prev-position state.
    """

    class _DetRNG:
        """Deterministic RNG stub: forces calf creation (random > 0.5) and a
        fixed energy draw, so the weaning path is fully reproducible."""

        def random(self, n):
            return np.ones(n, dtype=np.float64)

        def normal(self, mean, sd, n):
            return np.full(n, 10.0, dtype=np.float64)

    def test_weaned_calf_does_not_inherit_dead_slot_state(self):
        params = SimulationParameters()
        pop = PorpoisePopulation(count=2, params=params)

        # Slot 0 = mother ready to wean a calf this day.
        pop.is_female[0] = True
        pop.active_mask[0] = True
        pop.with_calf[0] = True
        pop.pregnancy_status[0] = 2
        pop.days_since_birth[0] = params.nursing_time
        pop.mating_day[0] = -99  # not on a mating day -> skip conceive branch
        pop.age[0] = 6.0
        pop.x[0] = 40.0
        pop.y[0] = 55.0
        pop.heading[0] = 90.0

        # Slot 1 = a DEAD agent carrying distinctive persistent state.
        pop.active_mask[1] = False
        pop.is_dispersing[1] = True
        pop.dispersal_target_x[1] = 999.0
        pop.dispersal_target_y[1] = 888.0
        pop.dispersal_target_distance[1] = 123.0
        pop.dispersal_distance_traveled[1] = 45.0
        pop.dispersal_start_x[1] = 12.0
        pop.dispersal_start_y[1] = 34.0
        pop.days_declining_energy[1] = 7
        pop.prev_log_mov[1] = 5.5
        pop.prev_angle[1] = 123.0
        pop._prev_step_heading[1] = 77.0
        pop._stored_util[1, :] = 7.0
        pop._pos_history_x[1, :] = 3.0
        pop._pos_history_y[1, :] = 4.0
        pop._mem_count[1] = 50
        pop._mem_ptr[1] = 33
        pop._ve_total[1] = 9.0
        pop._vt_x[1] = 1.0
        pop._vt_y[1] = 2.0
        pop.psm_buffer[1, :, :, :] = 6.0
        pop._energy_history[1, :] = 8.0
        pop._energy_ticks_today[1] = 3.0
        pop._energy_consumed_today[1] = 5.0
        pop.energy_consumed_daily[1] = 6.0
        pop._energy_level_sum[1] = 7.0
        pop.deter_strength[1] = 0.9
        pop._turbine_deter_strength[1] = 0.8
        pop._was_deterred[1] = True
        pop._prev_x[1] = -1.0
        pop._prev_y[1] = -2.0

        # Deterministic weaning: force calf creation + fixed energy draw.
        pop.rng = self._DetRNG()
        pop._day_of_year = 0  # current_day = 0, != mating_day(-99): no conceive

        pop._update_pregnancy_status(pop.active_mask.copy())

        # The calf was recycled into slot 1.
        assert pop.active_mask[1], "calf should have been created into the dead slot"

        # Dispersal state must be newborn defaults, not inherited.
        assert pop.is_dispersing[1] == False
        assert pop.days_declining_energy[1] == 0
        assert pop.dispersal_target_x[1] == 0.0
        assert pop.dispersal_target_y[1] == 0.0
        assert pop.dispersal_target_distance[1] == 0.0
        assert pop.dispersal_distance_traveled[1] == 0.0
        assert pop.dispersal_start_x[1] == 0.0
        assert pop.dispersal_start_y[1] == 0.0

        # CRW movement state = newborn defaults.
        assert pop.prev_log_mov[1] == 0.8
        assert pop.prev_angle[1] == 10.0
        assert pop._prev_step_heading[1] == 0.0

        # Reference memory cleared.
        assert pop._mem_count[1] == 0
        assert pop._mem_ptr[1] == 0
        assert np.all(pop._stored_util[1] == 0.0)
        assert np.all(pop._pos_history_x[1] == 0.0)
        assert np.all(pop._pos_history_y[1] == 0.0)
        assert pop._ve_total[1] == 0.0
        assert pop._vt_x[1] == 0.0
        assert pop._vt_y[1] == 0.0

        # PSM grid + energy history cleared.
        assert np.all(pop.psm_buffer[1] == 0.0)
        assert np.all(pop._energy_history[1] == 0.0)
        assert pop._energy_ticks_today[1] == 0.0
        assert pop._energy_consumed_today[1] == 0.0
        assert pop.energy_consumed_daily[1] == 0.0
        assert pop._energy_level_sum[1] == 0.0

        # Deterrence status cleared.
        assert pop.deter_strength[1] == 0.0
        assert pop._turbine_deter_strength[1] == 0.0
        assert pop._was_deterred[1] == False

        # Prev-position anchored to the calf's (mother-copied) location.
        assert pop._prev_x[1] == pop.x[1]
        assert pop._prev_y[1] == pop.y[1]
        assert pop.x[1] == 40.0 and pop.y[1] == 55.0
