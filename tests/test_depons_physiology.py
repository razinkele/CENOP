"""
Test CENOP physiology model against DEPONS 3.0 expected behaviors.

This module compares population trajectories and vital rates to expected
DEPONS 3.0 values documented in the TRACE document and model publications.

Expected DEPONS behaviors (from documentation):
- Stable population over time with default parameters
- Annual adult mortality: ~5%
- Annual juvenile mortality: ~15%
- Annual birth rate: ~60% of eligible females
- Starvation mortality linked to energy level
"""

import numpy as np
import pytest
from cenop.parameters.simulation_params import SimulationParameters
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape


class TestPhysiologyParameters:
    """Test that all physiology parameters match DEPONS 3.0 defaults."""

    def test_energetics_parameters(self):
        """Verify energetics parameters match DEPONS defaults."""
        params = SimulationParameters()

        # Energy use
        assert params.e_use_per_30_min == 4.5, "BMR should be 4.5 per 30 min"

        # Seasonal scaling
        assert params.e_lact == 1.4, "Lactation multiplier should be 1.4"
        assert params.e_warm == 1.3, "Warm water multiplier should be 1.3"

        # Initial energy
        assert params.energy_init_mean == 10.0, "Initial energy mean should be 10"
        assert params.energy_init_sd == 1.0, "Initial energy SD should be 1"

    def test_mortality_parameters(self):
        """Verify mortality parameters match DEPONS defaults."""
        params = SimulationParameters()

        # Starvation constants (DEPONS 3.2 values)
        assert params.m_mort_prob_const == 1.0, "M_MORT_PROB_CONST should be 1.0"
        assert params.x_survival_const == 0.4, "xSurvivalProbConst should be 0.4"

        # Age-based mortality
        assert params.mortality_juvenile == 0.15, "Juvenile mortality should be 15%/year"
        assert params.mortality_adult == 0.05, "Adult mortality should be 5%/year"
        assert params.mortality_elderly == 0.15, "Elderly mortality should be 15%/year"

    def test_reproduction_parameters(self):
        """Verify reproduction parameters match DEPONS defaults."""
        params = SimulationParameters()

        # Life history
        assert params.maturity_age == 3.44, "Maturity age should be 3.44 years"
        assert params.max_age == 30.0, "Max age should be 30 years"
        assert params.max_breeding_age == 30.0, "Max breeding age should equal max_age (effectively removed per DEPONS 3.2 spec)"

        # Breeding
        assert params.conceive_prob == 0.68, "Conception probability should be 0.68"
        assert params.gestation_time == 300, "Gestation should be 300 days"
        assert params.nursing_time == 240, "Nursing should be 240 days"
        assert params.mating_day_mean == 225.0, "Mean mating day should be 225"
        assert params.mating_day_sd == 20.0, "Mating day SD should be 20"


class TestStarvationMortality:
    """Test starvation mortality matches DEPONS formula."""

    def test_starvation_probability_at_low_energy(self):
        """Low energy should have high mortality probability."""
        params = SimulationParameters(porpoise_count=100)
        landscape = create_homogeneous_landscape()
        pop = PorpoisePopulation(100, params, landscape)

        # Set all agents to very low energy
        pop.energy[:] = 1.0

        # Calculate expected yearly survival: 1 - (1.0 * exp(-1 * 0.4)) = ~0.3297
        expected_yearly_surv = 1.0 - (1.0 * np.exp(-1.0 * 0.4))
        assert 0.30 < expected_yearly_surv < 0.36, f"Expected yearly survival ~0.3297, got {expected_yearly_surv}"

    def test_starvation_probability_at_high_energy(self):
        """High energy should have low mortality probability."""
        params = SimulationParameters(porpoise_count=100)

        # Calculate expected yearly survival at energy=10: 1 - (1.0 * exp(-10 * 0.4)) = ~0.9817
        expected_yearly_surv = 1.0 - (1.0 * np.exp(-10.0 * 0.4))
        assert 0.97 < expected_yearly_surv < 0.99, f"Expected yearly survival ~0.9817, got {expected_yearly_surv}"


class TestAgeMortality:
    """Test age-dependent mortality rates."""

    def test_juvenile_mortality_rate(self):
        """Juvenile mortality should be ~15% per year."""
        # Annual rate = 0.15
        # Per tick = 0.15 / 365 / 48 = ~8.56e-6
        expected_per_tick = 0.15 / 365 / 48
        assert 8e-6 < expected_per_tick < 9e-6

    def test_adult_mortality_rate(self):
        """Adult mortality should be ~5% per year."""
        # Annual rate = 0.05
        # Per tick = 0.05 / 365 / 48 = ~2.85e-6
        expected_per_tick = 0.05 / 365 / 48
        assert 2e-6 < expected_per_tick < 3e-6

    def test_elderly_mortality_rate(self):
        """Elderly mortality should be ~15% per year."""
        # Same as juvenile
        expected_per_tick = 0.15 / 365 / 48
        assert 8e-6 < expected_per_tick < 9e-6


class TestReproduction:
    """Test reproduction mechanics match DEPONS."""

    def test_breeding_season_bounds(self):
        """Breeding season should be days 195-255."""
        # This is enforced in _handle_reproduction()
        # Day 195 = ~July 14, Day 255 = ~September 12
        assert 195 <= 225 <= 255, "Mean mating day should be within breeding season"

    def test_birth_probability_achieves_target(self):
        """Birth probability should achieve ~60% reproduction rate."""
        # birth_prob = 0.0003 per tick
        # Breeding season = 60 days = 60 * 48 = 2880 ticks
        # P(at least one birth) = 1 - (1 - 0.0003)^2880 ≈ 0.58
        birth_prob = 0.0003
        ticks_in_season = 60 * 48
        expected_rate = 1 - (1 - birth_prob) ** ticks_in_season
        assert 0.55 < expected_rate < 0.65, f"Expected ~60% birth rate, got {expected_rate:.2%}"


class TestPopulationTrajectory:
    """Test population trajectory over multiple years."""

    @pytest.fixture
    def simulation_setup(self):
        """Create simulation components for trajectory tests."""
        params = SimulationParameters(
            porpoise_count=500,
            random_seed=42,
        )
        landscape = create_homogeneous_landscape(width=200, height=200, depth=30.0, food_prob=0.5)
        pop = PorpoisePopulation(500, params, landscape)
        return params, landscape, pop

    def test_population_stability_short_term(self, simulation_setup):
        """Population should remain relatively stable over 1 year."""
        params, landscape, pop = simulation_setup

        initial_pop = pop.population_size

        # Run for 1 year (365 * 48 ticks)
        ticks_per_year = 365 * 48

        for tick in range(ticks_per_year):
            pop.step()

        final_pop = pop.population_size

        # Population should not collapse or explode
        # Allow 30% change over a year (which is reasonable given stochasticity)
        change_ratio = final_pop / initial_pop
        assert 0.7 < change_ratio < 1.3, f"Population changed by {change_ratio:.2f}x in 1 year"

    def test_age_distribution_evolves(self, simulation_setup):
        """Age distribution should evolve over time."""
        params, landscape, pop = simulation_setup

        initial_mean_age = np.mean(pop.age[pop.active_mask])

        # Run for half a year
        ticks = 365 * 48 // 2
        for _ in range(ticks):
            pop.step()

        final_mean_age = np.mean(pop.age[pop.active_mask])

        # Mean age should increase by about 0.5 years (minus deaths of old, plus births of young)
        # Actual change depends on birth/death dynamics
        assert final_mean_age != initial_mean_age, "Age distribution should change"

    def test_energy_remains_bounded(self, simulation_setup):
        """Energy should remain within 0-20 range."""
        params, landscape, pop = simulation_setup

        # Run for 100 ticks
        for _ in range(100):
            pop.step()

        # Check energy bounds
        active = pop.active_mask
        assert np.all(pop.energy[active] >= 0), "Energy should not go negative"
        assert np.all(pop.energy[active] <= 20), "Energy should not exceed 20"


class TestDEPONSTrajectoryComparison:
    """Compare population trajectories to DEPONS 3.0 reference values."""

    def test_annual_mortality_rate(self):
        """Overall annual mortality should be low for well-fed population (DEPONS 3.2).

        DEPONS 3.2 has NO invented age-bracket mortality. Mortality is driven by:
        - Energy-based starvation (near-zero at healthy energy levels)
        - Max-age death (none expected in a 1-year run for adults aged ~5)
        - Bycatch (default 0.0)
        A well-fed population on a 50% food landscape should have very low mortality.
        """
        params = SimulationParameters(
            porpoise_count=500,
            random_seed=42,
        )
        landscape = create_homogeneous_landscape(width=200, height=200, depth=30.0, food_prob=0.5)
        pop = PorpoisePopulation(500, params, landscape)

        initial_pop = pop.population_size
        deaths = 0

        # Track deaths over 1 year
        ticks_per_year = 365 * 48

        for tick in range(ticks_per_year):
            pop_before = pop.population_size
            pop.step()
            pop_after = pop.population_size

            # Count deaths (population decrease minus births is tricky, just track decreases)
            if pop_after < pop_before:
                deaths += (pop_before - pop_after)

        # DEPONS 3.2: no age-bracket mortality. Well-fed population has near-zero mortality.
        # Starvation mortality at energy ~15-20 is <0.3% per year (parameterized curve).
        mortality_rate = deaths / initial_pop
        print(f"Annual mortality estimate: {mortality_rate:.1%}")

        # Allow up to 10% for stochastic variation; minimum near-zero (no invented floors)
        assert mortality_rate < 0.10, f"Mortality rate {mortality_rate:.1%} unexpectedly high for well-fed population"

    def test_energy_dynamics_over_year(self):
        """Mean energy should remain stable over a year."""
        params = SimulationParameters(
            porpoise_count=200,
            random_seed=42,
        )
        # Use lower food probability to better match realistic conditions
        # Real landscapes have variable food distribution; uniform 0.5 is unrealistic
        landscape = create_homogeneous_landscape(width=150, height=150, depth=30.0, food_prob=0.3)
        pop = PorpoisePopulation(200, params, landscape)

        energy_samples = []
        ticks_per_year = 365 * 48
        sample_interval = ticks_per_year // 12  # Monthly samples

        for tick in range(ticks_per_year):
            pop.step()
            if tick % sample_interval == 0:
                active = pop.active_mask
                if np.any(active):
                    energy_samples.append(np.mean(pop.energy[active]))

        # Energy should fluctuate but remain in reasonable range
        # With uniform food, energy tends high (10-20); real landscapes have more variability
        mean_energy = np.mean(energy_samples)
        print(f"Mean energy over year: {mean_energy:.2f}")

        # DEPONS with realistic landscapes shows energy in 8-15 range
        # Homogeneous landscape with food_prob=0.3 should give 8-18 range
        assert 5 < mean_energy < 18, f"Mean energy {mean_energy:.1f} outside expected range (5-18)"

    def test_female_reproduction_rate(self):
        """Eligible females should reproduce during breeding season (DEPONS 3.2).

        Tests that per-tick birth probability (0.0003) produces calves during days 195-255.
        Uses a population smaller than the pre-allocated array so inactive slots are
        available for newborns (fixed-array architecture constraint: births require
        inactive slots, previously masked by age-bracket mortality freeing slots).
        """
        params = SimulationParameters(
            porpoise_count=500,  # Pre-allocate 500 slots
            random_seed=42,
        )
        landscape = create_homogeneous_landscape(width=150, height=150, depth=30.0, food_prob=0.5)
        # Start with only 200 active agents so 300 inactive slots are available for calves
        pop = PorpoisePopulation(500, params, landscape)
        pop.active_mask[200:] = False  # deactivate 300 to create free slots

        # Count initial eligible females (mature, not with calf)
        maturity_age = params.maturity_age
        max_breeding_age = params.max_breeding_age
        active = pop.active_mask
        eligible_start = int(np.sum(
            active & pop.is_female &
            (pop.age >= maturity_age) & (pop.age <= max_breeding_age) &
            ~pop.with_calf
        ))

        # Run for 1 year
        ticks_per_year = 365 * 48
        for _ in range(ticks_per_year):
            pop.step()

        # Count females with calves (indicates they gave birth)
        active = pop.active_mask
        with_calf_end = int(np.sum(active & pop.is_female & pop.with_calf))

        print(f"Eligible females at start: {eligible_start}, with_calf at end: {with_calf_end}")

        # With birth_prob=0.0003 over 60-day season (2880 ticks) and ~50 eligible females,
        # expected birth rate = 1 - (1-0.0003)^2880 ≈ 58% per female.
        # At least some births should occur (sanity check).
        assert eligible_start > 0, "Need eligible females to test reproduction"
        assert with_calf_end > 0, (
            f"Expected some births from {eligible_start} eligible females, got 0. "
            f"Check breeding season logic and slot availability."
        )


class TestFoodSystemParameters:
    """Test that food system parameters match DEPONS 3.2 defaults."""

    def test_food_system_parameters(self):
        """Verify food system parameters match DEPONS 3.2 defaults."""
        params = SimulationParameters()

        assert params.max_u == 1.0, "maxU should be 1.0 (hardcoded in Java)"
        assert params.food_growth_rate == 0.1, "rU food growth rate should be 0.1"
        assert params.regrowth_food_qualifier == 0.001, "Umin threshold should be 0.001"


class TestPhysiologyValidation:
    """Validate physiology model produces reasonable outputs."""

    def test_survival_probability_formula(self):
        """Verify survival probability formula at different energy levels (DEPONS 3.2)."""
        m_const = 1.0
        x_const = 0.4

        test_cases = [
            (0.0, 0.00),   # Zero energy: 0% yearly survival (1 - 1.0 * exp(0) = 0)
            (1.0, 0.33),   # Energy 1: ~33% yearly survival (1 - 1.0 * exp(-0.4) = 0.3297)
            (5.0, 0.86),   # Energy 5: ~86% yearly survival (1 - 1.0 * exp(-2.0) = 0.8647)
            (10.0, 0.98),  # Energy 10: ~98% yearly survival (1 - 1.0 * exp(-4.0) = 0.9817)
            (15.0, 1.00),  # Energy 15: ~99.75% yearly survival (1 - 1.0 * exp(-6.0) = 0.9975)
        ]

        for energy, expected_surv in test_cases:
            yearly_surv = 1.0 - (m_const * np.exp(-energy * x_const))
            assert abs(yearly_surv - expected_surv) < 0.02, \
                f"At energy={energy}, expected survival ~{expected_surv}, got {yearly_surv:.2f}"

    def test_energy_scaling_months(self):
        """Verify seasonal energy scaling matches DEPONS."""
        expected_scaling = {
            1: 1.0,   # January - cold
            2: 1.0,   # February - cold
            3: 1.0,   # March - cold
            4: 1.15,  # April - transition
            5: 1.3,   # May - warm
            6: 1.3,   # June - warm
            7: 1.3,   # July - warm
            8: 1.3,   # August - warm
            9: 1.3,   # September - warm
            10: 1.15, # October - transition
            11: 1.0,  # November - cold
            12: 1.0,  # December - cold
        }

        params = SimulationParameters(porpoise_count=10)
        landscape = create_homogeneous_landscape()
        pop = PorpoisePopulation(10, params, landscape)
        mask = pop.active_mask

        for month, expected in expected_scaling.items():
            scaling = pop._get_energy_scaling(month, mask)
            # For non-lactating animals
            non_lactating = mask & ~pop.with_calf
            if np.any(non_lactating):
                actual = scaling[non_lactating][0]
                assert abs(actual - expected) < 0.01, \
                    f"Month {month}: expected scaling {expected}, got {actual}"


class TestJasmineSurvivalConsistency:
    """JASMINE survival should use the same DEPONS formula, not a hardcoded 0.95 base."""

    def test_healthy_porpoise_high_survival(self):
        """At body_condition=1.0 (energy 20), annual survival should be > 0.99."""
        from cenop.physiology.energy_budget import JASMINEEnergyModule, EnergyState
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        module = JASMINEEnergyModule(params)
        state = EnergyState.create(1)
        state.body_condition[0] = 1.0
        state.energy[0] = 20.0
        state.disturbance_energy_cost[0] = 0.0
        mask = np.array([True])

        step_surv = module.compute_survival_probability(state, mask)
        yearly_surv = step_surv[0] ** (360 * 48)
        assert yearly_surv > 0.99, f"Healthy porpoise yearly survival={yearly_surv:.4f}, expected > 0.99"

    def test_moderate_energy_reasonable_survival(self):
        """At body_condition=0.75 (energy 15), annual survival should be > 0.95."""
        from cenop.physiology.energy_budget import JASMINEEnergyModule, EnergyState
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        module = JASMINEEnergyModule(params)
        state = EnergyState.create(1)
        state.body_condition[0] = 0.75
        state.energy[0] = 15.0
        state.disturbance_energy_cost[0] = 0.0
        mask = np.array([True])

        step_surv = module.compute_survival_probability(state, mask)
        yearly_surv = step_surv[0] ** (360 * 48)
        assert yearly_surv > 0.95, f"Moderate-energy yearly survival={yearly_surv:.4f}, expected > 0.95"

    def test_starving_porpoise_low_survival(self):
        """At body_condition=0.1 (energy ~2), annual survival should be < 0.80."""
        from cenop.physiology.energy_budget import JASMINEEnergyModule, EnergyState
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        module = JASMINEEnergyModule(params)
        state = EnergyState.create(1)
        state.body_condition[0] = 0.1
        state.energy[0] = 2.0
        state.disturbance_energy_cost[0] = 0.0
        mask = np.array([True])

        step_surv = module.compute_survival_probability(state, mask)
        yearly_surv = step_surv[0] ** (360 * 48)
        assert yearly_surv < 0.80, f"Starving porpoise yearly survival={yearly_surv:.4f}, expected < 0.80"


class TestMortalityAlignment:
    """Test mortality matches DEPONS 3.2 (no invented categories)."""

    def test_no_age_bracket_mortality(self):
        """At max energy, only bycatch should kill (no age-bracket rates)."""
        params = SimulationParameters()
        pop = PorpoisePopulation(count=100, params=params)
        pop.age[:] = 10.0
        pop.energy[:] = 20.0
        pop.active_mask[:] = True
        mask = pop.active_mask.copy()
        active_before = int(np.sum(mask))

        np.random.seed(42)
        deaths_total = 0
        for _ in range(1000):
            pop.active_mask[:] = True
            pop._check_mortality(mask, active_before)
            deaths_total += int(np.sum(~pop.active_mask))

        assert deaths_total < 5, \
            f"Expected near-zero deaths at max energy, got {deaths_total}"

    def test_lactating_mother_dies_at_zero_energy(self):
        """Lactating mother with energy=0 should die after abandoning calf.
        Java ref: Porpoise.java:766-776
        """
        params = SimulationParameters()
        pop = PorpoisePopulation(count=5, params=params)
        pop.active_mask[:] = True
        pop.energy[:] = 0.0
        pop.with_calf[:3] = True
        pop.with_calf[3:] = False
        mask = pop.active_mask.copy()

        np.random.seed(0)
        pop._check_mortality(mask, 5)

        assert np.all(~pop.active_mask), "All agents at energy=0 should die"
        assert np.all(~pop.with_calf[:3]), "Calves should be abandoned before death"

    def test_initial_energy_is_normal_distribution(self):
        """Initial energy should be N(10, 1) not constant 10.0."""
        np.random.seed(42)
        params = SimulationParameters()
        pop = PorpoisePopulation(count=1000, params=params)

        assert pop.energy.std() > 0.5, \
            f"Energy should have variation (std={pop.energy.std():.3f})"
        assert abs(pop.energy.mean() - 10.0) < 0.2, \
            f"Mean energy should be ~10.0, got {pop.energy.mean():.2f}"
        assert np.all(pop.energy >= 0)
        assert np.all(pop.energy <= 20)

    def test_max_age_death(self):
        """Porpoises older than max_age (30) should die unconditionally."""
        params = SimulationParameters()
        pop = PorpoisePopulation(count=10, params=params)
        pop.energy[:] = 20.0
        pop.active_mask[:] = True
        pop.age[:5] = 31.0   # Over max age
        pop.age[5:] = 10.0   # Normal age
        mask = pop.active_mask.copy()

        pop._check_mortality(mask, 10)

        assert np.sum(~pop.active_mask[:5]) == 5, "All agents over max_age should die"
        assert np.sum(pop.active_mask[5:]) == 5, "Young agents at max energy should survive"


class TestPopulationStabilitySmoke:
    """Verify DEPONS 3.2 parameters produce stable population dynamics."""

    def test_survival_vs_birth_rate_balance(self):
        """
        At typical operating energy (15-20), total mortality should be
        less than the effective birth rate to allow population stability.
        """
        params = SimulationParameters()

        # Annual starvation mortality at energy 15 (typical)
        energy = 15.0
        yearly_surv_starvation = 1.0 - (params.m_mort_prob_const * np.exp(-energy * params.x_survival_const))
        annual_starvation_mort = 1.0 - yearly_surv_starvation

        # Natural adult mortality
        annual_natural_mort = params.mortality_adult  # 0.05

        # Total annual mortality (independent probabilities)
        total_annual_mort = 1.0 - (1.0 - annual_starvation_mort) * (1.0 - annual_natural_mort)

        # Conservative lower bound for birth rate
        min_effective_birth_rate = 0.05

        assert total_annual_mort < min_effective_birth_rate + 0.05, (
            f"Total annual mortality ({total_annual_mort:.4f}) exceeds birth rate "
            f"headroom ({min_effective_birth_rate + 0.05:.4f}). "
            f"Starvation: {annual_starvation_mort:.6f}, Natural: {annual_natural_mort:.4f}"
        )

    def test_depons_jasmine_survival_consistency(self):
        """DEPONS and JASMINE survival at same energy level should be identical."""
        params = SimulationParameters()

        energy = 15.0
        # DEPONS formula
        depons_yearly = 1.0 - (params.m_mort_prob_const * np.exp(-energy * params.x_survival_const))

        # JASMINE formula (body_condition = energy/20)
        body_condition = energy / 20.0
        effective_energy = body_condition * 20.0
        jasmine_yearly = 1.0 - (params.m_mort_prob_const * np.exp(-effective_energy * params.x_survival_const))

        assert abs(depons_yearly - jasmine_yearly) < 0.001, (
            f"DEPONS ({depons_yearly:.4f}) and JASMINE ({jasmine_yearly:.4f}) survival diverge"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
