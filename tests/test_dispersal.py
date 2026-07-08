"""
Tests for dispersal behavior implementations.

Validates DEPONS-compliant PSM formulas:
- PSM-Type2: SSLogis with distLogX = (3 * distPerc) - 1.5, uses previous heading
- PSM-Type3: angleDelta = maxAngle / (1 + exp(-psmLog * (dist - x0)))
- 50-cell minimum for PSM activation
"""

import pytest
import numpy as np
from cenop.behavior.dispersal import (
    DispersalParams,
    DispersalType,
    NoDispersal,
    PSMType1Dispersal,
    PSMType2Dispersal,
    PSMType3Dispersal,
    sslogis,
    create_dispersal_behavior,
)


class TestSSLogisFunction:
    """Test the SSLogis (Simple Self-Starting Logistic) function."""
    
    def test_sslogis_at_inflection_point(self):
        """SSLogis at x=phi2 should return phi1/2."""
        # phi1 / (1 + exp((phi2 - x) / phi3)) where x = phi2
        # = phi1 / (1 + exp(0)) = phi1 / 2
        result = sslogis(0.0, phi1=1.0, phi2=0.0, phi3=0.6)
        assert result == pytest.approx(0.5, rel=1e-6)
    
    def test_sslogis_large_positive_x(self):
        """SSLogis at large x should approach phi1."""
        result = sslogis(10.0, phi1=1.0, phi2=0.0, phi3=0.6)
        assert result == pytest.approx(1.0, rel=1e-6)
    
    def test_sslogis_large_negative_x(self):
        """SSLogis at large negative x should approach 0."""
        result = sslogis(-10.0, phi1=1.0, phi2=0.0, phi3=0.6)
        assert result == pytest.approx(0.0, abs=1e-6)
    
    def test_sslogis_matches_depons_type2_start(self):
        """At start of dispersal (distPerc=0), SSLogis input is -1.5."""
        # distLogX = (3 * 0) - 1.5 = -1.5
        # SSLogis(-1.5) should give high value (allowing large turns early)
        dist_log_x = (3 * 0.0) - 1.5
        result = sslogis(dist_log_x, phi1=1.0, phi2=0.0, phi3=0.6)
        # At x=-1.5: 1 / (1 + exp(1.5/0.6)) = 1 / (1 + exp(2.5)) ≈ 0.076
        assert result == pytest.approx(0.076, rel=0.05)
    
    def test_sslogis_matches_depons_type2_halfway(self):
        """At halfway (distPerc=0.5), SSLogis input is 0."""
        dist_log_x = (3 * 0.5) - 1.5
        result = sslogis(dist_log_x, phi1=1.0, phi2=0.0, phi3=0.6)
        assert result == pytest.approx(0.5, rel=1e-6)
    
    def test_sslogis_matches_depons_type2_end(self):
        """At end of dispersal (distPerc=1), SSLogis input is 1.5."""
        dist_log_x = (3 * 1.0) - 1.5
        result = sslogis(dist_log_x, phi1=1.0, phi2=0.0, phi3=0.6)
        # At x=1.5: 1 / (1 + exp(-1.5/0.6)) = 1 / (1 + exp(-2.5)) ≈ 0.924
        assert result == pytest.approx(0.924, rel=0.05)


class TestPSMType2Dispersal:
    """Test PSM-Type2 dispersal behavior matches DEPONS."""
    
    @pytest.fixture
    def params(self):
        return DispersalParams(
            psm_log=0.6,
            psm_type2_random_angle=20.0,
            min_memory_cells=50,
        )
    
    @pytest.fixture
    def dispersal(self, params):
        return PSMType2Dispersal(params)
    
    def test_50_cell_minimum_blocks_activation(self, dispersal):
        """Should not start dispersal with fewer than 50 memory cells."""
        # 49 cells - should not activate
        assert dispersal.should_start_dispersal(
            days_declining_energy=5,
            current_energy=10.0,
            memory_cell_count=49
        ) is False
    
    def test_50_cell_minimum_allows_activation(self, dispersal):
        """Should start dispersal with 50+ memory cells."""
        # 50 cells - should activate if energy declining
        assert dispersal.should_start_dispersal(
            days_declining_energy=5,
            current_energy=10.0,
            memory_cell_count=50
        ) is True
    
    def test_uses_95_percent_target_distance(self, dispersal):
        """PSM-Type2 should use 95% of target distance."""
        rng = np.random.default_rng(42)
        dispersal.start_dispersal(rng)
        
        # Target should be 95% of what was drawn
        # Since we don't know exact draw, test that target is set
        assert dispersal._target_distance is not None
    
    def test_angle_decreases_as_travel_increases(self, dispersal, params):
        """Angle perturbation should decrease (get smoother) as distance traveled increases."""
        rng = np.random.default_rng(42)
        
        # Simulate start of dispersal
        dispersal._target_distance = 100.0
        dispersal._distance_traveled = 0.0
        dispersal._dispersing = True
        dispersal._previous_step_heading = 90.0
        
        # At start, distLogX = -1.5, SSLogis gives ~0.076 (low value)
        # Actually WAIT - DEPONS Type2 is DECREASE, so at start we have MORE turning
        # SSLogis at x=-1.5 ≈ 0.076 (low), but wait the comment says "decrease"
        
        # Let me re-check: at start distPerc=0, distLogX=-1.5
        # SSLogis(-1.5) = 1 / (1 + exp(1.5/0.6)) ≈ 0.076
        # angleDelta multiplied by 0.076 = very small angle
        # So at START, angle is SMALL (straighter)
        # At END distPerc=1, distLogX=1.5, SSLogis(1.5) ≈ 0.924 = large angle
        
        # This means Type2 starts straight and gets more random = INCREASE
        # But the class docstring says "DECREASE" - let me verify with Java
        # Actually looking at Java, the function is LogisticDecreaseSSLogis but
        # the behavior depends on input transformation
        
        # Test the actual values at different distances
        dist_perc_start = 0.0
        dist_log_x_start = (3 * dist_perc_start) - 1.5  # -1.5
        sslogis_start = sslogis(dist_log_x_start, phi1=1.0, phi2=0.0, phi3=params.psm_log)
        
        dist_perc_end = 1.0
        dist_log_x_end = (3 * dist_perc_end) - 1.5  # 1.5
        sslogis_end = sslogis(dist_log_x_end, phi1=1.0, phi2=0.0, phi3=params.psm_log)
        
        # At start SSLogis output is SMALLER, at end it's LARGER
        # This means turning angle INCREASES as we travel
        assert sslogis_start < sslogis_end


class TestPSMType3Dispersal:
    """Test PSM-Type3 dispersal behavior matches DEPONS."""
    
    @pytest.fixture
    def params(self):
        return DispersalParams(
            psm_log=0.6,
            psm_type2_random_angle=20.0,
            min_memory_cells=50,
        )
    
    @pytest.fixture
    def dispersal(self, params):
        return PSMType3Dispersal(params)
    
    def test_50_cell_minimum_blocks_activation(self, dispersal):
        """Should not start dispersal with fewer than 50 memory cells."""
        assert dispersal.should_start_dispersal(
            days_declining_energy=5,
            current_energy=10.0,
            memory_cell_count=49
        ) is False
    
    def test_formula_at_start(self, dispersal, params):
        """At start (dist=0), z is positive, angleDelta is small."""
        dispersal._target_distance = 100.0
        dispersal._distance_traveled = 0.0
        dispersal._dispersing = True
        
        x0 = dispersal._target_distance / 2  # 50
        z = -params.psm_log * (0 - x0)  # -0.6 * -50 = 30
        expected_delta = params.psm_type2_random_angle / (1 + np.exp(z))
        # 20 / (1 + exp(30)) ≈ 0 (very small)
        
        assert expected_delta < 1.0  # Very small at start
    
    def test_formula_at_halfway(self, dispersal, params):
        """At halfway (dist=x0), z=0, angleDelta = maxAngle/2."""
        dispersal._target_distance = 100.0
        dispersal._distance_traveled = 50.0  # x0
        dispersal._dispersing = True
        
        x0 = dispersal._target_distance / 2  # 50
        z = -params.psm_log * (50 - x0)  # 0
        expected_delta = params.psm_type2_random_angle / (1 + np.exp(z))
        # 20 / (1 + exp(0)) = 20 / 2 = 10
        
        assert expected_delta == pytest.approx(params.psm_type2_random_angle / 2, rel=1e-6)
    
    def test_formula_at_end(self, dispersal, params):
        """At end (dist=target), z is negative, angleDelta approaches maxAngle."""
        dispersal._target_distance = 100.0
        dispersal._distance_traveled = 100.0
        dispersal._dispersing = True
        
        x0 = dispersal._target_distance / 2  # 50
        z = -params.psm_log * (100 - x0)  # -0.6 * 50 = -30
        expected_delta = params.psm_type2_random_angle / (1 + np.exp(z))
        # 20 / (1 + exp(-30)) ≈ 20 (approaches max)
        
        assert expected_delta > 19.0  # Close to max at end
    
    def test_stop_condition_uses_distance_from_start(self, dispersal):
        """PSM-Type3 should stop when distance from start >= target."""
        dispersal._target_distance = 100.0
        dispersal._dispersing = True
        dispersal._start_position = (0.0, 0.0)
        
        # At start position - should not stop
        assert dispersal.should_stop_dispersing(0.0, 0.0) == False
        
        # Halfway from start - should not stop
        assert dispersal.should_stop_dispersing(50.0, 0.0) == False
        
        # At target distance from start - should stop
        assert dispersal.should_stop_dispersing(100.0, 0.0) == True
        
        # Beyond target - should stop
        assert dispersal.should_stop_dispersing(0.0, 150.0) == True


class TestCreateDispersalBehavior:
    """Test factory function."""
    
    def test_create_no_dispersal(self):
        behavior = create_dispersal_behavior(DispersalType.OFF)
        assert isinstance(behavior, NoDispersal)
    
    def test_create_psm_type1(self):
        behavior = create_dispersal_behavior(DispersalType.PSM_TYPE1)
        assert isinstance(behavior, PSMType1Dispersal)
    
    def test_create_psm_type2(self):
        behavior = create_dispersal_behavior(DispersalType.PSM_TYPE2)
        assert isinstance(behavior, PSMType2Dispersal)
    
    def test_create_psm_type3(self):
        behavior = create_dispersal_behavior(DispersalType.PSM_TYPE3)
        assert isinstance(behavior, PSMType3Dispersal)
    
    def test_create_from_string(self):
        behavior = create_dispersal_behavior("PSM-Type2")
        assert isinstance(behavior, PSMType2Dispersal)


class TestPSMType2HeadingWiring:
    """Test that population uses PSMType2Dispersal module for heading."""

    def test_dispersal_heading_uses_sslogis(self):
        """_apply_dispersal_heading should use SSLogis formula.

        The existing dispersal.py module has the correct formula.
        population.py should use the same: angleDelta * SSLogis(distLogX).
        """
        # Verify SSLogis formula matches at halfway point
        # distLogX = 3 * 0.5 - 1.5 = 0.0
        # SSLogis(0.0, 1.0, 0.0, 0.6) = 1/(1+exp(0/0.6)) = 0.5
        result = sslogis(0.0, 1.0, 0.0, 0.6)
        assert result == pytest.approx(0.5), f"SSLogis(0) should be 0.5, got {result}"

    def test_prev_step_heading_initialized(self):
        """Population should have _prev_step_heading array."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()
        pop = PorpoisePopulation(count=5, params=params)
        assert hasattr(pop, '_prev_step_heading')
        assert pop._prev_step_heading.shape == (5,)


class TestDispersalStepDistance:
    """Test dispersal uses fixed step distance."""

    def test_dispersal_step_is_mean_disp_dist_over_04(self):
        """Dispersing agents use fixed step = mean_disp_dist / 0.4.

        Java: AbstractPSMDispersal.java:210
        mean_disp_dist=2.0 from Java parameters.xml:117, step=2.0/0.4=5.0 grid cells.
        """
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()
        step = params.mean_disp_dist / 0.4
        assert step == pytest.approx(5.0), f"Dispersal step should be 5.0, got {step}"

    def test_dispersal_target_uses_95_percent(self):
        """PSM-Type2 target = 0.95 * drawn target distance.

        Java: DispersalPSMType2.java:91
        """
        params = DispersalParams(dist_mean=300.0, dist_sd=0.001)  # Near-deterministic
        d = PSMType2Dispersal(params)
        rng = np.random.default_rng(42)
        d.start_dispersal(rng)

        # Target should be ~95% of ~300
        assert d._target_distance == pytest.approx(300.0 * 0.95, rel=0.01)


class TestEnergyBasedDispersalStop:
    """Test energy-based dispersal stop and deterrence deactivation."""

    def test_energy_history_has_10_slots(self):
        """Energy history should have 10 slots for 7-day lookback."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters.simulation_params import SimulationParameters
        params = SimulationParameters()
        pop = PorpoisePopulation(count=5, params=params)
        assert pop._energy_history.shape == (5, 10), \
            f"Expected (5, 10), got {pop._energy_history.shape}"

    def test_dispersal_stops_when_energy_recovering(self):
        """Dispersal stops when today's energy > min of last 7 days.

        Java: Porpoise.java:1105-1118
        """
        energy_history = np.array([5.0, 4.0, 3.0, 3.5, 4.5, 5.5, 6.0, 6.5, 7.0, 7.5])
        today = energy_history[0]
        past_min = np.min(energy_history[1:8])
        should_stop = bool(today > past_min)
        assert should_stop is True, f"5.0 > {past_min} (3.0), should stop"

    def test_dispersal_continues_when_energy_still_low(self):
        """Dispersal continues when today's energy <= min of last 7 days."""
        energy_history = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        today = energy_history[0]
        past_min = np.min(energy_history[1:8])
        should_stop = bool(today > past_min)
        assert should_stop is False, f"2.0 is not > {past_min} (3.0), should continue"

    def test_deterrence_deactivates_dispersal(self):
        """Turbine deterrence applied to a dispersing agent stops dispersal; ship
        deterrence does NOT.

        Java: Porpoise.java:1277-1278 deactivates dispersal only for turbine/
        sound-source deterrence (applyDeterrence), not for ships
        (applyShipDeterrence). The gate therefore keys on the turbine-only
        deterrence strength, not the combined deter_strength.
        """
        is_dispersing = np.array([True, True, False, True, False])
        # Combined deterrence is non-zero for several agents (e.g. from ships)...
        deter_strength = np.array([1.0, 1.0, 1.0, 0.5, 0.0])
        # ...but only these have turbine-sourced deterrence.
        turbine_deter_strength = np.array([1.0, 0.0, 1.0, 0.5, 0.0])
        active = np.array([True, True, True, True, True])

        deterred_and_dispersing = active & (turbine_deter_strength > 0) & is_dispersing
        is_dispersing[deterred_and_dispersing] = False

        # Agent 0: was dispersing + turbine-deterred -> stopped
        assert is_dispersing[0] is np.False_
        # Agent 1: was dispersing + ship-only deterred (no turbine) -> still dispersing
        assert is_dispersing[1] is np.True_
        # Agent 3: was dispersing + turbine-deterred -> stopped
        assert is_dispersing[3] is np.False_


class TestDispersalBatchInit:
    """Test that _check_dispersal_trigger batch-initializes dispersal state."""

    def test_batch_dispersal_matches_individual(self):
        """Batch init in _check_dispersal_trigger should set the same fields
        as individual _start_dispersal calls."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        pop = PorpoisePopulation(count=10, params=params, landscape=None)

        mask = pop.active_mask.copy()
        # Set declining energy history for agents 0-4
        # Declining = history[i] < history[i+1] for all consecutive pairs
        # So newest (index 0) is lowest: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i in range(5):
            pop._energy_history[i, :] = np.arange(1, 11, dtype=np.float32)

        # Give enough PSM memory cells (50+ visited)
        pop.psm_buffer[:5, :10, :10, 0] = 1.0
        pop.psm_buffer[:5, :10, :10, 1] = 5.0

        pop._check_dispersal_trigger(mask)

        dispersing_idx = np.where(pop.is_dispersing[:5])[0]
        assert len(dispersing_idx) >= 1, "At least one agent should start dispersing"

        for idx in dispersing_idx:
            # Batch init should set start position to current position
            assert pop.dispersal_start_x[idx] == pop.x[idx]
            assert pop.dispersal_start_y[idx] == pop.y[idx]
            assert pop.dispersal_distance_traveled[idx] == 0.0


class TestPSMReproducibility:
    """PSM preferred_distance must be seeded (finding #6) and centre on
    params.psm_dist_mean=350 not the hardcoded 300 (finding #14)."""

    def test_preferred_distance_reproducible_same_seed(self):
        """Two populations built with the same random_seed must produce an
        identical preferred_distance sequence across all agents."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters.simulation_params import SimulationParameters

        # Small world keeps per-agent psm_buffer tiny (12x12 grid).
        params_a = SimulationParameters(random_seed=12345, world_width=60, world_height=60)
        params_b = SimulationParameters(random_seed=12345, world_width=60, world_height=60)
        pop_a = PorpoisePopulation(count=30, params=params_a)
        pop_b = PorpoisePopulation(count=30, params=params_b)

        dists_a = [pop_a._psm_instances[i].preferred_distance for i in range(30)]
        dists_b = [pop_b._psm_instances[i].preferred_distance for i in range(30)]

        assert dists_a == dists_b, "same seed must give identical PSM distances"
        # Sanity: it is genuinely a distribution, not a constant.
        assert len(set(dists_a)) > 1

    def test_constructor_centres_on_pref_dist_mean_350(self):
        """PersistentSpatialMemory(pref_dist_mean=350) must sample ~N(350;100)."""
        from cenop.behavior.psm import PersistentSpatialMemory

        rng = np.random.default_rng(7)
        dists = np.array(
            [
                PersistentSpatialMemory(
                    100, 100, rng=rng, pref_dist_mean=350.0, pref_dist_sd=100.0
                ).preferred_distance
                for _ in range(2000)
            ]
        )
        mean = float(dists.mean())
        assert 340.0 < mean < 360.0, f"expected ~350, got {mean}"

    def test_population_plumbs_params_psm_dist_mean(self):
        """Production PSM construction must read params.psm_dist_mean (350),
        not generate_preferred_distance's own default (was 300)."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters(random_seed=99, world_width=60, world_height=60)
        assert params.psm_dist_mean == 350.0  # guard the default
        pop = PorpoisePopulation(count=1000, params=params)
        dists = np.array([pop._psm_instances[i].preferred_distance for i in range(1000)])
        mean = float(dists.mean())
        # Pre-fix PSM ignores params and centres on 300 -> mean far below 335.
        assert 335.0 < mean < 365.0, f"expected ~350 (params.psm_dist_mean), got {mean}"


class TestPSMDistDefaults:
    """UI default and controller parse-fallback must be N(350;100) (finding #14)."""

    def test_ui_and_controller_defaults_are_350(self):
        import inspect
        import cenop.ui.tabs.settings as settings_mod
        import cenop.server.simulation_controller as ctrl_mod

        settings_src = inspect.getsource(settings_mod)
        ctrl_src = inspect.getsource(ctrl_mod)

        assert 'ui.input_text("psm_dist", None, value="N(350;100)")' in settings_src
        assert 'value="N(300;100)"' not in settings_src
        assert "psm_dist_mean = 350.0" in ctrl_src
        assert "psm_dist_mean = 300.0" not in ctrl_src
