"""
Tests for TimeManager ensuring DEPONS reproducibility.

CRITICAL: These tests MUST pass before any production use.
They verify bit-exact reproducibility with the CENOP implementation.

The TimeManager is the foundation of the JASMINE-CENOP merge, providing:
- DEPONS mode: Fixed timestep, deterministic seeding, regulatory-compliant
- JASMINE mode: Flexible timestep, event scheduling, research mode
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from cenop.core.time_manager import TimeManager, TimeMode, TimeState
from cenop.core.simulation import Simulation
from cenop.parameters import SimulationParameters
from cenop.landscape.cell_data import create_homogeneous_landscape


class TestTimeState:
    """Test TimeState dataclass properties."""

    def test_time_state_immutable(self):
        """TimeState should be immutable (frozen dataclass)."""
        state = TimeState(tick=100, day=2, month=1, year=1)
        with pytest.raises(Exception):  # FrozenInstanceError
            state.tick = 200

    def test_hour_calculation(self):
        """Hour should be calculated correctly from tick."""
        # tick 0 = 00:00, tick 2 = 01:00, tick 12 = 06:00
        assert TimeState(tick=0).hour == 0
        assert TimeState(tick=1).hour == 0  # 00:30
        assert TimeState(tick=2).hour == 1  # 01:00
        assert TimeState(tick=12).hour == 6  # 06:00
        assert TimeState(tick=24).hour == 12  # 12:00
        assert TimeState(tick=47).hour == 23  # 23:30
        assert TimeState(tick=48).hour == 0  # Next day 00:00

    def test_minute_calculation(self):
        """Minute should be 0 or 30."""
        assert TimeState(tick=0).minute == 0
        assert TimeState(tick=1).minute == 30
        assert TimeState(tick=2).minute == 0
        assert TimeState(tick=3).minute == 30

    def test_is_daytime(self):
        """Daytime should be 6:00-18:00."""
        # Night: 00:00-05:59
        assert not TimeState(tick=0).is_daytime  # 00:00
        assert not TimeState(tick=11).is_daytime  # 05:30

        # Day: 06:00-17:59
        assert TimeState(tick=12).is_daytime  # 06:00
        assert TimeState(tick=35).is_daytime  # 17:30

        # Night: 18:00-23:59
        assert not TimeState(tick=36).is_daytime  # 18:00
        assert not TimeState(tick=47).is_daytime  # 23:30

    def test_quarter_calculation(self):
        """Quarter should be derived from month."""
        assert TimeState(month=1).quarter == 0
        assert TimeState(month=3).quarter == 0
        assert TimeState(month=4).quarter == 1
        assert TimeState(month=6).quarter == 1
        assert TimeState(month=7).quarter == 2
        assert TimeState(month=9).quarter == 2
        assert TimeState(month=10).quarter == 3
        assert TimeState(month=12).quarter == 3

    def test_total_days(self):
        """Total days should be tick / 48."""
        assert TimeState(tick=0).total_days == 0.0
        assert TimeState(tick=48).total_days == 1.0
        assert TimeState(tick=96).total_days == 2.0
        assert TimeState(tick=24).total_days == 0.5


class TestTimeManagerDEPONSMode:
    """Verify DEPONS mode matches original CENOP behavior."""

    def test_default_mode_is_depons(self):
        """Default mode should be DEPONS for backward compatibility."""
        tm = TimeManager()
        assert tm.mode == TimeMode.DEPONS

    def test_fixed_timestep_enforced(self):
        """DEPONS mode should enforce 30-minute timestep."""
        # Even if we try to set different dt, DEPONS mode forces 1800 seconds
        tm = TimeManager(mode=TimeMode.DEPONS, dt_seconds=60)
        assert tm.dt_seconds == 1800, "DEPONS mode must use 30-min timestep"

    def test_timestep_change_blocked(self):
        """Cannot change timestep in DEPONS mode."""
        tm = TimeManager(mode=TimeMode.DEPONS)
        with pytest.raises(RuntimeError, match="Cannot change timestep"):
            tm.set_dt(60)

    def test_event_scheduling_blocked(self):
        """Cannot schedule events in DEPONS mode."""
        tm = TimeManager(mode=TimeMode.DEPONS)
        with pytest.raises(RuntimeError, match="not available in DEPONS"):
            tm.schedule_event(100, lambda: None)

    def test_update_frequency_change_blocked(self):
        """Cannot change update frequencies in DEPONS mode."""
        tm = TimeManager(mode=TimeMode.DEPONS)
        with pytest.raises(RuntimeError, match="Cannot change update frequencies"):
            tm.set_update_frequency('movement', 10)

    def test_deterministic_seeding(self):
        """Same tick should produce same seed."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        seed_at_0 = tm.get_seed()
        assert seed_at_0 == 42, "Seed at tick 0 should be base_seed"

        tm.advance()
        seed_at_1 = tm.get_seed()
        assert seed_at_1 == 43, "Seed at tick 1 should be base_seed + 1"

        tm.advance()
        seed_at_2 = tm.get_seed()
        assert seed_at_2 == 44

        # Verify by creating new manager
        tm2 = TimeManager(mode=TimeMode.DEPONS, base_seed=42)
        assert tm2.get_seed() == seed_at_0

        tm2.advance()
        assert tm2.get_seed() == seed_at_1

    def test_agent_seed_deterministic(self):
        """Agent seeds should be deterministic and unique per agent."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        seed_agent_0 = tm.get_agent_seed(0)
        seed_agent_1 = tm.get_agent_seed(1)
        seed_agent_100 = tm.get_agent_seed(100)

        # Different agents should have different seeds
        assert seed_agent_0 != seed_agent_1
        assert seed_agent_0 != seed_agent_100
        assert seed_agent_1 != seed_agent_100

        # Same agent at same tick should have same seed
        tm2 = TimeManager(mode=TimeMode.DEPONS, base_seed=42)
        assert tm2.get_agent_seed(0) == seed_agent_0
        assert tm2.get_agent_seed(1) == seed_agent_1

    def test_time_advancement(self):
        """Time should advance correctly following DEPONS conventions."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        assert tm.tick == 0
        assert tm.day == 0
        assert tm.month == 1
        assert tm.year == 1

        # Advance 47 ticks (not yet day boundary)
        for _ in range(47):
            tm.advance()

        assert tm.tick == 47
        assert tm.day == 0
        assert not tm.is_day_boundary()

        # Tick 48 is day boundary
        tm.advance()
        assert tm.tick == 48
        assert tm.day == 1
        assert tm.is_day_boundary()

    def test_month_boundary(self):
        """Month boundary should occur at day 30."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        # Advance 30 days (30 * 48 = 1440 ticks)
        for _ in range(1440):
            tm.advance()

        assert tm.tick == 1440
        assert tm.day == 30
        assert tm.month == 2
        assert tm.is_month_boundary()

    def test_year_boundary(self):
        """Year boundary should occur at day 360."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        # Advance 360 days (360 * 48 = 17280 ticks)
        for _ in range(17280):
            tm.advance()

        assert tm.tick == 17280
        assert tm.day == 360
        assert tm.month == 1  # Wrapped to month 1
        assert tm.year == 2
        assert tm.is_year_boundary()

    def test_max_ticks_calculation(self):
        """Max ticks should be sim_years * 360 * 48."""
        tm = TimeManager(mode=TimeMode.DEPONS, sim_years=5)
        assert tm.max_ticks == 5 * 360 * 48

        tm2 = TimeManager(mode=TimeMode.DEPONS, sim_years=1)
        assert tm2.max_ticks == 17280

    def test_is_finished(self):
        """is_finished should return True when tick >= max_ticks."""
        tm = TimeManager(mode=TimeMode.DEPONS, sim_years=1)

        for _ in range(17279):
            assert not tm.is_finished()
            tm.advance()

        # At tick 17279
        assert not tm.is_finished()
        tm.advance()

        # At tick 17280
        assert tm.is_finished()


class TestTimeManagerJASMINEMode:
    """Verify JASMINE mode provides expected flexibility."""

    def test_variable_timestep(self):
        """JASMINE mode should allow timestep changes."""
        tm = TimeManager(mode=TimeMode.JASMINE, dt_seconds=60)
        assert tm.dt_seconds == 60

        tm.set_dt(30)
        assert tm.dt_seconds == 30

        tm.set_dt(120)
        assert tm.dt_seconds == 120

    def test_timestep_must_be_positive(self):
        """Timestep must be positive."""
        tm = TimeManager(mode=TimeMode.JASMINE)
        with pytest.raises(ValueError, match="must be positive"):
            tm.set_dt(0)
        with pytest.raises(ValueError, match="must be positive"):
            tm.set_dt(-1)

    def test_event_scheduling(self):
        """JASMINE mode should allow event scheduling."""
        tm = TimeManager(mode=TimeMode.JASMINE)

        events_fired = []
        tm.schedule_event(10, lambda: events_fired.append(10))
        tm.schedule_event(20, lambda: events_fired.append(20))
        tm.schedule_event(20, lambda: events_fired.append("20b"))  # Multiple at same tick

        for _ in range(25):
            for event in tm.get_scheduled_events():
                event()
            tm.advance()

        assert 10 in events_fired
        assert 20 in events_fired
        assert "20b" in events_fired

    def test_event_scheduling_datetime(self):
        """JASMINE mode should schedule events by datetime."""
        start = datetime(2020, 1, 1, 0, 0)
        tm = TimeManager(mode=TimeMode.JASMINE, dt_seconds=1800, start_datetime=start)

        events_fired = []
        # Schedule event 1 hour from start (2 ticks at 30-min timestep)
        event_time = start + timedelta(hours=1)
        tm.schedule_event_at_datetime(event_time, lambda: events_fired.append("1h"))

        # Advance to that time
        for _ in range(3):
            for event in tm.get_scheduled_events():
                event()
            tm.advance()

        assert "1h" in events_fired

    def test_subsystem_update_frequencies(self):
        """Subsystems should update at configured frequencies."""
        tm = TimeManager(mode=TimeMode.JASMINE)

        # Default: movement every tick, food every 48 ticks
        assert tm.should_update('movement')  # tick 0
        assert tm.should_update('food')      # tick 0 is also food update

        tm.advance()  # tick 1
        assert tm.should_update('movement')
        assert not tm.should_update('food')

        # Advance to tick 48
        for _ in range(47):
            tm.advance()

        assert tm.should_update('movement')
        assert tm.should_update('food')

    def test_custom_update_frequency(self):
        """JASMINE mode should allow custom update frequencies."""
        tm = TimeManager(mode=TimeMode.JASMINE)
        tm.set_update_frequency('custom_system', 10)

        assert tm.get_update_frequency('custom_system') == 10

        for i in range(25):
            if i % 10 == 0:
                assert tm.should_update('custom_system')
            else:
                assert not tm.should_update('custom_system')
            tm.advance()

    def test_jasmine_seeding_differs(self):
        """JASMINE mode uses different seeding formula."""
        tm_depons = TimeManager(mode=TimeMode.DEPONS, base_seed=42)
        tm_jasmine = TimeManager(mode=TimeMode.JASMINE, base_seed=42)

        # At tick 0, both use base_seed
        assert tm_depons.get_seed() == 42
        assert tm_jasmine.get_seed() == 42

        # At tick 1, they differ
        tm_depons.advance()
        tm_jasmine.advance()

        seed_depons = tm_depons.get_seed()   # 42 + 1 = 43
        seed_jasmine = tm_jasmine.get_seed() # 42 + 1*1000 = 1042

        assert seed_depons == 43
        assert seed_jasmine == 1042


class TestTimeManagerSerialization:
    """Test serialization and checkpointing."""

    def test_to_dict_from_dict_depons(self):
        """TimeManager state should survive serialization (DEPONS)."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=123, sim_years=5)

        # Advance to some arbitrary state
        for _ in range(1000):
            tm.advance()

        # Serialize
        data = tm.to_dict()

        # Restore
        tm2 = TimeManager.from_dict(data)

        assert tm2.mode == tm.mode
        assert tm2.tick == tm.tick
        assert tm2.day == tm.day
        assert tm2.month == tm.month
        assert tm2.year == tm.year
        assert tm2.base_seed == tm.base_seed
        assert tm2.dt_seconds == tm.dt_seconds
        assert tm2.get_seed() == tm.get_seed()

    def test_to_dict_from_dict_jasmine(self):
        """TimeManager state should survive serialization (JASMINE)."""
        tm = TimeManager(mode=TimeMode.JASMINE, base_seed=456, sim_years=3, dt_seconds=120)

        for _ in range(500):
            tm.advance()

        data = tm.to_dict()
        tm2 = TimeManager.from_dict(data)

        assert tm2.mode == TimeMode.JASMINE
        assert tm2.dt_seconds == 120
        assert tm2.tick == tm.tick

    def test_reset(self):
        """Reset should return to initial state."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        for _ in range(100):
            tm.advance()

        assert tm.tick == 100

        tm.reset()

        assert tm.tick == 0
        assert tm.day == 0
        assert tm.month == 1
        assert tm.year == 1


class TestDEPONSReproducibility:
    """
    Critical regression tests comparing TimeManager-based simulation
    to expected DEPONS behavior.

    These tests ensure regulatory compliance by verifying:
    1. Same seed produces identical trajectories
    2. Population dynamics are deterministic
    3. Random number sequences are reproducible
    """

    def test_same_seed_produces_same_random_sequence(self):
        """
        Identical seeds must produce identical random sequences.

        This is fundamental to DEPONS reproducibility.
        """
        tm1 = TimeManager(mode=TimeMode.DEPONS, base_seed=42)
        tm2 = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        # Generate random sequences
        seq1 = []
        seq2 = []

        for _ in range(100):
            np.random.seed(tm1.get_seed())
            seq1.append(np.random.random())
            tm1.advance()

            np.random.seed(tm2.get_seed())
            seq2.append(np.random.random())
            tm2.advance()

        np.testing.assert_array_equal(seq1, seq2)

    def test_same_seed_same_trajectories(self):
        """
        Identical seeds must produce identical simulation trajectories.

        This is the MOST CRITICAL test for regulatory compliance.
        """
        landscape = create_homogeneous_landscape(width=200, height=200, depth=20.0)

        params = SimulationParameters(
            porpoise_count=10,
            sim_years=1,
            random_seed=42
        )

        # Run simulation twice with same seed
        sim1 = Simulation(params=params, cell_data=landscape)
        sim1.initialize()

        positions_1 = []
        for _ in range(100):
            sim1.step()
            positions_1.append((
                sim1.population_manager.x.copy(),
                sim1.population_manager.y.copy()
            ))

        # Reset and run again with fresh landscape
        landscape2 = create_homogeneous_landscape(width=200, height=200, depth=20.0)
        sim2 = Simulation(params=params, cell_data=landscape2)
        sim2.initialize()

        positions_2 = []
        for _ in range(100):
            sim2.step()
            positions_2.append((
                sim2.population_manager.x.copy(),
                sim2.population_manager.y.copy()
            ))

        # Compare trajectories
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(positions_1, positions_2)):
            np.testing.assert_array_almost_equal(
                x1, x2, decimal=10,
                err_msg=f"X positions differ at step {i}"
            )
            np.testing.assert_array_almost_equal(
                y1, y2, decimal=10,
                err_msg=f"Y positions differ at step {i}"
            )

    def test_different_seeds_different_trajectories(self):
        """Different seeds should produce different trajectories."""
        landscape1 = create_homogeneous_landscape(width=200, height=200, depth=20.0)
        landscape2 = create_homogeneous_landscape(width=200, height=200, depth=20.0)

        params1 = SimulationParameters(porpoise_count=10, random_seed=42)
        params2 = SimulationParameters(porpoise_count=10, random_seed=123)

        sim1 = Simulation(params=params1, cell_data=landscape1)
        sim1.initialize()

        sim2 = Simulation(params=params2, cell_data=landscape2)
        sim2.initialize()

        # Run both
        for _ in range(50):
            sim1.step()
            sim2.step()

        # Should be different
        assert not np.allclose(sim1.population_manager.x, sim2.population_manager.x), \
            "Different seeds should produce different X positions"

    def test_population_dynamics_deterministic(self):
        """Population stats should be deterministic with same seed."""
        params = SimulationParameters(
            porpoise_count=50,
            sim_years=1,
            random_seed=12345
        )

        # Run twice
        results = []
        for _ in range(2):
            landscape = create_homogeneous_landscape(width=200, height=200, depth=20.0)
            sim = Simulation(params=params, cell_data=landscape)
            sim.initialize()

            for _ in range(1000):  # ~20 days
                sim.step()

            results.append({
                'tick': sim.state.tick,
                'population': sim.population_manager.population_size,
            })

        assert results[0]['tick'] == results[1]['tick']
        assert results[0]['population'] == results[1]['population']

    def test_time_manager_state_sync(self):
        """TimeManager state should sync with SimulationState."""
        landscape = create_homogeneous_landscape(width=200, height=200, depth=20.0)
        params = SimulationParameters(porpoise_count=5, random_seed=42)
        sim = Simulation(params=params, cell_data=landscape)
        sim.initialize()

        for _ in range(100):
            sim.step()

            # TimeManager and SimulationState should be in sync
            assert sim.time_manager.tick == sim.state.tick
            assert sim.time_manager.day == sim.state.day
            assert sim.time_manager.month == sim.state.month
            assert sim.time_manager.year == sim.state.year


class TestSimulationTimeManagerIntegration:
    """Test Simulation class integration with TimeManager."""

    def test_simulation_creates_time_manager(self):
        """Simulation should create TimeManager automatically."""
        landscape = create_homogeneous_landscape(width=100, height=100, depth=20.0)
        params = SimulationParameters(porpoise_count=5, random_seed=42)
        sim = Simulation(params=params, cell_data=landscape)

        assert hasattr(sim, 'time_manager')
        assert isinstance(sim.time_manager, TimeManager)
        assert sim.time_manager.mode == TimeMode.DEPONS

    def test_simulation_with_custom_time_manager(self):
        """Simulation should accept custom TimeManager."""
        landscape = create_homogeneous_landscape(width=100, height=100, depth=20.0)
        params = SimulationParameters(porpoise_count=5, random_seed=42)
        tm = TimeManager(mode=TimeMode.JASMINE, base_seed=42)

        sim = Simulation(params=params, cell_data=landscape, time_manager=tm)

        assert sim.time_manager is tm
        assert sim.time_manager.mode == TimeMode.JASMINE

    def test_simulation_max_ticks_from_time_manager(self):
        """Simulation.max_ticks should come from TimeManager."""
        landscape = create_homogeneous_landscape(width=100, height=100, depth=20.0)
        params = SimulationParameters(porpoise_count=5, sim_years=5)
        sim = Simulation(params=params, cell_data=landscape)

        assert sim.max_ticks == sim.time_manager.max_ticks
        assert sim.max_ticks == 5 * 360 * 48

    def test_simulation_step_advances_time(self):
        """Each step should advance TimeManager."""
        landscape = create_homogeneous_landscape(width=100, height=100, depth=20.0)
        params = SimulationParameters(porpoise_count=5, random_seed=42)
        sim = Simulation(params=params, cell_data=landscape)
        sim.initialize()

        assert sim.time_manager.tick == 0

        sim.step()
        assert sim.time_manager.tick == 1

        sim.step()
        assert sim.time_manager.tick == 2

    def test_simulation_is_daytime_property(self):
        """Simulation should use TimeManager for daytime."""
        landscape = create_homogeneous_landscape(width=100, height=100, depth=20.0)
        params = SimulationParameters(porpoise_count=5, random_seed=42)
        sim = Simulation(params=params, cell_data=landscape)
        sim.initialize()

        # At tick 0 (00:00), not daytime
        assert not sim.time_manager.is_daytime

        # Advance to tick 12 (06:00), daytime
        for _ in range(12):
            sim.step()

        assert sim.time_manager.is_daytime


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_sim_years(self):
        """Zero sim_years should result in zero max_ticks."""
        tm = TimeManager(mode=TimeMode.DEPONS, sim_years=0)
        assert tm.max_ticks == 0
        assert tm.is_finished()

    def test_large_sim_years(self):
        """Large sim_years should work correctly."""
        tm = TimeManager(mode=TimeMode.DEPONS, sim_years=100)
        assert tm.max_ticks == 100 * 360 * 48

    def test_progress_calculation(self):
        """Progress should be calculated correctly."""
        tm = TimeManager(mode=TimeMode.DEPONS, sim_years=1)

        assert tm.progress == 0.0

        for _ in range(8640):  # Half year
            tm.advance()

        assert abs(tm.progress - 0.5) < 0.001

        for _ in range(8640):  # Full year
            tm.advance()

        assert tm.progress == 1.0

    def test_current_datetime(self):
        """Current datetime should be calculated correctly."""
        start = datetime(2020, 6, 15, 12, 0)
        tm = TimeManager(mode=TimeMode.DEPONS, start_datetime=start)

        assert tm.current_datetime == start

        # Advance 1 hour (2 ticks of 30 min each)
        tm.advance()
        tm.advance()

        expected = start + timedelta(hours=1)
        assert tm.current_datetime == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
