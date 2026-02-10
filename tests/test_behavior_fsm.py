"""
Tests for behavioral state machine.

Tests the hybrid FSM including:
- State transitions in DEPONS mode
- State transitions in JASMINE mode
- Hybrid mode behavior
- Movement mode selection
"""

import pytest
import numpy as np

from cenop.behavior.states import (
    BehaviorState,
    BehaviorContext,
    BehaviorStateVector,
    StateTransition,
    STATE_PARAMETERS,
)
from cenop.behavior.hybrid_fsm import (
    HybridBehaviorFSM,
    FSMMode,
    create_behavior_fsm,
)
from cenop.core.time_manager import TimeMode
from cenop.parameters import SimulationParameters


class TestBehaviorState:
    """Test BehaviorState enum."""

    def test_all_states_defined(self):
        """All expected states should be defined."""
        expected = {'FORAGING', 'TRAVELING', 'RESTING', 'DISPERSING', 'DISTURBED'}
        actual = {s.name for s in BehaviorState}
        assert expected == actual

    def test_state_values_unique(self):
        """All state values should be unique."""
        values = [s.value for s in BehaviorState]
        assert len(values) == len(set(values))


class TestBehaviorStateVector:
    """Test BehaviorStateVector creation and manipulation."""

    def test_create_state_vector(self):
        """Should create state vector with correct shape."""
        sv = BehaviorStateVector.create(100)

        assert sv.state.shape == (100,)
        assert sv.state_duration.shape == (100,)
        assert sv.previous_state.shape == (100,)

    def test_initial_state_is_foraging(self):
        """All agents should start in FORAGING state."""
        sv = BehaviorStateVector.create(50)

        assert np.all(sv.state == BehaviorState.FORAGING.value)

    def test_get_state(self):
        """get_state should return correct BehaviorState."""
        sv = BehaviorStateVector.create(10)
        sv.state[5] = BehaviorState.DISTURBED.value

        assert sv.get_state(0) == BehaviorState.FORAGING
        assert sv.get_state(5) == BehaviorState.DISTURBED

    def test_set_state(self):
        """set_state should update state and reset duration."""
        sv = BehaviorStateVector.create(10)
        sv.state_duration[3] = 100

        sv.set_state(3, BehaviorState.TRAVELING)

        assert sv.state[3] == BehaviorState.TRAVELING.value
        assert sv.state_duration[3] == 0
        assert sv.previous_state[3] == BehaviorState.FORAGING.value

    def test_get_mask(self):
        """get_mask should return correct boolean mask."""
        sv = BehaviorStateVector.create(10)
        sv.state[2] = BehaviorState.DISTURBED.value
        sv.state[5] = BehaviorState.DISTURBED.value
        sv.state[8] = BehaviorState.DISTURBED.value

        mask = sv.get_mask(BehaviorState.DISTURBED)

        assert np.sum(mask) == 3
        assert mask[2] and mask[5] and mask[8]
        assert not mask[0]

    def test_advance_duration(self):
        """advance_duration should increment all durations."""
        sv = BehaviorStateVector.create(10)
        sv.advance_duration()
        sv.advance_duration()

        assert np.all(sv.state_duration == 2)

    def test_get_statistics(self):
        """get_statistics should return state counts."""
        sv = BehaviorStateVector.create(10)
        sv.state[0:3] = BehaviorState.DISTURBED.value
        sv.state[3:5] = BehaviorState.TRAVELING.value

        stats = sv.get_statistics()

        assert stats['state_counts']['FORAGING'] == 5
        assert stats['state_counts']['DISTURBED'] == 3
        assert stats['state_counts']['TRAVELING'] == 2


class TestBehaviorContext:
    """Test BehaviorContext creation."""

    def test_create_default(self):
        """Should create default context."""
        ctx = BehaviorContext.create_default(50)

        assert ctx.deterrence_magnitude.shape == (50,)
        assert ctx.current_energy.shape == (50,)
        assert np.all(ctx.deterrence_magnitude == 0)
        assert np.all(ctx.current_energy == 0.5)


class TestHybridBehaviorFSM:
    """Test HybridBehaviorFSM."""

    @pytest.fixture
    def params(self):
        """Create test parameters."""
        return SimulationParameters(porpoise_count=20)

    @pytest.fixture
    def depons_fsm(self, params):
        """Create DEPONS mode FSM."""
        return HybridBehaviorFSM(params, TimeMode.DEPONS)

    @pytest.fixture
    def jasmine_fsm(self, params):
        """Create JASMINE mode FSM."""
        return HybridBehaviorFSM(params, TimeMode.JASMINE)

    def test_depons_mode_from_time_mode(self, params):
        """DEPONS TimeMode should create DEPONS FSM."""
        fsm = HybridBehaviorFSM(params, TimeMode.DEPONS)
        assert fsm.mode == FSMMode.DEPONS

    def test_jasmine_mode_from_time_mode(self, params):
        """JASMINE TimeMode should create JASMINE FSM."""
        fsm = HybridBehaviorFSM(params, TimeMode.JASMINE)
        assert fsm.mode == FSMMode.JASMINE

    def test_explicit_mode_overrides(self, params):
        """Explicit FSM mode should override TimeMode inference."""
        fsm = HybridBehaviorFSM(params, TimeMode.DEPONS, FSMMode.JASMINE)
        assert fsm.mode == FSMMode.JASMINE


class TestDEPONSTransitions:
    """Test DEPONS mode state transitions."""

    @pytest.fixture
    def fsm(self):
        params = SimulationParameters(porpoise_count=20)
        return HybridBehaviorFSM(params, TimeMode.DEPONS)

    @pytest.fixture
    def state_vector(self):
        return BehaviorStateVector.create(20)

    @pytest.fixture
    def context(self):
        return BehaviorContext.create_default(20)

    @pytest.fixture
    def mask(self):
        return np.ones(20, dtype=bool)

    def test_foraging_to_disturbed(self, fsm, state_vector, context, mask):
        """Deterrence should trigger DISTURBED state."""
        # Apply deterrence to some agents
        context.deterrence_magnitude[5:10] = 0.5

        fsm.update_states(state_vector, context, mask)

        # Agents 5-9 should be DISTURBED
        assert np.all(state_vector.state[5:10] == BehaviorState.DISTURBED.value)
        # Others should remain FORAGING
        assert np.all(state_vector.state[0:5] == BehaviorState.FORAGING.value)
        assert np.all(state_vector.state[10:] == BehaviorState.FORAGING.value)

    def test_disturbed_to_foraging_recovery(self, fsm, state_vector, context, mask):
        """Agents should recover from DISTURBED after recovery period."""
        # Start in DISTURBED state
        state_vector.state[:] = BehaviorState.DISTURBED.value
        context.deterrence_magnitude[:] = 0  # No deterrence
        context.time_since_disturbance[:] = 100  # Long time since disturbance

        fsm.update_states(state_vector, context, mask)

        # All should recover to FORAGING
        assert np.all(state_vector.state == BehaviorState.FORAGING.value)

    def test_disturbed_stays_if_deterrence_continues(self, fsm, state_vector, context, mask):
        """Agents should stay DISTURBED while deterrence continues."""
        state_vector.state[:] = BehaviorState.DISTURBED.value
        context.deterrence_magnitude[:] = 0.5  # Still deterrence

        fsm.update_states(state_vector, context, mask)

        # All should remain DISTURBED
        assert np.all(state_vector.state == BehaviorState.DISTURBED.value)

    def test_foraging_to_dispersing(self, fsm, state_vector, context, mask):
        """Energy decline should trigger DISPERSING."""
        context.energy_declining_days[:] = 5  # > t_disp (3)
        context.memory_cell_count[:] = 100    # > min_memory_cells (50)

        fsm.update_states(state_vector, context, mask)

        # All should be DISPERSING
        assert np.all(state_vector.state == BehaviorState.DISPERSING.value)

    def test_dispersing_needs_memory(self, fsm, state_vector, context, mask):
        """DISPERSING requires minimum memory cells."""
        context.energy_declining_days[:] = 5
        context.memory_cell_count[:] = 10  # < min_memory_cells

        fsm.update_states(state_vector, context, mask)

        # All should remain FORAGING
        assert np.all(state_vector.state == BehaviorState.FORAGING.value)

    def test_dispersing_to_foraging(self, fsm, state_vector, context, mask):
        """Dispersal completion should return to FORAGING."""
        state_vector.state[:] = BehaviorState.DISPERSING.value
        context.dispersal_complete[:] = True

        fsm.update_states(state_vector, context, mask)

        # All should be FORAGING
        assert np.all(state_vector.state == BehaviorState.FORAGING.value)


class TestJASMINETransitions:
    """Test JASMINE mode state transitions."""

    @pytest.fixture
    def fsm(self):
        params = SimulationParameters(porpoise_count=20)
        return HybridBehaviorFSM(params, TimeMode.JASMINE)

    @pytest.fixture
    def state_vector(self):
        return BehaviorStateVector.create(20)

    @pytest.fixture
    def context(self):
        return BehaviorContext.create_default(20)

    @pytest.fixture
    def mask(self):
        return np.ones(20, dtype=bool)

    def test_foraging_to_traveling(self, fsm, state_vector, context, mask):
        """High speed should trigger TRAVELING state."""
        context.current_speed[:] = 3.0  # > speed_threshold (2.0)

        fsm.update_states(state_vector, context, mask)

        assert np.all(state_vector.state == BehaviorState.TRAVELING.value)

    def test_traveling_to_foraging(self, fsm, state_vector, context, mask):
        """Low speed should return to FORAGING."""
        state_vector.state[:] = BehaviorState.TRAVELING.value
        context.current_speed[:] = 1.0  # < speed_threshold

        fsm.update_states(state_vector, context, mask)

        assert np.all(state_vector.state == BehaviorState.FORAGING.value)

    def test_foraging_to_resting(self, fsm, state_vector, context, mask):
        """Low energy should trigger RESTING state."""
        context.current_energy[:] = 0.2  # < energy_threshold_low (0.3)
        context.current_speed[:] = 0.5   # Below speed threshold

        fsm.update_states(state_vector, context, mask)

        assert np.all(state_vector.state == BehaviorState.RESTING.value)

    def test_resting_to_foraging(self, fsm, state_vector, context, mask):
        """High energy should exit RESTING state."""
        state_vector.state[:] = BehaviorState.RESTING.value
        context.current_energy[:] = 0.8  # > energy_threshold_high (0.7)

        fsm.update_states(state_vector, context, mask)

        assert np.all(state_vector.state == BehaviorState.FORAGING.value)


class TestMovementModeSelection:
    """Test movement mode selection based on state."""

    @pytest.fixture
    def fsm(self):
        params = SimulationParameters(porpoise_count=20)
        return HybridBehaviorFSM(params, TimeMode.JASMINE)

    def test_foraging_uses_depons(self, fsm):
        """FORAGING state should use DEPONS movement."""
        sv = BehaviorStateVector.create(10)
        mask = np.ones(10, dtype=bool)

        modes = fsm.get_movement_mode(sv, mask)

        assert np.all(modes == 0)  # DEPONS_CRW

    def test_traveling_uses_jasmine(self, fsm):
        """TRAVELING state should use JASMINE movement."""
        sv = BehaviorStateVector.create(10)
        sv.state[:] = BehaviorState.TRAVELING.value
        mask = np.ones(10, dtype=bool)

        modes = fsm.get_movement_mode(sv, mask)

        assert np.all(modes == 1)  # JASMINE_PHYSICS

    def test_resting_uses_jasmine(self, fsm):
        """RESTING state should use JASMINE movement."""
        sv = BehaviorStateVector.create(10)
        sv.state[:] = BehaviorState.RESTING.value
        mask = np.ones(10, dtype=bool)

        modes = fsm.get_movement_mode(sv, mask)

        assert np.all(modes == 1)  # JASMINE_PHYSICS

    def test_disturbed_uses_depons(self, fsm):
        """DISTURBED state should use DEPONS movement."""
        sv = BehaviorStateVector.create(10)
        sv.state[:] = BehaviorState.DISTURBED.value
        mask = np.ones(10, dtype=bool)

        modes = fsm.get_movement_mode(sv, mask)

        assert np.all(modes == 0)  # DEPONS_CRW

    def test_mixed_states(self, fsm):
        """Mixed states should return appropriate modes."""
        sv = BehaviorStateVector.create(10)
        sv.state[0:3] = BehaviorState.FORAGING.value
        sv.state[3:6] = BehaviorState.TRAVELING.value
        sv.state[6:10] = BehaviorState.DISTURBED.value
        mask = np.ones(10, dtype=bool)

        modes = fsm.get_movement_mode(sv, mask)

        assert np.all(modes[0:3] == 0)   # DEPONS
        assert np.all(modes[3:6] == 1)   # JASMINE
        assert np.all(modes[6:10] == 0)  # DEPONS


class TestSpeedMultiplier:
    """Test speed multiplier based on state."""

    @pytest.fixture
    def fsm(self):
        params = SimulationParameters(porpoise_count=20)
        return HybridBehaviorFSM(params, TimeMode.JASMINE)

    def test_disturbed_has_highest_speed(self, fsm):
        """DISTURBED state should have highest speed multiplier."""
        sv = BehaviorStateVector.create(10)
        mask = np.ones(10, dtype=bool)

        # Get multipliers for each state
        multipliers = {}
        for state in BehaviorState:
            sv.state[:] = state.value
            mult = fsm.get_speed_multiplier(sv, mask)
            multipliers[state] = mult[0]

        # DISTURBED should be highest, RESTING lowest
        assert multipliers[BehaviorState.DISTURBED] > multipliers[BehaviorState.FORAGING]
        assert multipliers[BehaviorState.RESTING] < multipliers[BehaviorState.FORAGING]


class TestFactoryFunction:
    """Test factory function."""

    def test_create_behavior_fsm_depons(self):
        """Factory should create DEPONS FSM."""
        params = SimulationParameters(porpoise_count=10)
        fsm = create_behavior_fsm(params, TimeMode.DEPONS)

        assert fsm.mode == FSMMode.DEPONS

    def test_create_behavior_fsm_jasmine(self):
        """Factory should create JASMINE FSM."""
        params = SimulationParameters(porpoise_count=10)
        fsm = create_behavior_fsm(params, TimeMode.JASMINE)

        assert fsm.mode == FSMMode.JASMINE


class TestStateParameters:
    """Test state parameters dictionary."""

    def test_all_states_have_parameters(self):
        """All states should have movement parameters."""
        for state in BehaviorState:
            assert state in STATE_PARAMETERS
            assert 'movement_mode' in STATE_PARAMETERS[state]
            assert 'speed_multiplier' in STATE_PARAMETERS[state]
            assert 'energy_cost_multiplier' in STATE_PARAMETERS[state]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
