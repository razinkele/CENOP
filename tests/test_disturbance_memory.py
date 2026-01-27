"""
Tests for disturbance memory module.

Tests the memory/cognition system including:
- Disturbance memory recording
- Memory decay
- Learned avoidance behavior
- DEPONS vs JASMINE mode differences
"""

import pytest
import numpy as np

from cenop.behavior.disturbance_memory import (
    MemoryMode,
    DisturbanceMemoryState,
    DisturbanceMemoryContext,
    AvoidanceResult,
    DisturbanceMemoryModule,
    DEPONSMemoryModule,
    JASMINEMemoryModule,
    create_memory_module,
)
from cenop.parameters import SimulationParameters


class TestDisturbanceMemoryState:
    """Test DisturbanceMemoryState creation and properties."""

    def test_create_memory_state(self):
        """Should create memory state with correct shapes."""
        state = DisturbanceMemoryState.create(100)

        assert len(state.memory_grids) == 100
        assert state.total_disturbance_exposure.shape == (100,)
        assert state.disturbance_event_count.shape == (100,)
        assert state.last_disturbance_tick.shape == (100,)
        assert state.avoidance_strength.shape == (100,)

    def test_initial_values(self):
        """Should have correct initial values."""
        state = DisturbanceMemoryState.create(50)

        assert np.all(state.total_disturbance_exposure == 0.0)
        assert np.all(state.disturbance_event_count == 0)
        assert np.all(state.last_disturbance_tick == -9999)
        assert np.all(state.avoidance_strength == 0.0)

    def test_memory_grids_are_independent(self):
        """Each agent should have independent memory grid."""
        state = DisturbanceMemoryState.create(10)

        # Modify one agent's grid
        state.memory_grids[0][5] = 0.8
        state.memory_grids[1][5] = 0.3

        # They should be independent
        assert state.memory_grids[0][5] == 0.8
        assert state.memory_grids[1][5] == 0.3
        assert 5 not in state.memory_grids[2]


class TestDisturbanceMemoryContext:
    """Test DisturbanceMemoryContext creation."""

    def test_create_default_context(self):
        """Should create default context."""
        ctx = DisturbanceMemoryContext.create_default(50, tick=100)

        assert ctx.is_disturbed.shape == (50,)
        assert ctx.current_tick == 100
        assert not np.any(ctx.is_disturbed)

    def test_disturbance_flags(self):
        """Should correctly handle disturbance flags."""
        ctx = DisturbanceMemoryContext.create_default(20)

        # Set some agents as disturbed
        ctx.is_disturbed[:5] = True
        ctx.disturbance_intensity[:5] = 0.5

        assert np.sum(ctx.is_disturbed) == 5
        assert np.mean(ctx.disturbance_intensity[:5]) == 0.5


class TestDEPONSMemoryModule:
    """Test DEPONS memory module (no-op implementation)."""

    @pytest.fixture
    def params(self):
        return SimulationParameters(porpoise_count=20)

    @pytest.fixture
    def module(self, params):
        return DEPONSMemoryModule(params)

    @pytest.fixture
    def state(self):
        return DisturbanceMemoryState.create(20)

    @pytest.fixture
    def context(self):
        return DisturbanceMemoryContext.create_default(20)

    @pytest.fixture
    def mask(self):
        return np.ones(20, dtype=bool)

    def test_get_mode(self, module):
        """Should return DEPONS mode."""
        assert module.get_mode() == MemoryMode.DEPONS

    def test_record_disturbance_noop(self, module, state, context, mask):
        """Record should be no-op in DEPONS mode."""
        context.is_disturbed[:5] = True
        context.disturbance_intensity[:5] = 0.8

        module.record_disturbance(state, context, mask)

        # Memory should remain empty
        assert all(len(grid) == 0 for grid in state.memory_grids)
        assert np.all(state.disturbance_event_count == 0)

    def test_decay_noop(self, module, state, mask):
        """Decay should be no-op in DEPONS mode."""
        # Manually add some memory to verify no change
        state.memory_grids[0][5] = 0.5

        module.decay_memory(state, mask, ticks_elapsed=100)

        # Memory should remain unchanged
        assert state.memory_grids[0][5] == 0.5

    def test_avoidance_zero(self, module, state, mask):
        """Avoidance should be zero in DEPONS mode."""
        agent_x = np.random.uniform(0, 100, 20).astype(np.float32)
        agent_y = np.random.uniform(0, 100, 20).astype(np.float32)

        result = module.compute_avoidance(state, agent_x, agent_y, mask)

        assert isinstance(result, AvoidanceResult)
        assert np.all(result.avoidance_dx == 0.0)
        assert np.all(result.avoidance_dy == 0.0)
        assert np.all(result.avoidance_strength == 0.0)


class TestJASMINEMemoryModule:
    """Test JASMINE memory module (full implementation)."""

    @pytest.fixture
    def params(self):
        return SimulationParameters(porpoise_count=20)

    @pytest.fixture
    def module(self, params):
        return JASMINEMemoryModule(params)

    @pytest.fixture
    def state(self):
        return DisturbanceMemoryState.create(20)

    @pytest.fixture
    def mask(self):
        return np.ones(20, dtype=bool)

    def test_get_mode(self, module):
        """Should return JASMINE mode."""
        assert module.get_mode() == MemoryMode.JASMINE

    def test_record_disturbance(self, module, state, mask):
        """Should record disturbance events in memory."""
        context = DisturbanceMemoryContext(
            is_disturbed=np.array([True] * 5 + [False] * 15),
            disturbance_intensity=np.array([0.8] * 5 + [0.0] * 15, dtype=np.float32),
            disturbance_x=np.array([50.0] * 5 + [0.0] * 15, dtype=np.float32),
            disturbance_y=np.array([50.0] * 5 + [0.0] * 15, dtype=np.float32),
            agent_x=np.full(20, 50.0, dtype=np.float32),
            agent_y=np.full(20, 50.0, dtype=np.float32),
            current_tick=100,
        )

        module.record_disturbance(state, context, mask)

        # First 5 agents should have memory
        for i in range(5):
            assert len(state.memory_grids[i]) > 0
            assert state.disturbance_event_count[i] == 1
            assert state.last_disturbance_tick[i] == 100

        # Others should not
        for i in range(5, 20):
            assert len(state.memory_grids[i]) == 0
            assert state.disturbance_event_count[i] == 0

    def test_memory_decay(self, module, state, mask):
        """Should decay memory over time."""
        # Add initial memory
        state.memory_grids[0][10] = 1.0
        state.memory_grids[1][10] = 0.5

        # Apply decay
        module.decay_memory(state, mask, ticks_elapsed=100)

        # Memory should have decayed
        assert state.memory_grids[0][10] < 1.0
        assert state.memory_grids[1][10] < 0.5

    def test_memory_decay_removes_weak(self, module, state, mask):
        """Should remove very weak memories."""
        # Add very weak memory
        state.memory_grids[0][10] = 0.02

        # Apply heavy decay
        module.decay_memory(state, mask, ticks_elapsed=1000)

        # Weak memory should be removed
        assert 10 not in state.memory_grids[0]

    def test_avoidance_from_memory(self, module, state, mask):
        """Should compute avoidance based on remembered disturbances."""
        # Add disturbance memory at position (50, 50)
        state.memory_grids[0][10] = 0.8  # Strong memory

        # Agent at position close to remembered disturbance
        agent_x = np.array([45.0] + [100.0] * 19, dtype=np.float32)
        agent_y = np.array([45.0] + [100.0] * 19, dtype=np.float32)

        result = module.compute_avoidance(state, agent_x, agent_y, mask)

        # Agent 0 should have avoidance (direction away from cell 10)
        # The exact direction depends on cell position calculation
        assert result.avoidance_strength[0] > 0

    def test_no_avoidance_without_memory(self, module, state, mask):
        """Should have no avoidance when no disturbances remembered."""
        agent_x = np.full(20, 50.0, dtype=np.float32)
        agent_y = np.full(20, 50.0, dtype=np.float32)

        result = module.compute_avoidance(state, agent_x, agent_y, mask)

        assert np.all(result.avoidance_strength == 0)
        assert np.all(result.cells_avoided == 0)

    def test_exposure_accumulation(self, module, state, mask):
        """Should accumulate total disturbance exposure."""
        context = DisturbanceMemoryContext(
            is_disturbed=np.ones(20, dtype=bool),
            disturbance_intensity=np.full(20, 0.3, dtype=np.float32),
            disturbance_x=np.full(20, 50.0, dtype=np.float32),
            disturbance_y=np.full(20, 50.0, dtype=np.float32),
            agent_x=np.full(20, 50.0, dtype=np.float32),
            agent_y=np.full(20, 50.0, dtype=np.float32),
            current_tick=100,
        )

        # Record multiple disturbances
        for i in range(5):
            context.current_tick = 100 + i
            module.record_disturbance(state, context, mask)

        # Exposure should accumulate
        assert np.all(state.total_disturbance_exposure > 1.0)
        assert np.all(state.disturbance_event_count == 5)

    def test_get_statistics(self, module, state, mask):
        """Should return meaningful statistics."""
        # Add some memory
        state.total_disturbance_exposure[:10] = 0.5
        state.disturbance_event_count[:10] = 2

        stats = module.get_statistics(state, mask)

        assert 'mean_exposure' in stats
        assert 'mean_event_count' in stats
        assert 'agents_with_memory' in stats
        assert stats['agents_with_memory'] == 10


class TestFactoryFunction:
    """Test factory function."""

    def test_create_depons_module(self):
        """Factory should create DEPONS module."""
        params = SimulationParameters(porpoise_count=10)
        module = create_memory_module(params, MemoryMode.DEPONS)

        assert isinstance(module, DEPONSMemoryModule)
        assert module.get_mode() == MemoryMode.DEPONS

    def test_create_jasmine_module(self):
        """Factory should create JASMINE module."""
        params = SimulationParameters(porpoise_count=10)
        module = create_memory_module(params, MemoryMode.JASMINE)

        assert isinstance(module, JASMINEMemoryModule)
        assert module.get_mode() == MemoryMode.JASMINE


class TestAvoidanceResult:
    """Test AvoidanceResult properties."""

    def test_avoidance_result_creation(self):
        """Should create avoidance result with correct arrays."""
        result = AvoidanceResult(
            avoidance_dx=np.array([0.1, 0.2]),
            avoidance_dy=np.array([0.3, 0.4]),
            avoidance_strength=np.array([0.5, 0.6]),
            cells_avoided=np.array([1, 2]),
        )

        assert len(result.avoidance_dx) == 2
        assert result.avoidance_dx[0] == 0.1
        assert result.cells_avoided[1] == 2


class TestMemoryIntegration:
    """Integration tests for memory system."""

    def test_record_decay_avoid_cycle(self):
        """Test complete cycle: record -> decay -> avoid."""
        params = SimulationParameters(porpoise_count=10)
        module = JASMINEMemoryModule(params)
        state = DisturbanceMemoryState.create(10)
        mask = np.ones(10, dtype=bool)

        # 1. Record disturbance
        context = DisturbanceMemoryContext(
            is_disturbed=np.ones(10, dtype=bool),
            disturbance_intensity=np.full(10, 0.8, dtype=np.float32),
            disturbance_x=np.full(10, 50.0, dtype=np.float32),
            disturbance_y=np.full(10, 50.0, dtype=np.float32),
            agent_x=np.full(10, 50.0, dtype=np.float32),
            agent_y=np.full(10, 50.0, dtype=np.float32),
            current_tick=100,
        )
        module.record_disturbance(state, context, mask)

        # Verify recorded
        assert all(len(grid) > 0 for grid in state.memory_grids)

        # 2. Apply decay
        module.decay_memory(state, mask, ticks_elapsed=10)

        # Memory should still exist but be weaker
        for grid in state.memory_grids:
            for cell_id, strength in grid.items():
                assert strength < 0.8

        # 3. Compute avoidance
        agent_x = np.full(10, 45.0, dtype=np.float32)  # Near disturbance
        agent_y = np.full(10, 45.0, dtype=np.float32)
        result = module.compute_avoidance(state, agent_x, agent_y, mask)

        # Should have avoidance
        assert np.any(result.avoidance_strength > 0)

    def test_statistics_with_inactive_agents(self):
        """Statistics should only consider active agents."""
        params = SimulationParameters(porpoise_count=20)
        module = JASMINEMemoryModule(params)
        state = DisturbanceMemoryState.create(20)

        state.total_disturbance_exposure[:10] = 1.0
        state.total_disturbance_exposure[10:] = 5.0

        mask = np.zeros(20, dtype=bool)
        mask[:10] = True  # Only first 10 active

        stats = module.get_statistics(state, mask)

        assert stats['mean_exposure'] == 1.0  # Only active agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
