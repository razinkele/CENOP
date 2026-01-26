"""
Tests for movement modules.

Tests the modular movement system including:
- DEPONS CRW movement
- JASMINE physics-based movement
- Hybrid movement selector
- Movement state management
"""

import pytest
import numpy as np

from cenop.movement import (
    MovementModule,
    MovementMode,
    MovementState,
    EnvironmentContext,
    MovementResult,
    DEPONSCRWMovement,
    DEPONSCRWMovementVectorized,
    JASMINEPhysicsMovement,
    HybridMovementSelector,
    HybridStrategy,
    create_movement_module,
)
from cenop.core.time_manager import TimeMode
from cenop.parameters import SimulationParameters


class TestMovementState:
    """Test MovementState creation and properties."""

    def test_create_movement_state(self):
        """MovementState.create should initialize all arrays."""
        state = MovementState.create(100)

        assert state.prev_heading.shape == (100,)
        assert state.prev_log_mov.shape == (100,)
        assert state.heading.shape == (100,)
        assert state.dx.shape == (100,)
        assert state.dy.shape == (100,)
        assert state.is_dispersing.shape == (100,)

    def test_state_default_values(self):
        """MovementState should have sensible defaults."""
        state = MovementState.create(10)

        # Heading should be random 0-360
        assert np.all(state.heading >= 0)
        assert np.all(state.heading < 360)

        # prev_log_mov should be ~1.25 (DEPONS default)
        assert np.allclose(state.prev_log_mov, 1.25)

        # Not dispersing by default
        assert not np.any(state.is_dispersing)


class TestEnvironmentContext:
    """Test EnvironmentContext creation."""

    def test_create_homogeneous(self):
        """Should create homogeneous environment context."""
        env = EnvironmentContext.create_homogeneous(50, depth=25.0, salinity=32.0)

        assert env.depth.shape == (50,)
        assert env.salinity.shape == (50,)
        assert np.all(env.depth == 25.0)
        assert np.all(env.salinity == 32.0)


class TestDEPONSCRWMovement:
    """Test DEPONS CRW movement implementation."""

    @pytest.fixture
    def params(self):
        """Create test parameters."""
        return SimulationParameters(porpoise_count=10)

    @pytest.fixture
    def movement(self, params):
        """Create DEPONS CRW movement module."""
        return DEPONSCRWMovement(params)

    @pytest.fixture
    def state(self):
        """Create test movement state."""
        return MovementState.create(10)

    @pytest.fixture
    def environment(self):
        """Create test environment."""
        return EnvironmentContext.create_homogeneous(10)

    def test_compute_step_returns_result(self, movement, state, environment):
        """compute_step should return MovementResult."""
        x = np.random.uniform(100, 200, 10).astype(np.float32)
        y = np.random.uniform(100, 200, 10).astype(np.float32)
        mask = np.ones(10, dtype=bool)

        result = movement.compute_step(x, y, state, environment, mask)

        assert isinstance(result, MovementResult)
        assert result.dx.shape == (10,)
        assert result.dy.shape == (10,)
        assert result.new_heading.shape == (10,)

    def test_movement_produces_displacement(self, movement, state, environment):
        """Active agents should have non-zero displacement."""
        x = np.full(10, 150.0, dtype=np.float32)
        y = np.full(10, 150.0, dtype=np.float32)
        mask = np.ones(10, dtype=bool)

        result = movement.compute_step(x, y, state, environment, mask)

        # Most agents should move
        distances = np.sqrt(result.dx**2 + result.dy**2)
        assert np.mean(distances) > 0.1

    def test_masked_agents_not_updated(self, movement, state, environment):
        """Masked-out agents should not be updated."""
        x = np.full(10, 150.0, dtype=np.float32)
        y = np.full(10, 150.0, dtype=np.float32)
        mask = np.zeros(10, dtype=bool)
        mask[:5] = True  # Only first 5 active

        result = movement.compute_step(x, y, state, environment, mask)

        # Inactive agents should have zero displacement
        assert np.all(result.dx[5:] == 0)
        assert np.all(result.dy[5:] == 0)

    def test_deterrence_affects_movement(self, movement, state, environment):
        """Deterrence should modify movement direction."""
        x = np.full(10, 150.0, dtype=np.float32)
        y = np.full(10, 150.0, dtype=np.float32)
        mask = np.ones(10, dtype=bool)

        # No deterrence
        result1 = movement.compute_step(x, y, state, environment, mask)

        # Reset state
        state = MovementState.create(10)
        np.random.seed(42)  # Same seed

        # With deterrence pushing right
        deter_dx = np.full(10, 5.0, dtype=np.float32)
        deter_dy = np.zeros(10, dtype=np.float32)
        result2 = movement.compute_step(x, y, state, environment, mask, deter_dx, deter_dy)

        # X displacement should be larger with deterrence
        assert np.mean(result2.dx) > np.mean(result1.dx)

    def test_heading_wraps_correctly(self, movement, state, environment):
        """Heading should stay in [0, 360)."""
        x = np.full(10, 150.0, dtype=np.float32)
        y = np.full(10, 150.0, dtype=np.float32)
        mask = np.ones(10, dtype=bool)

        # Run many steps
        for _ in range(100):
            result = movement.compute_step(x, y, state, environment, mask)
            state.heading = result.new_heading
            state.prev_angle = result.turning_angle

        # Heading should be valid
        assert np.all(result.new_heading >= 0)
        assert np.all(result.new_heading < 360)

    def test_get_mode_returns_depons(self, movement):
        """get_mode should return DEPONS_CRW."""
        assert movement.get_mode() == MovementMode.DEPONS_CRW


class TestDEPONSCRWMovementVectorized:
    """Test vectorized DEPONS CRW implementation."""

    def test_vectorized_produces_valid_results(self):
        """Vectorized version should produce valid movement results."""
        params = SimulationParameters(porpoise_count=100)
        vectorized = DEPONSCRWMovementVectorized(params)

        np.random.seed(42)
        state = MovementState.create(100)
        env = EnvironmentContext.create_homogeneous(100)
        x = np.random.uniform(100, 200, 100).astype(np.float32)
        y = np.random.uniform(100, 200, 100).astype(np.float32)
        mask = np.ones(100, dtype=bool)

        result = vectorized.compute_step(x, y, state, env, mask)

        # Verify results are valid
        assert result.dx.shape == (100,)
        assert result.dy.shape == (100,)
        assert np.all(np.isfinite(result.dx))
        assert np.all(np.isfinite(result.dy))
        assert np.all(result.new_heading >= 0)
        assert np.all(result.new_heading < 360)

        # Verify movement happens
        distances = np.sqrt(result.dx**2 + result.dy**2)
        assert np.mean(distances) > 0.1


class TestJASMINEPhysicsMovement:
    """Test JASMINE physics-based movement."""

    @pytest.fixture
    def params(self):
        """Create test parameters."""
        return SimulationParameters(porpoise_count=10)

    @pytest.fixture
    def movement(self, params):
        """Create JASMINE physics movement module."""
        return JASMINEPhysicsMovement(params)

    @pytest.fixture
    def state(self):
        """Create test movement state."""
        return MovementState.create(10)

    @pytest.fixture
    def environment(self):
        """Create test environment."""
        return EnvironmentContext.create_homogeneous(10)

    def test_compute_step_returns_result(self, movement, state, environment):
        """compute_step should return MovementResult."""
        x = np.random.uniform(100, 200, 10).astype(np.float32)
        y = np.random.uniform(100, 200, 10).astype(np.float32)
        mask = np.ones(10, dtype=bool)

        result = movement.compute_step(x, y, state, environment, mask)

        assert isinstance(result, MovementResult)
        assert result.dx.shape == (10,)
        assert result.dy.shape == (10,)

    def test_get_mode_returns_jasmine(self, movement):
        """get_mode should return JASMINE_PHYSICS."""
        assert movement.get_mode() == MovementMode.JASMINE_PHYSICS


class TestHybridMovementSelector:
    """Test hybrid movement selector."""

    @pytest.fixture
    def params(self):
        """Create test parameters."""
        return SimulationParameters(porpoise_count=10)

    def test_depons_only_strategy(self, params):
        """DEPONS_ONLY should always use DEPONS."""
        selector = HybridMovementSelector(
            params, TimeMode.DEPONS, HybridStrategy.DEPONS_ONLY
        )

        state = MovementState.create(10)
        env = EnvironmentContext.create_homogeneous(10)
        x = np.full(10, 150.0, dtype=np.float32)
        y = np.full(10, 150.0, dtype=np.float32)
        mask = np.ones(10, dtype=bool)

        selector.compute_step(x, y, state, env, mask)

        stats = selector.get_statistics()
        assert stats['depons_steps'] > 0
        assert stats['jasmine_steps'] == 0

    def test_jasmine_only_strategy(self, params):
        """JASMINE_ONLY should always use JASMINE."""
        selector = HybridMovementSelector(
            params, TimeMode.JASMINE, HybridStrategy.JASMINE_ONLY
        )

        state = MovementState.create(10)
        env = EnvironmentContext.create_homogeneous(10)
        x = np.full(10, 150.0, dtype=np.float32)
        y = np.full(10, 150.0, dtype=np.float32)
        mask = np.ones(10, dtype=bool)

        selector.compute_step(x, y, state, env, mask)

        stats = selector.get_statistics()
        assert stats['jasmine_steps'] > 0
        assert stats['depons_steps'] == 0

    def test_disturbance_aware_uses_depons_under_disturbance(self, params):
        """DISTURBANCE_AWARE should use DEPONS when disturbed."""
        selector = HybridMovementSelector(
            params, TimeMode.JASMINE, HybridStrategy.DISTURBANCE_AWARE
        )

        state = MovementState.create(10)
        env = EnvironmentContext.create_homogeneous(10)
        x = np.full(10, 150.0, dtype=np.float32)
        y = np.full(10, 150.0, dtype=np.float32)
        mask = np.ones(10, dtype=bool)

        # With disturbance
        deter_dx = np.full(10, 1.0, dtype=np.float32)
        deter_dy = np.full(10, 1.0, dtype=np.float32)

        selector.compute_step(x, y, state, env, mask, deter_dx, deter_dy)

        stats = selector.get_statistics()
        assert stats['depons_steps'] > 0

    def test_from_time_mode_factory(self, params):
        """from_time_mode should create appropriate selector."""
        # DEPONS mode
        selector1 = HybridMovementSelector.from_time_mode(params, TimeMode.DEPONS)
        assert selector1.strategy == HybridStrategy.DEPONS_ONLY

        # JASMINE mode
        selector2 = HybridMovementSelector.from_time_mode(params, TimeMode.JASMINE)
        assert selector2.strategy == HybridStrategy.JASMINE_ONLY

    def test_get_mode_returns_hybrid(self, params):
        """get_mode should return HYBRID."""
        selector = HybridMovementSelector(params, TimeMode.DEPONS)
        assert selector.get_mode() == MovementMode.HYBRID


class TestCreateMovementModule:
    """Test factory function."""

    def test_depons_mode_creates_depons(self):
        """DEPONS mode should create DEPONS movement."""
        params = SimulationParameters(porpoise_count=10)
        movement = create_movement_module(params, TimeMode.DEPONS)

        assert isinstance(movement, DEPONSCRWMovementVectorized)

    def test_jasmine_mode_creates_hybrid(self):
        """JASMINE mode should create hybrid selector."""
        params = SimulationParameters(porpoise_count=10)
        movement = create_movement_module(params, TimeMode.JASMINE)

        assert isinstance(movement, HybridMovementSelector)

    def test_explicit_mode_override(self):
        """Explicit mode should override time mode."""
        params = SimulationParameters(porpoise_count=10)

        # Request JASMINE physics even in DEPONS time mode
        movement = create_movement_module(
            params, TimeMode.DEPONS, MovementMode.JASMINE_PHYSICS
        )

        assert isinstance(movement, JASMINEPhysicsMovement)


class TestMovementReproducibility:
    """Test movement reproducibility for regulatory compliance."""

    def test_same_seed_same_results(self):
        """Same seed should produce identical movement."""
        params = SimulationParameters(porpoise_count=50)
        movement = DEPONSCRWMovementVectorized(params)

        # Create identical states with fixed headings (avoid random consumption)
        np.random.seed(42)
        state1 = MovementState.create(50)
        np.random.seed(42)
        state2 = MovementState.create(50)

        env = EnvironmentContext.create_homogeneous(50)
        x = np.full(50, 150.0, dtype=np.float32)
        y = np.full(50, 150.0, dtype=np.float32)
        mask = np.ones(50, dtype=bool)

        # First run with seed 123
        np.random.seed(123)
        result1 = movement.compute_step(x, y, state1, env, mask)

        # Second run with same seed 123
        np.random.seed(123)
        result2 = movement.compute_step(x, y, state2, env, mask)

        np.testing.assert_array_almost_equal(result1.dx, result2.dx)
        np.testing.assert_array_almost_equal(result1.dy, result2.dy)
        np.testing.assert_array_almost_equal(result1.new_heading, result2.new_heading)

    def test_different_seeds_different_results(self):
        """Different seeds should produce different movement."""
        params = SimulationParameters(porpoise_count=50)
        movement = DEPONSCRWMovementVectorized(params)

        state = MovementState.create(50)
        env = EnvironmentContext.create_homogeneous(50)
        x = np.full(50, 150.0, dtype=np.float32)
        y = np.full(50, 150.0, dtype=np.float32)
        mask = np.ones(50, dtype=bool)

        np.random.seed(42)
        result1 = movement.compute_step(x, y, state, env, mask)

        state = MovementState.create(50)
        np.random.seed(123)
        result2 = movement.compute_step(x, y, state, env, mask)

        # Results should be different
        assert not np.allclose(result1.dx, result2.dx)


class TestSimulationIntegration:
    """Test movement module integration with Simulation."""

    def test_simulation_with_movement_module(self):
        """Simulation should use provided movement module."""
        from cenop.core.simulation import Simulation

        params = SimulationParameters(
            porpoise_count=10,
            landscape="Homogeneous",
            sim_years=1
        )

        # Create movement module
        movement = DEPONSCRWMovementVectorized(params)

        # Create simulation with movement module
        sim = Simulation(params, movement_module=movement)

        # Run a few steps
        for _ in range(5):
            sim.step()

        # Verify simulation ran
        assert sim.state.tick == 5

    def test_simulation_without_movement_module(self):
        """Simulation should work without movement module (backward compat)."""
        from cenop.core.simulation import Simulation

        params = SimulationParameters(
            porpoise_count=10,
            landscape="Homogeneous",
            sim_years=1
        )

        # Create simulation without movement module
        sim = Simulation(params)

        # Run a few steps
        for _ in range(5):
            sim.step()

        # Verify simulation ran
        assert sim.state.tick == 5

    def test_modular_movement_produces_movement(self):
        """Modular movement should produce agent movement."""
        from cenop.core.simulation import Simulation

        params = SimulationParameters(
            porpoise_count=20,
            landscape="Homogeneous",
            sim_years=1
        )

        # Create with JASMINE physics movement
        movement = JASMINEPhysicsMovement(params)
        sim = Simulation(params, movement_module=movement)

        # Record initial positions
        initial_x = sim.population_manager.x.copy()
        initial_y = sim.population_manager.y.copy()

        # Run several steps
        for _ in range(10):
            sim.step()

        # Agents should have moved
        final_x = sim.population_manager.x
        final_y = sim.population_manager.y
        mask = sim.population_manager.active_mask

        # Calculate distances moved
        distances = np.sqrt((final_x - initial_x)**2 + (final_y - initial_y)**2)
        mean_distance = np.mean(distances[mask])

        assert mean_distance > 0.1, "Agents should have moved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
