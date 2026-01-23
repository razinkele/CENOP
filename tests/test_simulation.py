"""Tests for CENOP simulation."""

import pytest
import numpy as np


class TestSimulationParameters:
    """Test simulation parameter configuration."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters()
        assert params.porpoise_count == 10000
        assert params.sim_years == 50
        assert params.landscape == "NorthSea"
        
    def test_custom_parameters(self):
        """Test custom parameter values."""
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(
            porpoise_count=1000,
            sim_years=10,
            landscape="Homogeneous"
        )
        assert params.porpoise_count == 1000
        assert params.sim_years == 10
        assert params.is_homogeneous
        
    def test_parameter_validation(self):
        """Test parameter validation."""
        from cenop.parameters import SimulationParameters
        
        with pytest.raises(ValueError):
            SimulationParameters(porpoise_count=-1)
            
        with pytest.raises(ValueError):
            SimulationParameters(sim_years=0)


class TestCellData:
    """Test landscape cell data."""
    
    def test_homogeneous_landscape(self):
        """Test homogeneous landscape creation."""
        from cenop.landscape import create_homogeneous_landscape
        
        landscape = create_homogeneous_landscape(
            width=100,
            height=100,
            depth=20.0,
            food_prob=0.5
        )
        
        assert landscape.width == 100
        assert landscape.height == 100
        assert landscape.get_depth(50, 50) == 20.0
        assert landscape.get_food_prob(50, 50) == 0.5
        
    def test_position_validation(self):
        """Test position validation."""
        from cenop.landscape import create_homogeneous_landscape
        
        landscape = create_homogeneous_landscape(width=100, height=100)
        
        assert landscape.is_valid_position(50, 50)
        assert not landscape.is_valid_position(-1, 50)
        assert not landscape.is_valid_position(50, 100)


class TestPorpoise:
    """Test porpoise agent."""
    
    def test_porpoise_creation(self):
        """Test porpoise creation with defaults."""
        from cenop.agents import Porpoise
        
        porpoise = Porpoise(id=1, x=50.0, y=50.0)
        
        assert porpoise.id == 1
        assert porpoise.x == 50.0
        assert porpoise.y == 50.0
        assert porpoise.energy > 0
        assert porpoise.is_alive
        
    def test_porpoise_movement(self):
        """Test porpoise movement."""
        from cenop.agents import Porpoise
        
        porpoise = Porpoise(id=1, x=50.0, y=50.0)
        initial_x, initial_y = porpoise.x, porpoise.y
        
        porpoise.move(distance=1.0)
        
        # Should have moved
        dist = np.sqrt((porpoise.x - initial_x)**2 + (porpoise.y - initial_y)**2)
        assert dist > 0


class TestRefMem:
    """Test reference memory."""
    
    def test_memory_add(self):
        """Test adding memories."""
        from cenop.behavior import RefMem
        
        mem = RefMem()
        assert len(mem) == 0
        
        mem.add(10.0, 20.0, 0.5)
        assert len(mem) == 1
        assert mem.satiation > 0
        
    def test_memory_decay(self):
        """Test memory decay."""
        from cenop.behavior import RefMem
        
        mem = RefMem()
        mem.add(10.0, 20.0, 0.5)
        initial_satiation = mem.satiation
        
        mem.update()
        assert mem.satiation < initial_satiation
        
    def test_best_memory(self):
        """Test getting best memory."""
        from cenop.behavior import RefMem
        
        mem = RefMem()
        mem.add(10.0, 20.0, 0.3)
        mem.add(30.0, 40.0, 0.8)
        
        best = mem.get_best_memory()
        assert best == (30.0, 40.0)


class TestDispersal:
    """Test dispersal behavior."""
    
    def test_dispersal_types(self):
        """Test dispersal type creation."""
        from cenop.behavior.dispersal import (
            create_dispersal_behavior, DispersalType, DispersalParams
        )
        
        for dtype in DispersalType:
            behavior = create_dispersal_behavior(dtype)
            assert behavior is not None
            
    def test_no_dispersal(self):
        """Test disabled dispersal."""
        from cenop.behavior.dispersal import (
            create_dispersal_behavior, DispersalType
        )
        
        behavior = create_dispersal_behavior(DispersalType.OFF)
        assert not behavior.should_start_dispersal(10, 5.0)


class TestSimulation:
    """Test full simulation."""
    
    def test_simulation_creation(self):
        """Test simulation creation."""
        from cenop import Simulation, SimulationParameters
        from cenop.landscape import create_homogeneous_landscape
        
        params = SimulationParameters(
            porpoise_count=100,
            sim_years=1,
            landscape="Homogeneous"
        )
        landscape = create_homogeneous_landscape()
        
        sim = Simulation(params, landscape)
        assert sim.population_size == 100
        
    def test_simulation_step(self):
        """Test single simulation step."""
        from cenop import Simulation, SimulationParameters
        from cenop.landscape import create_homogeneous_landscape
        
        params = SimulationParameters(
            porpoise_count=10,
            sim_years=1,
            landscape="Homogeneous"
        )
        landscape = create_homogeneous_landscape()
        
        sim = Simulation(params, landscape)
        initial_tick = sim.state.tick
        
        sim.step()
        assert sim.state.tick == initial_tick + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
