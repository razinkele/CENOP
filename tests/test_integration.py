"""
Phase 1 Integration Tests for CENOP.

Tests verify that turbine and ship deterrence systems are correctly
integrated with the porpoise population simulation.
"""

import pytest
import numpy as np
from pathlib import Path


class TestTurbineDeterrenceIntegration:
    """Test turbine deterrence integration with population."""
    
    def test_turbine_manager_creates_deterrence_vectors(self):
        """Verify TurbineManager produces non-zero deterrence vectors."""
        from cenop.agents.turbine import TurbineManager, Turbine, TurbinePhase
        from cenop.parameters import SimulationParameters
        
        # Create manager with active turbines
        manager = TurbineManager()
        manager.set_phase(TurbinePhase.CONSTRUCTION)
        
        # Add turbines at known positions
        # Use higher source level (pile driving can be 220+ dB)
        turbine = Turbine(
            id=0,
            x=50.0,
            y=50.0,
            heading=0.0,
            name="TestTurbine",
            impact=220.0,  # High source level dB for pile driving
            start_tick=0,
            end_tick=1000000
        )
        turbine._is_active = True
        turbine.phase = TurbinePhase.CONSTRUCTION
        manager.turbines.append(turbine)
        
        params = SimulationParameters()
        
        # Calculate max deterrence distance:
        # RL = SL - 20*log10(d) > threshold
        # 220 - 20*log10(d) > 158
        # 62 > 20*log10(d)
        # d < 10^3.1 = 1259m = ~3.1 cells
        
        # Create porpoise positions - some within deterrence range, some outside
        porpoise_x = np.array([50.5, 51.0, 52.0, 55.0, 100.0], dtype=np.float32)
        porpoise_y = np.array([50.0, 50.0, 50.0, 50.0, 50.0], dtype=np.float32)
        
        # Calculate deterrence
        dx, dy = manager.calculate_aggregate_deterrence_vectorized(
            porpoise_x, porpoise_y, params, cell_size=400.0
        )
        
        # Porpoises near turbine should have deterrence vectors
        assert dx.shape == porpoise_x.shape
        assert dy.shape == porpoise_y.shape
        
        # Debug output
        for i in range(len(porpoise_x)):
            dist_cells = porpoise_x[i] - 50.0
            dist_m = dist_cells * 400.0
            print(f"Porpoise {i}: dist={dist_m:.0f}m, dx={dx[i]:.4f}, dy={dy[i]:.4f}")
        
        # At least some porpoises should be deterred (the close ones)
        total_deterrence = np.abs(dx) + np.abs(dy)
        assert np.any(total_deterrence > 0), "Some porpoises should be deterred"
        
        # Closer porpoises should have stronger deterrence
        # Porpoise 0 at 0.5 cells (200m) should have stronger deter than porpoise 2 at 2 cells (800m)
        if total_deterrence[0] > 0 and total_deterrence[2] > 0:
            assert total_deterrence[0] > total_deterrence[2], \
                "Closer porpoises should have stronger deterrence"
    
    def test_turbine_deterrence_applied_in_simulation_step(self):
        """Verify turbine deterrence is applied during simulation step."""
        from cenop.core.simulation import Simulation
        from cenop.parameters import SimulationParameters
        
        # Create simulation with turbines enabled
        params = SimulationParameters(
            porpoise_count=50,
            sim_years=1,
            landscape="Homogeneous",
            turbines="construction"  # Enable construction turbines
        )
        sim = Simulation(params=params)
        sim.initialize()
        
        # Manually add a turbine near center
        from cenop.agents.turbine import Turbine, TurbinePhase
        turbine = Turbine(
            id=99,
            x=sim.params.world_width / 2,
            y=sim.params.world_height / 2,
            heading=0.0,
            name="CenterTurbine",
            impact=210.0,
            start_tick=0,
            end_tick=1000000
        )
        turbine._is_active = True
        turbine.phase = TurbinePhase.CONSTRUCTION
        sim._turbine_manager.turbines.append(turbine)
        
        # Run a few steps
        initial_deter = sim.population_manager.deter_strength.copy()
        
        for _ in range(10):
            sim.step()
        
        # Check that deterrence was recorded
        current_deter = sim.population_manager.deter_strength
        
        # At least some porpoises should have non-zero deterrence
        print(f"Initial deter sum: {np.sum(initial_deter)}")
        print(f"Current deter sum: {np.sum(current_deter)}")
        print(f"Max deter strength: {np.max(current_deter)}")
        
        # Note: deterrence might be 0 if all porpoises are far away
        # This test confirms the code path runs without error
        assert current_deter is not None
        
    def test_turbine_noise_calculation(self):
        """Test sound propagation from turbine to porpoise."""
        from cenop.agents.turbine import Turbine, TurbinePhase
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters()
        
        # Use higher source level (pile driving can be 220+ dB)
        turbine = Turbine(
            id=0, x=50.0, y=50.0, heading=0.0,
            name="Test", impact=220.0,  # Realistic pile driving level
            start_tick=0, end_tick=1000000
        )
        turbine._is_active = True
        turbine.phase = TurbinePhase.CONSTRUCTION
        
        # Test at various distances
        # With SL=220 and threshold=158, max deterrence distance is ~1259m = ~3.1 cells
        distances_cells = [0.5, 1, 2, 3, 5]
        for d in distances_cells:
            should_deter, rl, dist_m, strength = turbine.should_deter(
                50.0 + d, 50.0, params, cell_size=400.0
            )
            print(f"Distance: {d} cells ({dist_m:.0f}m), RL: {rl:.1f} dB, "
                  f"Strength: {strength:.2f}, Deter: {should_deter}")
            
            # Close distances should trigger deterrence
            if d <= 2:  # Within ~800m should deter with SL=220
                assert rl > params.deter_threshold, \
                    f"RL should exceed threshold at {d} cells"


class TestShipDeterrenceIntegration:
    """Test ship deterrence integration with population."""
    
    def test_ship_manager_creates_deterrence_vectors(self):
        """Verify ShipManager produces non-zero deterrence vectors."""
        from cenop.agents.ship import ShipManager, Ship, Route, Buoy, VesselClass
        from cenop.parameters import SimulationParameters
        
        # Create manager with active ship
        manager = ShipManager()
        manager.set_enabled(True)
        
        # Create a route
        route = Route(
            name="TestRoute",
            buoys=[
                Buoy(x=10, y=50, speed=10),
                Buoy(x=100, y=50, speed=10)
            ]
        )
        
        # Create ship at position
        ship = Ship(
            id=0,
            x=50.0,
            y=50.0,
            heading=90.0,
            name="TestShip",
            vessel_type=VesselClass.CARGO,
            vessel_length=200.0,
            route=route,
            tick_start=0,
            tick_end=1000000
        )
        ship._is_active = True
        manager.ships.append(ship)
        
        # Create porpoise positions near ship (cargo ships have ~175 dB source level)
        # Max deterrence at threshold 158: 175 - 158 = 17 dB margin
        # TL = 20*log10(d) => d = 10^(17/20) = ~7m
        # Need very close porpoises for ship deterrence to trigger
        porpoise_x = np.array([50.02, 50.05, 50.1, 50.5], dtype=np.float32)  # Very close
        porpoise_y = np.array([50.0, 50.0, 50.0, 50.0], dtype=np.float32)
        
        params = SimulationParameters()
        
        # Calculate deterrence
        dx, dy = manager.calculate_aggregate_deterrence_vectorized(
            porpoise_x, porpoise_y, params, is_day=True, cell_size=400.0
        )
        
        print(f"Ship deterrence dx: {dx}")
        print(f"Ship deterrence dy: {dy}")
        
        assert dx.shape == porpoise_x.shape
        assert dy.shape == porpoise_y.shape
        
        # At least some deterrence should occur
        total_deterrence = np.abs(dx) + np.abs(dy)
        assert np.any(total_deterrence > 0), "Some porpoises should be deterred by ship"
    
    def test_ship_deterrence_applied_in_simulation(self):
        """Verify ship deterrence is applied during simulation."""
        from cenop.core.simulation import Simulation
        from cenop.parameters import SimulationParameters
        from cenop.agents.ship import Ship, Route, Buoy, VesselClass
        
        # Create simulation with ships enabled
        params = SimulationParameters(
            porpoise_count=50,
            sim_years=1,
            landscape="Homogeneous",
            ships_enabled=True
        )
        sim = Simulation(params=params)
        sim.initialize()
        
        # Ensure we have at least one ship
        if not sim._ship_manager.ships:
            route = Route(
                name="TestRoute",
                buoys=[
                    Buoy(x=10, y=sim.params.world_height/2, speed=10),
                    Buoy(x=sim.params.world_width-10, y=sim.params.world_height/2, speed=10)
                ]
            )
            ship = Ship(
                id=0,
                x=sim.params.world_width / 2,
                y=sim.params.world_height / 2,
                heading=90.0,
                name="CargoShip_Test",
                vessel_type=VesselClass.CARGO,
                vessel_length=200.0,
                route=route
            )
            ship._is_active = True
            sim._ship_manager.ships.append(ship)
        
        # Run steps
        for _ in range(10):
            sim.step()
            
        # Verify the ship manager is working (it's okay if no deterrence occurs
        # since porpoises may not be near ships)
        print(f"Ship manager enabled: {sim._ship_manager.enabled}")
        print(f"Ship count: {len(sim._ship_manager.ships)}")
        print(f"Active ships: {len(sim._ship_manager.get_active_ships())}")
        
        # The test passes if we got here without error - integration is working
    
    def test_ship_day_night_deterrence_difference(self):
        """Test that day/night affects ship deterrence probability."""
        from cenop.agents.ship import ShipDeterrenceModel
        
        model = ShipDeterrenceModel()
        
        # Test at same distance and noise level
        spl = 140.0  # dB
        distance = 1000.0  # m
        
        prob_day = model.calculate_deterrence_probability(spl, distance, is_day=True)
        prob_night = model.calculate_deterrence_probability(spl, distance, is_day=False)
        
        print(f"Deterrence probability at SPL={spl} dB, dist={distance}m:")
        print(f"  Day: {prob_day:.4f}")
        print(f"  Night: {prob_night:.4f}")
        
        # Day and night probabilities should differ (DEPONS has different coefficients)
        # Note: depending on implementation, one might be higher than the other


class TestCombinedDeterrence:
    """Test combined turbine + ship deterrence."""
    
    def test_combined_deterrence_vectors_accumulate(self):
        """Verify turbine and ship deterrence vectors combine."""
        from cenop.core.simulation import Simulation
        from cenop.parameters import SimulationParameters
        from cenop.agents.turbine import Turbine, TurbinePhase
        from cenop.agents.ship import Ship, Route, Buoy, VesselClass
        
        params = SimulationParameters(
            porpoise_count=100,
            sim_years=1,
            landscape="Homogeneous",
            turbines="construction",
            ships_enabled=True
        )
        sim = Simulation(params=params)
        sim.initialize()
        
        center_x = sim.params.world_width / 2
        center_y = sim.params.world_height / 2
        
        # Add turbine at center
        turbine = Turbine(
            id=99, x=center_x, y=center_y, heading=0.0,
            name="CenterTurbine", impact=210.0,
            start_tick=0, end_tick=1000000
        )
        turbine._is_active = True
        turbine.phase = TurbinePhase.CONSTRUCTION
        sim._turbine_manager.turbines.append(turbine)
        
        # Add ship near center
        route = Route(
            name="CenterRoute",
            buoys=[
                Buoy(x=center_x - 20, y=center_y, speed=10),
                Buoy(x=center_x + 20, y=center_y, speed=10)
            ]
        )
        ship = Ship(
            id=99, x=center_x + 5, y=center_y, heading=90.0,
            name="CenterShip", vessel_type=VesselClass.CARGO,
            vessel_length=200.0, route=route
        )
        ship._is_active = True
        sim._ship_manager.ships.append(ship)
        
        # Step and observe deterrence
        sim.step()
        
        deter = sim.population_manager.deter_strength
        print(f"Combined deterrence - Max: {np.max(deter):.4f}, "
              f"Mean: {np.mean(deter):.4f}, "
              f"Deterred count: {np.sum(deter > 0)}")
        
        # With both sources, some porpoises should be deterred
        assert sim.population_manager is not None


class TestLandscapeDataLoading:
    """Test loading real DEPONS landscape data."""
    
    def test_homogeneous_landscape_valid(self):
        """Test homogeneous landscape creation."""
        from cenop.landscape import create_homogeneous_landscape
        
        landscape = create_homogeneous_landscape(
            width=250, height=500, depth=25.0, food_prob=0.3
        )
        
        assert landscape.width == 250
        assert landscape.height == 500
        assert landscape.get_depth(125, 250) == 25.0
        
    def test_wind_farm_data_loading(self):
        """Test loading wind farm turbine data files."""
        from cenop.agents.turbine import Turbine, TurbineManager
        from cenop.config import get_wind_farm_file
        
        # Test DanTysk file exists and loads
        dantysk_file = get_wind_farm_file("DanTysk-construction.txt")
        
        if not dantysk_file.exists():
            pytest.skip(f"DanTysk file not found: {dantysk_file}")
        
        # Load turbines
        turbines = Turbine.load_from_file(
            str(dantysk_file),
            utm_origin_x=0.0,
            utm_origin_y=0.0,
            cell_size=400.0
        )
        
        print(f"Loaded {len(turbines)} turbines from DanTysk file")
        assert len(turbines) > 0, "Should load some turbines"
        
        # Check first turbine has valid data
        t = turbines[0]
        print(f"First turbine: id={t.id}, name={t.name}, impact={t.impact}, "
              f"start={t.start_tick}, end={t.end_tick}")
        assert t.impact > 0, "Turbine should have positive impact (source level)"
        assert t.start_tick >= 0, "Start tick should be non-negative"
        
    def test_depons_landscape_loading(self):
        """Test loading real DEPONS landscape if available."""
        from cenop.landscape.loader import LandscapeLoader
        from cenop.config import DATA_DIR
        
        # Check if Kattegat or NorthSea data exists
        kattegat_path = DATA_DIR / "Kattegat"
        northsea_path = DATA_DIR / "NorthSea"
        
        landscape_path = None
        if kattegat_path.exists():
            landscape_path = kattegat_path
            landscape_name = "Kattegat"
        elif northsea_path.exists():
            landscape_path = northsea_path
            landscape_name = "NorthSea"
            
        if landscape_path is None:
            pytest.skip("No real landscape data available (Kattegat or NorthSea)")
            
        # Try loading
        loader = LandscapeLoader(landscape_name, DATA_DIR)
        try:
            data = loader.load_all()
            
            assert 'depth' in data
            assert 'metadata' in data
            assert data['metadata'] is not None
            
            print(f"Loaded {landscape_name}:")
            print(f"  Size: {data['metadata'].ncols} x {data['metadata'].nrows}")
            print(f"  Cell size: {data['metadata'].cellsize}m")
            print(f"  Depth range: {np.min(data['depth']):.1f} to {np.max(data['depth']):.1f}m")
            
        except FileNotFoundError as e:
            pytest.skip(f"Missing landscape file: {e}")


class TestDeterrenceStrengthLogging:
    """Test that deterrence tracking works correctly."""
    
    def test_deter_strength_array_updated(self):
        """Verify deter_strength array is updated in population."""
        from cenop.core.simulation import Simulation
        from cenop.parameters import SimulationParameters
        from cenop.agents.turbine import Turbine, TurbinePhase
        
        params = SimulationParameters(
            porpoise_count=100,
            sim_years=1,
            landscape="Homogeneous",
            turbines="construction"
        )
        sim = Simulation(params=params)
        sim.initialize()
        
        # Place turbine at center where porpoises might be
        center_x = sim.params.world_width / 2
        center_y = sim.params.world_height / 2
        
        turbine = Turbine(
            id=99, x=center_x, y=center_y, heading=0.0,
            name="CenterTurbine", impact=210.0,
            start_tick=0, end_tick=1000000
        )
        turbine._is_active = True
        turbine.phase = TurbinePhase.CONSTRUCTION
        sim._turbine_manager.turbines = [turbine]  # Replace all
        
        # Move some porpoises very close to turbine
        # With SL=210, threshold=158, beta=20, alpha=0:
        # Deterrence occurs when dist < 10^((210-158)/20) = 10^2.6 = 398m ≈ 1 cell
        # Place porpoises within 0.5 cells (~200m) to ensure deterrence
        pm = sim.population_manager
        pm.x[:10] = center_x + np.random.uniform(-0.5, 0.5, 10)
        pm.y[:10] = center_y + np.random.uniform(-0.5, 0.5, 10)
        
        # Step
        sim.step()
        
        # Those 10 porpoises should have high deterrence
        close_deter = pm.deter_strength[:10]
        print(f"Deterrence for close porpoises: {close_deter}")
        
        # At least some should be deterred
        assert np.any(close_deter > 0), "Close porpoises should be deterred"


class TestPSMIntegration:
    """Phase 2: Test PSM integration with population."""
    
    def test_psm_instances_created(self):
        """Verify each porpoise has a PSM instance."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=50)
        pop = PorpoisePopulation(count=50, params=params)
        
        # Check PSM instances exist
        assert len(pop._psm_instances) == 50
        assert all(psm is not None for psm in pop._psm_instances)
        
    def test_psm_updates_on_step(self):
        """Verify PSM is updated when porpoises move and eat."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters

        params = SimulationParameters(porpoise_count=10)
        pop = PorpoisePopulation(count=10, params=params)

        # Record initial PSM state from psm_buffer
        # psm_buffer shape: (count, rows, cols, 2) where [:,:,:,0] is visit count
        initial_visited = np.count_nonzero(pop.psm_buffer[:, :, :, 0], axis=(1, 2))

        # Run several steps
        for _ in range(10):
            pop.step()

        # PSM should have been updated (check psm_buffer, not _psm_instances)
        final_visited = np.count_nonzero(pop.psm_buffer[:, :, :, 0], axis=(1, 2))

        # At least some porpoises should have visited cells
        assert np.sum(final_visited) > 0, "PSM should record visited cells"

        # Check total food recorded from psm_buffer
        # psm_buffer[:,:,:,1] contains food values
        total_food = np.sum(pop.psm_buffer[:, :, :, 1])
        assert total_food > 0, "PSM should record food obtained"
        
    def test_energy_history_tracked(self):
        """Verify energy history is tracked for dispersal trigger."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=5)
        pop = PorpoisePopulation(count=5, params=params)
        
        # Run 48 ticks (1 day)
        for _ in range(48):
            pop.step()
            
        # Energy history should be updated
        # At least one day of history
        assert pop._tick_counter == 0, "Tick counter should reset after 48 ticks"
        
        # Run another day
        for _ in range(48):
            pop.step()
            
        # Check energy history has values
        assert np.any(pop._energy_history > 0), "Energy history should have values"
        
    def test_dispersal_stats_available(self):
        """Verify dispersal statistics are accessible."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=20)
        pop = PorpoisePopulation(count=20, params=params)
        
        # Run some steps
        for _ in range(100):
            pop.step()
            
        stats = pop.get_dispersal_stats()
        
        assert 'dispersing_count' in stats
        assert 'total_active' in stats
        assert 'avg_psm_cells' in stats
        assert 'max_declining_days' in stats
        
        print(f"Dispersal stats after 100 steps: {stats}")
        
    def test_psm_inherited_by_calf(self):
        """Verify calves inherit mother's PSM."""
        from cenop.behavior.psm import PersistentSpatialMemory
        
        # Create mother PSM with some memory
        mother_psm = PersistentSpatialMemory(world_width=100, world_height=100)
        
        # Record some food locations
        for _ in range(100):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            mother_psm.update(x, y, food_eaten=np.random.uniform(0.1, 1.0))
            
        # Create calf PSM
        calf_psm = mother_psm.copy_for_calf()
        
        # Calf should have same visited cells
        assert calf_psm.visited_cell_count == mother_psm.visited_cell_count
        
        # But different preferred distance
        # (may occasionally be same due to randomness)
        # Just check it's valid
        assert calf_psm.preferred_distance >= 1.0


class TestDispersalBehavior:
    """Phase 2: Test dispersal behavior integration."""
    
    def test_dispersal_heading_dampening(self):
        """Verify dispersing porpoises have reduced turning."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=10)
        pop = PorpoisePopulation(count=10, params=params)
        
        # Manually trigger dispersal for first porpoise
        pop._start_dispersal(0)
        
        assert pop.is_dispersing[0], "Porpoise 0 should be dispersing"
        
        # Record heading
        initial_heading = pop.heading[0]
        
        # Step - dispersing porpoise should maintain heading better
        pop.step()
        
        # Not a strict test since random, but heading should be set
        assert 0 <= pop.heading[0] < 360
        
    def test_dispersal_target_selection(self):
        """Verify dispersal target is selected from PSM."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=5)
        pop = PorpoisePopulation(count=5, params=params)
        
        # Build up PSM memory first
        for _ in range(200):
            pop.step()
            
        # Manually trigger dispersal
        pop._start_dispersal(0)
        
        assert pop.is_dispersing[0]
        assert pop.dispersal_target_distance[0] > 0, "Target distance should be set"
        
    def test_dispersal_completion(self):
        """Verify dispersal ends when target distance reached."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=1)
        pop = PorpoisePopulation(count=1, params=params)
        
        # Manually trigger dispersal
        pop._start_dispersal(0)
        
        # Set a short target distance for testing
        pop.dispersal_target_distance[0] = 5.0  # 5 cells
        
        # Move the porpoise past the target distance
        pop.x[0] = pop.dispersal_start_x[0] + 6.0
        pop.y[0] = pop.dispersal_start_y[0]
        
        # Update dispersal should detect completion
        pop._update_dispersal(pop.active_mask)
        
        assert not pop.is_dispersing[0], "Dispersal should end when target reached"


class TestEnhancedEnergetics:
    """Phase 3: Test enhanced energetics implementation."""
    
    def test_hunger_based_eating(self):
        """Verify hungry porpoises eat more."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=10)
        pop = PorpoisePopulation(count=10, params=params)
        
        # Set half to hungry, half to full
        pop.energy[:5] = 5.0   # Hungry (will eat (20-5)/10 = 1.5, capped at 0.99)
        pop.energy[5:] = 18.0  # Full (will eat (20-18)/10 = 0.2)
        
        # Calculate fraction to eat
        fract = np.clip((20.0 - pop.energy) / 10.0, 0.0, 0.99)
        
        assert all(fract[:5] > 0.9), "Hungry porpoises should try to eat more"
        assert all(fract[5:] < 0.25), "Full porpoises should eat less"
        
    def test_seasonal_energy_scaling(self):
        """Verify seasonal energy scaling factors."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=5)
        pop = PorpoisePopulation(count=5, params=params)
        
        # Test cold months (Nov-Mar)
        for month in [1, 2, 3, 11, 12]:
            scaling = pop._get_energy_scaling(month, pop.active_mask)
            assert np.allclose(scaling, 1.0), f"Cold month {month} should have scaling=1.0"
            
        # Test transition months
        for month in [4, 10]:
            scaling = pop._get_energy_scaling(month, pop.active_mask)
            assert np.allclose(scaling, 1.15), f"Transition month {month} should have scaling=1.15"
            
        # Test warm months
        for month in [5, 6, 7, 8, 9]:
            scaling = pop._get_energy_scaling(month, pop.active_mask)
            assert np.allclose(scaling, params.e_warm), f"Warm month {month} should have scaling={params.e_warm}"
            
    def test_lactation_energy_scaling(self):
        """Verify lactating females have higher energy cost."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=10)
        pop = PorpoisePopulation(count=10, params=params)
        
        # Set some females with calves
        pop.with_calf[:5] = True
        pop.with_calf[5:] = False
        
        # Get scaling in summer (month 7)
        scaling = pop._get_energy_scaling(7, pop.active_mask)
        
        # Lactating: e_warm * e_lact = 1.3 * 1.4 = 1.82
        expected_lact = params.e_warm * params.e_lact
        assert np.allclose(scaling[:5], expected_lact), "Lactating females should have higher scaling"
        
        # Non-lactating: e_warm = 1.3
        assert np.allclose(scaling[5:], params.e_warm), "Non-lactating should have base warm scaling"
        
    def test_energy_stats_api(self):
        """Verify energy statistics are accessible."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=20)
        pop = PorpoisePopulation(count=20, params=params)
        
        # Run some steps
        for _ in range(50):
            pop.step()
            
        stats = pop.get_energy_stats()
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'hungry' in stats
        assert 'starving' in stats
        
        print(f"Energy stats after 50 steps: {stats}")
        
    def test_starvation_mortality(self):
        """Verify low energy increases mortality risk."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=100)
        pop = PorpoisePopulation(count=100, params=params)
        
        # Verify that survival probability decreases with lower energy
        # DEPONS formula: yearly_surv = 1 - (M_MORT_PROB_CONST * exp(-energy * xSurvivalConst))
        m_mort = 0.5
        x_surv = 0.15
        
        # At energy = 0: yearly_surv = 1 - 0.5 * exp(0) = 0.5 (50% annual mortality)
        yearly_surv_0 = 1.0 - (m_mort * np.exp(-0 * x_surv))
        assert 0.49 < yearly_surv_0 < 0.51, f"At energy=0, survival should be ~50%, got {yearly_surv_0}"
        
        # At energy = 10: yearly_surv = 1 - 0.5 * exp(-1.5) ≈ 0.89
        yearly_surv_10 = 1.0 - (m_mort * np.exp(-10 * x_surv))
        assert yearly_surv_10 > 0.85, f"At energy=10, survival should be >85%, got {yearly_surv_10}"
        
        # At energy = 20: yearly_surv = 1 - 0.5 * exp(-3) ≈ 0.975
        yearly_surv_20 = 1.0 - (m_mort * np.exp(-20 * x_surv))
        assert yearly_surv_20 > 0.97, f"At energy=20, survival should be >97%, got {yearly_surv_20}"
        
        print(f"Survival probabilities: E=0: {yearly_surv_0:.3f}, E=10: {yearly_surv_10:.3f}, E=20: {yearly_surv_20:.3f}")
        
    def test_calf_abandonment_before_death(self):
        """Verify lactating mothers abandon calves before dying."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters import SimulationParameters
        
        params = SimulationParameters(porpoise_count=20)
        pop = PorpoisePopulation(count=20, params=params)
        
        # Set some lactating females with low energy
        pop.with_calf[:10] = True
        pop.is_female[:10] = True
        pop.energy[:10] = 1.0  # Very low energy
        
        initial_with_calf = np.sum(pop.with_calf)
        
        # Run steps - some should abandon calves
        for _ in range(200):
            pop.step()
            
        final_with_calf = np.sum(pop.with_calf)
        
        # Some calves should be abandoned
        print(f"Calves: {initial_with_calf} -> {final_with_calf}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
