"""
Tests for Phase 5: Advanced Features

Tests batch runner, output writer, and histogram functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import csv
import json


class TestBatchRunner:
    """Tests for BatchRunner functionality."""
    
    def test_batch_configuration_combinations(self):
        """Test parameter combination generation."""
        from cenop.core.batch_runner import BatchConfiguration
        
        config = BatchConfiguration(
            base_params={"sim_years": 1, "landscape": "Homogeneous"},
            variations={
                "porpoise_count": [50, 100],
                "turbines": ["off", "construction"]
            },
            replicates=2
        )
        
        combos = config.get_all_combinations()
        
        # Should have 2 * 2 = 4 combinations
        assert len(combos) == 4
        
        # Total runs = 4 combos * 2 replicates = 8
        assert config.total_runs == 8
        
        # Check all combos have the base params
        for combo in combos:
            assert combo["sim_years"] == 1
            assert combo["landscape"] == "Homogeneous"
    
    def test_batch_configuration_single_param(self):
        """Test single parameter variation."""
        from cenop.core.batch_runner import BatchConfiguration
        
        config = BatchConfiguration(
            base_params={"sim_years": 1},
            variations={"porpoise_count": [50, 100, 150]},
            replicates=1
        )
        
        combos = config.get_all_combinations()
        
        assert len(combos) == 3
        assert config.total_runs == 3
    
    def test_batch_configuration_no_variations(self):
        """Test configuration with no variations (single run)."""
        from cenop.core.batch_runner import BatchConfiguration
        
        config = BatchConfiguration(
            base_params={"sim_years": 1, "porpoise_count": 50},
            variations={},
            replicates=3
        )
        
        combos = config.get_all_combinations()
        
        # One combination (base params only)
        assert len(combos) == 1
        assert config.total_runs == 3
    
    def test_batch_result_calculations(self):
        """Test BatchResult derived metric calculations."""
        from cenop.core.batch_runner import BatchResult
        
        result = BatchResult(
            run_id=0,
            parameters={"test": 1},
            final_population=110,
            initial_population=100,
            total_births=20,
            total_deaths=10,
            total_ticks=17280,  # 1 year
            execution_time_seconds=10.0,
            random_seed=42
        )
        
        # Population change should be +10%
        assert abs(result.population_change_pct - 10.0) < 0.1
        
        # Annual growth rate for 1 year = (110/100)^1 - 1 = 10%
        assert abs(result.annual_growth_rate - 10.0) < 0.1
    
    def test_batch_result_zero_initial_population(self):
        """Test BatchResult handles zero initial population."""
        from cenop.core.batch_runner import BatchResult
        
        result = BatchResult(
            run_id=0,
            parameters={},
            final_population=0,
            initial_population=0,
            total_births=0,
            total_deaths=0,
            total_ticks=1000,
            execution_time_seconds=1.0,
            random_seed=42
        )
        
        # Should not crash, should be 0
        assert result.population_change_pct == 0.0
        assert result.annual_growth_rate == 0.0
    
    def test_batch_runner_short_simulation(self):
        """Test running a very short batch (1 tick per sim)."""
        from cenop.core.batch_runner import BatchRunner, BatchConfiguration
        from cenop.parameters import SimulationParameters
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BatchConfiguration(
                base_params={
                    "sim_years": 1,
                    "porpoise_count": 10,
                    "landscape": "Homogeneous"
                },
                variations={},
                replicates=2,
                output_dir=tmpdir
            )
            
            runner = BatchRunner(config)
            
            # Run with progress=False for cleaner test output
            # Note: This will run 2 full 1-year simulations
            # For a quick test, we'd need to mock the simulation
            # Here we just test the runner setup
            assert runner.config == config
            assert len(runner.seeds) == 2
    
    def test_batch_export_results_empty(self):
        """Test export with no results raises error."""
        from cenop.core.batch_runner import BatchRunner, BatchConfiguration
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BatchConfiguration(output_dir=tmpdir)
            runner = BatchRunner(config)
            
            with pytest.raises(ValueError, match="No results"):
                runner.export_results()
    
    def test_batch_get_results_by_parameter(self):
        """Test grouping results by parameter."""
        from cenop.core.batch_runner import BatchRunner, BatchConfiguration, BatchResult
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = BatchConfiguration(output_dir=tmpdir)
            runner = BatchRunner(config)
            
            # Create mock results
            runner.results = [
                BatchResult(0, {"p": "A"}, 100, 100, 0, 0, 1000, 1.0, 1),
                BatchResult(1, {"p": "A"}, 110, 100, 0, 0, 1000, 1.0, 2),
                BatchResult(2, {"p": "B"}, 90, 100, 0, 0, 1000, 1.0, 3),
            ]
            
            grouped = runner.get_results_by_parameter("p")
            
            assert len(grouped["A"]) == 2
            assert len(grouped["B"]) == 1


class TestOutputWriter:
    """Tests for OutputWriter functionality."""
    
    def test_output_config_defaults(self):
        """Test OutputConfig default values."""
        from cenop.core.output_writer import OutputConfig
        
        config = OutputConfig()
        
        assert config.population == True
        assert config.porpoise_statistics == True
        assert config.mortality == True
        assert config.dispersal == True
        assert config.energy == True
        assert config.population_interval == 48
    
    def test_output_config_get_filename(self):
        """Test filename generation."""
        from cenop.core.output_writer import OutputConfig
        
        config = OutputConfig(run_id="test123", timestamp=False)
        
        filename = config.get_filename("Population")
        assert filename == "Population_test123.txt"
        
        config2 = OutputConfig(run_id=None, timestamp=False)
        filename2 = config2.get_filename("Mortality")
        assert filename2 == "Mortality.txt"
    
    def test_output_writer_creates_files(self):
        """Test that OutputWriter creates expected files."""
        from cenop.core.output_writer import OutputWriter, OutputConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OutputConfig(
                output_dir=tmpdir,
                population=True,
                porpoise_statistics=True,
                mortality=True,
                dispersal=True,
                energy=True
            )
            
            writer = OutputWriter(config)
            writer.open()
            writer.close()
            
            output_path = Path(tmpdir)
            
            # Check files were created
            assert (output_path / "Population.txt").exists()
            assert (output_path / "PorpoiseStatistics.txt").exists()
            assert (output_path / "Mortality.txt").exists()
            assert (output_path / "Dispersal.txt").exists()
            assert (output_path / "Energy.txt").exists()
    
    def test_output_writer_context_manager(self):
        """Test OutputWriter as context manager."""
        from cenop.core.output_writer import OutputWriter, OutputConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OutputConfig(output_dir=tmpdir)
            
            with OutputWriter(config) as writer:
                assert writer._is_open == True
            
            # After context exit, should be closed
            assert writer._is_open == False
    
    def test_output_writer_population_file_header(self):
        """Test Population.txt has correct header."""
        from cenop.core.output_writer import OutputWriter, OutputConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = OutputConfig(output_dir=tmpdir)
            
            with OutputWriter(config):
                pass
            
            # Read and check header
            pop_file = Path(tmpdir) / "Population.txt"
            with open(pop_file, 'r') as f:
                header = f.readline().strip()
                
            expected_cols = ["tick", "day", "month", "year", "population",
                           "births", "deaths", "deaths_starvation",
                           "deaths_old_age", "deaths_bycatch"]
            
            for col in expected_cols:
                assert col in header
    
    def test_mortality_event_dataclass(self):
        """Test MortalityEvent dataclass."""
        from cenop.core.output_writer import MortalityEvent
        
        event = MortalityEvent(
            tick=100,
            day=2,
            porpoise_id=5,
            age=10.5,
            energy=0.1,
            cause="starvation",
            x=50.0,
            y=75.0
        )
        
        assert event.tick == 100
        assert event.cause == "starvation"
        assert event.age == 10.5
    
    def test_dispersal_event_dataclass(self):
        """Test DispersalEvent dataclass."""
        from cenop.core.output_writer import DispersalEvent
        
        event = DispersalEvent(
            tick=200,
            day=4,
            porpoise_id=10,
            start_x=30.0,
            start_y=40.0,
            target_x=80.0,
            target_y=90.0,
            target_distance_km=50.0,
            dispersal_type="PSM"
        )
        
        assert event.tick == 200
        assert event.target_distance_km == 50.0


class TestHistogramCharts:
    """Tests for histogram chart rendering."""
    
    def test_histogram_chart_creation(self):
        """Test histogram chart is created correctly."""
        from server.renderers import create_histogram_chart
        
        data = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10]
        
        result = create_histogram_chart(
            data=data,
            title="Test Histogram",
            x_title="Values",
            y_title="Count",
            x_range=(0, 15),
            nbins=10,
            color="red",
            height=300
        )
        
        # Result should be HTML output
        assert result is not None
    
    def test_histogram_empty_data(self):
        """Test histogram handles empty data gracefully."""
        from server.renderers import create_histogram_chart
        
        # Empty list should still return something
        result = create_histogram_chart(
            data=[],
            title="Empty",
            x_title="X",
            y_title="Y",
            x_range=(0, 10)  # Required parameter
        )
        
        # Should return some UI element (may be empty chart or placeholder)
        assert result is not None


class TestIntegration:
    """Integration tests for Phase 5 components."""
    
    def test_batch_runner_with_output_writer(self):
        """Test that batch runner can be combined with output writer."""
        from cenop.core.batch_runner import BatchConfiguration, BatchRunner
        from cenop.core.output_writer import OutputConfig, OutputWriter
        
        # Just test that both modules can be imported together
        config = BatchConfiguration(
            base_params={"sim_years": 1, "porpoise_count": 10},
            variations={},
            replicates=1
        )
        
        output_config = OutputConfig(
            output_dir="output/test",
            population=True
        )
        
        assert config is not None
        assert output_config is not None
    
    def test_sensitivity_analysis_function_exists(self):
        """Test sensitivity analysis helper function exists."""
        from cenop.core.batch_runner import run_sensitivity_analysis
        
        # Just check it's callable
        assert callable(run_sensitivity_analysis)
    
    def test_scenario_comparison_function_exists(self):
        """Test scenario comparison helper function exists."""
        from cenop.core.batch_runner import run_scenario_comparison
        
        # Just check it's callable
        assert callable(run_scenario_comparison)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
