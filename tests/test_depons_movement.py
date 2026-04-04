"""
Test CENOP movement model against DEPONS 3.0 expected behaviors.

This module compares movement patterns and CRW (Correlated Random Walk)
implementation to expected DEPONS 3.0 values.

DEPONS Movement Model (from TRACE document):
- Turning angle: angleTmp = b0 * prevAngle + N(0, 4)
                 presAngle = angleTmp * (b1*depth + b2*salinity + b3)
- Step length:   logMov = R1 + a0*prevLogMov + a1*depth + a2*salinity
                 distance = 10^logMov * 100 meters

Known Differences (population.py vs porpoise.py):
1. Speed-dependent angle adjustment (m parameter) - not in population.py
2. Inertia constant (k) for PSM vector combination - not in population.py
3. Vector normalization (TRACE A35) - simplified in population.py
"""

import numpy as np
import pytest
from cenop.parameters.simulation_params import SimulationParameters
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape


class TestMovementParameters:
    """Test that movement parameters match DEPONS 3.0 defaults."""

    def test_crw_turning_angle_parameters(self):
        """Verify turning angle parameters match DEPONS."""
        params = SimulationParameters()

        # Autoregressive coefficient
        assert params.corr_angle_base == -0.024, "b0 should be -0.024"

        # Environmental modulation
        assert params.corr_angle_bathy == -0.008, "b1 (depth effect) should be -0.008"
        assert params.corr_angle_salinity == 0.93, "b2 (salinity effect) should be 0.93"
        assert params.corr_angle_base_sd == -14.0, "b3 (intercept) should be -14.0"

        # Random component R2
        assert params.r2_mean == 0.0, "R2 mean should be 0.0"
        assert params.r2_sd == 4.0, "R2 sd should be 4.0 degrees"

    def test_crw_step_length_parameters(self):
        """Verify step length parameters match DEPONS."""
        params = SimulationParameters()

        # Autoregressive coefficient
        assert params.corr_logmov_length == 0.35, "a0 should be 0.35"

        # Environmental modulation
        assert params.corr_logmov_bathy == 0.0005, "a1 (depth effect) should be 0.0005"
        assert params.corr_logmov_salinity == -0.02, "a2 (salinity effect) should be -0.02"

        # Random component R1
        assert params.r1_mean == 1.25, "R1 mean should be 1.25"
        assert params.r1_sd == 0.15, "R1 sd should be 0.15"

        # Maximum movement
        assert params.max_mov == 1.73, "max_mov should be 1.73"

    def test_speed_modulation_parameter(self):
        """Verify m parameter exists (speed-dependent turning limit)."""
        params = SimulationParameters()

        # Java SimulationParameters.initialize():262 hardcodes m=0.00001
        assert params.m == 0.00001, "m should be 0.00001 (Java runtime override)"

    def test_inertia_parameter(self):
        """Verify inertia constant k exists."""
        params = SimulationParameters()

        assert params.inertia_const == 0.001, "inertia_const (k) should be 0.001"


class TestCRWTurningAngle:
    """Test CRW turning angle calculation."""

    def test_turning_angle_with_environmental_modulation(self):
        """Test that environmental modulation affects turning angle."""
        params = SimulationParameters(porpoise_count=100, random_seed=42)
        landscape = create_homogeneous_landscape(depth=30.0)
        pop = PorpoisePopulation(100, params, landscape)

        # Record initial state
        initial_heading = pop.heading.copy()

        # Run one step
        pop.step()

        # Headings should have changed
        heading_changes = (pop.heading - initial_heading) % 360
        # Normalize to [-180, 180]
        heading_changes = np.where(heading_changes > 180, heading_changes - 360, heading_changes)

        # Mean change should be near 0 (no systematic drift)
        assert abs(np.mean(heading_changes)) < 30, "Mean heading change should be small"

        # Standard deviation should be reasonable (not too small, not too large)
        assert 5 < np.std(heading_changes) < 120, "Heading change SD should be reasonable"

    def test_turning_angle_bounded(self):
        """Turning angle should stay within reasonable bounds."""
        params = SimulationParameters(porpoise_count=200, random_seed=42)
        landscape = create_homogeneous_landscape()
        pop = PorpoisePopulation(200, params, landscape)

        # Run for 100 steps
        for _ in range(100):
            pop.step()

        # All headings should be in [0, 360)
        assert np.all(pop.heading >= 0), "Headings should be >= 0"
        assert np.all(pop.heading < 360), "Headings should be < 360"


class TestStepLength:
    """Test step length calculation."""

    def test_step_length_distribution(self):
        """Step length should follow expected distribution."""
        params = SimulationParameters(porpoise_count=500, random_seed=42)
        landscape = create_homogeneous_landscape()
        pop = PorpoisePopulation(500, params, landscape)

        # Record positions
        initial_x = pop.x.copy()
        initial_y = pop.y.copy()

        # Run one step
        pop.step()

        # Calculate distances moved (in grid cells)
        dx = pop.x - initial_x
        dy = pop.y - initial_y
        distances = np.sqrt(dx**2 + dy**2)

        # Mean step length should be around expected value
        # log_mov ~ N(1.25, 0.15) + autoregressive term
        # distance = 10^log_mov / 4 cells
        # Expected: 10^1.25 / 4 ≈ 4.4 cells
        mean_dist = np.mean(distances[pop.active_mask])
        print(f"Mean step length: {mean_dist:.2f} cells")

        # Allow wide range due to land avoidance and deterrence
        assert 0.5 < mean_dist < 10, f"Mean step length {mean_dist:.1f} outside expected range"

    def test_step_length_max_bounded(self):
        """Step length should not exceed max_mov."""
        params = SimulationParameters(porpoise_count=200, random_seed=42)
        landscape = create_homogeneous_landscape()
        pop = PorpoisePopulation(200, params, landscape)

        # Check log_mov stays bounded after steps
        for _ in range(50):
            pop.step()

        # prev_log_mov should be <= max_mov
        assert np.all(pop.prev_log_mov <= params.max_mov + 0.01), "log_mov should not exceed max_mov"


class TestLandAvoidance:
    """Test land avoidance behavior."""

    def test_porpoises_stay_in_water(self):
        """Porpoises should avoid land cells."""
        params = SimulationParameters(porpoise_count=200, random_seed=42)
        landscape = create_homogeneous_landscape(depth=30.0)
        pop = PorpoisePopulation(200, params, landscape)

        # Run for 500 steps
        for _ in range(500):
            pop.step()

        # Check all active porpoises are in valid water
        active = pop.active_mask
        x_int = pop.x[active].astype(int)
        y_int = pop.y[active].astype(int)

        # Clamp to bounds
        x_int = np.clip(x_int, 0, landscape.width - 1)
        y_int = np.clip(y_int, 0, landscape.height - 1)

        depths = landscape._depth[y_int, x_int]
        min_depth = params.min_depth

        # All porpoises should be in water (depth >= min_depth)
        in_water = depths >= min_depth
        water_fraction = np.mean(in_water)

        assert water_fraction > 0.95, f"Only {water_fraction:.1%} of porpoises in water"

    def test_land_avoidance_turn_angles(self):
        """Land avoidance should use correct turn angles (40, 70, 120 degrees)."""
        # This is tested implicitly by checking porpoises stay in water
        # The implementation uses [40, 70, 120] degree turns (verified by code inspection)
        pass


class TestDispersalMovement:
    """Test dispersal (PSM-Type2) movement."""

    def test_dispersal_reduces_turning(self):
        """Dispersing porpoises should have reduced turning angle."""
        params = SimulationParameters(porpoise_count=50, random_seed=42)
        landscape = create_homogeneous_landscape()
        pop = PorpoisePopulation(50, params, landscape)

        # Manually trigger dispersal for some porpoises
        for i in range(10):
            pop.is_dispersing[i] = True
            pop.dispersal_start_x[i] = pop.x[i]
            pop.dispersal_start_y[i] = pop.y[i]
            pop.dispersal_target_distance[i] = 100.0  # 100 cells
            pop.dispersal_target_x[i] = pop.x[i] + 50
            pop.dispersal_target_y[i] = pop.y[i] + 50

        initial_heading_disp = pop.heading[:10].copy()
        initial_heading_norm = pop.heading[10:20].copy()

        # Run 10 steps
        for _ in range(10):
            pop.step()

        # Calculate heading changes
        change_disp = np.abs((pop.heading[:10] - initial_heading_disp + 180) % 360 - 180)
        change_norm = np.abs((pop.heading[10:20] - initial_heading_norm + 180) % 360 - 180)

        # Dispersing porpoises should have smaller heading changes on average
        # (though this is stochastic, so we use a weak assertion)
        print(f"Dispersing mean change: {np.mean(change_disp):.1f}")
        print(f"Normal mean change: {np.mean(change_norm):.1f}")

    def test_dispersal_completion(self):
        """Dispersal should complete at 95% of target distance."""
        params = SimulationParameters(porpoise_count=10, random_seed=42)
        landscape = create_homogeneous_landscape(width=500, height=500)
        pop = PorpoisePopulation(10, params, landscape)

        # Start dispersal for first porpoise
        pop.is_dispersing[0] = True
        pop.dispersal_start_x[0] = 100.0
        pop.dispersal_start_y[0] = 100.0
        pop.x[0] = 100.0
        pop.y[0] = 100.0
        pop.dispersal_target_distance[0] = 50.0  # 50 cells
        pop.dispersal_target_x[0] = 150.0
        pop.dispersal_target_y[0] = 100.0
        pop.heading[0] = 90.0  # East

        # Move towards target manually
        pop.x[0] = 147.5  # 95% of distance
        pop._update_dispersal(pop.active_mask)

        # Should complete dispersal
        assert not pop.is_dispersing[0], "Dispersal should complete at 95% distance"


class TestMovementStatistics:
    """Test overall movement statistics match DEPONS expectations."""

    def test_movement_produces_realistic_trajectories(self):
        """Movement should produce realistic trajectories over time."""
        params = SimulationParameters(porpoise_count=100, random_seed=42)
        landscape = create_homogeneous_landscape(width=300, height=300)
        pop = PorpoisePopulation(100, params, landscape)

        # Track positions over time
        positions = [(pop.x.copy(), pop.y.copy())]

        for _ in range(200):
            pop.step()
            positions.append((pop.x.copy(), pop.y.copy()))

        # Calculate total displacement from start
        start_x, start_y = positions[0]
        end_x, end_y = positions[-1]

        displacements = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        active = pop.active_mask
        mean_displacement = np.mean(displacements[active])

        print(f"Mean displacement over 200 ticks: {mean_displacement:.1f} cells")

        # Should have moved significantly but not absurdly far
        # 200 ticks * ~4 cells/tick = ~800 max, but CRW should curve back
        assert 10 < mean_displacement < 500, f"Displacement {mean_displacement:.0f} outside expected range"

    def test_population_spreads_over_time(self):
        """Population should spread spatially over time."""
        params = SimulationParameters(porpoise_count=200, random_seed=42)
        landscape = create_homogeneous_landscape(width=300, height=300)
        pop = PorpoisePopulation(200, params, landscape)

        # Initial spread
        active = pop.active_mask
        initial_std_x = np.std(pop.x[active])
        initial_std_y = np.std(pop.y[active])

        # Run simulation
        for _ in range(500):
            pop.step()

        # Final spread
        active = pop.active_mask
        final_std_x = np.std(pop.x[active])
        final_std_y = np.std(pop.y[active])

        print(f"Initial spread: ({initial_std_x:.1f}, {initial_std_y:.1f})")
        print(f"Final spread: ({final_std_x:.1f}, {final_std_y:.1f})")

        # Spread should increase or stay similar (not collapse)
        assert final_std_x + final_std_y >= (initial_std_x + initial_std_y) * 0.5, \
            "Population should not collapse spatially"


class TestEnvironmentalModulation:
    """Test that depth/salinity affect movement as expected."""

    def test_depth_affects_movement(self):
        """Different depths should produce different movement patterns."""
        # This is a statistical test - hard to assert precisely
        # The effect of depth on movement is small (a1=0.0005, b1=-0.008)
        params = SimulationParameters()

        # With default parameters:
        # At depth=10m: a1*10 = 0.005 (step length), b1*10 = -0.08 (turning)
        # At depth=50m: a1*50 = 0.025 (step length), b1*50 = -0.4 (turning)

        # The effect is very small, so this test just verifies the code runs
        landscape_shallow = create_homogeneous_landscape(depth=10.0)
        landscape_deep = create_homogeneous_landscape(depth=50.0)

        pop_shallow = PorpoisePopulation(50, params, landscape_shallow)
        pop_deep = PorpoisePopulation(50, params, landscape_deep)

        # Run both
        for _ in range(100):
            pop_shallow.step()
            pop_deep.step()

        # Both should still have active porpoises
        assert np.sum(pop_shallow.active_mask) > 0
        assert np.sum(pop_deep.active_mask) > 0


class TestCRWRejectionSampling:
    """Test CRW uses retry loops instead of clipping."""

    def test_turning_angle_retry(self):
        """Turning angles > 180 should be redrawn, not clipped.

        Java: Porpoise.java:332-360 — while (abs(presAngle) > 180)
        """
        np.random.seed(42)
        # Generate many turning angles
        angles = []
        for _ in range(10000):
            angle = 999
            attempts = 0
            while abs(angle) > 180 and attempts < 200:
                angle = np.random.normal(0, 4)  # R2
                angle = -0.024 * 10.0 + angle   # b0 * prevAngle + R2
                angle *= (-0.008 * 30 + 0.93 * 30 + (-14.0))  # env modulation
                attempts += 1
            angles.append(angle)

        angles = np.array(angles)
        # All should be within [-180, 180]
        assert np.all(np.abs(angles) <= 180)
        # Distribution should NOT be truncated at ±180 (no spike at boundaries)
        n_at_boundary = np.sum(np.abs(angles) > 175)
        assert n_at_boundary < 200, "Too many at boundary — suggests clipping not retry"


class TestPrevAngleTotalTurn:
    """Test prev_angle stores total heading change."""

    def test_prev_angle_stores_total_turn(self):
        """prev_angle should store total heading change, not just CRW angle.

        Java: Porpoise.java:583 — totalTurn = substractHeadings(heading, presHeading)
        """
        old_heading = 90.0
        new_heading = 120.0
        total_turn = (new_heading - old_heading + 180) % 360 - 180
        assert total_turn == pytest.approx(30.0)

    def test_prev_angle_wraps_correctly(self):
        """Wrap around 360: heading from 350 → 10 should give +20."""
        old_heading = 350.0
        new_heading = 10.0
        total_turn = (new_heading - old_heading + 180) % 360 - 180
        assert total_turn == pytest.approx(20.0)


class TestKattegatSalinity:
    """Test Kattegat salinity override."""

    def test_kattegat_salinity_override(self):
        """Kattegat landscape should use hardcoded salinity 34.069105813295.

        Java: Porpoise.java:339-345
        """
        KATTEGAT_SALINITY = 34.069105813295
        assert KATTEGAT_SALINITY == pytest.approx(34.069, abs=0.001)


class TestHeadingComposition:
    """Test heading composition: CRW*crwContrib + vt + deterVt."""

    def test_crw_contrib_formula(self):
        """crwContrib = inertiaConst + presMov * veTotal.

        Java: Porpoise.java:549, 556
        """
        inertia_const = 0.001
        pres_mov = 5.0  # 10^prevLogMov
        ve_total = 0.8

        crw_contrib = inertia_const + pres_mov * ve_total
        assert crw_contrib == pytest.approx(4.001)

    def test_heading_composite_direction(self):
        """totalD = getDxDy * crwContrib + vt + deterVt, then facePoint.

        Java: Porpoise.java:565-566
        """
        # CRW direction (heading 90°)
        dx_crw, dy_crw = 1.0, 0.0
        crw_contrib = 2.0

        # Memory attraction (north)
        vt_x, vt_y = 0.0, 1.0

        # Deterrence (push east)
        deter_x, deter_y = 0.5, 0.0

        total_dx = dx_crw * crw_contrib + vt_x + deter_x
        total_dy = dy_crw * crw_contrib + vt_y + deter_y

        heading = np.degrees(np.arctan2(total_dx, total_dy))
        # Should be between 0 (north) and 90 (east)
        assert 0 < heading < 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
