"""Tests for DEPONS 3.2 reference memory system."""

import numpy as np
import pytest


class TestDecayTables:
    """Test precomputed decay tables matching Java RefMem.java."""

    def test_ref_mem_strength_first_values(self):
        """refMemStrength table starts at 0.999, decays with rR=0.03.

        Java: RefMem.java:67-69 — initMemLists recomputes with actual rR param
        Default rR=0.03 from parameters.xml (NOT the hardcoded 0.10 in RefMem comments)
        Formula: s[i+1] = s[i] - rR * s[i] * (1 - s[i])
        """
        from cenop.behavior.ref_mem import get_ref_mem_strength_table

        table = get_ref_mem_strength_table(r_r=0.03, size=120)
        assert len(table) == 120
        assert table[0] == pytest.approx(0.999, abs=0.001)
        # Hand-computed: s[1] = 0.999 - 0.03*0.999*0.001 = 0.999 - 0.00002997 ≈ 0.999
        assert table[1] == pytest.approx(0.999, abs=0.001)
        # With rR=0.03, decay is slow — still >0.5 at index 50
        assert table[50] > 0.5, "rR=0.03 should decay slowly"

    def test_work_mem_strength_first_values(self):
        """workMemStrength table starts at 0.999, decays with rS=0.03.

        Java: RefMem.java:71-73 — initMemLists recomputes with actual rS param
        Default rS=0.03 from parameters.xml
        """
        from cenop.behavior.ref_mem import get_work_mem_strength_table

        # With rS=0.03 (DEPONS 3.2 default, same as rR)
        table = get_work_mem_strength_table(r_s=0.03, size=120)
        assert len(table) == 120
        assert table[0] == pytest.approx(0.999, abs=0.001)
        # With rS=0.03, decay is slow
        assert table[50] > 0.5, "rS=0.03 should decay slowly"

    def test_decay_formula(self):
        """Verify logistic decay: s[i+1] = s[i] - r * s[i] * (1 - s[i])."""
        from cenop.behavior.ref_mem import get_ref_mem_strength_table

        table = get_ref_mem_strength_table(r_r=0.10, size=5)
        for i in range(4):
            expected = table[i] - 0.10 * table[i] * (1 - table[i])
            assert table[i + 1] == pytest.approx(expected, abs=1e-4)

    def test_faster_decay_rate(self):
        """Higher rate should produce faster decay."""
        from cenop.behavior.ref_mem import get_ref_mem_strength_table

        slow = get_ref_mem_strength_table(r_r=0.03, size=120)
        fast = get_ref_mem_strength_table(r_r=0.10, size=120)
        # At index 50, fast should be much lower
        assert fast[50] < slow[50], "Higher rR should decay faster"


class TestMemoryArrays:
    """Test SoA memory arrays exist in population."""

    def test_memory_arrays_initialized(self):
        """Population should have circular buffer arrays for reference memory."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        pop = PorpoisePopulation(count=50, params=params)

        # Circular buffer for food utility
        assert hasattr(pop, "_stored_util")
        assert pop._stored_util.shape == (50, 120)

        # Position history
        assert hasattr(pop, "_pos_history_x")
        assert pop._pos_history_x.shape == (50, 120)
        assert hasattr(pop, "_pos_history_y")
        assert pop._pos_history_y.shape == (50, 120)

        # Buffer management
        assert hasattr(pop, "_mem_ptr")
        assert pop._mem_ptr.shape == (50,)
        assert hasattr(pop, "_mem_count")
        assert pop._mem_count.shape == (50,)


class TestMemoryUpdate:
    """Test memory update and veTotal/vt computation."""

    def test_stored_util_records_food(self):
        """Each tick, current cell's food level is stored in circular buffer."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        pop = PorpoisePopulation(count=5, params=params)
        pop.landscape = create_homogeneous_landscape(width=100, height=100, food_prob=0.5)
        pop.active_mask[:] = True

        pop._update_reference_memory(pop.active_mask)

        # After one update, first entry should have food level (0.5)
        assert pop._mem_count[0] == 1
        ptr = pop._mem_ptr[0]
        assert pop._stored_util[0, (ptr - 1) % 120] == pytest.approx(0.5, abs=0.1)

    def test_ve_total_increases_with_food_history(self):
        """veTotal should be positive when agent has food history."""
        from cenop.agents.population import PorpoisePopulation
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.parameters.simulation_params import SimulationParameters

        params = SimulationParameters()
        pop = PorpoisePopulation(count=3, params=params)
        pop.landscape = create_homogeneous_landscape(width=100, height=100, food_prob=0.5)
        pop.active_mask[:] = True

        # Record several ticks of food history
        for _ in range(10):
            pop._update_reference_memory(pop.active_mask)

        assert pop._ve_total[0] > 0, "veTotal should be positive with food history"


def test_attraction_zero_distance_skips():
    """When an agent is at the exact position of a memory entry, that entry should be skipped."""
    from cenop.behavior.ref_mem import compute_attraction_vector

    n_agents = 2
    mem_size = 5
    stored_util = np.zeros((n_agents, mem_size), dtype=np.float32)
    pos_x = np.zeros((n_agents, mem_size), dtype=np.float32)
    pos_y = np.zeros((n_agents, mem_size), dtype=np.float32)
    mem_ptr = np.array([1, 0], dtype=np.int32)
    mem_count = np.array([1, 0], dtype=np.int32)
    x = np.array([100.0, 200.0], dtype=np.float32)
    y = np.array([100.0, 200.0], dtype=np.float32)
    pos_x[0, 0] = 100.0
    pos_y[0, 0] = 100.0
    stored_util[0, 0] = 5.0
    ref_mem_table = np.ones(mem_size, dtype=np.float32)
    mask = np.array([True, True])
    vt_x, vt_y = compute_attraction_vector(
        stored_util, pos_x, pos_y, mem_ptr, mem_count, x, y, ref_mem_table, mask
    )
    assert vt_x[0] == 0.0, f"Expected 0.0, got {vt_x[0]}"
    assert vt_y[0] == 0.0, f"Expected 0.0, got {vt_y[0]}"
