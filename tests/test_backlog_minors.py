"""Small fixes from the 2026-11-19 backlog pass."""

from types import SimpleNamespace

import numpy as np
import pytest

from cenop.agents.population import PorpoisePopulation
from cenop.core.batch_runner import BatchConfiguration, BatchRunner
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters


class TestDispersalEventDistance:
    @pytest.mark.parametrize("cellsize, want_km", [(400.0, 100.0), (1000.0, 250.0)])
    def test_target_distance_is_in_km(self, tmp_path, cellsize, want_km):
        # dispersal_target_distance is in cells; the column is km
        from cenop.core.output_writer import OutputConfig, OutputWriter

        params = SimulationParameters(random_seed=1, porpoise_count=4)
        land = create_homogeneous_landscape(width=50, height=50, food_prob=0.5)
        land.metadata.cellsize = cellsize
        pop = PorpoisePopulation(4, params, land)
        pop.dispersal_target_distance[2] = 250.0
        sim = SimpleNamespace(
            population_manager=pop,
            params=params,
            state=SimpleNamespace(tick=48, day=1),
            _cell_data=land,
        )
        writer = OutputWriter(OutputConfig(output_dir=str(tmp_path)))
        events = []
        writer._write_dispersal_event = events.append
        writer._prev_dispersing = np.zeros(4, dtype=bool)
        pop.is_dispersing[2] = True
        writer._detect_dispersal(sim)
        assert len(events) == 1
        assert events[0].target_distance_km == pytest.approx(want_km)


class TestBatchRunner:
    def test_too_few_seeds_is_rejected_at_construction(self, tmp_path):
        config = BatchConfiguration(output_dir=str(tmp_path), replicates=5, seeds=[42, 43])
        with pytest.raises(ValueError, match="seeds"):
            BatchRunner(config)

    def test_progress_callback_works_with_progress_off(self, tmp_path):
        calls = []
        config = BatchConfiguration(
            output_dir=str(tmp_path),
            replicates=1,
            seeds=[1],
            progress_callback=lambda done, total, label: calls.append((done, total, label)),
        )
        runner = BatchRunner(config)
        runner._run_single = lambda run_id, combo, seed: "result"
        out = runner._run_sequential([{"porpoise_count": 5}], progress=False)
        assert out == ["result"]
        assert calls == [(1, 1, "")]


class TestRecoveryGateWithoutMemoryModule:
    def test_time_since_disturbance_counts_without_jasmine_memory(self):
        from cenop.behavior import create_behavior_fsm
        from cenop.core.time_manager import TimeMode

        params = SimulationParameters(random_seed=1, porpoise_count=4)
        land = create_homogeneous_landscape(width=50, height=50, food_prob=0.5)
        fsm = create_behavior_fsm(params, TimeMode.DEPONS)
        pop = PorpoisePopulation(4, params, land, behavior_fsm=fsm)
        seen = []
        real = fsm.update_states

        def spy(state, context, *a, **k):
            seen.append(context.time_since_disturbance.copy())
            return real(state, context, *a, **k)

        fsm.update_states = spy
        mask = pop.active_mask

        pop._global_tick = 10
        pop.deter_strength[:] = 0.0
        pop.deter_strength[1] = 0.5
        pop._update_behavior_fsm(mask)
        pop._global_tick = 15
        pop.deter_strength[:] = 0.0
        pop._update_behavior_fsm(mask)

        assert seen[0].tolist() == [9999, 0, 9999, 9999]
        assert seen[1].tolist() == [9999, 5, 9999, 9999]


def test_help_text_gives_the_default_dispersal_step():
    import inspect

    from cenop.ui import layout

    text = inspect.getsource(layout)
    assert "<td>mean_disp_dist</td><td>2.0 km</td>" in text
    assert SimulationParameters().mean_disp_dist == 2.0
