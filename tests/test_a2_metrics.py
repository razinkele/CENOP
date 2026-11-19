"""A2 metrics and bands on synthetic inputs."""

import numpy as np
import pytest

from validation.a2 import metrics as m


def test_checkpoint_passes_when_within_two_se_and_ten_percent():
    java = np.array([100.0, 104, 96, 101, 99, 103, 97, 100])
    row = m.checkpoint("x", java, java + 2.0, max_rel=0.10)
    assert row.verdict == m.PASS
    assert row.diff == pytest.approx(2.0)
    assert row.java == pytest.approx(100.0) and row.cenop == pytest.approx(102.0)


def test_checkpoint_fails_on_a_large_difference_even_if_noisy():
    java = np.array([100.0, 140, 60, 120, 80, 130, 70, 100])
    row = m.checkpoint("x", java, java + 15.0, max_rel=0.10)  # 15% off, though under 2 SE
    assert abs(row.in_se) < 2
    assert row.verdict == m.FAIL


def test_checkpoint_fails_on_a_small_but_significant_difference():
    java = np.array([100.0, 100.2, 99.8, 100.1, 99.9, 100.0, 100.1, 99.9])
    row = m.checkpoint("x", java, java + 3.0, max_rel=0.10)  # 3%, but 40 SE
    assert row.verdict == m.FAIL


def test_checkpoint_is_inconclusive_when_noise_exceeds_the_band():
    java = np.array([100.0, 140, 60, 120, 80, 130, 70, 100])
    row = m.checkpoint("x", java, java[::-1] + 1.0, max_rel=0.10)
    assert row.verdict == m.INCONCLUSIVE and row.note


def test_energy_band_is_absolute():
    java = np.array([[0.0, 5.0], [0.0, 5.1], [0.0, 4.9], [0.0, 5.0]])
    assert m.energy_rows(java, java + 0.01, [1])[0].verdict == m.PASS
    assert m.energy_rows(java, java + 0.30, [1])[0].verdict == m.FAIL


def test_population_rows_use_the_given_days_and_labels():
    java = np.tile(np.array([2000.0, 1500, 900]), (4, 1)) + np.arange(4)[:, None]
    rows = m.population_rows(java, java, [1, 2], ["a", "b"])
    assert [r.label for r in rows] == ["a", "b"]
    assert [r.java for r in rows] == [pytest.approx(1501.5), pytest.approx(901.5)]


def test_trajectory_counts_days_inside_the_java_band():
    rng = np.random.default_rng(1)
    java = 1000 + rng.normal(0, 10, (8, 101))
    inside, verdict = m.trajectory(java, java + 1.0)
    assert inside == 1.0 and verdict == m.PASS
    shifted = java.copy()
    shifted[:, 51:] += 100.0  # half the days far outside
    inside, verdict = m.trajectory(java, shifted)
    assert inside == pytest.approx(0.5) and verdict == m.FAIL


def test_trajectory_does_not_fail_on_one_animal_when_java_seeds_agree():
    java = np.full((8, 11), 2000.0)
    inside, verdict = m.trajectory(java, java - 1.0)
    assert inside == 1.0 and verdict == m.PASS


def test_block_shares_average_daily_shares_over_the_window():
    per_block = np.array([[10.0, 0.0], [5.0, 5.0], [0.0, 10.0], [0.0, 0.0]])
    np.testing.assert_allclose(m.block_shares(per_block, 1, 2), [0.25, 0.75])
    np.testing.assert_allclose(m.block_shares(per_block, 3, 3), [0.0, 0.0])  # nobody left


def test_location_distance_and_correlation():
    a = np.array([[0.5, 0.3, 0.2, 0.0]])
    tvd, corr, verdict = m.location(a, a)
    assert tvd == 0.0 and corr == pytest.approx(1.0) and verdict == m.PASS
    b = np.array([[0.0, 0.2, 0.3, 0.5]])
    tvd, corr, verdict = m.location(a, b)
    assert tvd == pytest.approx(0.6) and verdict == m.FAIL


def test_age_at_death_ks():
    rng = np.random.default_rng(2)
    ages = rng.integers(0, 20, 4000)
    _, p, verdict = m.age_at_death(ages, rng.integers(0, 20, 4000))
    assert p > 0.01 and verdict == m.PASS
    _, p, verdict = m.age_at_death(ages, rng.integers(5, 25, 4000))
    assert p < 0.01 and verdict == m.FAIL


def test_ratio_row_compares_the_effect_of_the_disturbance():
    und = np.array([100.0, 104, 96, 101, 99, 103, 97, 100])
    row = m.ratio_row("r", und * 0.9, und, und * 0.9, und)
    assert row.java == pytest.approx(0.9) and row.diff == pytest.approx(0.0)
    assert row.verdict == m.PASS
    row = m.ratio_row("r", und * 0.9, und, und * 0.5, und)
    assert row.verdict == m.FAIL


def test_exposure_band_is_twenty_percent_of_java():
    java = np.array([0.10, 0.11, 0.09, 0.10])
    assert m.exposure_row(java, java * 1.15).verdict == m.PASS
    assert m.exposure_row(java, java * 1.30).verdict == m.FAIL


def test_overall_verdict():
    assert m.overall([m.PASS, m.PASS]) == m.PASS
    assert m.overall([m.PASS, m.INCONCLUSIVE]) == m.INCONCLUSIVE
    assert m.overall([m.INCONCLUSIVE, m.FAIL]) == m.FAIL


class TestLoaders:
    def test_java_seed_uses_the_last_tick_of_each_day(self, tmp_path):
        from validation.a2.report import load_java_seed

        (tmp_path / "Statistics.csv").write_text(
            '"run","tick","PorpoiseCount","FoodEnergyLevel","PorpoiseEnergyLevel"\n'
            "1,46.0,10,0,90\n1,47.0,8,0,80\n1,48.0,7,0,70\n1,95.0,4,0,20\n"
        )
        (tmp_path / "PorpoisePerBlock.csv").write_text(
            '"run","tick","Block","PorpoiseCount"\n1,47.0,1,5\n1,47.0,60,3\n1,48.0,1,9\n1,95.0,2,4\n'
        )
        (tmp_path / "Reproduction.csv").write_text(
            '"run","tick","TickAtDeath","AgeAtDeath","CalvesBorn","CalvesWeaned","LongDistMovType"\n'
            '1,5.0,5.0,3.9,0,0,"IDW"\n1,9.0,9.0,0.2,0,0,"IDW"\n'
        )
        run = load_java_seed(tmp_path, 2)
        assert run["pop"][1:].tolist() == [8, 4]
        assert run["energy"][1:].tolist() == [10.0, 5.0]
        assert run["blocks"][1, 0] == 5 and run["blocks"][1, 59] == 3 and run["blocks"][2, 1] == 4
        assert run["ages"].tolist() == [3, 0]

    def test_java_exposure_counts_only_ticks_while_alive(self, tmp_path):
        from validation.a2.report import java_exposure

        head = '"run","tick","Id","UtmX","UtmY","EnergyLevel","DeterTurbineStrength","DeterShipStrength","LoudestShipSPL","DispersalMode","PSMActive","PSMTargetUtmX","PSMTargetUtmY"\n'
        rows = []
        # animal 1: moves on ticks 0-3 (alive 4 ticks), then its row repeats; ship noise at 1, 5
        for t, x, s in [(0, 1, 0), (1, 2, 0.5), (2, 3, 0), (3, 4, 0), (4, 4, 0), (5, 4, 0.5)]:
            rows.append(f'1,{t}.0,1,{x},0,9.0,0,{s},0,0,"false",-1,-1\n')
        # animal 2: alive all 6 ticks, ship noise on 3 of them
        for t in range(6):
            rows.append(f'1,{t}.0,2,{10 + t},0,9.0,0,{0.5 if t < 3 else 0},0,0,"false",-1,-1\n')
        (tmp_path / "RandomPorpoise.csv").write_text(head + "".join(rows))
        assert java_exposure(tmp_path) == pytest.approx((1 + 3) / (4 + 6))

    def test_cenop_seed(self, tmp_path):
        from validation.a2.report import load_cenop_seed

        head = "day,n,mean_energy,ship_ticks,alive_ticks," + ",".join(f"b{i}" for i in range(60))
        zeros = ",".join(["0"] * 59)
        (tmp_path / "seed-42.csv").write_text(
            head + f"\n1,10,9.5,20,480,10,{zeros}\n2,8,9.0,10,400,8,{zeros}\n"
        )
        (tmp_path / "seed-42-deaths.csv").write_text("day,age\n1,3\n2,0\n")
        run = load_cenop_seed(tmp_path / "seed-42.csv")
        assert run["pop"][1:].tolist() == [10, 8]
        assert run["blocks"][2, 0] == 8
        assert run["ages"].tolist() == [3, 0]
        assert run["exposure"] == pytest.approx(30 / 880)


def test_figure_is_written(tmp_path):
    from validation.a2.report import figure

    rng = np.random.default_rng(3)
    days = np.arange(361)
    pop = 2000 * np.exp(-days / 120.0) + 50
    data = {
        "pop": pop + rng.normal(0, 5, (8, 361)),
        "energy": 10 - 5 * np.sin(days / 115.0) + rng.normal(0, 0.1, (8, 361)),
    }
    out = tmp_path / "fig" / "x.png"
    figure("Synthetic", data, {k: v + 1 for k, v in data.items()}, out)
    assert out.stat().st_size > 10_000
