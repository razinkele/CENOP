import numpy as np
import pytest
from cenop.behavior.sound import ShipDeterrenceModel


class TestModelExpAndArrays:
    def test_magnitude_uses_exp_link_day(self):
        """DEPONS predictMag returns exp(Mag) (Ship.java:395), not the raw linear term."""
        m = ShipDeterrenceModel()
        std = m.STD_MAG_DAY
        ns = (100.0 - std['noise_mean']) / std['noise_sd']
        ds = (5.0 - std['dist_mean']) / std['dist_sd']
        linear = (m.cship_int_day + m.cship_noise_day * ns
                  + m.cship_dist_day * ds + m.cship_dist_x_noise_day * ns * ds)
        expected = np.exp(linear)
        got = m.calculate_deterrence_magnitude(100.0, 5.0, is_day=True)
        assert got == pytest.approx(expected), f"magnitude should be exp(Mag)={expected}, got {got}"

    def test_probability_and_magnitude_accept_arrays(self):
        """Both model functions must vectorize over arrays for the kernel."""
        m = ShipDeterrenceModel()
        rl = np.array([90.0, 110.0, 130.0])
        dist_km = np.array([1.0, 3.0, 8.0])
        p = m.calculate_deterrence_probability(rl, dist_km, is_day=True)
        mag = m.calculate_deterrence_magnitude(rl, dist_km, is_day=True)
        assert p.shape == (3,) and mag.shape == (3,)
        assert np.all((p >= 0.0) & (p <= 1.0))
        assert np.all(mag > 0.0)  # exp(x) > 0 always


class TestDeterrenceComponents:
    def _model(self):
        return ShipDeterrenceModel()

    def test_gate_excludes_rl_at_or_below_tships(self):
        m = self._model()
        rl = np.array([80.0, 80.0001])
        dist_m = np.array([2000.0, 2000.0])
        gdx = np.array([5.0, 5.0]); gdy = np.array([0.0, 0.0])
        u = np.array([0.0, 0.0])  # always reacts if gated & prob>0
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, is_day=True, u_draw=u, tships=80.0)
        assert react[0] == False and react[1] == True  # strict > 80

    def test_reacts_iff_u_below_prob(self):
        m = self._model()
        rl = np.array([130.0, 130.0]); dist_m = np.array([1500.0, 1500.0])
        gdx = np.array([3.0, 3.0]); gdy = np.array([0.0, 0.0])
        p = m.calculate_deterrence_probability(130.0, 1.5, True)
        u = np.array([float(p) - 1e-6, float(p) + 1e-6])
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, u, tships=80.0)
        assert react[0] == True and react[1] == False  # strict u < prob

    def test_vector_is_grid_disp_over_metre_distance_times_mag(self):
        """DEPONS Ship.java:231-235: unit vector = grid displacement / metre distance."""
        m = self._model()
        rl = np.array([130.0]); dist_m = np.array([2000.0])
        gdx = np.array([5.0]); gdy = np.array([0.0])  # 5 cells east
        u = np.array([0.0])
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, u, tships=80.0)
        assert react[0]
        assert vx[0] == pytest.approx((5.0 / 2000.0) * mag[0])
        assert vy[0] == pytest.approx(0.0)
        assert vx[0] > 0  # pushed east, away from ship

    def test_zero_vector_when_not_reacting(self):
        m = self._model()
        rl = np.array([130.0]); dist_m = np.array([2000.0])
        gdx = np.array([5.0]); gdy = np.array([0.0])
        u = np.array([1.0])  # never reacts
        vx, vy, prob, mag, react = m.deterrence_components(
            rl, dist_m, gdx, gdy, True, u, tships=80.0)
        assert react[0] == False
        assert vx[0] == 0.0 and vy[0] == 0.0
        assert mag[0] > 0.0  # magnitude still computed, just not applied
