"""theory.py against the numbers printed in covis-change-detection-theory.pdf.

    python -m pytest dynamic_updates/covis_gate/test_theory.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory as T  # noqa: E402


def test_figure2_survival_at_two_days():
    """Figure 2: a = 2, b = 2 / rate, dt = 2 days. The caption prints 0.980
    for the monitor (rate 0.015); eq. 3 gives (133.3 / 135.3)^2 = 0.971, so
    the caption is off, not the formula."""
    rates = [0.002, 0.015, 0.3, 0.5, 1.0, 1.5]
    expected = [0.996, 0.971, 0.592, 0.444, 0.250, 0.160]
    for lam, s in zip(rates, expected):
        a, b = T.init_rate(lam, 2)
        assert T.survival(a, b, 2.0) == pytest.approx(s, abs=1e-3)


def test_lomax_heavier_than_plugin_exponential_and_limit():
    """§3.3 items 2 and 3."""
    for dt in (0.5, 2.0, 6.0):
        assert T.survival(2, 2, dt) >= math.exp(-1.0 * dt)
        assert T.survival(1e5, 1e5, dt) == pytest.approx(math.exp(-dt), rel=1e-3)
    assert T.hazard(2, 2, 0) > T.hazard(2, 2, 5)
    assert T.median_time_to_move(1, 1) == pytest.approx(1.0)


def test_sensor_model_and_kish():
    """§4.2: l ~ 2.36; §4.3: n = 100, rho_c = 0.2 -> alpha ~ 0.05."""
    assert T.voxel_llr() == pytest.approx(2.36, abs=0.01)
    assert float(T.kish_alpha(100, 0.2)) == pytest.approx(0.048, abs=0.002)
    assert float(T.kish_alpha(1, 0.2)) == 1.0


def test_ransac_table():
    """§5.4 table: p = 0.99, s = 3."""
    for w, n in ((0.8, 7), (0.6, 19), (0.5, 35), (0.3, 169)):
        assert T.ransac_iterations(w) == n


def test_signed_residual():
    sigma = 0.01
    r = np.array([0.0, 0.019, 0.05, -0.05])
    assert T.classify_residual(r, sigma).tolist() == [T.HIT, T.HIT, T.THROUGH, T.OCCLUDED]
    assert T.classify_residual(r, sigma, signed=False).tolist() == [T.HIT, T.HIT, T.THROUGH, T.THROUGH]


def test_worked_example_7_5():
    """Pen on desk, dt = 2 days, prior S = 0.444."""
    a, b = T.init_rate(0.5, 2)
    L0 = T.logit(T.survival(a, b, 2.0))
    assert L0 == pytest.approx(-0.225, abs=0.005)
    x = -0.06 * 17 * T.voxel_llr()
    assert x == pytest.approx(-2.41, abs=0.01)
    rho_clear = T.noisy_or([0.9], [0.98], [0.94])
    assert rho_clear == pytest.approx(0.83, abs=0.005)
    rho_blocked = T.noisy_or([0.9], [0.98], [0.05])
    assert rho_blocked == pytest.approx(0.044, abs=0.001)
    # tempered vs mixture, §7.3 text
    assert float(T.gated_absence(x, 0.83)) == pytest.approx(-2.00, abs=0.01)
    assert float(T.gated_absence(x, 0.83, "mixture")) == pytest.approx(-1.40, abs=0.01)
    assert float(T.gated_absence(x, 0.044)) == pytest.approx(-0.106, abs=0.002)
    assert float(T.gated_absence(x, 0.044, "mixture")) == pytest.approx(-0.041, abs=0.002)
    # five frames
    L = L0
    for _ in range(5):
        L = float(T.clip_logodds(L + T.gated_absence(x, rho_clear), 8))
    assert L == -8
    L = L0
    for _ in range(5):
        L = float(T.clip_logodds(L + T.gated_absence(x, rho_blocked), 8))
    assert float(T.sigmoid(L)) == pytest.approx(0.32, abs=0.01)


def test_noisy_or_properties():
    """§7.2: one good neighbour suffices, more can only help, an absent one adds nothing."""
    one = T.noisy_or([0.9], [1.0], [1.0])
    two = T.noisy_or([0.9, 0.4], [1.0, 1.0], [1.0, 1.0])
    absent = T.noisy_or([0.9, 0.4], [1.0, 0.0], [1.0, 1.0])
    assert two >= one and absent == pytest.approx(one) and 0 <= two <= 1
    assert T.noisy_or([], [], []) == 0.0


def test_chow():
    """§8.1: costs 1, 1, 0.2 -> 0.2 / 0.8; false removal twice as costly -> 0.1."""
    assert T.chow_thresholds(1, 1, 0.2) == pytest.approx((0.2, 0.8))
    assert T.chow_thresholds(2, 1, 0.2)[0] == pytest.approx(0.1)


def test_rate_updates_table_9_2():
    """§9.2: mug a = b = 2 moved after 2 days."""
    a, b = T.update_moved(2, 2, 2)
    assert (a, b) == pytest.approx((2.65, 2.27), abs=0.01)
    assert a / b == pytest.approx(1.17, abs=0.01)
    an, bn = T.update_moved_naive(2, 2, 2)
    assert an / bn == pytest.approx(0.75)
    # small dt: exact mean -> (a + 1) / b
    m1, _ = T.moved_moments(2, 2, 1e-6)
    assert m1 == pytest.approx(3 / 2, rel=1e-4)
    assert T.update_stayed(2, 2, 3) == (2, 5)
    a2, b2 = T.forget(2, 4, 0.5)
    assert a2 / b2 == pytest.approx(0.5)


def test_receptacle_prior_sums_to_one():
    counts = {"desk_0": 3, "shelf_0": 1}
    recs = ["desk_0", "shelf_0", "floor_0"]
    total = sum(T.receptacle_prior(counts, c, len(recs), 1.0) for c in recs)
    assert total == pytest.approx(1.0)
