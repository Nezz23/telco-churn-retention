import numpy as np
import pandas as pd

from churn.economics import Economics, best_margin
from churn.evaluation import ece
from churn.models import Calibrator, choose_calibrator


def test_expected_value_and_realised_profit():
    e = Economics()
    assert np.isclose(e.expected_value(0.5, 100.0), 0.5 * 0.3 * 12 * 100 - 100)
    assert np.allclose(e.realised([1, 1, 0], [1, 0, 1], [100.0, 100.0, 100.0]), [260.0, -100.0, 0.0])
    assert e.revenue_at_risk([0.5], [100.0]) == 600.0


def test_margin_search_prefers_zero_on_ties():
    margin, _ = best_margin(np.array([50.0, -50.0]), np.array([1, 0]), np.array([100.0, 100.0]), Economics())
    assert margin == 0.0  # every margin in [-50, 50) earns the same; 0 is the Bayes-optimal one


def test_calibration_repairs_a_miscalibrated_score():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(size=4000) < truth).astype(int)
    raw = truth**3  # badly under-confident score with the right ranking
    method, _ = choose_calibrator(raw, y)
    assert method != "identity"
    assert ece(y, Calibrator(method).fit(raw, y)(raw)) < 0.5 * ece(y, raw)


def test_ece_is_zero_for_perfect_bins():
    assert ece(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.0
    assert ece(pd.Series([0, 0]).to_numpy(), np.array([0.9, 0.9])) > 0.8
