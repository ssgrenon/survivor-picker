import numpy as np
import pandas as pd
import pytest

from models import win_prob as wp

FIXED_MODEL = wp.SpreadModel(intercept=0.0, slope=0.15)


def test_moneyline_to_implied_prob_favorite_and_underdog():
    # -200 favorite: implied prob 200/300
    assert wp.moneyline_to_implied_prob(-200) == pytest.approx(200 / 300)
    # +150 underdog: implied prob 100/250
    assert wp.moneyline_to_implied_prob(150) == pytest.approx(100 / 250)


def test_moneyline_to_implied_prob_rejects_zero():
    with pytest.raises(ValueError):
        wp.moneyline_to_implied_prob(0)


def test_devig_moneylines_sums_to_one_and_favors_favorite():
    home_prob, away_prob = wp.devig_moneylines(-198, 164)
    assert home_prob + away_prob == pytest.approx(1.0)
    assert home_prob > away_prob  # -198 is the favorite


def test_fit_logistic_1d_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    true_intercept, true_slope = 0.05, 0.18
    x = rng.uniform(-14, 14, size=5000)
    z = true_intercept + true_slope * x
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=x.shape) < p).astype(float)

    model = wp._fit_logistic_1d(x, y)
    assert model.intercept == pytest.approx(true_intercept, abs=0.05)
    assert model.slope == pytest.approx(true_slope, abs=0.02)


def test_spread_to_home_win_probability_monotonic():
    low = wp.spread_to_home_win_probability(-7, model=FIXED_MODEL)
    mid = wp.spread_to_home_win_probability(0, model=FIXED_MODEL)
    high = wp.spread_to_home_win_probability(7, model=FIXED_MODEL)
    assert low < mid < high
    assert mid == pytest.approx(0.5)


def test_get_win_probability_uses_moneylines_when_present():
    game = {
        "home_team": "KC",
        "away_team": "DET",
        "spread_line": 4.0,
        "home_moneyline": -198,
        "away_moneyline": 164,
    }
    home_prob = wp.get_win_probability(game, "KC")
    away_prob = wp.get_win_probability(game, "DET")
    assert home_prob + away_prob == pytest.approx(1.0)
    assert home_prob > 0.5


def test_get_win_probability_falls_back_to_spread_when_moneylines_missing():
    game = pd.Series(
        {
            "home_team": "KC",
            "away_team": "DET",
            "spread_line": 4.0,
            "home_moneyline": np.nan,
            "away_moneyline": np.nan,
        }
    )
    home_prob = wp.get_win_probability(game, "KC", spread_model=FIXED_MODEL)
    away_prob = wp.get_win_probability(game, "DET", spread_model=FIXED_MODEL)
    assert home_prob == pytest.approx(FIXED_MODEL.home_win_probability(4.0))
    assert home_prob + away_prob == pytest.approx(1.0)


def test_get_win_probability_raises_for_unknown_team():
    game = {
        "home_team": "KC",
        "away_team": "DET",
        "spread_line": 4.0,
        "home_moneyline": None,
        "away_moneyline": None,
    }
    with pytest.raises(ValueError):
        wp.get_win_probability(game, "SEA", spread_model=FIXED_MODEL)


def test_get_win_probability_raises_when_no_data_available():
    game = {
        "home_team": "KC",
        "away_team": "DET",
        "spread_line": None,
        "home_moneyline": None,
        "away_moneyline": None,
    }
    with pytest.raises(ValueError):
        wp.get_win_probability(game, "KC", spread_model=FIXED_MODEL)
