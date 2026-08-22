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


def test_home_win_probability_to_spread_line_is_the_inverse():
    for spread in (-10.0, -3.5, 0.0, 3.5, 10.0):
        prob = wp.spread_to_home_win_probability(spread, model=FIXED_MODEL)
        recovered = wp.home_win_probability_to_spread_line(prob, model=FIXED_MODEL)
        assert recovered == pytest.approx(spread, abs=1e-6)


def test_get_win_probability_uses_moneylines_when_present():
    game = {
        "home_team": "KC",
        "away_team": "DET",
        "spread_line": 4.0,
        "home_moneyline": -198,
        "away_moneyline": 164,
    }
    home_prob = wp.get_win_probability(game, "KC").win_probability
    away_prob = wp.get_win_probability(game, "DET").win_probability
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
    home_prob = wp.get_win_probability(game, "KC", spread_model=FIXED_MODEL).win_probability
    away_prob = wp.get_win_probability(game, "DET", spread_model=FIXED_MODEL).win_probability
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


GAME_WITH_ID = {
    "game_id": "2023_01_DET_KC",
    "season": 2023,
    "week": 1,
    "home_team": "KC",
    "away_team": "DET",
    "spread_line": 4.0,
    "home_moneyline": -198,
    "away_moneyline": 164,
}


def _elo_games(home_prob: float = 0.80) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": "2023_01_DET_KC", "season": 2023, "week": 1, "team": "KC", "opponent": "DET",
             "is_home": True, "elo_win_probability": home_prob},
            {"game_id": "2023_01_DET_KC", "season": 2023, "week": 1, "team": "DET", "opponent": "KC",
             "is_home": False, "elo_win_probability": 1.0 - home_prob},
        ]
    )


def test_get_win_probability_rejects_invalid_market_weight():
    with pytest.raises(ValueError):
        wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=0.3, elo_games=_elo_games())


def test_get_win_probability_default_market_weight_ignores_elo_games():
    # market_weight defaults to 1.0 -- elo_games is never consulted for the
    # probability itself, so omitting it (or passing garbage) doesn't matter.
    market_prob = wp.get_win_probability(GAME_WITH_ID, "KC").win_probability
    assert market_prob == pytest.approx(wp.get_win_probability(GAME_WITH_ID, "KC", elo_games=None).win_probability)


def test_get_win_probability_requires_elo_games_when_blending():
    with pytest.raises(ValueError):
        wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=0.5, elo_games=None)


def test_get_win_probability_blends_market_and_elo():
    elo_games = _elo_games(home_prob=0.80)
    market_prob = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=1.0, elo_games=elo_games).win_probability
    elo_only = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=0.0, elo_games=elo_games).win_probability
    assert elo_only == pytest.approx(0.80)

    for weight in (0.75, 0.5, 0.25):
        blended = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=weight, elo_games=elo_games).win_probability
        assert blended == pytest.approx(weight * market_prob + (1 - weight) * elo_only)


def test_get_win_probability_blend_is_symmetric_for_both_teams():
    elo_games = _elo_games(home_prob=0.80)
    home_prob = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=0.5, elo_games=elo_games).win_probability
    away_prob = wp.get_win_probability(GAME_WITH_ID, "DET", market_weight=0.5, elo_games=elo_games).win_probability
    assert home_prob + away_prob == pytest.approx(1.0)


def test_get_win_probability_falls_back_to_market_when_elo_missing(caplog):
    empty_elo = pd.DataFrame(columns=["game_id", "season", "week", "team", "opponent", "is_home", "elo_win_probability"])
    market_prob = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=1.0).win_probability

    with caplog.at_level("WARNING"):
        result = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=0.5, elo_games=empty_elo)

    assert result.win_probability == pytest.approx(market_prob)
    assert result.elo_spread is None
    assert result.divergence is None
    assert any("nfelo rating" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# market_spread / elo_spread / divergence
# ---------------------------------------------------------------------------


def test_market_spread_uses_the_posted_spread_line_directly():
    game = {**GAME_WITH_ID, "spread_line": 6.5}
    result = wp.get_win_probability(game, "KC")  # KC is home
    assert result.market_spread == pytest.approx(6.5)
    away_result = wp.get_win_probability(game, "DET")
    assert away_result.market_spread == pytest.approx(-6.5)  # flipped for the away team


def test_market_spread_derived_from_moneylines_when_no_spread_line_posted():
    game = {
        "game_id": "2023_01_DET_KC",
        "season": 2023,
        "week": 1,
        "home_team": "KC",
        "away_team": "DET",
        "spread_line": None,
        "home_moneyline": -198,
        "away_moneyline": 164,
    }
    result = wp.get_win_probability(game, "KC", spread_model=FIXED_MODEL)
    home_prob, _ = wp.devig_moneylines(-198, 164)
    expected = wp.home_win_probability_to_spread_line(home_prob, model=FIXED_MODEL)
    assert result.market_spread == pytest.approx(expected)


def test_elo_spread_and_divergence_computed_regardless_of_market_weight():
    # nfelo's 80% win probability, at FIXED_MODEL's calibration, implies a
    # specific spread; divergence should be the same whether market_weight
    # is 1.0 (elo unused for the probability) or a blend -- it's purely a
    # display signal, not a scoring input.
    game = {**GAME_WITH_ID, "spread_line": 1.0}  # a small market spread
    elo_games = _elo_games(home_prob=0.80)
    expected_elo_spread = wp.home_win_probability_to_spread_line(0.80, model=None)

    for weight in (1.0, 0.5, 0.0):
        result = wp.get_win_probability(game, "KC", market_weight=weight, elo_games=elo_games)
        assert result.elo_spread == pytest.approx(expected_elo_spread)
        assert result.divergence == pytest.approx(abs(result.market_spread - expected_elo_spread))


def test_divergence_is_symmetric_in_magnitude_for_both_teams():
    game = {**GAME_WITH_ID, "spread_line": 1.0}
    elo_games = _elo_games(home_prob=0.80)
    home_result = wp.get_win_probability(game, "KC", market_weight=0.5, elo_games=elo_games)
    away_result = wp.get_win_probability(game, "DET", market_weight=0.5, elo_games=elo_games)
    assert home_result.divergence == pytest.approx(away_result.divergence)


def test_divergence_is_none_without_elo_games():
    result = wp.get_win_probability(GAME_WITH_ID, "KC")
    assert result.elo_spread is None
    assert result.divergence is None
    assert result.market_spread is not None
