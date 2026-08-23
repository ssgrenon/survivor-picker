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
    home_prob = wp.get_win_probability(game, "KC", market_weight=1.0).win_probability
    away_prob = wp.get_win_probability(game, "DET", market_weight=1.0).win_probability
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
    home_prob = wp.get_win_probability(game, "KC", market_weight=1.0, spread_model=FIXED_MODEL).win_probability
    away_prob = wp.get_win_probability(game, "DET", market_weight=1.0, spread_model=FIXED_MODEL).win_probability
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


def test_get_win_probability_default_market_weight_requires_elo_games():
    # market_weight now defaults to 0.75 (< 1.0), so elo_games must be supplied.
    with pytest.raises(ValueError):
        wp.get_win_probability(GAME_WITH_ID, "KC")


def test_get_win_probability_default_market_weight_blends_when_elo_supplied():
    elo_games = _elo_games(home_prob=0.80)
    market_only = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=1.0, elo_games=elo_games).win_probability
    default_result = wp.get_win_probability(GAME_WITH_ID, "KC", elo_games=elo_games).win_probability
    assert default_result != pytest.approx(market_only)


def test_get_win_probability_requires_elo_games_when_blending():
    with pytest.raises(ValueError):
        wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=0.5, elo_games=None)


def test_get_win_probability_blends_market_and_elo():
    # The blend happens in spread space: adjusted_spread = market_spread +
    # (1 - weight) * divergence, then converted back to a probability via
    # the same calibrated model -- not a linear blend of probabilities.
    elo_games = _elo_games(home_prob=0.80)
    baseline = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=1.0, elo_games=elo_games)
    elo_only = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=0.0, elo_games=elo_games).win_probability
    assert elo_only == pytest.approx(0.80)

    model = wp.get_spread_model()
    for weight in (0.75, 0.5, 0.25):
        result = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=weight, elo_games=elo_games)
        adjusted_spread = baseline.market_spread + (1 - weight) * result.divergence
        expected = model.home_win_probability(adjusted_spread)
        assert result.win_probability == pytest.approx(expected)


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
    # market_spread is always in the home team's convention, regardless of
    # which team's win probability was requested.
    game = {**GAME_WITH_ID, "spread_line": 6.5}
    result = wp.get_win_probability(game, "KC", market_weight=1.0)  # KC is home
    assert result.market_spread == pytest.approx(6.5)
    away_result = wp.get_win_probability(game, "DET", market_weight=1.0)
    assert away_result.market_spread == pytest.approx(6.5)


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
    result = wp.get_win_probability(game, "KC", market_weight=1.0, spread_model=FIXED_MODEL)
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
        assert result.divergence == pytest.approx(expected_elo_spread - result.market_spread)


def test_divergence_is_signed_positive_when_elo_favors_home_more_than_market():
    # market has KC (home) favored by 1; nfelo's 80% home win probability
    # implies a much bigger home-favored spread -- elo rates the home team
    # more favorably than the market, so divergence should be positive.
    game = {**GAME_WITH_ID, "spread_line": 1.0}
    elo_games = _elo_games(home_prob=0.80)
    result = wp.get_win_probability(game, "KC", market_weight=1.0, elo_games=elo_games)
    assert result.divergence > 0
    assert result.divergence == pytest.approx(result.elo_spread - result.market_spread)


def test_divergence_is_signed_negative_when_elo_favors_home_less_than_market():
    # market has KC (home) favored by 10; nfelo instead sees the home team
    # as a near coinflip -- elo rates the home team less favorably than the
    # market, so divergence should be negative.
    game = {**GAME_WITH_ID, "spread_line": 10.0}
    elo_games = _elo_games(home_prob=0.51)
    result = wp.get_win_probability(game, "KC", market_weight=1.0, elo_games=elo_games)
    assert result.divergence < 0


def test_divergence_is_identical_regardless_of_which_team_is_evaluated():
    # Computed entirely in the home team's convention, so it must be the
    # exact same number for both sides of the same game -- not merely equal
    # in magnitude with a flipped sign.
    game = {**GAME_WITH_ID, "spread_line": 1.0}
    elo_games = _elo_games(home_prob=0.80)
    home_result = wp.get_win_probability(game, "KC", market_weight=0.5, elo_games=elo_games)
    away_result = wp.get_win_probability(game, "DET", market_weight=0.5, elo_games=elo_games)
    assert home_result.divergence == pytest.approx(away_result.divergence)
    assert home_result.market_spread == pytest.approx(away_result.market_spread)
    assert home_result.elo_spread == pytest.approx(away_result.elo_spread)


def test_divergence_is_none_without_elo_games():
    result = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=1.0)
    assert result.elo_spread is None
    assert result.divergence is None
    assert result.market_spread is not None


# ---------------------------------------------------------------------------
# compute_team_bias / get_team_bias / team_bias_adjustment
# ---------------------------------------------------------------------------


def _bias_row(season, home_team, away_team, home_score, away_score, home_ml=-150, away_ml=130, spread=3.0):
    return {
        "season": season,
        "week": 1,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "spread_line": spread,
    }


def test_compute_team_bias_matches_hand_calculation():
    # TEST is priced as a 58% home favorite (moneyline -150) but wins
    # outright every time across three seasons -- a consistent positive
    # residual that recency-weighting and shrinkage should only partially
    # erode, matching a hand-computed expected value exactly.
    games = pd.DataFrame(
        [_bias_row(season, "TEST", "OPP", 20, 10) for season in (2020, 2021, 2022)]
    )
    decay, k, max_adj = 0.85, 15.0, 0.10  # high cap so the true shrunk value isn't clipped

    home_prob, _ = wp.devig_moneylines(-150, 130)
    residual = 1.0 - home_prob
    weights = [decay ** (2022 - s) for s in (2020, 2021, 2022)]
    expected_avg = sum(w * residual for w in weights) / sum(weights)
    expected = expected_avg * (sum(weights) / (sum(weights) + k))

    got = wp.compute_team_bias("TEST", True, games, decay_per_season=decay, shrinkage_k=k, max_adjustment=max_adj)
    assert got == pytest.approx(expected)
    assert got > 0  # market undersold TEST relative to its actual results


def test_compute_team_bias_caps_at_max_adjustment():
    # An extreme, long, undiluted run of upsets should hit the cap.
    games = pd.DataFrame(
        [_bias_row(season, "TEST", "OPP", 20, 10, home_ml=200, away_ml=-260) for season in range(2015, 2023)]
    )
    got = wp.compute_team_bias("TEST", True, games, max_adjustment=0.04)
    assert got == pytest.approx(0.04)


def test_compute_team_bias_is_negative_when_team_underperforms_market():
    games = pd.DataFrame(
        [_bias_row(season, "OPP", "TEST", 20, 10) for season in (2020, 2021, 2022)]
    )  # TEST (away, priced ~42%) loses every time -- market was still too generous
    got = wp.compute_team_bias("TEST", False, games, max_adjustment=0.10)
    assert got < 0


def test_compute_team_bias_shrinks_small_samples_toward_zero():
    # One game's raw residual is large, but shrinkage (k=15) should pull a
    # single-game sample (n≈1) most of the way back to zero.
    games = pd.DataFrame([_bias_row(2022, "TEST", "OPP", 20, 10)])
    got = wp.compute_team_bias("TEST", True, games, shrinkage_k=15.0, max_adjustment=1.0)
    home_prob, _ = wp.devig_moneylines(-150, 130)
    raw_residual = 1.0 - home_prob
    assert 0 < got < raw_residual * 0.2  # heavily shrunk relative to the raw residual


def test_compute_team_bias_excludes_ties():
    games = pd.DataFrame([_bias_row(2022, "TEST", "OPP", 20, 20)])  # tie
    assert wp.compute_team_bias("TEST", True, games) == 0.0


def test_compute_team_bias_returns_zero_for_unknown_team():
    games = pd.DataFrame([_bias_row(2022, "OTHER", "OPP", 20, 10)])
    assert wp.compute_team_bias("TEST", True, games) == 0.0


def test_compute_team_bias_skips_games_with_no_market_data():
    row = _bias_row(2022, "TEST", "OPP", 20, 10)
    row["home_moneyline"] = None
    row["away_moneyline"] = None
    row["spread_line"] = None
    games = pd.DataFrame([row])
    assert wp.compute_team_bias("TEST", True, games) == 0.0


def test_get_team_bias_caches_and_recomputes_only_when_stale():
    games = pd.DataFrame([_bias_row(2022, "TEST", "OPP", 20, 10)])
    calls = {"count": 0}
    real_compute = wp.compute_team_bias

    def _spy(*args, **kwargs):
        calls["count"] += 1
        return real_compute(*args, **kwargs)

    import unittest.mock as mock

    with mock.patch.object(wp, "compute_team_bias", side_effect=_spy):
        first = wp.get_team_bias("TEST", True, games, force_refresh=True)
        second = wp.get_team_bias("TEST", True, games)
        assert calls["count"] == 1  # second call served from cache
        assert first == second

        # A different games_df object invalidates the cache even though
        # the content is identical -- avoids cross-test/cross-caller bleed.
        other_games = pd.DataFrame([_bias_row(2022, "TEST", "OPP", 20, 10)])
        wp.get_team_bias("TEST", True, other_games)
        assert calls["count"] == 2


def test_get_win_probability_applies_team_bias_as_final_step():
    game = {**GAME_WITH_ID, "spread_line": None}  # force moneyline pricing
    bias_games = pd.DataFrame(
        [_bias_row(season, "KC", "OPP", 20, 10) for season in (2020, 2021, 2022)]
    )
    without_bias = wp.get_win_probability(game, "KC", market_weight=1.0)
    with_bias = wp.get_win_probability(
        game, "KC", market_weight=1.0, team_bias_games=bias_games, team_bias_max_adjustment=0.10
    )

    assert with_bias.team_bias_adjustment > 0
    assert with_bias.win_probability == pytest.approx(
        without_bias.win_probability + with_bias.team_bias_adjustment
    )


def test_get_win_probability_team_bias_defaults_to_zero_when_not_supplied():
    result = wp.get_win_probability(GAME_WITH_ID, "KC", market_weight=1.0)
    assert result.team_bias_adjustment == 0.0


def test_get_win_probability_clamps_final_probability_to_valid_range():
    # An extreme, unclamped market probability plus the max positive bias
    # must still land inside [0.01, 0.99].
    game = {
        "game_id": "2023_01_DET_KC",
        "season": 2023,
        "week": 1,
        "home_team": "KC",
        "away_team": "DET",
        "spread_line": None,
        "home_moneyline": -100000,
        "away_moneyline": 100000,
    }
    bias_games = pd.DataFrame(
        [_bias_row(season, "KC", "OPP", 20, 10) for season in (2020, 2021, 2022)]
    )
    result = wp.get_win_probability(game, "KC", market_weight=1.0, team_bias_games=bias_games)
    assert 0.01 <= result.win_probability <= 0.99
