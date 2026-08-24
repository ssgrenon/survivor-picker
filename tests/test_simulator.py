import pandas as pd
import pytest

from backtest import simulator as sim
from strategy import entry_b_hedge as hedge

SEASON = 2099


def _row(week, home_team, away_team, home_score=None, away_score=None, home_ml=-150, away_ml=130, spread=3.0):
    return {
        "season": SEASON,
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "spread_line": spread,
    }


def _schedule(rows):
    return pd.DataFrame(rows)


def _scripted_algorithm(picks_by_week):
    def _pick(season, week, used_teams, available):
        return picks_by_week[week]

    return _pick


def test_simulate_survives_full_season():
    schedule = _schedule(
        [
            _row(1, "KC", "DEN", 30, 10),
            _row(2, "SF", "LV", 27, 20),
            _row(3, "BUF", "MIA", 24, 21),
        ]
    )
    algorithm = _scripted_algorithm({1: "KC", 2: "SF", 3: "BUF"})
    result = sim.simulate(
        SEASON, 1, algorithm, algorithm_name="picks_kc_sf_buf", schedule=schedule, market_weight=1.0
    )

    assert result.stop_reason == "survived_full_season"
    assert result.survived_full_season is True
    assert result.eliminated_week is None
    assert result.weeks_survived == 3
    assert [r.week for r in result.records] == [1, 2, 3]
    assert all(r.outcome == "WIN" and r.still_alive for r in result.records)
    assert result.records[0].actual_result == pytest.approx(20.0)
    assert result.records[0].opponent == "DEN"


def test_simulate_eliminated_on_loss_stops_advancing():
    schedule = _schedule(
        [
            _row(1, "KC", "DEN", 30, 10),  # win
            _row(2, "SF", "LV", 10, 20),  # SF loses -> eliminated
            _row(3, "BUF", "MIA", 24, 21),  # never reached
        ]
    )
    algorithm = _scripted_algorithm({1: "KC", 2: "SF", 3: "BUF"})
    result = sim.simulate(SEASON, 1, algorithm, schedule=schedule, market_weight=1.0)

    assert result.stop_reason == "eliminated"
    assert result.eliminated_week == 2
    assert result.weeks_survived == 1
    assert result.survived_full_season is False
    assert [r.week for r in result.records] == [1, 2]
    assert result.records[-1].outcome == "LOSS"
    assert result.records[-1].still_alive is False


def test_simulate_tie_counts_as_a_win_and_does_not_eliminate():
    schedule = _schedule(
        [
            _row(1, "KC", "DEN", 30, 10),
            _row(2, "SF", "LV", 20, 20),  # tie -- counts as a win in this survivor pool variant
            _row(3, "BUF", "MIA", 24, 21),
        ]
    )
    algorithm = _scripted_algorithm({1: "KC", 2: "SF", 3: "BUF"})
    result = sim.simulate(SEASON, 1, algorithm, schedule=schedule, market_weight=1.0)

    assert result.eliminated_week is None
    assert [r.week for r in result.records] == [1, 2, 3]
    assert result.records[1].outcome == "WIN"
    assert result.records[1].actual_result == pytest.approx(0.0)
    assert result.records[1].still_alive is True
    assert result.survived_full_season is True
    assert result.weeks_survived == 3


def test_simulate_stops_on_unplayed_game():
    schedule = _schedule(
        [
            _row(1, "KC", "DEN", 30, 10),
            _row(2, "SF", "LV", None, None),  # future/unplayed
        ]
    )
    algorithm = _scripted_algorithm({1: "KC", 2: "SF"})
    result = sim.simulate(SEASON, 1, algorithm, schedule=schedule, market_weight=1.0)

    assert result.stop_reason == "hit_unplayed_game"
    assert [r.week for r in result.records] == [1]
    assert result.eliminated_week is None
    assert result.survived_full_season is False


def test_simulate_raises_when_algorithm_picks_unavailable_team():
    schedule = _schedule([_row(1, "KC", "DEN", 30, 10)])
    algorithm = _scripted_algorithm({1: "SF"})  # SF isn't playing this week
    with pytest.raises(ValueError):
        sim.simulate(SEASON, 1, algorithm, schedule=schedule, market_weight=1.0)


def test_simulate_respects_initial_used_teams():
    schedule = _schedule([_row(1, "KC", "DEN", 30, 10)])
    algorithm = sim.highest_win_probability_algorithm
    result = sim.simulate(
        SEASON, 1, algorithm, schedule=schedule, initial_used_teams={"KC"}, market_weight=1.0
    )
    assert result.records[0].pick == "DEN"


def test_highest_win_probability_algorithm_picks_the_favorite():
    schedule = _schedule([_row(1, "KC", "DEN", home_ml=-400, away_ml=320)])
    available = hedge.build_candidates(SEASON, 1, used_teams=set(), schedule=schedule, market_weight=1.0)
    pick = sim.highest_win_probability_algorithm(SEASON, 1, set(), available)
    assert pick == "KC"


def test_highest_win_probability_algorithm_raises_on_empty_available():
    with pytest.raises(ValueError):
        sim.highest_win_probability_algorithm(SEASON, 1, set(), [])


def test_make_entry_b_algorithm_respects_floor():
    schedule = _schedule([_row(1, "KC", "DEN", home_ml=110, away_ml=-130)])  # KC underdog, below floor
    algorithm = sim.make_entry_b_algorithm(min_win_probability=0.65)
    available = hedge.build_candidates(SEASON, 1, used_teams=set(), schedule=schedule)
    with pytest.raises(ValueError):
        algorithm(SEASON, 1, set(), available)


def test_make_entry_a_algorithm_returns_available_team():
    schedule = _schedule(
        [
            _row(1, "KC", "DEN", home_ml=-200, away_ml=170),
            _row(2, "KC", "LV", home_ml=-900, away_ml=650),  # much better future matchup
        ]
    )
    algorithm = sim.make_entry_a_algorithm(schedule=schedule, market_weight=1.0)
    available = hedge.build_candidates(SEASON, 1, used_teams=set(), schedule=schedule, market_weight=1.0)
    pick = algorithm(SEASON, 1, set(), available)
    assert pick in {c.team for c in available}


def test_simulate_threads_market_weight_to_available_candidates():
    # KC is a heavy market favorite here (-400); with market_weight=0.0 and
    # an elo table that instead rates KC as a near-lock loser, the win
    # probability `simulate` hands the algorithm for KC should reflect elo,
    # not the market line -- proving the parameter reaches build_candidates
    # inside `simulate` itself.
    schedule = _schedule([_row(1, "KC", "DEN", home_ml=-400, away_ml=320)])
    schedule["game_id"] = ["2099_01_DEN_KC"]
    elo_games = pd.DataFrame(
        [
            {"game_id": "2099_01_DEN_KC", "season": SEASON, "week": 1, "team": "KC", "opponent": "DEN",
             "is_home": True, "elo_win_probability": 0.10},
            {"game_id": "2099_01_DEN_KC", "season": SEASON, "week": 1, "team": "DEN", "opponent": "KC",
             "is_home": False, "elo_win_probability": 0.90},
        ]
    )

    seen = {}

    def _spy_algorithm(season, week, used_teams, available):
        seen["kc_prob"] = next(c.win_probability for c in available if c.team == "KC")
        return "KC"

    sim.simulate(SEASON, 1, _spy_algorithm, schedule=schedule, market_weight=0.0, elo_games=elo_games)
    assert seen["kc_prob"] == pytest.approx(0.10)


def test_simulate_carries_model_divergence_into_week_records():
    # spread=None forces market_spread to derive from the -400 moneyline
    # (a big home favorite) rather than a posted spread_line, so it's
    # directly comparable to nfelo's much less confident 60% home rating.
    schedule = _schedule([_row(1, "KC", "DEN", 30, 10, home_ml=-400, away_ml=320, spread=None)])
    schedule["game_id"] = ["2099_01_DEN_KC"]
    elo_games = pd.DataFrame(
        [
            {"game_id": "2099_01_DEN_KC", "season": SEASON, "week": 1, "team": "KC", "opponent": "DEN",
             "is_home": True, "elo_win_probability": 0.60},
            {"game_id": "2099_01_DEN_KC", "season": SEASON, "week": 1, "team": "DEN", "opponent": "KC",
             "is_home": False, "elo_win_probability": 0.40},
        ]
    )
    algorithm = _scripted_algorithm({1: "KC"})

    without_elo = sim.simulate(SEASON, 1, algorithm, schedule=schedule, market_weight=1.0)
    assert without_elo.records[0].model_divergence is None

    with_elo = sim.simulate(SEASON, 1, algorithm, schedule=schedule, market_weight=1.0, elo_games=elo_games)
    assert with_elo.records[0].model_divergence is not None
    # Market favors KC (home) far more heavily (-400) than nfelo does (60%),
    # so nfelo's home-team spread is smaller than the market's -- negative
    # under the signed (elo_spread - market_spread) convention.
    assert with_elo.records[0].model_divergence < 0


def test_make_entry_a_algorithm_threads_market_weight():
    schedule = _schedule(
        [
            _row(1, "KC", "DEN", home_ml=-400, away_ml=320),
            _row(1, "SF", "SEA", home_ml=-150, away_ml=130),
        ]
    )
    schedule["game_id"] = ["2099_01_DEN_KC", "2099_01_SEA_SF"]
    elo_games = pd.DataFrame(
        [
            {"game_id": "2099_01_DEN_KC", "season": SEASON, "week": 1, "team": "KC", "opponent": "DEN",
             "is_home": True, "elo_win_probability": 0.20},
            {"game_id": "2099_01_DEN_KC", "season": SEASON, "week": 1, "team": "DEN", "opponent": "KC",
             "is_home": False, "elo_win_probability": 0.80},
            {"game_id": "2099_01_SEA_SF", "season": SEASON, "week": 1, "team": "SF", "opponent": "SEA",
             "is_home": True, "elo_win_probability": 0.95},
            {"game_id": "2099_01_SEA_SF", "season": SEASON, "week": 1, "team": "SEA", "opponent": "SF",
             "is_home": False, "elo_win_probability": 0.05},
        ]
    )
    available = hedge.build_candidates(SEASON, 1, used_teams=set(), schedule=schedule, market_weight=1.0)

    market_algorithm = sim.make_entry_a_algorithm(schedule=schedule, market_weight=1.0)
    assert market_algorithm(SEASON, 1, set(), available) == "KC"

    blended_algorithm = sim.make_entry_a_algorithm(schedule=schedule, market_weight=0.0, elo_games=elo_games)
    assert blended_algorithm(SEASON, 1, set(), available) == "SF"


def test_compare_algorithms_runs_each_and_keys_by_name():
    schedule = _schedule([_row(1, "KC", "DEN", 30, 10)])
    results = sim.compare_algorithms(
        SEASON,
        1,
        {
            "baseline": sim.highest_win_probability_algorithm,
            "always_den": _scripted_algorithm({1: "DEN"}),
        },
        schedule=schedule,
        market_weight=1.0,
    )
    assert set(results.keys()) == {"baseline", "always_den"}
    assert results["baseline"].records[0].pick == "KC"
    assert results["always_den"].records[0].pick == "DEN"
    assert results["always_den"].records[0].outcome == "LOSS"


def test_simulate_carries_team_bias_adjustment_into_week_records():
    schedule = _schedule([_row(1, "KC", "DEN", 30, 10, home_ml=-400, away_ml=320)])
    bias_games = pd.DataFrame(
        [{"season": season, "week": 1, "home_team": "KC", "away_team": "OPP",
          "home_score": 10, "away_score": 24, "home_moneyline": -400, "away_moneyline": 320,
          "spread_line": 9.0}
         for season in (2020, 2021, 2022)]
    )
    algorithm = _scripted_algorithm({1: "KC"})

    without_bias = sim.simulate(SEASON, 1, algorithm, schedule=schedule, market_weight=1.0)
    assert without_bias.records[0].team_bias_adjustment == 0.0

    with_bias = sim.simulate(
        SEASON, 1, algorithm, schedule=schedule, market_weight=1.0, team_bias_games=bias_games
    )
    assert with_bias.records[0].team_bias_adjustment < 0
