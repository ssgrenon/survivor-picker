import pandas as pd
import pytest

from strategy import entry_b_hedge as hedge
from strategy import joint_optimizer as jo


def _candidate(team, opponent, is_home, win_probability):
    return hedge.TeamCandidate(
        team=team, opponent=opponent, is_home=is_home, win_probability=win_probability, spread_line=None
    )


def test_evaluate_pair_matches_independent_probability_formula():
    cand_a = _candidate("KC", "DEN", True, 0.8)
    cand_b = _candidate("SF", "SEA", True, 0.7)
    pick = jo.evaluate_pair(cand_a, cand_b)
    assert pick.both_survive == pytest.approx(0.8 * 0.7)
    assert pick.both_eliminated == pytest.approx(0.2 * 0.3)
    assert pick.one_survives == pytest.approx(1 - 0.8 * 0.7 - 0.2 * 0.3)
    assert pick.objective == pytest.approx(0.8 + 0.7 - 0.2 * 0.3)


def test_find_valid_pairs_excludes_same_team():
    candidates_a = [_candidate("KC", "DEN", True, 0.8)]
    candidates_b = [_candidate("KC", "DEN", True, 0.8)]
    pairs = jo.find_valid_pairs(candidates_a, candidates_b, min_win_probability_b=0.0)
    assert pairs == []


def test_find_valid_pairs_excludes_opposing_sides_of_same_game():
    candidates_a = [_candidate("KC", "DEN", True, 0.8)]
    candidates_b = [_candidate("DEN", "KC", False, 0.2)]
    pairs = jo.find_valid_pairs(candidates_a, candidates_b, min_win_probability_b=0.0)
    assert pairs == []


def test_find_valid_pairs_enforces_entry_b_floor():
    candidates_a = [_candidate("KC", "DEN", True, 0.8)]
    candidates_b = [_candidate("SF", "SEA", True, 0.60)]  # below default 65% floor
    pairs = jo.find_valid_pairs(candidates_a, candidates_b)
    assert pairs == []

    pairs_lenient = jo.find_valid_pairs(candidates_a, candidates_b, min_win_probability_b=0.5)
    assert len(pairs_lenient) == 1


def test_find_valid_pairs_allows_independent_games():
    candidates_a = [_candidate("KC", "DEN", True, 0.8)]
    candidates_b = [_candidate("SF", "SEA", True, 0.75)]
    pairs = jo.find_valid_pairs(candidates_a, candidates_b)
    assert len(pairs) == 1
    assert pairs[0].team_a == "KC"
    assert pairs[0].team_b == "SF"


def _week_schedule():
    return pd.DataFrame(
        [
            {
                "week": 5,
                "home_team": "KC",
                "away_team": "DEN",
                "home_moneyline": -1000,
                "away_moneyline": 700,
                "spread_line": 15.0,
            },
            {
                "week": 5,
                "home_team": "SF",
                "away_team": "SEA",
                "home_moneyline": -220,
                "away_moneyline": 180,
                "spread_line": 4.5,
            },
            {
                "week": 5,
                "home_team": "BUF",
                "away_team": "NYJ",
                "home_moneyline": 105,
                "away_moneyline": -125,  # BUF is a slight underdog, below floor
                "spread_line": -1.0,
            },
        ]
    )


def test_recommend_joint_pick_end_to_end():
    schedule = _week_schedule()
    rec = jo.recommend_joint_pick(
        2026, 5, used_teams_a=set(), used_teams_b=set(), schedule=schedule, market_weight=1.0
    )

    assert rec.pick_a != rec.pick_b
    # picks must not be opposing sides of the same game
    game_pairs = {("KC", "DEN"), ("DEN", "KC"), ("SF", "SEA"), ("SEA", "SF"), ("BUF", "NYJ"), ("NYJ", "BUF")}
    assert (rec.pick_a, rec.pick_b) not in game_pairs
    assert rec.win_probability_b >= jo.DEFAULT_MIN_WIN_PROBABILITY_B
    assert rec.both_survive_probability == pytest.approx(rec.win_probability_a * rec.win_probability_b)
    assert rec.both_eliminated_probability == pytest.approx(
        (1 - rec.win_probability_a) * (1 - rec.win_probability_b)
    )
    assert rec.pick_a in rec.reasoning and rec.pick_b in rec.reasoning
    # KC is the heaviest favorite and should end up picked for one of the two entries
    assert "KC" in (rec.pick_a, rec.pick_b)


def test_recommend_joint_pick_raises_when_no_valid_pairing():
    # Single game, so any A/B pairing is either the same team or opposing sides.
    schedule = _week_schedule().iloc[[0]]
    with pytest.raises(ValueError):
        jo.recommend_joint_pick(2026, 5, used_teams_a=set(), used_teams_b=set(), schedule=schedule)


def test_recommend_joint_pick_respects_used_teams_per_entry():
    schedule = _week_schedule()
    rec = jo.recommend_joint_pick(
        2026, 5, used_teams_a={"KC"}, used_teams_b=set(), schedule=schedule, market_weight=1.0
    )
    assert rec.pick_a != "KC"


def test_recommend_joint_pick_threads_market_weight():
    schedule = _week_schedule()
    schedule["game_id"] = ["2026_05_DEN_KC", "2026_05_SEA_SF", "2026_05_NYJ_BUF"]
    elo_games = pd.DataFrame(
        [
            {"game_id": "2026_05_DEN_KC", "season": 2026, "week": 5, "team": "KC", "opponent": "DEN",
             "is_home": True, "elo_win_probability": 0.99},
            {"game_id": "2026_05_DEN_KC", "season": 2026, "week": 5, "team": "DEN", "opponent": "KC",
             "is_home": False, "elo_win_probability": 0.01},
        ]
    )
    rec = jo.recommend_joint_pick(
        2026, 5, used_teams_a=set(), used_teams_b=set(), schedule=schedule,
        market_weight=0.0, elo_games=elo_games,
    )
    kc_prob = rec.win_probability_a if rec.pick_a == "KC" else rec.win_probability_b
    assert kc_prob == pytest.approx(0.99)


def test_recommend_joint_pick_threads_team_bias_games():
    schedule = _week_schedule()
    bias_games = pd.DataFrame(
        [{"season": season, "week": 1, "home_team": "KC", "away_team": "OPP",
          "home_score": 10, "away_score": 24, "home_moneyline": -1000, "away_moneyline": 700,
          "spread_line": 15.0}
         for season in (2018, 2019, 2020, 2021, 2022)]
    )
    rec = jo.recommend_joint_pick(
        2026, 5, used_teams_a=set(), used_teams_b=set(), schedule=schedule,
        market_weight=1.0, team_bias_games=bias_games,
    )
    kc_prob = rec.win_probability_a if rec.pick_a == "KC" else (
        rec.win_probability_b if rec.pick_b == "KC" else None
    )
    if kc_prob is not None:
        assert kc_prob < 1000 / (1000 + 100)  # below KC's raw devigged market probability
