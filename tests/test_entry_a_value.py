import json

import pandas as pd
import pytest

from models import dp_optimizer as dpo
from strategy import entry_a_value as strat

SEASON = 2026


def _candidate(team, opponent, is_home, win_probability, spread_line=None):
    return strat.TeamCandidate(
        team=team, opponent=opponent, is_home=is_home, win_probability=win_probability, spread_line=spread_line
    )


def test_team_spread_flips_sign_for_away_team():
    home = _candidate("KC", "DEN", True, 0.8, spread_line=6.0)
    away = _candidate("DEN", "KC", False, 0.2, spread_line=6.0)
    assert home.team_spread == pytest.approx(6.0)
    assert away.team_spread == pytest.approx(-6.0)


def test_team_spread_none_when_spread_unavailable():
    c = _candidate("KC", "DEN", True, 0.8, spread_line=None)
    assert c.team_spread is None


def test_load_used_teams_reads_state_file(tmp_path):
    state_path = tmp_path / "used_teams_a.json"
    state_path.write_text(json.dumps({"entry": "A", "season": 2026, "used_teams": {"1": "KC", "2": "SF"}}))
    used = strat.load_used_teams(state_path)
    assert used == {"KC", "SF"}


def test_load_used_teams_empty_state(tmp_path):
    state_path = tmp_path / "used_teams_a.json"
    state_path.write_text(json.dumps({"entry": "A", "season": 2026, "used_teams": {}}))
    assert strat.load_used_teams(state_path) == set()


def _week_schedule():
    return pd.DataFrame(
        [
            {
                "week": 5,
                "home_team": "KC",
                "away_team": "DEN",
                "home_moneyline": -400,
                "away_moneyline": 320,
                "spread_line": 9.5,
            },
            {
                "week": 5,
                "home_team": "SF",
                "away_team": "SEA",
                "home_moneyline": -150,
                "away_moneyline": 130,
                "spread_line": 3.0,
            },
        ]
    )


def test_build_candidates_excludes_used_teams():
    schedule = _week_schedule()
    candidates = strat.build_candidates(SEASON, 5, used_teams={"DEN"}, schedule=schedule)
    teams = {c.team for c in candidates}
    assert "DEN" not in teams
    assert {"KC", "SF", "SEA"} <= teams


def test_build_candidates_skips_games_with_no_odds_yet():
    schedule = pd.concat(
        [
            _week_schedule(),
            pd.DataFrame(
                [
                    {
                        "week": 5,
                        "home_team": "MIA",
                        "away_team": "NYJ",
                        "home_moneyline": None,
                        "away_moneyline": None,
                        "spread_line": None,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    candidates = strat.build_candidates(SEASON, 5, used_teams=set(), schedule=schedule)
    teams = {c.team for c in candidates}
    assert "MIA" not in teams
    assert "NYJ" not in teams


def _sequence_result(path_probs, week_offset=1):
    """Build a minimal OptimizedSequence for build_reasoning tests."""
    path = [
        dpo.WeekPick(
            week=week_offset + i,
            team=team,
            opponent="OPP",
            is_home=True,
            win_probability=prob,
            spread_line=9.5 if i == 0 else None,
        )
        for i, (team, prob) in enumerate(path_probs)
    ]
    survival = 1.0
    for _, prob in path_probs:
        survival *= prob
    return dpo.OptimizedSequence(
        current_week=week_offset, survival_probability=survival, path=path, candidate_universe=[t for t, _ in path_probs]
    )


def test_build_reasoning_when_recommendation_matches_greedy_best():
    result = _sequence_result([("KC", 0.85)])
    greedy_best = _candidate("KC", "OPP", True, 0.85, spread_line=9.5)
    reasoning = strat.build_reasoning(result, greedy_best)
    assert "KC" in reasoning
    assert "85.0%" in reasoning
    assert "favored by 9.5" in reasoning
    assert "Also the single best option" in reasoning


def test_build_reasoning_explains_a_hold():
    result = _sequence_result([("SF", 0.70), ("KC", 0.95)], week_offset=5)
    greedy_best = _candidate("KC", "OPP", True, 0.85)  # KC was the raw best this week, but held for week 6
    reasoning = strat.build_reasoning(result, greedy_best)
    assert "SF" in reasoning
    assert "KC is actually this week's single best option" in reasoning
    assert "Week 6" in reasoning
    assert "Projected path after this week" in reasoning


def test_recommend_pick_end_to_end_holds_team_for_better_matchup():
    schedule = pd.DataFrame(
        [
            {
                "season": SEASON, "week": 1, "home_team": "X", "away_team": "AAA",
                "home_moneyline": -400, "away_moneyline": 320, "spread_line": 9.0,
            },
            {
                "season": SEASON, "week": 1, "home_team": "Y", "away_team": "BBB",
                "home_moneyline": -150, "away_moneyline": 130, "spread_line": 3.0,
            },
            {
                "season": SEASON, "week": 2, "home_team": "X", "away_team": "CCC",
                "home_moneyline": -1900, "away_moneyline": 1200, "spread_line": 17.0,
            },
            {
                "season": SEASON, "week": 2, "home_team": "Z", "away_team": "DDD",
                "home_moneyline": -125, "away_moneyline": 105, "spread_line": 1.5,
            },
        ]
    )
    rec = strat.recommend_pick(SEASON, 1, used_teams=set(), schedule=schedule, lookahead_weeks=2)

    assert rec.entry == "A"
    assert rec.week == 1
    assert rec.team == "Y"  # holds X for its much better week-2 matchup
    assert [p.team for p in rec.projected_path] == ["Y", "X"]
    assert rec.survival_probability == pytest.approx(rec.projected_path[0].win_probability * rec.projected_path[1].win_probability)
    assert "X" in rec.reasoning and "Week 2" in rec.reasoning
    assert {c.team for c in rec.available} == {"X", "AAA", "Y", "BBB"}


def test_recommend_pick_raises_when_no_games_that_week():
    schedule = _week_schedule()
    with pytest.raises(ValueError):
        strat.recommend_pick(SEASON, 99, used_teams=set(), schedule=schedule)


def test_recommend_pick_raises_when_all_teams_used():
    schedule = _week_schedule()
    with pytest.raises(ValueError):
        strat.recommend_pick(SEASON, 5, used_teams={"KC", "DEN", "SF", "SEA"}, schedule=schedule)


def test_build_candidates_and_recommend_pick_thread_market_weight():
    schedule = _week_schedule()
    schedule["game_id"] = ["2026_05_DEN_KC", "2026_05_SEA_SF"]
    elo_games = pd.DataFrame(
        [
            {"game_id": "2026_05_DEN_KC", "season": SEASON, "week": 5, "team": "KC", "opponent": "DEN",
             "is_home": True, "elo_win_probability": 0.15},
            {"game_id": "2026_05_DEN_KC", "season": SEASON, "week": 5, "team": "DEN", "opponent": "KC",
             "is_home": False, "elo_win_probability": 0.85},
        ]
    )

    candidates = strat.build_candidates(
        SEASON, 5, used_teams=set(), schedule=schedule, market_weight=0.0, elo_games=elo_games
    )
    assert next(c for c in candidates if c.team == "KC").win_probability == pytest.approx(0.15)

    rec = strat.recommend_pick(
        SEASON, 5, used_teams=set(), schedule=schedule, market_weight=0.0, elo_games=elo_games
    )
    kc_available = next(c for c in rec.available if c.team == "KC")
    assert kc_available.win_probability == pytest.approx(0.15)


def test_recommend_pick_carries_divergence_through():
    schedule = _week_schedule()
    schedule["game_id"] = ["2026_05_DEN_KC", "2026_05_SEA_SF"]
    elo_games = pd.DataFrame(
        [
            {"game_id": "2026_05_DEN_KC", "season": SEASON, "week": 5, "team": "KC", "opponent": "DEN",
             "is_home": True, "elo_win_probability": 0.15},
            {"game_id": "2026_05_DEN_KC", "season": SEASON, "week": 5, "team": "DEN", "opponent": "KC",
             "is_home": False, "elo_win_probability": 0.85},
        ]
    )
    # market_weight=1.0 so the DP optimizer still picks KC on its market
    # merits, while divergence is still populated (computed independently
    # of market_weight whenever elo_games is supplied).
    rec = strat.recommend_pick(SEASON, 5, used_teams=set(), schedule=schedule, market_weight=1.0, elo_games=elo_games)
    assert rec.team == "KC"
    assert rec.divergence is not None
    assert rec.divergence == next(c for c in rec.available if c.team == "KC").divergence


def test_build_candidates_and_recommend_pick_thread_team_bias_games():
    schedule = _week_schedule()
    # KC priced at ~80% (-400) but this history shows it losing every one
    # of its home games -- market has been overrating KC at home.
    bias_games = pd.DataFrame(
        [
            {"season": season, "week": 1, "home_team": "KC", "away_team": "OPP",
             "home_score": 10, "away_score": 24, "home_moneyline": -400, "away_moneyline": 320,
             "spread_line": 9.5}
            for season in (2020, 2021, 2022)
        ]
    )

    candidates = strat.build_candidates(SEASON, 5, used_teams=set(), schedule=schedule, team_bias_games=bias_games)
    kc = next(c for c in candidates if c.team == "KC")
    assert kc.team_bias_adjustment < 0

    rec = strat.recommend_pick(SEASON, 5, used_teams=set(), schedule=schedule, team_bias_games=bias_games)
    kc_available = next(c for c in rec.available if c.team == "KC")
    assert kc_available.team_bias_adjustment == kc.team_bias_adjustment
