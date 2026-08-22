import json

import pandas as pd
import pytest

from strategy import entry_b_hedge as hedge


def _candidate(team, opponent, is_home, win_probability, spread_line=None):
    return hedge.TeamCandidate(
        team=team,
        opponent=opponent,
        is_home=is_home,
        win_probability=win_probability,
        spread_line=spread_line,
    )


def test_team_spread_flips_sign_for_away_team():
    home = _candidate("KC", "DEN", True, 0.8, spread_line=6.0)
    away = _candidate("DEN", "KC", False, 0.2, spread_line=6.0)
    assert home.team_spread == pytest.approx(6.0)
    assert away.team_spread == pytest.approx(-6.0)


def test_rank_picks_filters_below_floor_and_sorts_descending():
    candidates = [
        _candidate("KC", "DEN", True, 0.90),
        _candidate("SF", "SEA", True, 0.60),  # below default 65% floor
        _candidate("BUF", "NYJ", True, 0.70),
    ]
    ranked = hedge.rank_picks(candidates)
    assert [c.team for c in ranked] == ["KC", "BUF"]


def test_rank_picks_custom_floor():
    candidates = [_candidate("KC", "DEN", True, 0.55), _candidate("SF", "SEA", True, 0.50)]
    assert hedge.rank_picks(candidates, min_win_probability=0.5) == [
        candidates[0],
        candidates[1],
    ]
    assert hedge.rank_picks(candidates, min_win_probability=0.6) == []


def test_build_reasoning_mentions_floor_and_runner_up():
    ranked = hedge.rank_picks(
        [
            _candidate("KC", "DEN", True, 0.85, spread_line=9.5),
            _candidate("SF", "SEA", True, 0.75, spread_line=5.5),
        ]
    )
    reasoning = hedge.build_reasoning(ranked[0], ranked[1], min_win_probability=0.65)
    assert "KC" in reasoning
    assert "85.0%" in reasoning
    assert "65%" in reasoning
    assert "SF" in reasoning


def test_build_reasoning_handles_sole_candidate():
    ranked = hedge.rank_picks([_candidate("KC", "DEN", True, 0.85, spread_line=9.5)])
    reasoning = hedge.build_reasoning(ranked[0], None)
    assert "only team" in reasoning


def test_load_used_teams_reads_state_file(tmp_path):
    state_path = tmp_path / "used_teams_b.json"
    state_path.write_text(json.dumps({"entry": "B", "season": 2026, "used_teams": {"1": "KC"}}))
    assert hedge.load_used_teams(state_path) == {"KC"}


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
                "home_moneyline": -105,
                "away_moneyline": -115,
                "spread_line": 1.0,
            },
        ]
    )


def test_build_candidates_excludes_used_teams():
    schedule = _week_schedule()
    candidates = hedge.build_candidates(2026, 5, used_teams={"DEN"}, schedule=schedule)
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
    candidates = hedge.build_candidates(2026, 5, used_teams=set(), schedule=schedule)
    teams = {c.team for c in candidates}
    assert "MIA" not in teams
    assert "NYJ" not in teams


def test_recommend_pick_picks_highest_prob_above_floor():
    schedule = _week_schedule()
    rec = hedge.recommend_pick(2026, 5, used_teams=set(), schedule=schedule)
    assert rec.entry == "B"
    assert rec.team == "KC"  # heavy favorite, clears the floor
    assert rec.win_probability >= 0.65


def test_recommend_pick_raises_when_nothing_clears_floor():
    schedule = _week_schedule()
    with pytest.raises(ValueError):
        hedge.recommend_pick(2026, 5, used_teams=set(), schedule=schedule, min_win_probability=0.99)


def test_recommend_pick_raises_when_no_games_that_week():
    schedule = _week_schedule()
    with pytest.raises(ValueError):
        hedge.recommend_pick(2026, 99, used_teams=set(), schedule=schedule)
