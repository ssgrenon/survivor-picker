import json

import pandas as pd
import pytest

from strategy import entry_a_value as strat


def _candidate(team, opponent, is_home, win_probability, future_value, spread_line=None):
    return strat.TeamCandidate(
        team=team,
        opponent=opponent,
        is_home=is_home,
        win_probability=win_probability,
        future_value=future_value,
        spread_line=spread_line,
    )


def test_future_value_penalty_clips_negative_to_zero():
    assert strat.future_value_penalty(-0.4) == 0.0
    assert strat.future_value_penalty(0.0) == 0.0


def test_future_value_penalty_scales_positive_and_caps_at_one():
    assert strat.future_value_penalty(0.3) == pytest.approx(0.3)
    assert strat.future_value_penalty(2.0) == 1.0  # capped


def test_team_spread_flips_sign_for_away_team():
    home = _candidate("KC", "DEN", True, 0.8, 0.0, spread_line=6.0)
    away = _candidate("DEN", "KC", False, 0.2, 0.0, spread_line=6.0)
    assert home.team_spread == pytest.approx(6.0)
    assert away.team_spread == pytest.approx(-6.0)


def test_team_spread_none_when_spread_unavailable():
    c = _candidate("KC", "DEN", True, 0.8, 0.0, spread_line=None)
    assert c.team_spread is None


def test_rank_picks_orders_by_penalized_score():
    candidates = [
        _candidate("KC", "DEN", True, 0.90, future_value=0.5),   # score 0.90*0.5=0.45
        _candidate("SF", "SEA", True, 0.70, future_value=0.0),   # score 0.70*1.0=0.70
        _candidate("BUF", "NYJ", True, 0.60, future_value=-0.2),  # score 0.60*1.0=0.60
    ]
    ranked = strat.rank_picks(candidates)
    assert [p.team for p in ranked] == ["SF", "BUF", "KC"]
    assert ranked[0].score == pytest.approx(0.70)
    assert ranked[2].future_value_penalty == pytest.approx(0.5)


def test_rank_picks_rejects_empty_list():
    with pytest.raises(ValueError):
        strat.rank_picks([])


def test_build_reasoning_mentions_win_prob_spread_and_runner_up():
    ranked = strat.rank_picks(
        [
            _candidate("KC", "DEN", True, 0.85, future_value=-0.1, spread_line=9.5),
            _candidate("SF", "SEA", True, 0.75, future_value=0.0, spread_line=5.5),
        ]
    )
    reasoning = strat.build_reasoning(ranked[0], ranked[1])
    assert "KC" in reasoning
    assert "85.0%" in reasoning
    assert "favored by 9.5" in reasoning
    assert "SF" in reasoning  # runner-up comparison


def test_build_reasoning_handles_sole_candidate():
    ranked = strat.rank_picks([_candidate("KC", "DEN", True, 0.85, future_value=0.0, spread_line=9.5)])
    reasoning = strat.build_reasoning(ranked[0], None)
    assert "only available team" in reasoning


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
            # Future weeks so future_value has something to look at.
            {
                "week": 6,
                "home_team": "KC",
                "away_team": "LAC",
                "home_moneyline": -110,
                "away_moneyline": -110,
                "spread_line": 0.0,
            },
            {
                "week": 6,
                "home_team": "SF",
                "away_team": "ARI",
                "home_moneyline": -900,
                "away_moneyline": 650,
                "spread_line": 15.0,
            },
        ]
    )


def test_build_candidates_excludes_used_teams():
    schedule = _week_schedule()
    candidates = strat.build_candidates(2026, 5, used_teams={"DEN"}, schedule=schedule)
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
    candidates = strat.build_candidates(2026, 5, used_teams=set(), schedule=schedule)
    teams = {c.team for c in candidates}
    assert "MIA" not in teams
    assert "NYJ" not in teams


def test_recommend_pick_end_to_end_with_injected_schedule_and_used_teams():
    schedule = _week_schedule()
    rec = strat.recommend_pick(2026, 5, used_teams=set(), schedule=schedule)
    assert rec.entry == "A"
    assert rec.week == 5
    assert rec.team in {"KC", "SF", "SEA", "DEN"}
    assert 0.0 <= rec.win_probability <= 1.0
    assert rec.team in rec.reasoning
    assert len(rec.ranked_picks) == 4


def test_recommend_pick_raises_when_no_games_that_week():
    schedule = _week_schedule()
    with pytest.raises(ValueError):
        strat.recommend_pick(2026, 99, used_teams=set(), schedule=schedule)


def test_recommend_pick_raises_when_all_teams_used():
    schedule = _week_schedule()[:2]  # week 5 only, KC/DEN and SF/SEA
    with pytest.raises(ValueError):
        strat.recommend_pick(2026, 5, used_teams={"KC", "DEN", "SF", "SEA"}, schedule=schedule)
