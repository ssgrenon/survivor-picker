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


def _elo_games_for_week_schedule():
    # KC/DEN game_id below matches nflverse's default "None" game_id column
    # absence in _week_schedule's rows -- so build_candidates constructs the
    # game_row without a game_id, meaning the elo lookup always misses and
    # falls back to market. To actually exercise a match, give the schedule
    # rows an explicit game_id and mirror it here.
    return pd.DataFrame(
        [
            {"game_id": "2026_05_DEN_KC", "season": 2026, "week": 5, "team": "KC", "opponent": "DEN",
             "is_home": True, "elo_win_probability": 0.20},
            {"game_id": "2026_05_DEN_KC", "season": 2026, "week": 5, "team": "DEN", "opponent": "KC",
             "is_home": False, "elo_win_probability": 0.80},
        ]
    )


def _week_schedule_with_game_ids():
    schedule = _week_schedule()
    schedule["game_id"] = ["2026_05_DEN_KC", "2026_05_SEA_SF"]
    return schedule


def test_build_candidates_threads_market_weight_and_elo_games():
    schedule = _week_schedule_with_game_ids()
    elo_games = _elo_games_for_week_schedule()

    market_only = hedge.build_candidates(2026, 5, used_teams=set(), schedule=schedule)
    kc_market = next(c for c in market_only if c.team == "KC").win_probability

    elo_only = hedge.build_candidates(
        2026, 5, used_teams=set(), schedule=schedule, market_weight=0.0, elo_games=elo_games
    )
    kc_elo = next(c for c in elo_only if c.team == "KC").win_probability
    assert kc_elo == pytest.approx(0.20)
    assert kc_elo != pytest.approx(kc_market)

    blended = hedge.build_candidates(
        2026, 5, used_teams=set(), schedule=schedule, market_weight=0.5, elo_games=elo_games
    )
    kc_blended = next(c for c in blended if c.team == "KC").win_probability
    assert kc_blended == pytest.approx(0.5 * kc_market + 0.5 * kc_elo)


def test_build_candidates_populates_divergence_independent_of_market_weight():
    schedule = _week_schedule_with_game_ids()
    elo_games = _elo_games_for_week_schedule()

    # KC has no divergence info without elo_games.
    market_only = hedge.build_candidates(2026, 5, used_teams=set(), schedule=schedule)
    assert next(c for c in market_only if c.team == "KC").divergence is None

    # With elo_games supplied, divergence is populated the same regardless
    # of market_weight -- it's a display signal, not a scoring input.
    divergences = set()
    for weight in (1.0, 0.5, 0.0):
        candidates = hedge.build_candidates(
            2026, 5, used_teams=set(), schedule=schedule, market_weight=weight, elo_games=elo_games
        )
        kc = next(c for c in candidates if c.team == "KC")
        assert kc.divergence is not None
        divergences.add(round(kc.divergence, 9))
    assert len(divergences) == 1


def test_recommend_pick_carries_divergence_through():
    # market_weight=1.0 keeps KC's win probability at its market value (so it
    # still clears the floor) while still exercising divergence, which is
    # computed independently of market_weight whenever elo_games is given.
    schedule = _week_schedule_with_game_ids()
    elo_games = _elo_games_for_week_schedule()
    rec = hedge.recommend_pick(2026, 5, used_teams=set(), schedule=schedule, market_weight=1.0, elo_games=elo_games)
    assert rec.team == "KC"
    assert rec.divergence is not None
    assert rec.divergence == next(c for c in rec.ranked_picks if c.team == "KC").divergence
