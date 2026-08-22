import pandas as pd
import pytest

from models import dp_optimizer as dpo

SEASON = 2099


def _pick(team, week, prob, opponent="OPP", is_home=True, spread=None):
    return dpo.WeekPick(
        week=week, team=team, opponent=opponent, is_home=is_home, win_probability=prob, spread_line=spread
    )


# ---------------------------------------------------------------------------
# _solve_dp: the core algorithm, tested with hand-picked probabilities.
# ---------------------------------------------------------------------------


def test_solve_dp_holds_a_team_for_a_better_future_matchup():
    # X is the single best option THIS week (0.80 > Y's 0.60), but X has an
    # even better matchup next week (0.95) while next week's alternative (Z)
    # is mediocre (0.55). A naive "always take this week's best" policy would
    # spend X now (0.80 * 0.55 = 0.440); the optimal sequence holds X for
    # next week instead (0.60 * 0.95 = 0.570).
    weekly_options = {
        1: [_pick("X", 1, 0.80), _pick("Y", 1, 0.60)],
        2: [_pick("X", 2, 0.95), _pick("Z", 2, 0.55)],
    }
    survival_probability, path = dpo._solve_dp(weekly_options)

    assert [p.team for p in path] == ["Y", "X"]
    assert survival_probability == pytest.approx(0.60 * 0.95)
    # confirm this genuinely beats the naive "always take this week's best" sequence
    naive_probability = 0.80 * 0.55
    assert survival_probability > naive_probability


def test_solve_dp_single_week_picks_the_single_best():
    weekly_options = {1: [_pick("X", 1, 0.80), _pick("Y", 1, 0.60)]}
    survival_probability, path = dpo._solve_dp(weekly_options)
    assert [p.team for p in path] == ["X"]
    assert survival_probability == pytest.approx(0.80)


def test_solve_dp_picks_greedy_when_no_holding_tension():
    # Y is best both weeks and nothing else is remotely competitive -- no
    # reason to ever prefer holding it, so the optimal sequence just takes
    # the best team each week (using distinct teams).
    weekly_options = {
        1: [_pick("Y", 1, 0.90), _pick("X", 1, 0.50)],
        2: [_pick("Z", 2, 0.85), _pick("X", 2, 0.40)],
    }
    survival_probability, path = dpo._solve_dp(weekly_options)
    assert [p.team for p in path] == ["Y", "Z"]
    assert survival_probability == pytest.approx(0.90 * 0.85)


def test_solve_dp_raises_when_a_later_week_has_no_viable_candidate():
    # Team X is the *only* candidate in both weeks -- week 1 must spend it,
    # leaving week 2 with nothing to pick from.
    weekly_options = {
        1: [_pick("X", 1, 0.80)],
        2: [_pick("X", 2, 0.95)],
    }
    with pytest.raises(ValueError):
        dpo._solve_dp(weekly_options)


# ---------------------------------------------------------------------------
# build_candidate_universe: pruning.
# ---------------------------------------------------------------------------


def test_build_candidate_universe_respects_per_week_top_k():
    weekly_options = {
        1: [_pick(f"T{i}", 1, 1.0 - i * 0.01) for i in range(10)],
    }
    universe = dpo.build_candidate_universe(weekly_options, per_week_top_k=3, max_candidate_teams=20)
    assert len(universe[1]) == 3
    assert [p.team for p in universe[1]] == ["T0", "T1", "T2"]


def test_build_candidate_universe_caps_total_teams_by_best_probability():
    # Each week has two candidates, so trimming the globally weakest ones
    # never has to fight the "keep >= 1 per week" invariant.
    weekly_options = {
        1: [_pick("A", 1, 0.95), _pick("A2", 1, 0.50)],
        2: [_pick("B", 2, 0.90), _pick("B2", 2, 0.50)],
        3: [_pick("C", 3, 0.60), _pick("C2", 3, 0.40)],
    }
    universe = dpo.build_candidate_universe(weekly_options, per_week_top_k=5, max_candidate_teams=3)
    kept = {p.team for options in universe.values() for p in options}
    assert kept == {"A", "B", "C"}  # the three highest best-any-week probabilities


def test_build_candidate_universe_never_leaves_a_week_with_zero_candidates():
    # Week 3's only team (C, 0.55) is globally the weakest, but it must be
    # kept anyway since it's week 3's sole representative.
    weekly_options = {
        1: [_pick("A", 1, 0.95), _pick("B1", 1, 0.90)],
        2: [_pick("B", 2, 0.93), _pick("A2", 2, 0.88)],
        3: [_pick("C", 3, 0.55)],
    }
    universe = dpo.build_candidate_universe(weekly_options, per_week_top_k=5, max_candidate_teams=2)
    assert universe[3] != []
    assert universe[3][0].team == "C"


def test_build_candidate_universe_returns_everything_under_the_cap():
    weekly_options = {1: [_pick("A", 1, 0.9)], 2: [_pick("B", 2, 0.8)]}
    universe = dpo.build_candidate_universe(weekly_options, per_week_top_k=5, max_candidate_teams=10)
    assert universe == {1: [_pick("A", 1, 0.9)], 2: [_pick("B", 2, 0.8)]}


# ---------------------------------------------------------------------------
# optimize_pick_sequence: end-to-end through a synthetic schedule.
# ---------------------------------------------------------------------------


def _row(week, home_team, away_team, home_ml, away_ml, spread=0.0):
    return {
        "season": SEASON,
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "spread_line": spread,
    }


def _holding_schedule():
    # X is a big home favorite this week (~0.80) and an even bigger one next
    # week (~0.95); Y is a modest favorite this week (~0.60); Z is a modest
    # favorite next week (~0.55). Mirrors the hand-picked _solve_dp scenario
    # above through the real win_prob pipeline.
    return pd.DataFrame(
        [
            _row(1, "X", "AAA", -400, 320, spread=9.0),
            _row(1, "Y", "BBB", -150, 130, spread=3.0),
            _row(2, "X", "CCC", -1900, 1200, spread=17.0),
            _row(2, "Z", "DDD", -125, 105, spread=1.5),
        ]
    )


def test_optimize_pick_sequence_holds_team_end_to_end():
    schedule = _holding_schedule()
    result = dpo.optimize_pick_sequence(SEASON, 1, used_teams=set(), schedule=schedule, lookahead_weeks=2)

    assert result.recommended_pick == "Y"
    assert [p.team for p in result.path] == ["Y", "X"]
    assert result.path[0].week == 1
    assert result.path[1].week == 2


def test_optimize_pick_sequence_excludes_used_teams():
    schedule = _holding_schedule()
    result = dpo.optimize_pick_sequence(SEASON, 1, used_teams={"Y"}, schedule=schedule, lookahead_weeks=2)
    assert "Y" not in [p.team for p in result.path]


def test_optimize_pick_sequence_truncates_at_season_end():
    schedule = _holding_schedule()  # only weeks 1-2 exist
    result = dpo.optimize_pick_sequence(SEASON, 1, used_teams=set(), schedule=schedule, lookahead_weeks=7)
    assert max(p.week for p in result.path) == 2


def test_optimize_pick_sequence_raises_when_nothing_available():
    schedule = _holding_schedule()
    all_teams = {"X", "Y", "Z", "AAA", "BBB", "CCC", "DDD"}
    with pytest.raises(ValueError):
        dpo.optimize_pick_sequence(
            SEASON, 1, used_teams=all_teams, schedule=schedule, lookahead_weeks=2
        )


def test_optimize_pick_sequence_threads_market_weight_to_change_the_plan():
    # With pure market probabilities, week 1's optimal pick is Y (holding X
    # for its stronger week-2 matchup, see test_optimize_pick_sequence_holds_
    # team_end_to_end above). If nfelo instead rates X as only a coinflip
    # both weeks and Y as a near-lock this week, a 0%-market/100%-elo blend
    # should flip week 1's pick to Y for a *different* reason: because X is
    # no longer worth holding at all under Elo's numbers.
    schedule = _holding_schedule()
    schedule["game_id"] = ["2099_01_AAA_X", "2099_01_BBB_Y", "2099_02_CCC_X", "2099_02_DDD_Z"]
    elo_games = pd.DataFrame(
        [
            {"game_id": "2099_01_AAA_X", "season": SEASON, "week": 1, "team": "X", "opponent": "AAA",
             "is_home": True, "elo_win_probability": 0.50},
            {"game_id": "2099_01_AAA_X", "season": SEASON, "week": 1, "team": "AAA", "opponent": "X",
             "is_home": False, "elo_win_probability": 0.50},
            {"game_id": "2099_01_BBB_Y", "season": SEASON, "week": 1, "team": "Y", "opponent": "BBB",
             "is_home": True, "elo_win_probability": 0.97},
            {"game_id": "2099_01_BBB_Y", "season": SEASON, "week": 1, "team": "BBB", "opponent": "Y",
             "is_home": False, "elo_win_probability": 0.03},
            {"game_id": "2099_02_CCC_X", "season": SEASON, "week": 2, "team": "X", "opponent": "CCC",
             "is_home": True, "elo_win_probability": 0.50},
            {"game_id": "2099_02_CCC_X", "season": SEASON, "week": 2, "team": "CCC", "opponent": "X",
             "is_home": False, "elo_win_probability": 0.50},
            {"game_id": "2099_02_DDD_Z", "season": SEASON, "week": 2, "team": "Z", "opponent": "DDD",
             "is_home": True, "elo_win_probability": 0.55},
            {"game_id": "2099_02_DDD_Z", "season": SEASON, "week": 2, "team": "DDD", "opponent": "Z",
             "is_home": False, "elo_win_probability": 0.45},
        ]
    )

    result = dpo.optimize_pick_sequence(
        SEASON, 1, used_teams=set(), schedule=schedule, lookahead_weeks=2,
        market_weight=0.0, elo_games=elo_games,
    )
    assert result.recommended_pick == "Y"
    assert [p.team for p in result.path] == ["Y", "Z"]
