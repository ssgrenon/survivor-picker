import pandas as pd
import pytest

from strategy import draft_order as do

SEASON = 2099


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


def _basic_week_schedule():
    return pd.DataFrame(
        [
            _row(1, "KC", "DEN", -900, 650, spread=15.0),  # KC ~90%
            _row(1, "SF", "SEA", -400, 320, spread=9.0),  # SF ~80%
            _row(1, "BUF", "NYJ", -250, 210, spread=6.0),  # BUF ~71%
            _row(1, "DAL", "NYG", -180, 150, spread=4.0),  # DAL ~64%
            _row(1, "MIA", "NE", -150, 130, spread=3.0),  # MIA ~60%
        ]
    )


def test_draft_picks_are_all_distinct_teams():
    schedule = _basic_week_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule)
    teams = [p.team for p in picks]
    assert len(teams) == len(set(teams)) == 4


def test_draft_picks_follow_priority_order():
    # Entry A drafts all of its picks first, then Entry B drafts all of its.
    schedule = _basic_week_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule)
    assert [p.entry for p in picks] == ["A", "A", "B", "B"]
    assert [p.pick_number for p in picks] == [1, 2, 3, 4]
    assert [p.round for p in picks] == [1, 2, 1, 2]


def test_draft_picks_respects_rounds_parameter():
    schedule = _basic_week_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=1, schedule=schedule)
    assert len(picks) == 2
    assert [p.entry for p in picks] == ["A", "B"]

    picks3 = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=3, schedule=schedule)
    assert len(picks3) == 6
    assert [p.entry for p in picks3] == ["A", "A", "A", "B", "B", "B"]


def test_draft_picks_respects_existing_used_teams_per_entry():
    schedule = _basic_week_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a={"KC"}, used_teams_b=set(), rounds=1, schedule=schedule)
    assert picks[0].team != "KC"


def _holding_tension_schedule():
    # X is the single best team this week (~0.80) but has an even better
    # matchup next week (~0.95); Y is a modest favorite this week (~0.60).
    # Only these 4 teams (X, AAA, Y, BBB) play in week 1 at all.
    return pd.DataFrame(
        [
            _row(1, "X", "AAA", -400, 320, spread=9.0),
            _row(1, "Y", "BBB", -150, 130, spread=3.0),
            _row(2, "X", "CCC", -1900, 1200, spread=17.0),
            _row(2, "Z", "DDD", -125, 105, spread=1.5),
        ]
    )


def test_second_pick_reruns_recommendation_excluding_the_first():
    # Unconstrained, Entry A holds X and picks Y for week 1 (see
    # test_dp_optimizer.py's equivalent scenario). For pick #2, Entry A's
    # DP re-runs with Y excluded: AAA (~0.20) is too weak a week-1
    # substitute to be worth holding X for its week-2 game anymore
    # (0.80*0.55 beats 0.20*0.95), so pick #2 switches to X instead of
    # continuing to hold it -- proving it's a genuine re-optimization, not
    # a stale index into A's original (unconstrained) list.
    schedule = _holding_tension_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule)
    teams = [p.team for p in picks]

    assert teams[0] == "Y"  # Entry A's unconstrained pick #1: holds X, takes Y
    assert teams[1] == "X"  # Entry A's pick #2: no longer worth holding X once Y is gone
    assert len(set(teams)) == 4


def test_falls_back_when_entry_b_floor_excludes_everything_remaining():
    # Entry A's two picks (Y, then X) claim both of week 1's floor-clearing
    # options. Entry B is left with only AAA (~0.20) and BBB (~0.40),
    # neither of which clears its 65% floor -- both of B's picks should
    # still resolve via the documented fallback rather than raising.
    schedule = _holding_tension_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule)
    third, fourth = picks[2], picks[3]

    assert third.entry == "B" and third.team == "BBB" and "Fallback" in third.reasoning
    assert fourth.entry == "B" and fourth.team == "AAA" and "Fallback" in fourth.reasoning


def test_draft_picks_works_when_both_entries_share_the_same_used_teams():
    # Once one entry is eliminated, the UI has both algorithms draft from
    # the surviving entry's used-teams history for *both* arguments -- this
    # must still yield four distinct picks, not error or double-count.
    schedule = _basic_week_schedule()
    shared_used = {"MIA"}
    picks = do.draft_picks(
        SEASON, 1, used_teams_a=shared_used, used_teams_b=shared_used, rounds=2, schedule=schedule
    )
    teams = [p.team for p in picks]
    assert "MIA" not in teams
    assert len(teams) == len(set(teams)) == 4
    assert [p.entry for p in picks] == ["A", "A", "B", "B"]


def test_draft_picks_raises_when_nothing_left_to_draft():
    schedule = _basic_week_schedule()  # 5 games -> 10 total teams
    with pytest.raises(ValueError):
        do.draft_picks(
            SEASON,
            1,
            used_teams_a=set(),
            used_teams_b=set(),
            rounds=6,  # 12 picks requested, only 10 teams exist
            schedule=schedule,
        )
