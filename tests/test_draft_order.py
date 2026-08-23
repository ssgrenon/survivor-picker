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
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule, market_weight=1.0)
    teams = [p.team for p in picks]
    assert len(teams) == len(set(teams)) == 4


def test_draft_picks_follow_priority_order():
    # Entry A drafts all of its picks first, then Entry B drafts all of its.
    schedule = _basic_week_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule, market_weight=1.0)
    assert [p.entry for p in picks] == ["A", "A", "B", "B"]
    assert [p.pick_number for p in picks] == [1, 2, 3, 4]
    assert [p.round for p in picks] == [1, 2, 1, 2]


def test_draft_picks_respects_rounds_parameter():
    schedule = _basic_week_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=1, schedule=schedule, market_weight=1.0)
    assert len(picks) == 2
    assert [p.entry for p in picks] == ["A", "B"]

    picks3 = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=3, schedule=schedule, market_weight=1.0)
    assert len(picks3) == 6
    assert [p.entry for p in picks3] == ["A", "A", "A", "B", "B", "B"]


def test_draft_picks_respects_existing_used_teams_per_entry():
    schedule = _basic_week_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a={"KC"}, used_teams_b=set(), rounds=1, schedule=schedule, market_weight=1.0)
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
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule, market_weight=1.0)
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
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule, market_weight=1.0)
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
        SEASON, 1, used_teams_a=shared_used, used_teams_b=shared_used, rounds=2, schedule=schedule,
        market_weight=1.0,
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
            market_weight=1.0,
        )


def test_draft_picks_threads_market_weight_and_changes_the_pick():
    # Under pure market probability, KC (~90%) is Entry A's pick #1 (see
    # _basic_week_schedule). If nfelo instead rates KC as a coinflip and BUF
    # (a modest ~71% market favorite) as a near-lock, a 0%-market/100%-elo
    # draft should promote BUF to pick #1 instead -- proving market_weight
    # actually reaches the DP optimizer draft_order delegates to for Entry A.
    schedule = _basic_week_schedule()
    schedule["game_id"] = [
        "2099_01_DEN_KC", "2099_01_SEA_SF", "2099_01_NYJ_BUF", "2099_01_NYG_DAL", "2099_01_NE_MIA",
    ]
    elo_rows = []
    for game_id, home, away, home_prob in (
        ("2099_01_DEN_KC", "KC", "DEN", 0.50),
        ("2099_01_SEA_SF", "SF", "SEA", 0.80),
        ("2099_01_NYJ_BUF", "BUF", "NYJ", 0.99),
        ("2099_01_NYG_DAL", "DAL", "NYG", 0.64),
        ("2099_01_NE_MIA", "MIA", "NE", 0.60),
    ):
        elo_rows.append({"game_id": game_id, "season": SEASON, "week": 1, "team": home, "opponent": away,
                          "is_home": True, "elo_win_probability": home_prob})
        elo_rows.append({"game_id": game_id, "season": SEASON, "week": 1, "team": away, "opponent": home,
                          "is_home": False, "elo_win_probability": 1.0 - home_prob})
    elo_games = pd.DataFrame(elo_rows)

    market_picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=1, schedule=schedule, market_weight=1.0)
    assert market_picks[0].team == "KC"

    blended_picks = do.draft_picks(
        SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=1, schedule=schedule,
        market_weight=0.0, elo_games=elo_games,
    )
    assert blended_picks[0].team == "BUF"


def _home_override_schedule():
    # Pick #1 candidates: AWAYFAV (away, ~90%) beats everything else, so
    # both entries' round-1 pick lands on an away team.
    # HOMEFAV (home, ~65%) is the best *home* candidate overall, but
    # AWAYFAV2 (away, ~80%) out-ranks it on raw win probability -- so a
    # normal (non-overridden) round-2 pick would take AWAYFAV2, while the
    # home-game override must instead force HOMEFAV / WEAK3.
    return pd.DataFrame(
        [
            _row(1, "WEAK1", "AWAYFAV", 650, -900),  # AWAYFAV ~90% (away)
            _row(1, "HOMEFAV", "WEAK2", -190, 160),  # HOMEFAV ~65% (home)
            _row(1, "WEAK3", "AWAYFAV2", 210, -250),  # AWAYFAV2 ~71% (away)
        ]
    )


def test_pick_2_forces_best_home_game_when_pick_1_is_away():
    schedule = _home_override_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule, market_weight=1.0)
    pick1, pick2 = picks[0], picks[1]

    assert pick1.team == "AWAYFAV" and pick1.is_home is False
    assert pick2.team == "HOMEFAV" and pick2.is_home is True
    assert "Home-game override" in pick2.reasoning


def test_pick_4_forces_best_home_game_when_pick_3_is_away():
    schedule = _home_override_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule, market_weight=1.0)
    pick3, pick4 = picks[2], picks[3]

    # Entry A already drafted AWAYFAV and HOMEFAV, so Entry B's best
    # remaining option is AWAYFAV2 (away, ~71%), clearing its 65% floor.
    assert pick3.team == "AWAYFAV2" and pick3.is_home is False
    assert pick4.team == "WEAK3" and pick4.is_home is True
    assert "Home-game override" in pick4.reasoning


def test_no_home_override_when_pick_1_is_already_home():
    # _basic_week_schedule's pick #1 (KC, ~90% home favorite) is already a
    # home game, so pick #2 should follow the normal (unconstrained)
    # recommendation logic -- no override language in its reasoning.
    schedule = _basic_week_schedule()
    picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=2, schedule=schedule, market_weight=1.0)
    assert picks[0].is_home is True
    assert "Home-game override" not in picks[1].reasoning


def test_home_override_falls_back_to_normal_pick_when_no_home_game_remains():
    schedule = _home_override_schedule()
    # All three home teams from the schedule are already used, so once
    # AWAYFAV is drafted as pick #1 there is no home candidate left for
    # pick #2 -- it must fall back to the normal recommendation instead of
    # raising.
    picks = do.draft_picks(
        SEASON, 1, used_teams_a={"WEAK1", "HOMEFAV", "WEAK3"}, used_teams_b=set(), rounds=2, schedule=schedule,
        market_weight=1.0,
    )
    pick1, pick2 = picks[0], picks[1]

    assert pick1.team == "AWAYFAV" and pick1.is_home is False
    assert pick2.team == "AWAYFAV2"  # only team left for A once home teams are excluded
    assert "Home-game override" not in pick2.reasoning


def test_draft_picks_threads_team_bias_games():
    schedule = _basic_week_schedule()
    # KC (~90% market favorite) has a consistent losing history at home in
    # this synthetic bias table -- strong enough to knock it off pick #1.
    bias_games = pd.DataFrame(
        [
            {"season": season, "week": 1, "home_team": "KC", "away_team": "OPP",
             "home_score": 10, "away_score": 30, "home_moneyline": -900, "away_moneyline": 650,
             "spread_line": 15.0}
            for season in (2018, 2019, 2020, 2021, 2022)
        ]
    )

    market_picks = do.draft_picks(SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=1, schedule=schedule, market_weight=1.0)
    assert market_picks[0].team == "KC"
    assert market_picks[0].team_bias_adjustment == 0.0

    biased_picks = do.draft_picks(
        SEASON, 1, used_teams_a=set(), used_teams_b=set(), rounds=1, schedule=schedule,
        team_bias_games=bias_games, market_weight=1.0,
    )
    kc_pick = next((p for p in biased_picks if p.team == "KC"), None)
    if kc_pick is not None:
        assert kc_pick.team_bias_adjustment < 0
    else:
        assert biased_picks[0].team != "KC"  # bias was strong enough to change pick #1
