import pandas as pd
import pytest

from models import future_value as fv
from models import win_prob as wp


def _row(week, home_team, away_team, home_ml, away_ml):
    return {
        "week": week,
        "home_team": home_team,
        "away_team": away_team,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
        "spread_line": None,
    }


def _kc_schedule():
    # KC's remaining games, weeks 1-8. Week 2 is a bye (no row).
    return pd.DataFrame(
        [
            _row(1, "KC", "LAC", 120, -140),   # KC home, slight underdog
            _row(3, "KC", "DEN", -300, 250),   # KC home, big favorite -> best matchup
            _row(4, "KC", "LV", -150, 130),    # KC home, moderate favorite
            _row(5, "BUF", "KC", 110, -130),   # KC away, moderate favorite
            _row(7, "KC", "CIN", -110, -110),  # outside default 6-week lookahead from week1
            _row(8, "MIA", "KC", 100, -120),
        ]
    )


def test_decay_weight_full_at_next_week_and_decreasing():
    w1 = fv.decay_weight(1, decay_rate=0.8)
    w2 = fv.decay_weight(2, decay_rate=0.8)
    w6 = fv.decay_weight(6, decay_rate=0.8)
    assert w1 == pytest.approx(1.0)
    assert w1 > w2 > w6


def test_decay_weight_rejects_non_positive_weeks_out():
    with pytest.raises(ValueError):
        fv.decay_weight(0)


def test_compute_future_value_identifies_best_future_matchup():
    schedule = _kc_schedule()
    result = fv.compute_future_value("KC", schedule, current_week=1, lookahead_weeks=6, decay_rate=0.8)

    # Week 3's big-favorite matchup should win out even after decay,
    # since it's a much stronger spot than weeks 4/5/7(excluded).
    assert result.best_future_week == 3
    week3_prob = wp.get_win_probability(schedule[schedule["week"] == 3].iloc[0], "KC")
    expected_weight = fv.decay_weight(3 - 1, decay_rate=0.8)
    assert result.best_future_probability == pytest.approx(week3_prob)
    assert result.best_future_weighted_value == pytest.approx(week3_prob * expected_weight)


def test_compute_future_value_respects_lookahead_window():
    schedule = _kc_schedule()
    result = fv.compute_future_value("KC", schedule, current_week=1, lookahead_weeks=6, decay_rate=0.8)
    weeks_considered = {o.week for o in result.opportunities}
    assert weeks_considered == {3, 4, 5, 7}  # week 8 falls outside the weeks 2-7 window
    assert 8 not in weeks_considered


def test_future_value_is_difference_between_best_future_and_now():
    schedule = _kc_schedule()
    result = fv.compute_future_value("KC", schedule, current_week=1, lookahead_weeks=6, decay_rate=0.8)
    assert result.future_value == pytest.approx(
        result.best_future_weighted_value - result.current_week_probability
    )
    # KC's week-1 matchup is mediocre and week 3 is much better even discounted,
    # so future_value should be clearly positive -> hold signal.
    assert result.future_value > 0


def test_bye_week_uses_zero_baseline():
    schedule = _kc_schedule()
    result = fv.compute_future_value("KC", schedule, current_week=2, lookahead_weeks=6, decay_rate=0.8)
    assert result.current_week_probability is None
    assert result.future_value == pytest.approx(result.best_future_weighted_value)


def test_no_future_games_in_window_gives_nonpositive_future_value():
    # Only a current-week game, nothing else in range.
    schedule = pd.DataFrame([_row(10, "KC", "LV", -200, 170)])
    result = fv.compute_future_value("KC", schedule, current_week=10, lookahead_weeks=6, decay_rate=0.8)
    assert result.opportunities == []
    assert result.best_future_week is None
    assert result.future_value <= 0


def test_get_future_value_matches_compute_future_value():
    schedule = _kc_schedule()
    scalar = fv.get_future_value("KC", schedule, current_week=1, lookahead_weeks=6, decay_rate=0.8)
    full = fv.compute_future_value("KC", schedule, current_week=1, lookahead_weeks=6, decay_rate=0.8)
    assert scalar == pytest.approx(full.future_value)
