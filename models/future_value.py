"""Future value model.

Scores how much better a team's best *remaining* matchup is compared to
using that team's pick in the current week, so a survivor-pool optimizer
can decide when it's worth holding a strong team back rather than
spending them immediately.

The score is a decaying lookahead: nearby future weeks (the next
4-6 weeks) are weighted close to full value, while matchups further out
are discounted more heavily, since spread/moneyline data that far out is
sparser and a team's outlook is more likely to change (injuries,
schedule strength, form).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from data import nflverse_client
from models import win_prob as wp

# Matchups more than this many weeks past "now" are ignored entirely.
DEFAULT_LOOKAHEAD_WEEKS = 6

# Per-week multiplicative decay applied to a future matchup's win
# probability. weeks_out=1 (next week) keeps full weight (decay**0 == 1);
# weeks_out=6 is discounted to decay_rate**5.
DEFAULT_DECAY_RATE = 0.80


@dataclass(frozen=True)
class WeekOpportunity:
    """A single future week's matchup opportunity for a team."""

    week: int
    win_probability: float
    weight: float

    @property
    def weighted_value(self) -> float:
        return self.win_probability * self.weight


@dataclass(frozen=True)
class FutureValue:
    """Result of a future-value computation for one team as of one week."""

    team: str
    current_week: int
    current_week_probability: Optional[float]
    best_future_week: Optional[int]
    best_future_probability: Optional[float]
    best_future_weighted_value: Optional[float]
    future_value: float
    opportunities: Sequence[WeekOpportunity]


def _safe_win_probability(
    row: pd.Series, team: str, spread_model: Optional[wp.SpreadModel]
) -> Optional[float]:
    """`win_prob.get_win_probability`, returning None instead of raising when
    the game has neither moneylines nor a spread_line yet (too far out for
    odds to be posted) -- treated the same as a bye: nothing to evaluate.
    """
    try:
        return wp.get_win_probability(row, team, spread_model=spread_model)
    except ValueError:
        return None


def decay_weight(weeks_out: int, decay_rate: float = DEFAULT_DECAY_RATE) -> float:
    """Weight for a matchup `weeks_out` weeks from the current week (1 = next week)."""
    if weeks_out < 1:
        raise ValueError("weeks_out must be >= 1")
    return decay_rate ** (weeks_out - 1)


def load_remaining_schedule(
    team: str,
    season: int,
    current_week: int,
    schedule: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return `team`'s games from `current_week` through the end of `season`.

    Args:
        schedule: Optional pre-loaded full-season schedule (as returned by
            `data.nflverse_client.load_games(season=season)`), to avoid
            re-downloading when scoring many teams/weeks. Loaded
            automatically if omitted.
    """
    if schedule is None:
        schedule = nflverse_client.load_games(season=season)

    team_games = schedule[
        (schedule["week"] >= current_week)
        & ((schedule["home_team"] == team) | (schedule["away_team"] == team))
    ]
    return team_games.sort_values("week").reset_index(drop=True)


def compute_future_value(
    team: str,
    remaining_schedule: pd.DataFrame,
    current_week: int,
    lookahead_weeks: int = DEFAULT_LOOKAHEAD_WEEKS,
    decay_rate: float = DEFAULT_DECAY_RATE,
    spread_model: Optional[wp.SpreadModel] = None,
) -> FutureValue:
    """Score how much better `team`'s best remaining matchup is vs. using them now.

    A positive `future_value` means a materially better matchup is coming up
    (even after discounting for how far out it is) than what's on the board
    this week -- a signal to hold the team back. A value at or below zero
    means using the team now is at least as good as waiting.

    Args:
        team: Team abbreviation, e.g. "KC".
        remaining_schedule: `team`'s remaining games for the season, e.g.
            from `load_remaining_schedule()`. Rows for other teams are
            ignored; only weeks >= `current_week` are considered.
        current_week: The week being evaluated as "now". If `team` has a
            bye that week, `current_week_probability` is None and the
            comparison baseline is 0 (there's no pick being given up).
        lookahead_weeks: How many weeks past `current_week` to consider.
            Matchups beyond this are ignored, not just discounted.
        decay_rate: Per-week decay applied to future matchups (0 < rate <= 1).
            Lower values concentrate value on the next week or two; values
            closer to 1 keep weight spread across the full lookahead window.
        spread_model: Optional pre-fit `win_prob.SpreadModel` to reuse
            across many calls instead of implicitly fitting/caching one
            per call inside `win_prob.get_win_probability`.
    """
    team_games = remaining_schedule[
        (remaining_schedule["home_team"] == team) | (remaining_schedule["away_team"] == team)
    ]

    current_rows = team_games[team_games["week"] == current_week]
    current_prob = (
        _safe_win_probability(current_rows.iloc[0], team, spread_model)
        if not current_rows.empty
        else None
    )

    last_week = current_week + lookahead_weeks
    future_games = team_games[
        (team_games["week"] > current_week) & (team_games["week"] <= last_week)
    ].sort_values("week")

    opportunities = []
    for _, row in future_games.iterrows():
        win_probability = _safe_win_probability(row, team, spread_model)
        if win_probability is None:
            continue  # no odds posted yet for that far out -- can't evaluate it
        opportunities.append(
            WeekOpportunity(
                week=int(row["week"]),
                win_probability=win_probability,
                weight=decay_weight(int(row["week"]) - current_week, decay_rate),
            )
        )

    if opportunities:
        best = max(opportunities, key=lambda o: o.weighted_value)
        best_future_week = best.week
        best_future_probability = best.win_probability
        best_future_weighted_value = best.weighted_value
    else:
        best_future_week = None
        best_future_probability = None
        best_future_weighted_value = None

    baseline = current_prob if current_prob is not None else 0.0
    future_value = (best_future_weighted_value or 0.0) - baseline

    return FutureValue(
        team=team,
        current_week=current_week,
        current_week_probability=current_prob,
        best_future_week=best_future_week,
        best_future_probability=best_future_probability,
        best_future_weighted_value=best_future_weighted_value,
        future_value=future_value,
        opportunities=opportunities,
    )


def get_future_value(
    team: str,
    remaining_schedule: pd.DataFrame,
    current_week: int,
    lookahead_weeks: int = DEFAULT_LOOKAHEAD_WEEKS,
    decay_rate: float = DEFAULT_DECAY_RATE,
    spread_model: Optional[wp.SpreadModel] = None,
) -> float:
    """Convenience wrapper returning just the scalar `future_value` score."""
    return compute_future_value(
        team,
        remaining_schedule,
        current_week,
        lookahead_weeks=lookahead_weeks,
        decay_rate=decay_rate,
        spread_model=spread_model,
    ).future_value
