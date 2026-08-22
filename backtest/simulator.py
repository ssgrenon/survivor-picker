"""Survivor pool backtester.

Simulates a full survivor run for a pluggable pick-selection algorithm:
starting from a given week, each week the algorithm is handed the
season, the current week, the set of already-used teams, and every
available (not-yet-used) team's win probability and matchup info; it
returns the team to pick. The pick's actual result is looked up via
`data.nflverse_client`, and the entry is marked eliminated the first
week its pick loses (ties are eliminating too, by default).

Three ready-made algorithms are provided for comparison:
`highest_win_probability_algorithm` (a naive baseline), and factories
that replicate Entry A's (`make_entry_a_algorithm`) and Entry B's
(`make_entry_b_algorithm`) live strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from data import nflverse_client
from models import future_value
from models import win_prob as wp
from strategy import entry_a_value
from strategy import entry_b_hedge

# An algorithm receives (season, week, used_teams, available_candidates)
# and returns the team abbreviation it picks for that week.
PickAlgorithm = Callable[[int, int, Set[str], Sequence[entry_b_hedge.TeamCandidate]], str]


@dataclass(frozen=True)
class WeekRecord:
    """What happened on one week's pick."""

    week: int
    pick: str
    opponent: str
    is_home: bool
    predicted_win_probability: float
    spread_line: Optional[float]
    actual_result: Optional[float]  # home_score - away_score
    outcome: str  # "WIN" | "LOSS" | "TIE"
    still_alive: bool


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of running one algorithm through a season."""

    season: int
    starting_week: int
    algorithm_name: str
    weeks_survived: int
    eliminated_week: Optional[int]
    survived_full_season: bool
    stop_reason: str  # "eliminated" | "survived_full_season" | "ran_out_of_available_teams" | "hit_unplayed_game"
    records: Sequence[WeekRecord]


def _find_game_row(schedule: pd.DataFrame, week: int, team: str) -> pd.Series:
    rows = schedule[
        (schedule["week"] == week) & ((schedule["home_team"] == team) | (schedule["away_team"] == team))
    ]
    if rows.empty:
        raise ValueError(f"No game found for {team} in week {week}")
    return rows.iloc[0]


def _score_pick(game_row: pd.Series, team: str) -> Tuple[Optional[float], str]:
    """Return (actual_result, outcome) for `team` in `game_row`.

    actual_result is home_score - away_score (nflverse's `result` convention).
    outcome is "UNPLAYED" when the game has no final score yet.
    """
    home_score = game_row["home_score"]
    away_score = game_row["away_score"]
    if pd.isna(home_score) or pd.isna(away_score):
        return None, "UNPLAYED"

    result = float(home_score) - float(away_score)
    if result == 0:
        return result, "TIE"

    home_won = result > 0
    team_is_home = team == game_row["home_team"]
    team_won = home_won == team_is_home
    return result, "WIN" if team_won else "LOSS"


def simulate(
    season: int,
    starting_week: int,
    algorithm: PickAlgorithm,
    algorithm_name: Optional[str] = None,
    schedule: Optional[pd.DataFrame] = None,
    initial_used_teams: Optional[Iterable[str]] = None,
    eliminate_on_tie: bool = True,
    spread_model: Optional[wp.SpreadModel] = None,
) -> BacktestResult:
    """Run `algorithm` through `season` starting at `starting_week` until it's eliminated.

    Args:
        algorithm: Called each week as
            `algorithm(season, week, used_teams, available_candidates)` and
            must return one of the available candidates' team abbreviations.
        algorithm_name: Label for the result; defaults to `algorithm.__name__`.
        schedule: Optional pre-loaded full-season schedule, to avoid
            re-downloading across repeated/compared runs.
        initial_used_teams: Teams to treat as already spent before
            `starting_week` (e.g. picks made earlier in the real season).
        eliminate_on_tie: Whether a tied pick counts as elimination.
    """
    if algorithm_name is None:
        algorithm_name = getattr(algorithm, "__name__", "algorithm")

    if schedule is None:
        schedule = nflverse_client.load_games(season=season)
    season_schedule = schedule[schedule["season"] == season]
    if season_schedule.empty:
        raise ValueError(f"No games found for season {season}")
    max_week = int(season_schedule["week"].max())

    used_teams = set(initial_used_teams or ())
    records: List[WeekRecord] = []
    eliminated_week: Optional[int] = None
    stop_reason = "survived_full_season"

    for week in range(starting_week, max_week + 1):
        try:
            available = entry_b_hedge.build_candidates(
                season, week, used_teams, schedule=season_schedule, spread_model=spread_model
            )
        except ValueError:
            continue  # no games at all this week (schedule gap) -- nothing to pick

        if not available:
            stop_reason = "ran_out_of_available_teams"
            break

        pick = algorithm(season, week, set(used_teams), available)
        candidate = next((c for c in available if c.team == pick), None)
        if candidate is None:
            raise ValueError(
                f"Algorithm {algorithm_name!r} picked {pick!r}, which is not an available "
                f"team for season {season} week {week}"
            )

        game_row = _find_game_row(season_schedule, week, pick)
        actual_result, outcome = _score_pick(game_row, pick)

        if outcome == "UNPLAYED":
            stop_reason = "hit_unplayed_game"
            break

        used_teams.add(pick)
        eliminated = outcome == "LOSS" or (outcome == "TIE" and eliminate_on_tie)

        records.append(
            WeekRecord(
                week=week,
                pick=pick,
                opponent=candidate.opponent,
                is_home=candidate.is_home,
                predicted_win_probability=candidate.win_probability,
                spread_line=candidate.spread_line,
                actual_result=actual_result,
                outcome=outcome,
                still_alive=not eliminated,
            )
        )

        if eliminated:
            eliminated_week = week
            stop_reason = "eliminated"
            break

    weeks_survived = sum(1 for r in records if r.outcome == "WIN")

    return BacktestResult(
        season=season,
        starting_week=starting_week,
        algorithm_name=algorithm_name,
        weeks_survived=weeks_survived,
        eliminated_week=eliminated_week,
        survived_full_season=(stop_reason == "survived_full_season" and bool(records)),
        stop_reason=stop_reason,
        records=records,
    )


def compare_algorithms(
    season: int,
    starting_week: int,
    algorithms: Mapping[str, PickAlgorithm],
    schedule: Optional[pd.DataFrame] = None,
    initial_used_teams: Optional[Iterable[str]] = None,
    eliminate_on_tie: bool = True,
    spread_model: Optional[wp.SpreadModel] = None,
) -> Dict[str, BacktestResult]:
    """Run several algorithms through the same season/week for side-by-side comparison."""
    if schedule is None:
        schedule = nflverse_client.load_games(season=season)

    return {
        name: simulate(
            season,
            starting_week,
            algorithm,
            algorithm_name=name,
            schedule=schedule,
            initial_used_teams=initial_used_teams,
            eliminate_on_tie=eliminate_on_tie,
            spread_model=spread_model,
        )
        for name, algorithm in algorithms.items()
    }


def highest_win_probability_algorithm(
    season: int,
    week: int,
    used_teams: Set[str],
    available: Sequence[entry_b_hedge.TeamCandidate],
) -> str:
    """Naive baseline: always take the highest raw win-probability team available."""
    if not available:
        raise ValueError(f"No available teams for season {season} week {week}")
    return max(available, key=lambda c: c.win_probability).team


def make_entry_a_algorithm(
    schedule: Optional[pd.DataFrame] = None,
    lookahead_weeks: int = future_value.DEFAULT_LOOKAHEAD_WEEKS,
    decay_rate: float = future_value.DEFAULT_DECAY_RATE,
    penalty_weight: float = entry_a_value.DEFAULT_PENALTY_WEIGHT,
    spread_model: Optional[wp.SpreadModel] = None,
) -> PickAlgorithm:
    """Build an algorithm replicating Entry A's win-prob-minus-hold-penalty strategy."""

    def _pick(
        season: int,
        week: int,
        used_teams: Set[str],
        available: Sequence[entry_b_hedge.TeamCandidate],
    ) -> str:
        sched = schedule if schedule is not None else nflverse_client.load_games(season=season)
        candidates = entry_a_value.build_candidates(
            season,
            week,
            used_teams,
            schedule=sched,
            lookahead_weeks=lookahead_weeks,
            decay_rate=decay_rate,
            spread_model=spread_model,
        )
        if not candidates:
            raise ValueError(f"No available teams for season {season} week {week}")
        return entry_a_value.rank_picks(candidates, penalty_weight=penalty_weight)[0].team

    return _pick


def make_entry_b_algorithm(
    min_win_probability: float = entry_b_hedge.DEFAULT_MIN_WIN_PROBABILITY,
) -> PickAlgorithm:
    """Build an algorithm replicating Entry B's floor-then-highest-win-prob strategy."""

    def _pick(
        season: int,
        week: int,
        used_teams: Set[str],
        available: Sequence[entry_b_hedge.TeamCandidate],
    ) -> str:
        eligible = entry_b_hedge.rank_picks(available, min_win_probability=min_win_probability)
        if not eligible:
            raise ValueError(
                f"No available team clears the {min_win_probability:.0%} floor "
                f"for season {season} week {week}"
            )
        return eligible[0].team

    return _pick
