"""N-week dynamic-programming pick-sequence optimizer.

Replaces the old decaying-lookahead "future value" heuristic
(`models.future_value`) with an (approximately) exact optimizer: given
the current used-teams state, it searches for the sequence of distinct
teams across the next `lookahead_weeks` (default 7) that maximizes the
*product* of win probabilities -- i.e. the probability of surviving the
whole window -- rather than scoring each team's hold-vs-spend tradeoff
in isolation.

The search is a bitmask DP over (week, used-teams-subset) states. Doing
this exactly over the full ~32-team league would be intractable
(2^32 states), so each week's candidates are first pruned to its
`per_week_top_k` highest win-probability teams, the per-week pools are
unioned, and that union is capped at `max_candidate_teams` (keeping the
highest-value teams first, but always guaranteeing every week retains
at least one candidate) before the DP runs over that reduced universe.

Only the current week's pick is meant to actually be used -- next
week's real state (results, updated used-teams, refreshed odds) will
differ, so the plan should be recomputed each week. The full projected
path is returned alongside it purely for transparency/display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from data import nflverse_client
from models import win_prob as wp

# How many weeks ahead (including the current week) to plan over.
DEFAULT_LOOKAHEAD_WEEKS = 7

# How many of a week's highest win-probability teams are even considered
# as candidates for that week, before the cross-week universe cap.
DEFAULT_PER_WEEK_TOP_K = 6

# Soft cap on the total number of distinct teams tracked by the DP across
# the whole window. Kept "soft" because every week is still guaranteed at
# least one candidate even if that pushes the realized universe slightly
# over this number.
DEFAULT_MAX_CANDIDATE_TEAMS = 14


@dataclass(frozen=True)
class WeekPick:
    """One team's matchup/win-probability for one specific week."""

    week: int
    team: str
    opponent: str
    is_home: bool
    win_probability: float
    spread_line: Optional[float]


@dataclass(frozen=True)
class OptimizedSequence:
    """Result of optimizing a pick sequence over a lookahead window."""

    current_week: int
    survival_probability: float
    path: Sequence[WeekPick]
    candidate_universe: Sequence[str]

    @property
    def recommended_pick(self) -> str:
        """This week's pick -- the first step of the optimal path."""
        return self.path[0].team


def _week_candidates(
    season: int,
    week: int,
    excluded_teams: Set[str],
    schedule: pd.DataFrame,
    spread_model: Optional[wp.SpreadModel],
) -> List[WeekPick]:
    """Every not-excluded team with a usable win probability for `week`, best-first."""
    week_games = schedule[schedule["week"] == week]
    picks: List[WeekPick] = []
    for _, row in week_games.iterrows():
        for team, opponent, is_home in (
            (row["home_team"], row["away_team"], True),
            (row["away_team"], row["home_team"], False),
        ):
            if team in excluded_teams:
                continue
            try:
                win_probability = wp.get_win_probability(row, team, spread_model=spread_model)
            except ValueError:
                continue  # no odds posted yet for this game
            spread_line = row.get("spread_line")
            picks.append(
                WeekPick(
                    week=week,
                    team=team,
                    opponent=opponent,
                    is_home=is_home,
                    win_probability=win_probability,
                    spread_line=(float(spread_line) if pd.notna(spread_line) else None),
                )
            )
    picks.sort(key=lambda p: p.win_probability, reverse=True)
    return picks


def build_candidate_universe(
    weekly_options: Dict[int, List[WeekPick]],
    per_week_top_k: int = DEFAULT_PER_WEEK_TOP_K,
    max_candidate_teams: int = DEFAULT_MAX_CANDIDATE_TEAMS,
) -> Dict[int, List[WeekPick]]:
    """Prune each week's options to a small, DP-tractable cross-week universe.

    Each week is first cut to its own `per_week_top_k` best options (each
    input list is assumed already sorted best-first). The union of teams
    surviving that cut is then capped at `max_candidate_teams`, keeping
    the teams with the highest best-any-week win probability first -- but
    a week is never left with zero candidates: if trimming would strip a
    week bare, that week's own single best team is added back regardless
    of the cap (an additive-only top-up, so a team already kept for one
    week's sake is never evicted to make room for another).
    """
    topk = {week: options[:per_week_top_k] for week, options in weekly_options.items()}

    team_best_prob: Dict[str, float] = {}
    for options in topk.values():
        for pick in options:
            team_best_prob[pick.team] = max(team_best_prob.get(pick.team, 0.0), pick.win_probability)

    kept: Set[str] = set()
    for team in sorted(team_best_prob, key=lambda t: team_best_prob[t], reverse=True):
        if len(kept) >= max_candidate_teams:
            break
        kept.add(team)

    for week, options in topk.items():
        if options and not any(pick.team in kept for pick in options):
            kept.add(options[0].team)

    return {week: [pick for pick in options if pick.team in kept] for week, options in topk.items()}


def _solve_dp(weekly_options: Dict[int, List[WeekPick]]) -> Tuple[float, List[WeekPick]]:
    """Core bitmask DP: the week-ordered, all-distinct-teams sequence maximizing
    the product of win probabilities. `weekly_options` should already be pruned
    to a small enough universe (see `build_candidate_universe`).

    Returns (survival_probability, path). Raises ValueError if some week has
    no candidates left to pick from given teams already used earlier in the
    same sequence.
    """
    ordered_weeks = sorted(weekly_options)
    universe_teams = sorted({pick.team for options in weekly_options.values() for pick in options})
    index_of = {team: i for i, team in enumerate(universe_teams)}

    # mask -> (product so far, path so far)
    dp: Dict[int, Tuple[float, List[WeekPick]]] = {0: (1.0, [])}
    for week in ordered_weeks:
        new_dp: Dict[int, Tuple[float, List[WeekPick]]] = {}
        for mask, (prod, path) in dp.items():
            for pick in weekly_options[week]:
                bit = 1 << index_of[pick.team]
                if mask & bit:
                    continue  # already used earlier in this candidate sequence
                new_mask = mask | bit
                new_prod = prod * pick.win_probability
                if new_mask not in new_dp or new_prod > new_dp[new_mask][0]:
                    new_dp[new_mask] = (new_prod, path + [pick])
        if not new_dp:
            raise ValueError(
                f"No viable pick sequence: week {week} has no candidate left that "
                "isn't already used earlier in this window"
            )
        dp = new_dp

    best_mask = max(dp, key=lambda m: dp[m][0])
    return dp[best_mask]


def optimize_pick_sequence(
    season: int,
    current_week: int,
    used_teams: Iterable[str],
    schedule: Optional[pd.DataFrame] = None,
    lookahead_weeks: int = DEFAULT_LOOKAHEAD_WEEKS,
    per_week_top_k: int = DEFAULT_PER_WEEK_TOP_K,
    max_candidate_teams: int = DEFAULT_MAX_CANDIDATE_TEAMS,
    spread_model: Optional[wp.SpreadModel] = None,
) -> OptimizedSequence:
    """Find the optimal pick sequence starting at `current_week`.

    Args:
        used_teams: Teams already spent -- excluded from every week in
            the window.
        schedule: Optional pre-loaded full-season schedule, to avoid
            re-downloading across repeated calls.
        lookahead_weeks: Size of the planning window, including
            `current_week` itself.
        per_week_top_k / max_candidate_teams: Pruning knobs -- see
            `build_candidate_universe`.

    Returns:
        An `OptimizedSequence` whose `path` covers `current_week` through
        min(current_week + lookahead_weeks - 1, the season's last week),
        possibly with weeks skipped if literally no team was available
        that week (fully used league, or no odds posted). Only
        `.recommended_pick` (== `path[0].team`) is meant to actually be
        acted on now -- the rest of `path` is a projection for display,
        since it will be recomputed against real results next week.
    """
    if schedule is None:
        schedule = nflverse_client.load_games(season=season)

    excluded = set(used_teams)
    max_week = int(schedule["week"].max())
    weeks = [w for w in range(current_week, current_week + lookahead_weeks) if w <= max_week]
    if not weeks:
        raise ValueError(f"No weeks remain from week {current_week} in season {season}")

    weekly_options = {week: _week_candidates(season, week, excluded, schedule, spread_model) for week in weeks}
    weekly_options = {week: options for week, options in weekly_options.items() if options}
    if not weekly_options:
        raise ValueError(
            f"No available teams have a usable game in weeks {weeks} of season {season}"
        )

    universe_options = build_candidate_universe(weekly_options, per_week_top_k, max_candidate_teams)
    survival_probability, path = _solve_dp(universe_options)

    return OptimizedSequence(
        current_week=current_week,
        survival_probability=survival_probability,
        path=path,
        candidate_universe=sorted({pick.team for options in universe_options.values() for pick in options}),
    )
