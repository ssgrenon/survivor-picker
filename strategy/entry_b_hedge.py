"""Weekly pick strategy for Entry B: the hedge / safety entry.

Unlike Entry A (which weighs future value to decide when to hold a team
back), Entry B's job is to be the pool's high-floor pick each week: rank
available teams purely by win probability and enforce a minimum win
probability floor (default 65%) below which a team isn't even considered.

`build_candidates()` here is entry-agnostic -- it only depends on which
teams are already used, not on any ranking strategy -- so
`strategy.joint_optimizer` reuses it for both entries' pools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd

from data import nflverse_client
from models import win_prob as wp

ENTRY_NAME = "B"
DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "used_teams_b.json"
DEFAULT_MIN_WIN_PROBABILITY = 0.65


@dataclass(frozen=True)
class TeamCandidate:
    """One available team's matchup for the week under consideration."""

    team: str
    opponent: str
    is_home: bool
    win_probability: float
    spread_line: Optional[float]

    @property
    def team_spread(self) -> Optional[float]:
        """Points `team` is favored by (negative means `team` is the underdog)."""
        if self.spread_line is None:
            return None
        return self.spread_line if self.is_home else -self.spread_line


@dataclass(frozen=True)
class PickRecommendation:
    week: int
    entry: str
    team: str
    win_probability: float
    spread_line: Optional[float]
    reasoning: str
    ranked_picks: Sequence[TeamCandidate]


def load_used_teams(state_path: Path = DEFAULT_STATE_PATH) -> Set[str]:
    """Return the set of teams already used, from an entry's state file."""
    with open(state_path) as f:
        state = json.load(f)
    return set(state.get("used_teams", {}).values())


def build_candidates(
    season: int,
    week: int,
    used_teams: Iterable[str] = (),
    schedule: Optional[pd.DataFrame] = None,
    spread_model: Optional[wp.SpreadModel] = None,
    market_weight: float = 1.0,
    elo_games: Optional[pd.DataFrame] = None,
) -> List[TeamCandidate]:
    """Build the list of available (not-yet-used) teams' candidates for `week`.

    Teams whose game has neither moneylines nor a spread_line yet (too far
    out for odds to be posted) are skipped -- there's nothing to rank them
    by until a line exists.

    `market_weight` / `elo_games`: see `models.win_prob.get_win_probability`.
    """
    if schedule is None:
        schedule = nflverse_client.load_games(season=season)

    used = set(used_teams)
    week_games = schedule[schedule["week"] == week]
    if week_games.empty:
        raise ValueError(f"No games found for season {season} week {week}")

    candidates = []
    for _, row in week_games.iterrows():
        for team, opponent, is_home in (
            (row["home_team"], row["away_team"], True),
            (row["away_team"], row["home_team"], False),
        ):
            if team in used:
                continue
            try:
                win_probability = wp.get_win_probability(
                    row, team, market_weight=market_weight, spread_model=spread_model, elo_games=elo_games
                )
            except ValueError:
                continue
            spread_line = row.get("spread_line")
            candidates.append(
                TeamCandidate(
                    team=team,
                    opponent=opponent,
                    is_home=is_home,
                    win_probability=win_probability,
                    spread_line=(float(spread_line) if pd.notna(spread_line) else None),
                )
            )
    return candidates


def rank_picks(
    candidates: Sequence[TeamCandidate],
    min_win_probability: float = DEFAULT_MIN_WIN_PROBABILITY,
) -> List[TeamCandidate]:
    """Filter to candidates clearing the win-probability floor, sorted best-first."""
    eligible = [c for c in candidates if c.win_probability >= min_win_probability]
    eligible.sort(key=lambda c: c.win_probability, reverse=True)
    return eligible


def _format_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _format_team_spread(team_spread: Optional[float]) -> str:
    if team_spread is None:
        return "spread unavailable"
    if team_spread > 0:
        return f"favored by {team_spread:g}"
    if team_spread < 0:
        return f"underdog by {abs(team_spread):g}"
    return "a pick'em"


def build_reasoning(
    top: TeamCandidate,
    runner_up: Optional[TeamCandidate],
    min_win_probability: float = DEFAULT_MIN_WIN_PROBABILITY,
) -> str:
    """Plain-language explanation of the top pick: win prob, spread, and the floor."""
    parts = [
        f"{top.team} vs {top.opponent} ({'home' if top.is_home else 'away'}): "
        f"{_format_pct(top.win_probability)} win probability, {_format_team_spread(top.team_spread)}, "
        f"clears the {min_win_probability:.0%} floor."
    ]
    if runner_up is not None:
        parts.append(
            f"Safer than the next-best option, {runner_up.team} "
            f"({_format_pct(runner_up.win_probability)} win prob), "
            f"by {_format_pct(top.win_probability - runner_up.win_probability)}."
        )
    else:
        parts.append("It is the only team clearing the floor this week.")
    return " ".join(parts)


def recommend_pick(
    season: int,
    week: int,
    used_teams: Optional[Iterable[str]] = None,
    schedule: Optional[pd.DataFrame] = None,
    state_path: Path = DEFAULT_STATE_PATH,
    min_win_probability: float = DEFAULT_MIN_WIN_PROBABILITY,
    spread_model: Optional[wp.SpreadModel] = None,
    market_weight: float = 1.0,
    elo_games: Optional[pd.DataFrame] = None,
) -> PickRecommendation:
    """Recommend Entry B's safest pick for `season`/`week`.

    Args:
        used_teams: Teams already spent by Entry B. Loaded from
            `state_path` (Entry B's state file) if omitted.
        schedule: Optional pre-loaded full-season schedule, to avoid
            re-downloading across repeated calls.
        market_weight / elo_games: see `models.win_prob.get_win_probability`.
    """
    if used_teams is None:
        used_teams = load_used_teams(state_path)

    candidates = build_candidates(
        season,
        week,
        used_teams,
        schedule=schedule,
        spread_model=spread_model,
        market_weight=market_weight,
        elo_games=elo_games,
    )
    eligible = rank_picks(candidates, min_win_probability=min_win_probability)
    if not eligible:
        raise ValueError(
            f"No available team clears the {min_win_probability:.0%} win probability floor "
            f"for season {season} week {week}"
        )

    top = eligible[0]
    runner_up = eligible[1] if len(eligible) > 1 else None

    return PickRecommendation(
        week=week,
        entry=ENTRY_NAME,
        team=top.team,
        win_probability=top.win_probability,
        spread_line=top.spread_line,
        reasoning=build_reasoning(top, runner_up, min_win_probability),
        ranked_picks=eligible,
    )
