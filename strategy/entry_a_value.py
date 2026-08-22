"""Weekly pick strategy for Entry A.

Ranks this week's available (not-yet-used) teams by a value score that
rewards a high win probability but discounts teams that have a
materially better matchup coming up soon, per `models.future_value`:

    score = win_probability * (1 - future_value_penalty)

where `future_value_penalty` is the team's (clipped, non-negative)
`future_value` score -- 0 when there's nothing better ahead, up to 1 when
holding the team would be clearly better than using them now. The top
score is returned as the recommended pick, along with plain-language
reasoning referencing win probability, spread, and the runner-up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd

from data import nflverse_client
from models import future_value
from models import win_prob as wp

ENTRY_NAME = "A"
DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "used_teams_a.json"
DEFAULT_PENALTY_WEIGHT = 1.0


@dataclass(frozen=True)
class TeamCandidate:
    """One available team's matchup for the week under consideration."""

    team: str
    opponent: str
    is_home: bool
    win_probability: float
    future_value: float
    spread_line: Optional[float]

    @property
    def team_spread(self) -> Optional[float]:
        """Points `team` is favored by (negative means `team` is the underdog)."""
        if self.spread_line is None:
            return None
        return self.spread_line if self.is_home else -self.spread_line


@dataclass(frozen=True)
class RankedPick(TeamCandidate):
    """A candidate plus its computed hold penalty and final ranking score."""

    future_value_penalty: float = 0.0
    score: float = 0.0


@dataclass(frozen=True)
class PickRecommendation:
    week: int
    entry: str
    team: str
    win_probability: float
    spread_line: Optional[float]
    score: float
    reasoning: str
    ranked_picks: Sequence[RankedPick]


def load_used_teams(state_path: Path = DEFAULT_STATE_PATH) -> Set[str]:
    """Return the set of teams Entry A has already used, from its state file."""
    with open(state_path) as f:
        state = json.load(f)
    return set(state.get("used_teams", {}).values())


def future_value_penalty(raw_future_value: float, penalty_weight: float = DEFAULT_PENALTY_WEIGHT) -> float:
    """Convert a raw `future_value` score into a [0, 1] penalty fraction.

    Only a *positive* future_value (a materially better matchup coming up)
    penalizes a team's current-week score; future_value <= 0 (nothing
    better ahead) gets a 0 penalty.
    """
    return min(1.0, max(0.0, raw_future_value) * penalty_weight)


def rank_picks(
    candidates: Sequence[TeamCandidate], penalty_weight: float = DEFAULT_PENALTY_WEIGHT
) -> List[RankedPick]:
    """Score and sort candidates best-first by win_probability * (1 - penalty)."""
    if not candidates:
        raise ValueError("no candidates to rank")

    ranked = []
    for c in candidates:
        penalty = future_value_penalty(c.future_value, penalty_weight)
        score = c.win_probability * (1 - penalty)
        ranked.append(
            RankedPick(
                team=c.team,
                opponent=c.opponent,
                is_home=c.is_home,
                win_probability=c.win_probability,
                future_value=c.future_value,
                spread_line=c.spread_line,
                future_value_penalty=penalty,
                score=score,
            )
        )
    ranked.sort(key=lambda p: p.score, reverse=True)
    return ranked


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


def build_reasoning(top: RankedPick, runner_up: Optional[RankedPick]) -> str:
    """Plain-language explanation of the top pick: win prob, spread, why it wins out."""
    parts = [
        f"{top.team} vs {top.opponent} ({'home' if top.is_home else 'away'}): "
        f"{_format_pct(top.win_probability)} win probability, {_format_team_spread(top.team_spread)}."
    ]

    if top.future_value_penalty > 0:
        parts.append(
            f"A hold penalty of {top.future_value_penalty:.2f} was applied "
            f"(future_value={top.future_value:+.3f}, meaning a stronger matchup is on the "
            f"horizon), but {top.team} still ranks best this week."
        )
    else:
        parts.append(
            f"No better matchup is on the horizon for {top.team}, so no hold penalty applies."
        )

    if runner_up is not None:
        margin = top.score - runner_up.score
        parts.append(
            f"Beats the next-best option, {runner_up.team} "
            f"({_format_pct(runner_up.win_probability)} win prob, score {runner_up.score:.3f}), "
            f"by {margin:.3f} points of adjusted score."
        )
    else:
        parts.append("It is the only available team with a game this week.")

    return " ".join(parts)


def build_candidates(
    season: int,
    week: int,
    used_teams: Iterable[str],
    schedule: Optional[pd.DataFrame] = None,
    lookahead_weeks: int = future_value.DEFAULT_LOOKAHEAD_WEEKS,
    decay_rate: float = future_value.DEFAULT_DECAY_RATE,
    spread_model: Optional[wp.SpreadModel] = None,
) -> List[TeamCandidate]:
    """Build the list of available (not-yet-used) teams' candidates for `week`.

    Teams whose game has neither moneylines nor a spread_line yet (too far
    out for odds to be posted) are skipped -- there's nothing to rank them
    by until a line exists.
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
                win_probability = wp.get_win_probability(row, team, spread_model=spread_model)
            except ValueError:
                continue  # no odds posted yet for this game -- nothing to rank it by
            remaining = future_value.load_remaining_schedule(team, season, week, schedule=schedule)
            fv_score = future_value.get_future_value(
                team,
                remaining,
                week,
                lookahead_weeks=lookahead_weeks,
                decay_rate=decay_rate,
                spread_model=spread_model,
            )
            spread_line = row.get("spread_line")
            candidates.append(
                TeamCandidate(
                    team=team,
                    opponent=opponent,
                    is_home=is_home,
                    win_probability=win_probability,
                    future_value=fv_score,
                    spread_line=(float(spread_line) if pd.notna(spread_line) else None),
                )
            )

    return candidates


def recommend_pick(
    season: int,
    week: int,
    used_teams: Optional[Iterable[str]] = None,
    schedule: Optional[pd.DataFrame] = None,
    state_path: Path = DEFAULT_STATE_PATH,
    lookahead_weeks: int = future_value.DEFAULT_LOOKAHEAD_WEEKS,
    decay_rate: float = future_value.DEFAULT_DECAY_RATE,
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
    spread_model: Optional[wp.SpreadModel] = None,
) -> PickRecommendation:
    """Recommend Entry A's best pick for `season`/`week`.

    Args:
        used_teams: Teams already spent by Entry A. Loaded from
            `state_path` (Entry A's state file) if omitted.
        schedule: Optional pre-loaded full-season schedule, to avoid
            re-downloading across repeated calls.
    """
    if used_teams is None:
        used_teams = load_used_teams(state_path)

    candidates = build_candidates(
        season,
        week,
        used_teams,
        schedule=schedule,
        lookahead_weeks=lookahead_weeks,
        decay_rate=decay_rate,
        spread_model=spread_model,
    )
    if not candidates:
        raise ValueError(
            f"No available (unused) teams have games in season {season} week {week}"
        )

    ranked = rank_picks(candidates, penalty_weight=penalty_weight)
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    return PickRecommendation(
        week=week,
        entry=ENTRY_NAME,
        team=top.team,
        win_probability=top.win_probability,
        spread_line=top.spread_line,
        score=top.score,
        reasoning=build_reasoning(top, runner_up),
        ranked_picks=ranked,
    )
