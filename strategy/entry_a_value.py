"""Weekly pick strategy for Entry A.

Recommends this week's pick by running an N-week dynamic-programming
optimizer (`models.dp_optimizer`) over the current used-teams state: it
finds the sequence of distinct teams across the lookahead window that
maximizes the *product* of win probabilities -- i.e. the probability of
surviving the whole window -- and this week's pick is simply the first
step of that optimal sequence. The full projected path comes along for
display (e.g. in the UI), and the reasoning explains whether -- and
why -- the optimizer is holding back the single best team available
this week in favor of a stronger matchup later on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd

from models import dp_optimizer
from models import win_prob as wp
from strategy import entry_b_hedge

ENTRY_NAME = "A"
DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "used_teams_a.json"

# Entry A's candidate pool is just win-probability/matchup info, identical
# in shape to Entry B's -- the DP optimizer (not a per-candidate score) is
# what decides the hold-vs-spend tradeoff now.
TeamCandidate = entry_b_hedge.TeamCandidate


@dataclass(frozen=True)
class PickRecommendation:
    week: int
    entry: str
    team: str
    win_probability: float
    spread_line: Optional[float]
    reasoning: str
    survival_probability: float
    projected_path: Sequence[dp_optimizer.WeekPick]
    available: Sequence[TeamCandidate]


def load_used_teams(state_path: Path = DEFAULT_STATE_PATH) -> Set[str]:
    """Return the set of teams Entry A has already used, from its state file."""
    with open(state_path) as f:
        state = json.load(f)
    return set(state.get("used_teams", {}).values())


def build_candidates(
    season: int,
    week: int,
    used_teams: Iterable[str],
    schedule: Optional[pd.DataFrame] = None,
    spread_model: Optional[wp.SpreadModel] = None,
) -> List[TeamCandidate]:
    """Every available (not-yet-used) team's matchup/win-probability for `week`."""
    return entry_b_hedge.build_candidates(season, week, used_teams, schedule=schedule, spread_model=spread_model)


def _format_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _format_team_spread(spread_line: Optional[float], is_home: bool) -> str:
    if spread_line is None:
        return "spread unavailable"
    team_spread = spread_line if is_home else -spread_line
    if team_spread > 0:
        return f"favored by {team_spread:g}"
    if team_spread < 0:
        return f"underdog by {abs(team_spread):g}"
    return "a pick'em"


def build_reasoning(
    result: dp_optimizer.OptimizedSequence,
    greedy_best: Optional[TeamCandidate],
) -> str:
    """Plain-language explanation: win prob/spread for this week's pick, and why
    the optimizer picked it over the single best team available now (if
    different), referencing the projected path.
    """
    top = result.path[0]
    window = len(result.path)
    parts = [
        f"{top.team} vs {top.opponent} ({'home' if top.is_home else 'away'}): "
        f"{_format_pct(top.win_probability)} win probability, "
        f"{_format_team_spread(top.spread_line, top.is_home)}."
    ]

    if greedy_best is not None and greedy_best.team != top.team:
        held_week = next((p.week for p in result.path[1:] if p.team == greedy_best.team), None)
        if held_week is not None:
            held_prob = next(p.win_probability for p in result.path if p.team == greedy_best.team)
            parts.append(
                f"{greedy_best.team} is actually this week's single best option "
                f"({_format_pct(greedy_best.win_probability)}), but the {window}-week optimizer holds "
                f"them for Week {held_week} ({_format_pct(held_prob)} there) since that raises the "
                f"overall {window}-week survival odds to {_format_pct(result.survival_probability)}."
            )
        else:
            parts.append(
                f"{greedy_best.team} is actually this week's single best option "
                f"({_format_pct(greedy_best.win_probability)}), but the {window}-week optimizer favors "
                f"{top.team} instead to raise the overall survival odds across the window to "
                f"{_format_pct(result.survival_probability)}."
            )
    else:
        parts.append(
            f"Also the single best option available this week; the {window}-week projected "
            f"survival probability along this path is {_format_pct(result.survival_probability)}."
        )

    if window > 1:
        rest = ", ".join(f"Wk{p.week} {p.team} ({_format_pct(p.win_probability)})" for p in result.path[1:])
        parts.append(f"Projected path after this week: {rest}.")

    return " ".join(parts)


def recommend_pick(
    season: int,
    week: int,
    used_teams: Optional[Iterable[str]] = None,
    schedule: Optional[pd.DataFrame] = None,
    state_path: Path = DEFAULT_STATE_PATH,
    lookahead_weeks: int = dp_optimizer.DEFAULT_LOOKAHEAD_WEEKS,
    per_week_top_k: int = dp_optimizer.DEFAULT_PER_WEEK_TOP_K,
    max_candidate_teams: int = dp_optimizer.DEFAULT_MAX_CANDIDATE_TEAMS,
    spread_model: Optional[wp.SpreadModel] = None,
) -> PickRecommendation:
    """Recommend Entry A's best pick for `season`/`week` via the DP optimizer.

    Args:
        used_teams: Teams already spent by Entry A. Loaded from
            `state_path` (Entry A's state file) if omitted.
        schedule: Optional pre-loaded full-season schedule, to avoid
            re-downloading across repeated calls.
        lookahead_weeks / per_week_top_k / max_candidate_teams: Passed
            straight through to `dp_optimizer.optimize_pick_sequence`.
    """
    if used_teams is None:
        used_teams = load_used_teams(state_path)
    used_teams = set(used_teams)

    result = dp_optimizer.optimize_pick_sequence(
        season,
        week,
        used_teams,
        schedule=schedule,
        lookahead_weeks=lookahead_weeks,
        per_week_top_k=per_week_top_k,
        max_candidate_teams=max_candidate_teams,
        spread_model=spread_model,
    )

    available = build_candidates(season, week, used_teams, schedule=schedule, spread_model=spread_model)
    greedy_best = max(available, key=lambda c: c.win_probability) if available else None

    top = result.path[0]
    return PickRecommendation(
        week=week,
        entry=ENTRY_NAME,
        team=top.team,
        win_probability=top.win_probability,
        spread_line=top.spread_line,
        reasoning=build_reasoning(result, greedy_best),
        survival_probability=result.survival_probability,
        projected_path=result.path,
        available=available,
    )
