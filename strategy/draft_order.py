"""Sequential draft-order pick allocation across both entries.

Computes the top-N picks for each entry such that all picks across both
entries are guaranteed to be distinct teams, using a strict priority
order: Entry A drafts all of its picks first (Pick 1 = its best, Pick 2
= its next best excluding Pick 1, ...), then Entry B drafts all of its
picks (excluding every team Entry A already took).

A pick is not simply the next item on some static ranked list: for
Entry A in particular (whose recommendation comes from a multi-week DP
optimizer, see `models.dp_optimizer`), excluding an already-drafted
team can shift which team is genuinely optimal now, so each pick
re-runs that entry's full recommendation logic with the already-drafted
teams folded into its used-teams set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

import pandas as pd

from data import nflverse_client
from models import win_prob as wp
from strategy import entry_a_value
from strategy import entry_b_hedge

DEFAULT_ROUNDS = 2


@dataclass(frozen=True)
class DraftPick:
    """One pick in the draft sequence."""

    pick_number: int
    round: int
    entry: str  # "A" or "B"
    team: str
    opponent: str
    is_home: bool
    win_probability: float
    spread_line: Optional[float]
    reasoning: str


def _draft_pick_for_entry(
    entry: str,
    pick_number: int,
    round_number: int,
    season: int,
    week: int,
    used_teams: Set[str],
    drafted_so_far: Set[str],
    schedule: pd.DataFrame,
    spread_model: Optional[wp.SpreadModel],
) -> DraftPick:
    """Compute one entry's next pick, treating already-drafted teams as unavailable.

    Falls back to the plain highest-win-probability remaining team if the
    entry's own recommendation logic finds nothing that meets its usual
    criteria (Entry A: no viable DP sequence; Entry B: nothing clears its
    win-probability floor) once drafted teams are excluded.
    """
    combined_used = used_teams | drafted_so_far

    if entry == "A":
        try:
            rec = entry_a_value.recommend_pick(
                season, week, used_teams=combined_used, schedule=schedule, spread_model=spread_model
            )
            top = rec.projected_path[0]
            reasoning = rec.reasoning
        except ValueError:
            top = None
            reasoning = None
    else:
        try:
            rec = entry_b_hedge.recommend_pick(
                season, week, used_teams=combined_used, schedule=schedule, spread_model=spread_model
            )
            top = rec.ranked_picks[0]
            reasoning = rec.reasoning
        except ValueError:
            top = None
            reasoning = None

    if top is None:
        available = entry_b_hedge.build_candidates(
            season, week, combined_used, schedule=schedule, spread_model=spread_model
        )
        if not available:
            raise ValueError(
                f"No team left to draft for Entry {entry} at pick #{pick_number} "
                f"(season {season} week {week})"
            )
        top = max(available, key=lambda c: c.win_probability)
        reasoning = (
            f"Fallback: highest remaining win probability once already-drafted teams "
            f"are excluded ({top.win_probability:.1%})."
        )

    return DraftPick(
        pick_number=pick_number,
        round=round_number,
        entry=entry,
        team=top.team,
        opponent=top.opponent,
        is_home=top.is_home,
        win_probability=top.win_probability,
        spread_line=top.spread_line,
        reasoning=reasoning,
    )


def draft_picks(
    season: int,
    week: int,
    used_teams_a: Iterable[str],
    used_teams_b: Iterable[str],
    rounds: int = DEFAULT_ROUNDS,
    schedule: Optional[pd.DataFrame] = None,
    spread_model: Optional[wp.SpreadModel] = None,
) -> List[DraftPick]:
    """Draft `rounds` picks for Entry A, then `rounds` picks for Entry B.

    Every pick across both entries is guaranteed to be a distinct team:
    each pick excludes every team drafted earlier in this order (by
    either entry) on top of that entry's own real used-teams history.

    Args:
        used_teams_a / used_teams_b: Each entry's already-spent teams
            (from prior weeks), independent of this draft.
        rounds: How many picks to draft per entry (2 -> 4 total picks).
        schedule: Optional pre-loaded full-season schedule, to avoid
            re-downloading across repeated calls.

    Returns:
        A flat list of `rounds * 2` `DraftPick`s in priority order:
        [A round 1, A round 2, ..., B round 1, B round 2, ...].
    """
    if schedule is None:
        schedule = nflverse_client.load_games(season=season)

    used_a: Set[str] = set(used_teams_a)
    used_b: Set[str] = set(used_teams_b)
    drafted: Set[str] = set()

    picks: List[DraftPick] = []
    pick_number = 0
    for entry, used_teams in (("A", used_a), ("B", used_b)):
        for round_number in range(1, rounds + 1):
            pick_number += 1
            pick = _draft_pick_for_entry(
                entry,
                pick_number,
                round_number,
                season,
                week,
                used_teams,
                drafted,
                schedule,
                spread_model,
            )
            picks.append(pick)
            drafted.add(pick.team)

    return picks
