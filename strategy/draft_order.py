"""Sequential draft-order pick allocation across both entries.

Computes the top-N picks for each entry such that all picks across both
entries are guaranteed to be distinct teams. Both entries use the exact
same algorithm (Entry A's multi-week DP optimizer, see
`models.dp_optimizer` / `strategy.entry_a_value`) run independently
against their own used-teams history; picks are allocated round by round
-- Entry A's round-1 pick, then Entry B's round-1 pick (excluding
whatever Entry A just took), then Entry A's round-2 pick (excluding both
round-1 picks), then Entry B's round-2 pick (excluding all three prior
picks), and so on.

A pick is not simply the next item on some static ranked list: since
each entry's recommendation comes from a multi-week DP optimizer,
excluding an already-drafted team can shift which team is genuinely
optimal now, so each pick re-runs the full recommendation logic with the
already-drafted teams folded into that entry's used-teams set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

import pandas as pd

from data import nflverse_client
from models import dp_optimizer
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
    divergence: Optional[float] = None
    team_bias_adjustment: float = 0.0


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
    lookahead_weeks: int,
    market_weight: float = 0.5,
    elo_games: Optional[pd.DataFrame] = None,
    team_bias_games: Optional[pd.DataFrame] = None,
) -> DraftPick:
    """Compute one entry's next pick, treating already-drafted teams as unavailable.

    Falls back to the plain highest-win-probability remaining team if the
    multi-week optimizer finds no viable sequence once drafted teams are
    excluded.
    """
    combined_used = used_teams | drafted_so_far

    try:
        rec = entry_a_value.recommend_pick(
            season,
            week,
            used_teams=combined_used,
            schedule=schedule,
            spread_model=spread_model,
            lookahead_weeks=lookahead_weeks,
            market_weight=market_weight,
            elo_games=elo_games,
            team_bias_games=team_bias_games,
        )
        top = rec.projected_path[0]
        reasoning = rec.reasoning
    except ValueError:
        top = None
        reasoning = None

    if top is None:
        available = entry_b_hedge.build_candidates(
            season,
            week,
            combined_used,
            schedule=schedule,
            spread_model=spread_model,
            market_weight=market_weight,
            elo_games=elo_games,
            team_bias_games=team_bias_games,
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
        divergence=top.divergence,
        team_bias_adjustment=top.team_bias_adjustment,
    )


def draft_picks(
    season: int,
    week: int,
    used_teams_a: Iterable[str],
    used_teams_b: Iterable[str],
    rounds: int = DEFAULT_ROUNDS,
    schedule: Optional[pd.DataFrame] = None,
    spread_model: Optional[wp.SpreadModel] = None,
    lookahead_weeks: int = dp_optimizer.DEFAULT_LOOKAHEAD_WEEKS,
    market_weight: float = 0.5,
    elo_games: Optional[pd.DataFrame] = None,
    team_bias_games: Optional[pd.DataFrame] = None,
) -> List[DraftPick]:
    """Draft `rounds` picks per entry, alternating A/B each round.

    Every pick across both entries is guaranteed to be a distinct team:
    each pick excludes every team drafted earlier in this order (by
    either entry) on top of that entry's own real used-teams history.

    Args:
        used_teams_a / used_teams_b: Each entry's already-spent teams
            (from prior weeks), independent of this draft.
        rounds: How many picks to draft per entry (2 -> 4 total picks).
        schedule: Optional pre-loaded full-season schedule, to avoid
            re-downloading across repeated calls.
        lookahead_weeks: Planning window (N) passed through to the DP
            optimizer (see `models.dp_optimizer`), used identically for
            both entries.
        market_weight / elo_games / team_bias_games: see
            `models.win_prob.get_win_probability`.

    Returns:
        A flat list of `rounds * 2` `DraftPick`s in priority order:
        [A round 1, B round 1, A round 2, B round 2, ...].
    """
    if schedule is None:
        schedule = nflverse_client.load_games(season=season)

    used_a: Set[str] = set(used_teams_a)
    used_b: Set[str] = set(used_teams_b)
    drafted: Set[str] = set()

    picks: List[DraftPick] = []
    pick_number = 0
    for round_number in range(1, rounds + 1):
        for entry, used_teams in (("A", used_a), ("B", used_b)):
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
                lookahead_weeks,
                market_weight=market_weight,
                elo_games=elo_games,
                team_bias_games=team_bias_games,
            )
            picks.append(pick)
            drafted.add(pick.team)

    return picks
