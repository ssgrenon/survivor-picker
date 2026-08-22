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

Home-game override: if and only if an entry's round-1 pick is an away
game, its round-2 pick is forced to the single best available home game
instead of that entry's normal next-best recommendation (falling back to
the normal recommendation if no home game remains).
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
    market_weight: float = 1.0,
    elo_games: Optional[pd.DataFrame] = None,
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
                season,
                week,
                used_teams=combined_used,
                schedule=schedule,
                spread_model=spread_model,
                lookahead_weeks=lookahead_weeks,
                market_weight=market_weight,
                elo_games=elo_games,
            )
            top = rec.projected_path[0]
            reasoning = rec.reasoning
        except ValueError:
            top = None
            reasoning = None
    else:
        try:
            rec = entry_b_hedge.recommend_pick(
                season,
                week,
                used_teams=combined_used,
                schedule=schedule,
                spread_model=spread_model,
                market_weight=market_weight,
                elo_games=elo_games,
            )
            top = rec.ranked_picks[0]
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
    )


def _draft_best_home_pick_for_entry(
    entry: str,
    pick_number: int,
    round_number: int,
    round1_pick: DraftPick,
    season: int,
    week: int,
    used_teams: Set[str],
    drafted_so_far: Set[str],
    schedule: pd.DataFrame,
    spread_model: Optional[wp.SpreadModel],
    market_weight: float = 1.0,
    elo_games: Optional[pd.DataFrame] = None,
) -> Optional[DraftPick]:
    """The single best remaining home-team candidate for `entry`, or None if none remain.

    Used for round 2 when `round1_pick` was an away game (see module
    docstring). "Best" means highest win probability, matching how every
    other fallback in this module picks among an undifferentiated pool.
    """
    combined_used = used_teams | drafted_so_far
    available = entry_b_hedge.build_candidates(
        season,
        week,
        combined_used,
        schedule=schedule,
        spread_model=spread_model,
        market_weight=market_weight,
        elo_games=elo_games,
    )
    home_candidates = [c for c in available if c.is_home]
    if not home_candidates:
        return None

    top = max(home_candidates, key=lambda c: c.win_probability)
    return DraftPick(
        pick_number=pick_number,
        round=round_number,
        entry=entry,
        team=top.team,
        opponent=top.opponent,
        is_home=top.is_home,
        win_probability=top.win_probability,
        spread_line=top.spread_line,
        reasoning=(
            f"Home-game override: Pick #{round1_pick.pick_number} ({round1_pick.team}) is an "
            f"away game, so this pick is forced to the best available home game instead "
            f"({top.win_probability:.1%})."
        ),
        divergence=top.divergence,
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
    market_weight: float = 1.0,
    elo_games: Optional[pd.DataFrame] = None,
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
        lookahead_weeks: Planning window (N) passed through to Entry A's
            DP optimizer (see `models.dp_optimizer`). Entry B's hedge
            strategy doesn't use a lookahead, so this only affects "A"
            picks.
        market_weight / elo_games: see `models.win_prob.get_win_probability`.

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
        round1_pick: Optional[DraftPick] = None
        for round_number in range(1, rounds + 1):
            pick_number += 1

            pick = None
            if round_number == 2 and round1_pick is not None and not round1_pick.is_home:
                pick = _draft_best_home_pick_for_entry(
                    entry,
                    pick_number,
                    round_number,
                    round1_pick,
                    season,
                    week,
                    used_teams,
                    drafted,
                    schedule,
                    spread_model,
                    market_weight=market_weight,
                    elo_games=elo_games,
                )
            if pick is None:
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
                )

            picks.append(pick)
            drafted.add(pick.team)
            if round_number == 1:
                round1_pick = pick

    return picks
