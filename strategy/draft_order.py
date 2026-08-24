"""Sequential draft-order pick allocation across both entries.

Both entries are now advised by the same algorithm: Entry A's multi-week
DP optimizer (`models.dp_optimizer`, via `strategy.entry_a_value`).
Entry B keeps its own independent used-teams history (a real second
entry with its own picks), but its *recommendation* is no longer
computed by the separate floor-based hedge strategy -- it's the same
DP-optimizer logic run against Entry B's own pool.

Four picks are drafted in strict dependency order, each excluding every
team drafted earlier in the sequence on top of that entry's own real
used-teams history:

  1. Entry A recommended -- Entry A's algorithm, from Entry A's pool.
  2. Entry B recommended -- Entry A's algorithm, from Entry B's pool,
     excluding pick 1 (if the two entries' pools would otherwise agree,
     Entry B's pick falls through to its own next-best choice instead).
  3. Entry A alternative -- Entry A's algorithm, from Entry A's pool,
     excluding picks 1 and 2.
  4. Entry B alternative -- Entry A's algorithm, from Entry B's pool,
     excluding picks 1, 2, and 3.

A pick is not simply the next item on some static ranked list: excluding
an already-drafted team can shift which team is genuinely optimal now
(the DP optimizer may choose to hold a different team back), so each
pick re-runs the full recommendation logic with the already-drafted
teams folded into its used-teams set.
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
    market_weight: float = 0.75,
    elo_games: Optional[pd.DataFrame] = None,
    team_bias_games: Optional[pd.DataFrame] = None,
) -> DraftPick:
    """Compute one entry's next pick, treating already-drafted teams as unavailable.

    Always uses Entry A's DP-optimizer recommendation logic (see module
    docstring), applied to `entry`'s own pool. Falls back to the plain
    highest-win-probability remaining team if no viable DP sequence
    exists once drafted teams are excluded.
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
    market_weight: float = 0.75,
    elo_games: Optional[pd.DataFrame] = None,
    team_bias_games: Optional[pd.DataFrame] = None,
) -> List[DraftPick]:
    """Draft `rounds` picks for Entry A and `rounds` picks for Entry B, round by round.

    Both entries are advised by the same algorithm (see module
    docstring). Every pick across both entries is guaranteed to be a
    distinct team: each pick excludes every team drafted earlier in this
    order on top of that entry's own real used-teams history. Picks are
    computed round-major (round 1: A then B, round 2: A then B, ...) so
    that each round's Entry B pick already excludes that same round's
    Entry A pick -- e.g. if the two entries' pools would otherwise agree
    on round 1, Entry B falls through to its own next-best choice.

    Args:
        used_teams_a / used_teams_b: Each entry's already-spent teams
            (from prior weeks), independent of this draft.
        rounds: How many picks to draft per entry (2 -> 4 total picks).
        schedule: Optional pre-loaded full-season schedule, to avoid
            re-downloading across repeated calls.
        lookahead_weeks: Planning window (N) passed through to the DP
            optimizer (see `models.dp_optimizer`).
        market_weight / elo_games / team_bias_games: see
            `models.win_prob.get_win_probability`.

    Returns:
        A flat list of `rounds * 2` `DraftPick`s: Entry A's `rounds`
        picks (pick numbers 1..rounds) followed by Entry B's `rounds`
        picks (pick numbers rounds+1..2*rounds) -- e.g. at the default
        `rounds=2`: [A pick 1, A pick 2, B pick 1 (numbered 3), B pick 2
        (numbered 4)], even though B's picks are computed interleaved
        with A's round by round (see above) to get their exclusions right.
    """
    if schedule is None:
        schedule = nflverse_client.load_games(season=season)

    used_a: Set[str] = set(used_teams_a)
    used_b: Set[str] = set(used_teams_b)
    drafted: Set[str] = set()

    a_picks: List[DraftPick] = []
    b_picks: List[DraftPick] = []
    for round_number in range(1, rounds + 1):
        pick_a = _draft_pick_for_entry(
            "A",
            round_number,
            round_number,
            season,
            week,
            used_a,
            drafted,
            schedule,
            spread_model,
            lookahead_weeks,
            market_weight=market_weight,
            elo_games=elo_games,
            team_bias_games=team_bias_games,
        )
        a_picks.append(pick_a)
        drafted.add(pick_a.team)

        pick_b = _draft_pick_for_entry(
            "B",
            rounds + round_number,
            round_number,
            season,
            week,
            used_b,
            drafted,
            schedule,
            spread_model,
            lookahead_weeks,
            market_weight=market_weight,
            elo_games=elo_games,
            team_bias_games=team_bias_games,
        )
        b_picks.append(pick_b)
        drafted.add(pick_b.team)

    return a_picks + b_picks
