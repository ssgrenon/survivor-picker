"""Joint weekly optimizer across Entry A and Entry B.

Rather than letting each entry pick independently, this searches every
valid (team_for_A, team_for_B) pairing for the week and picks the pair
that maximizes:

    P(A survives) + P(B survives) - P(A eliminated AND B eliminated)

subject to:
  - neither team has already been used by its own entry
  - the two entries are never assigned the same team this week
  - the two entries are never put on opposing sides of the same game
    (that would guarantee exactly one survives every time, trading away
    the "both survive" upside for no added safety)
  - Entry B's pick must clear a minimum win probability floor (default
    65%), since B is the pool's safety net

Every valid pair is, by construction, drawn from two different games, so
the two outcomes are treated as independent when combining probabilities.

Candidate pools are built with `strategy.entry_b_hedge.build_candidates`,
which is entry-agnostic (it only depends on which teams are already
used, not on either entry's individual ranking strategy).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from data import nflverse_client
from models import win_prob as wp
from strategy import entry_a_value
from strategy import entry_b_hedge

DEFAULT_MIN_WIN_PROBABILITY_B = entry_b_hedge.DEFAULT_MIN_WIN_PROBABILITY


@dataclass(frozen=True)
class JointPick:
    """One candidate (team_A, team_B) pairing and its joint outcome probabilities."""

    team_a: str
    team_b: str
    win_probability_a: float
    win_probability_b: float
    both_survive: float
    one_survives: float
    both_eliminated: float
    objective: float


@dataclass(frozen=True)
class JointRecommendation:
    week: int
    pick_a: str
    pick_b: str
    win_probability_a: float
    win_probability_b: float
    spread_a: Optional[float]
    spread_b: Optional[float]
    both_survive_probability: float
    one_survives_probability: float
    both_eliminated_probability: float
    reasoning: str
    considered: Sequence[JointPick]


def _same_game(cand_a: entry_b_hedge.TeamCandidate, cand_b: entry_b_hedge.TeamCandidate) -> bool:
    """True if the two candidates are opposing sides of the same game."""
    return cand_a.opponent == cand_b.team


def evaluate_pair(
    cand_a: entry_b_hedge.TeamCandidate, cand_b: entry_b_hedge.TeamCandidate
) -> JointPick:
    """Score one (team_A, team_B) pairing, assuming independent outcomes."""
    a = cand_a.win_probability
    b = cand_b.win_probability
    both_survive = a * b
    both_eliminated = (1 - a) * (1 - b)
    one_survives = 1.0 - both_survive - both_eliminated
    objective = a + b - both_eliminated
    return JointPick(
        team_a=cand_a.team,
        team_b=cand_b.team,
        win_probability_a=a,
        win_probability_b=b,
        both_survive=both_survive,
        one_survives=one_survives,
        both_eliminated=both_eliminated,
        objective=objective,
    )


def find_valid_pairs(
    candidates_a: Sequence[entry_b_hedge.TeamCandidate],
    candidates_b: Sequence[entry_b_hedge.TeamCandidate],
    min_win_probability_b: float = DEFAULT_MIN_WIN_PROBABILITY_B,
) -> List[JointPick]:
    """Enumerate every constraint-satisfying (team_A, team_B) pairing, scored."""
    eligible_b = [c for c in candidates_b if c.win_probability >= min_win_probability_b]

    pairs = []
    for cand_a, cand_b in itertools.product(candidates_a, eligible_b):
        if cand_a.team == cand_b.team:
            continue  # can't spend the same team on both entries this week
        if _same_game(cand_a, cand_b):
            continue  # don't put the two entries on opposing sides of one game
        pairs.append(evaluate_pair(cand_a, cand_b))
    return pairs


def _format_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def build_reasoning(
    best: JointPick,
    runner_up: Optional[JointPick],
    min_win_probability_b: float = DEFAULT_MIN_WIN_PROBABILITY_B,
) -> str:
    parts = [
        f"Pick {best.team_a} for Entry A ({_format_pct(best.win_probability_a)} win prob) and "
        f"{best.team_b} for Entry B ({_format_pct(best.win_probability_b)} win prob, "
        f"clears the {min_win_probability_b:.0%} floor).",
        f"Joint outlook: both survive {_format_pct(best.both_survive)}, one survives "
        f"{_format_pct(best.one_survives)}, both eliminated {_format_pct(best.both_eliminated)} "
        f"-- objective score {best.objective:.3f}.",
    ]
    if runner_up is not None:
        parts.append(
            f"Best alternative pairing was {runner_up.team_a}/{runner_up.team_b} "
            f"(objective {runner_up.objective:.3f}), beaten by "
            f"{best.objective - runner_up.objective:.3f}."
        )
    else:
        parts.append("No other pairing satisfied every constraint.")
    return " ".join(parts)


def recommend_joint_pick(
    season: int,
    week: int,
    used_teams_a: Optional[Iterable[str]] = None,
    used_teams_b: Optional[Iterable[str]] = None,
    schedule: Optional[pd.DataFrame] = None,
    state_path_a: Path = entry_a_value.DEFAULT_STATE_PATH,
    state_path_b: Path = entry_b_hedge.DEFAULT_STATE_PATH,
    min_win_probability_b: float = DEFAULT_MIN_WIN_PROBABILITY_B,
    spread_model: Optional[wp.SpreadModel] = None,
    market_weight: float = 1.0,
    elo_games: Optional[pd.DataFrame] = None,
    team_bias_games: Optional[pd.DataFrame] = None,
) -> JointRecommendation:
    """Recommend the joint-optimal (Entry A, Entry B) pick pair for `season`/`week`.

    Args:
        used_teams_a / used_teams_b: Each entry's already-spent teams.
            Loaded from that entry's state file if omitted.
        schedule: Optional pre-loaded full-season schedule, to avoid
            re-downloading across repeated calls.
        market_weight / elo_games / team_bias_games: see
            `models.win_prob.get_win_probability`.
    """
    if schedule is None:
        schedule = nflverse_client.load_games(season=season)
    if used_teams_a is None:
        used_teams_a = entry_a_value.load_used_teams(state_path_a)
    if used_teams_b is None:
        used_teams_b = entry_b_hedge.load_used_teams(state_path_b)

    candidates_a = entry_b_hedge.build_candidates(
        season, week, used_teams_a, schedule=schedule, spread_model=spread_model,
        market_weight=market_weight, elo_games=elo_games, team_bias_games=team_bias_games,
    )
    candidates_b = entry_b_hedge.build_candidates(
        season, week, used_teams_b, schedule=schedule, spread_model=spread_model,
        market_weight=market_weight, elo_games=elo_games, team_bias_games=team_bias_games,
    )

    pairs = find_valid_pairs(
        candidates_a, candidates_b, min_win_probability_b=min_win_probability_b
    )
    if not pairs:
        raise ValueError(
            f"No valid (Entry A, Entry B) pairing found for season {season} week {week} "
            f"under the current constraints (Entry B floor {min_win_probability_b:.0%})"
        )

    pairs.sort(key=lambda p: p.objective, reverse=True)
    best = pairs[0]
    runner_up = pairs[1] if len(pairs) > 1 else None

    spread_a = next((c.spread_line for c in candidates_a if c.team == best.team_a), None)
    spread_b = next((c.spread_line for c in candidates_b if c.team == best.team_b), None)

    return JointRecommendation(
        week=week,
        pick_a=best.team_a,
        pick_b=best.team_b,
        win_probability_a=best.win_probability_a,
        win_probability_b=best.win_probability_b,
        spread_a=spread_a,
        spread_b=spread_b,
        both_survive_probability=best.both_survive,
        one_survives_probability=best.one_survives,
        both_eliminated_probability=best.both_eliminated,
        reasoning=build_reasoning(best, runner_up, min_win_probability_b),
        considered=pairs,
    )
