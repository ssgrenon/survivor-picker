"""Pregame win probability model.

The "market" probability comes from two sources of truth, in priority order:

1. Moneyline-implied probability, de-vigged by normalizing the home/away
   implied probabilities to sum to 1, when both `home_moneyline` and
   `away_moneyline` are present on the game row.
2. A spread-to-win-probability logistic model, used as a fallback when
   moneylines are unavailable. It is calibrated against actual outcomes
   (`result`) of completed games pulled from several recent nflverse
   seasons via `data.nflverse_client`.

`get_win_probability` can optionally blend that market probability with
nfelo's Elo-model win probability (`data.nfelo_client`) via `market_weight`:
`blended = market_weight * market_prob + (1 - market_weight) * elo_prob`.
When nfelo data is unavailable for a game (data lag, early-season gap, or
a game outside nfelo's coverage), it falls back to 100% market probability
for that game and logs the fallback so it's visible during backtesting.

Whenever `elo_games` is supplied, `get_win_probability` also reports each
model's implied point spread (`market_spread`/`elo_spread`, both on the
same spread-point scale via the calibrated logistic model, and both
always expressed in the home team's spread convention -- positive means
the home team is favored -- regardless of which team's win probability
was requested) and their signed difference (`divergence` = elo_spread -
market_spread: positive when nfelo rates the home team more favorably
than the market, negative when less favorably). Because both spreads use
the fixed home-team convention rather than the requested team's
perspective, the same game always produces the same `divergence` no
matter which team is being evaluated. `divergence` is a display/
awareness signal only, computed independently of `market_weight` and
never used to alter the blended probability itself.

Whenever `team_bias_games` is supplied, `get_win_probability` applies one
more, final adjustment: `team`'s own historical market-calibration bias
for the given home/away context (see `compute_team_bias`) -- a small,
shrunk, recency-weighted correction for teams the market has
systematically over- or under-rated in that context historically. This
is genuinely a scoring input (unlike `divergence`), added on top of the
market/Elo blend and clamped to [0.01, 0.99]; the adjustment itself is
also reported (`team_bias_adjustment`) so it's visible when the model is
leaning on it. It defaults to 0.0 (no adjustment) when `team_bias_games`
isn't supplied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data import nfelo_client
from data import nflverse_client

logger = logging.getLogger(__name__)

# Number of most-recent seasons (with available data) to calibrate the
# spread fallback model against by default.
DEFAULT_CALIBRATION_SEASONS = 10

# Valid market_weight values: fraction of the blended probability drawn
# from the market probability, with the remainder (1 - market_weight)
# drawn from nfelo's Elo win probability.
VALID_MARKET_WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)

# Team-bias defaults (see compute_team_bias): per-season recency-decay
# multiplier, shrinkage constant (larger = more skepticism of small
# samples), and the +/- cap on the final adjustment.
DEFAULT_BIAS_DECAY_PER_SEASON = 0.85
DEFAULT_BIAS_SHRINKAGE_K = 15.0
DEFAULT_BIAS_MAX_ADJUSTMENT = 0.04

# How often get_team_bias() recomputes, per (games_df identity, team,
# is_home) key -- this is a slow-ish historical calculation, so it's
# cached rather than redone on every get_win_probability call.
BIAS_CACHE_MAX_AGE = timedelta(days=1)


@dataclass(frozen=True)
class SpreadModel:
    """Logistic model: P(home team wins) = sigmoid(intercept + slope * spread_line)."""

    intercept: float
    slope: float

    def home_win_probability(self, spread_line: float) -> float:
        z = self.intercept + self.slope * spread_line
        return 1.0 / (1.0 + np.exp(-z))


_spread_model_cache: Optional[SpreadModel] = None


def moneyline_to_implied_prob(moneyline: float) -> float:
    """Convert a single American moneyline into its raw (vig-included) implied probability."""
    if moneyline == 0:
        raise ValueError("moneyline cannot be 0")
    if moneyline > 0:
        return 100.0 / (moneyline + 100.0)
    return -moneyline / (-moneyline + 100.0)


def devig_moneylines(home_moneyline: float, away_moneyline: float) -> Tuple[float, float]:
    """Remove the bookmaker's vig by normalizing implied probabilities to sum to 1.

    Returns (home_win_prob, away_win_prob).
    """
    home_raw = moneyline_to_implied_prob(home_moneyline)
    away_raw = moneyline_to_implied_prob(away_moneyline)
    total = home_raw + away_raw
    return home_raw / total, away_raw / total


def _fit_logistic_1d(
    x: np.ndarray, y: np.ndarray, iterations: int = 50, tol: float = 1e-10
) -> SpreadModel:
    """Fit P(y=1) = sigmoid(b0 + b1*x) via Newton-Raphson (IRLS).

    A hand-rolled fit avoids pulling in sklearn/statsmodels for a single
    two-parameter logistic regression.
    """
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iterations):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        weights = p * (1 - p)
        gradient = X.T @ (y - p)
        hessian = -(X.T * weights) @ X
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            logger.warning("Hessian singular during spread model fit; stopping early")
            break
        beta = beta - delta
        if np.max(np.abs(delta)) < tol:
            break
    return SpreadModel(intercept=float(beta[0]), slope=float(beta[1]))


def calibrate_spread_model(
    seasons: Optional[Sequence[int]] = None,
    n_recent_seasons: int = DEFAULT_CALIBRATION_SEASONS,
) -> SpreadModel:
    """Fit the spread -> home-win-probability logistic model on completed historical games.

    Args:
        seasons: Explicit seasons to calibrate on. If omitted, the most
            recent `n_recent_seasons` available seasons are used.
        n_recent_seasons: How many recent seasons to use when `seasons`
            is not given.
    """
    if seasons is None:
        available = nflverse_client.get_available_seasons()
        seasons = available[-n_recent_seasons:]

    games = pd.concat(
        [nflverse_client.load_games(season=season) for season in seasons],
        ignore_index=True,
    )
    completed = games.dropna(subset=["spread_line", "home_score", "away_score", "result"])
    completed = completed[completed["result"] != 0]  # drop ties: no win/loss label
    if completed.empty:
        raise ValueError("No completed games with spread_line found to calibrate against")

    x = completed["spread_line"].to_numpy(dtype=float)
    y = (completed["result"] > 0).to_numpy(dtype=float)
    return _fit_logistic_1d(x, y)


def get_spread_model(force_refresh: bool = False) -> SpreadModel:
    """Return the process-cached calibrated spread model, fitting it lazily on first use."""
    global _spread_model_cache
    if _spread_model_cache is None or force_refresh:
        _spread_model_cache = calibrate_spread_model()
    return _spread_model_cache


def spread_to_home_win_probability(
    spread_line: float, model: Optional[SpreadModel] = None
) -> float:
    """Convert a spread_line (positive = home favored) into a home team win probability."""
    model = model or get_spread_model()
    return float(model.home_win_probability(spread_line))


def home_win_probability_to_spread_line(
    home_win_probability: float, model: Optional[SpreadModel] = None
) -> float:
    """Inverse of `spread_to_home_win_probability`: map a home-team win probability back
    to its equivalent spread_line, via the same calibrated logistic model. Used to put a
    non-spread probability (e.g. nfelo's) on the same spread-point scale as the market's
    own spread_line, for `model_divergence` comparisons.
    """
    model = model or get_spread_model()
    p = min(max(float(home_win_probability), 1e-9), 1.0 - 1e-9)
    logit = float(np.log(p / (1.0 - p)))
    return (logit - model.intercept) / model.slope


def _has_moneylines(game_row: Mapping[str, Any]) -> bool:
    home_ml = game_row.get("home_moneyline")
    away_ml = game_row.get("away_moneyline")
    return pd.notna(home_ml) and pd.notna(away_ml)


def _market_home_win_probability(
    game_row: Mapping[str, Any],
    spread_model: Optional[SpreadModel],
) -> float:
    """The market-derived home-team win probability: de-vigged moneylines, or spread-model fallback."""
    if _has_moneylines(game_row):
        home_prob, _away_prob = devig_moneylines(
            float(game_row["home_moneyline"]), float(game_row["away_moneyline"])
        )
        return home_prob

    spread_line = game_row.get("spread_line")
    if spread_line is None or pd.isna(spread_line):
        raise ValueError(
            "game_row has neither moneylines nor a spread_line to fall back on"
        )
    return spread_to_home_win_probability(float(spread_line), model=spread_model)


def _home_market_spread(
    game_row: Mapping[str, Any],
    home_market_prob: float,
    spread_model: Optional[SpreadModel],
) -> float:
    """The market-implied point spread, in the home team's convention (positive = home
    favored): the game's actual posted spread_line when available (it's already in
    spread-point units), else the market probability's logistic-equivalent spread.
    """
    spread_line = game_row.get("spread_line")
    if spread_line is not None and not pd.isna(spread_line):
        return float(spread_line)
    return home_win_probability_to_spread_line(home_market_prob, spread_model)


def compute_team_bias(
    team: str,
    is_home: bool,
    games_df: pd.DataFrame,
    decay_per_season: float = DEFAULT_BIAS_DECAY_PER_SEASON,
    shrinkage_k: float = DEFAULT_BIAS_SHRINKAGE_K,
    max_adjustment: float = DEFAULT_BIAS_MAX_ADJUSTMENT,
    spread_model: Optional[SpreadModel] = None,
) -> float:
    """`team`'s historical market-calibration bias in the given home/away context.

    For every completed game in `games_df` where `team` played at home (if
    `is_home`) or away (otherwise), compares the actual outcome to the
    market-implied win probability at the time (de-vigged moneylines, or
    the calibrated spread model when moneylines aren't posted -- see
    `_market_home_win_probability`). Ties are excluded (no win/loss
    label), matching `calibrate_spread_model`. Games with neither
    moneylines nor a spread_line are skipped.

    The per-game residuals (actual_win - market_prob) are combined with
    exponential recency weighting -- `decay_per_season ** season_age`,
    where `season_age` is measured against the most recent season present
    in `games_df` -- into a weighted average, then shrunk toward zero by
    `n / (n + shrinkage_k)` (n = the weighted sample size, i.e. the sum of
    the weights) so teams with a thin history pull close to no
    adjustment, and finally clamped to +/- `max_adjustment`.

    Returns 0.0 if `team` has no usable games in `games_df` for this
    context.
    """
    side_col = "home_team" if is_home else "away_team"
    team_games = games_df[games_df[side_col] == team]
    team_games = team_games.dropna(subset=["home_score", "away_score"])
    if team_games.empty:
        return 0.0

    latest_season = games_df["season"].max()

    weighted_sum = 0.0
    weight_total = 0.0
    for _, row in team_games.iterrows():
        result = float(row["home_score"]) - float(row["away_score"])
        if result == 0:
            continue  # tie: no win/loss label, matching calibrate_spread_model

        home_won = result > 0
        actual_win = 1.0 if (home_won == is_home) else 0.0

        try:
            home_market_prob = _market_home_win_probability(row, spread_model)
        except ValueError:
            continue  # neither moneylines nor a spread_line for this historical game
        market_prob = home_market_prob if is_home else 1.0 - home_market_prob

        season_age = latest_season - row["season"]
        weight = decay_per_season ** season_age

        weighted_sum += weight * (actual_win - market_prob)
        weight_total += weight

    if weight_total <= 0:
        return 0.0

    weighted_avg_residual = weighted_sum / weight_total
    shrunk = weighted_avg_residual * (weight_total / (weight_total + shrinkage_k))
    return float(max(-max_adjustment, min(max_adjustment, shrunk)))


_team_bias_cache: Dict[Tuple[str, bool], float] = {}
_team_bias_cache_computed_at: Optional[datetime] = None
_team_bias_cache_games_id: Optional[int] = None


def get_team_bias(
    team: str,
    is_home: bool,
    games_df: pd.DataFrame,
    decay_per_season: float = DEFAULT_BIAS_DECAY_PER_SEASON,
    shrinkage_k: float = DEFAULT_BIAS_SHRINKAGE_K,
    max_adjustment: float = DEFAULT_BIAS_MAX_ADJUSTMENT,
    spread_model: Optional[SpreadModel] = None,
    force_refresh: bool = False,
) -> float:
    """Process-cached wrapper around `compute_team_bias`.

    Recomputed at most once per `BIAS_CACHE_MAX_AGE` (default: daily), or
    immediately whenever a different `games_df` object is passed in (so
    tests/callers using distinct DataFrames never see another caller's
    stale entries).
    """
    global _team_bias_cache, _team_bias_cache_computed_at, _team_bias_cache_games_id

    now = datetime.now(timezone.utc)
    games_id = id(games_df)
    stale = (
        force_refresh
        or _team_bias_cache_computed_at is None
        or games_id != _team_bias_cache_games_id
        or now - _team_bias_cache_computed_at > BIAS_CACHE_MAX_AGE
    )
    if stale:
        _team_bias_cache = {}
        _team_bias_cache_computed_at = now
        _team_bias_cache_games_id = games_id

    key = (team, is_home)
    if key not in _team_bias_cache:
        _team_bias_cache[key] = compute_team_bias(
            team,
            is_home,
            games_df,
            decay_per_season=decay_per_season,
            shrinkage_k=shrinkage_k,
            max_adjustment=max_adjustment,
            spread_model=spread_model,
        )
    return _team_bias_cache[key]


@dataclass(frozen=True)
class WinProbabilityResult:
    """Result of `get_win_probability`: the blended probability plus model-divergence context.

    `market_spread`/`elo_spread` are always in the home team's spread
    convention (positive = home favored), regardless of which team's win
    probability was requested -- so `divergence` (= elo_spread -
    market_spread) is identical for the same game no matter which team is
    being evaluated. `market_spread` is always populated. `elo_spread`/
    `divergence` are None whenever `elo_games` wasn't supplied, or nfelo
    has no rating for this particular game -- `divergence` is a display/
    awareness signal only, computed independently of `market_weight`,
    never a scoring input.
    """

    win_probability: float
    market_spread: float
    elo_spread: Optional[float]
    divergence: Optional[float]
    team_bias_adjustment: float = 0.0


def get_win_probability(
    game_row: Mapping[str, Any],
    team: str,
    market_weight: float = 1.0,
    spread_model: Optional[SpreadModel] = None,
    elo_games: Optional[pd.DataFrame] = None,
    team_bias_games: Optional[pd.DataFrame] = None,
    team_bias_decay_per_season: float = DEFAULT_BIAS_DECAY_PER_SEASON,
    team_bias_shrinkage_k: float = DEFAULT_BIAS_SHRINKAGE_K,
    team_bias_max_adjustment: float = DEFAULT_BIAS_MAX_ADJUSTMENT,
) -> WinProbabilityResult:
    """Return `team`'s pregame win probability for `game_row`, plus model-divergence context.

    `game_row` is a mapping (e.g. a pandas Series or dict) following the
    schema returned by `data.nflverse_client.load_games()`: at minimum
    game_id, home_team, away_team, spread_line, and optionally
    home_moneyline/away_moneyline.

    The market component uses de-vigged moneyline-implied probability
    when both moneylines are present; otherwise it falls back to the
    calibrated spread model.

    Args:
        market_weight: Fraction of `.win_probability` drawn from the
            market probability; the rest (1 - market_weight) is drawn
            from nfelo's Elo win probability (`elo_games`, see
            `data.nfelo_client.load_nfelo_games`). Must be one of
            `VALID_MARKET_WEIGHTS`.
        elo_games: Pre-loaded nfelo table from
            `data.nfelo_client.load_nfelo_games()`. Required (non-None)
            whenever `market_weight < 1.0`. When supplied (at any
            `market_weight`), it's also used to compute `.elo_spread`/
            `.divergence` for display, independent of the blend. If nfelo
            has no rating for this game, `.win_probability` falls back to
            100% market probability and `.elo_spread`/`.divergence` come
            back None; a warning identifying the game is logged either
            way, so fallbacks are visible during backtesting.
        team_bias_games: Pre-loaded historical games table (schema as
            returned by `data.nflverse_client.load_games()`, ideally
            covering 5+ seasons) used to compute `team`'s historical
            market-calibration bias for this home/away context (see
            `compute_team_bias`). When supplied, the bias is added to
            `.win_probability` as a final step (clamped to [0.01, 0.99])
            and reported as `.team_bias_adjustment`. Defaults to no
            adjustment (0.0) when omitted.
        team_bias_decay_per_season / team_bias_shrinkage_k /
            team_bias_max_adjustment: Passed through to `compute_team_bias`
            (via the cached `get_team_bias`) when `team_bias_games` is
            supplied.

    Returns:
        A `WinProbabilityResult` (`.win_probability`, `.market_spread`,
        `.elo_spread`, `.divergence`, `.team_bias_adjustment`).
    """
    if market_weight not in VALID_MARKET_WEIGHTS:
        raise ValueError(f"market_weight must be one of {VALID_MARKET_WEIGHTS}, got {market_weight!r}")

    home_team = game_row["home_team"]
    away_team = game_row["away_team"]
    if team not in (home_team, away_team):
        raise ValueError(
            f"team {team!r} is not playing in this game ({away_team} @ {home_team})"
        )

    home_market_prob = _market_home_win_probability(game_row, spread_model)
    market_prob = home_market_prob if team == home_team else 1.0 - home_market_prob
    market_spread = _home_market_spread(game_row, home_market_prob, spread_model)

    win_probability = market_prob
    elo_spread = None
    divergence = None

    if elo_games is not None:
        game_id = game_row.get("game_id")
        elo_prob = nfelo_client.get_team_elo_win_probability(elo_games, game_id, team)
        if elo_prob is None:
            logger.warning(
                "No nfelo rating for game_id=%r team=%r (season=%r week=%r); "
                "falling back to 100%% market probability (if a blend was requested) and "
                "model_divergence cannot be computed for this game.",
                game_id,
                team,
                game_row.get("season"),
                game_row.get("week"),
            )
        else:
            home_elo_prob = elo_prob if team == home_team else 1.0 - elo_prob
            elo_spread = home_win_probability_to_spread_line(home_elo_prob, spread_model)
            # Signed and home-team-relative (not team-relative), so the same game always
            # produces the same divergence regardless of which team is being evaluated.
            divergence = elo_spread - market_spread
            if market_weight < 1.0:
                win_probability = market_weight * market_prob + (1.0 - market_weight) * elo_prob
    elif market_weight < 1.0:
        raise ValueError("elo_games is required when market_weight < 1.0")

    team_bias_adjustment = 0.0
    if team_bias_games is not None:
        is_home_team = team == home_team
        team_bias_adjustment = get_team_bias(
            team,
            is_home_team,
            team_bias_games,
            decay_per_season=team_bias_decay_per_season,
            shrinkage_k=team_bias_shrinkage_k,
            max_adjustment=team_bias_max_adjustment,
            spread_model=spread_model,
        )
        win_probability = min(0.99, max(0.01, win_probability + team_bias_adjustment))

    return WinProbabilityResult(
        win_probability=win_probability,
        market_spread=market_spread,
        elo_spread=elo_spread,
        divergence=divergence,
        team_bias_adjustment=team_bias_adjustment,
    )
