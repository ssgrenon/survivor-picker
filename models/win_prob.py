"""Pregame win probability model.

Two sources of truth, in priority order:

1. Moneyline-implied probability, de-vigged by normalizing the home/away
   implied probabilities to sum to 1, when both `home_moneyline` and
   `away_moneyline` are present on the game row.
2. A spread-to-win-probability logistic model, used as a fallback when
   moneylines are unavailable. It is calibrated against actual outcomes
   (`result`) of completed games pulled from several recent nflverse
   seasons via `data.nflverse_client`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data import nflverse_client

logger = logging.getLogger(__name__)

# Number of most-recent seasons (with available data) to calibrate the
# spread fallback model against by default.
DEFAULT_CALIBRATION_SEASONS = 10


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


def _has_moneylines(game_row: Mapping[str, Any]) -> bool:
    home_ml = game_row.get("home_moneyline")
    away_ml = game_row.get("away_moneyline")
    return pd.notna(home_ml) and pd.notna(away_ml)


def get_win_probability(
    game_row: Mapping[str, Any],
    team: str,
    spread_model: Optional[SpreadModel] = None,
) -> float:
    """Return `team`'s pregame win probability for `game_row`.

    `game_row` is a mapping (e.g. a pandas Series or dict) following the
    schema returned by `data.nflverse_client.load_games()`: at minimum
    home_team, away_team, spread_line, and optionally home_moneyline/
    away_moneyline.

    Uses de-vigged moneyline-implied probability when both moneylines are
    present; otherwise falls back to the calibrated spread model.
    """
    home_team = game_row["home_team"]
    away_team = game_row["away_team"]
    if team not in (home_team, away_team):
        raise ValueError(
            f"team {team!r} is not playing in this game ({away_team} @ {home_team})"
        )

    if _has_moneylines(game_row):
        home_prob, away_prob = devig_moneylines(
            float(game_row["home_moneyline"]), float(game_row["away_moneyline"])
        )
    else:
        spread_line = game_row.get("spread_line")
        if spread_line is None or pd.isna(spread_line):
            raise ValueError(
                "game_row has neither moneylines nor a spread_line to fall back on"
            )
        home_prob = spread_to_home_win_probability(float(spread_line), model=spread_model)
        away_prob = 1.0 - home_prob

    return home_prob if team == home_team else away_prob
