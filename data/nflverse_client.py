"""Client for loading NFL schedule/game data from the nflverse `nfldata` project.

Downloads the games.csv file that backs nflreadr's `load_schedules()` directly
from GitHub (https://raw.githubusercontent.com/nflverse/nfldata) and caches it locally,
refreshing once per day. The same file covers both completed historical
seasons and the in-progress current season, so a single daily-refreshed
cache serves both.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GAMES_CSV_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_FILENAME = "games.csv"
CACHE_MAX_AGE = timedelta(days=1)
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {"User-Agent": "survivor-picker/0.1 (+https://github.com/nflverse/nfldata)"}

OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "result",
    "spread_line",
    "away_moneyline",
    "home_moneyline",
]

_NUMERIC_COLUMNS = [
    "away_score",
    "home_score",
    "result",
    "spread_line",
    "away_moneyline",
    "home_moneyline",
]


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / CACHE_FILENAME


def _is_cache_fresh(path: Path, max_age: timedelta = CACHE_MAX_AGE) -> bool:
    if not path.exists():
        return False
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    )
    return age < max_age


def _download_games_csv(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = _cache_path(cache_dir)
    response = requests.get(
        GAMES_CSV_URL, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS
    )
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def _load_raw_games(
    cache_dir: Path = DEFAULT_CACHE_DIR, force_refresh: bool = False
) -> pd.DataFrame:
    """Return the raw nflverse games table, downloading/refreshing the cache as needed."""
    dest = _cache_path(cache_dir)
    if force_refresh or not _is_cache_fresh(dest):
        try:
            _download_games_csv(cache_dir)
        except requests.RequestException:
            if dest.exists():
                logger.warning(
                    "Failed to refresh nflverse games.csv (network error); "
                    "using stale cache at %s",
                    dest,
                )
            else:
                raise
    return pd.read_csv(dest, low_memory=False)


def _clean_games(df: pd.DataFrame) -> pd.DataFrame:
    """Select/coerce the columns survivor-picker cares about.

    Unplayed games have null away_score/home_score/result in the source data;
    those are preserved as NaN rather than raising or being dropped.
    """
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"nflverse games.csv is missing expected columns: {missing}")

    games = df[OUTPUT_COLUMNS].copy()

    for col in _NUMERIC_COLUMNS:
        games[col] = pd.to_numeric(games[col], errors="coerce")

    games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    games["week"] = pd.to_numeric(games["week"], errors="coerce").astype("Int64")
    games["gameday"] = pd.to_datetime(games["gameday"], errors="coerce")

    games = games.sort_values(
        ["season", "week", "gameday", "game_id"], na_position="last"
    ).reset_index(drop=True)
    return games


def load_games(
    season: Optional[int] = None,
    week: Optional[int] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load NFL games/schedule data from nflverse.

    Args:
        season: If given, restrict results to this season (e.g. 2024).
        week: If given, restrict results to this week number. Only applied
            when combined with games that have a week value.
        cache_dir: Directory to store the cached CSV in.
        force_refresh: Force a re-download even if the cache is fresh.

    Returns:
        DataFrame with columns: game_id, season, week, gameday, away_team,
        home_team, away_score, home_score, result, spread_line,
        away_moneyline, home_moneyline. Future/unplayed games have NaN
        score/result values rather than raising an error.
    """
    raw = _load_raw_games(cache_dir=cache_dir, force_refresh=force_refresh)
    games = _clean_games(raw)

    if season is not None:
        games = games[games["season"] == season]
    if week is not None:
        games = games[games["week"] == week]

    return games.reset_index(drop=True)


def get_available_seasons(
    cache_dir: Path = DEFAULT_CACHE_DIR, force_refresh: bool = False
) -> List[int]:
    """Return the sorted list of seasons present in the nflverse games data."""
    raw = _load_raw_games(cache_dir=cache_dir, force_refresh=force_refresh)
    seasons = pd.to_numeric(raw["season"], errors="coerce").dropna().astype(int)
    return sorted(seasons.unique().tolist())


def get_week_games(
    season: int,
    week: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return games for a single season/week."""
    return load_games(
        season=season, week=week, cache_dir=cache_dir, force_refresh=force_refresh
    )
