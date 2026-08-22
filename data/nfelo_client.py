"""Client for loading nfelo Elo-model game ratings/win probabilities.

Source discovery: nfelo (https://www.nfeloapp.com) is greerreNFL's public
NFL Elo rating and win-probability model. Its underlying data is published
in the open-source GitHub repo https://github.com/greerreNFL/nfelo, whose
`output_data/nfelo_games.csv` is regenerated as part of that repo's own
pipeline and covers every game from the 2009 season through the
in-progress current season (verified against the live file as of this
writing: seasons 2009-2026). That CSV is fetched directly from GitHub raw
content, mirroring how `data.nflverse_client` fetches nflverse's
`games.csv` -- no API key or scraping of nfeloapp.com itself is required.

The raw file has no separate home_team/away_team columns; each game is
identified only by a `game_id` string of the form "YYYY_WW_AWAY_HOME"
(e.g. "2023_01_DET_KC"), the same format nflverse's `games.csv` uses.

Team abbreviation caveat: nfelo assigns one stable abbreviation per
franchise across relocations, while nflverse uses the season-accurate
abbreviation. Confirmed (by diffing team sets season-by-season, 2019-2025)
for the two active relocated franchises:
  - Raiders: nfelo always uses "OAK"; nflverse uses "LV" from 2020 on.
  - Rams: nfelo always uses "LAR"; nflverse uses "LA" from 2016 on.
`TEAM_ALIASES` below translates nfelo's abbreviation to nflverse's so the
reconstructed `game_id` matches nflverse's `games.csv` and can be joined
on directly. This mapping is unconditional (not season-gated), so it does
NOT correctly translate the Rams' pre-2016 St. Louis seasons (where
nflverse uses "STL" but nfelo's file, per its "one stable abbreviation"
design, is expected to still say "LAR") -- games from that window simply
won't match by game_id and will fall back to market-only probability,
same as any other missing-data case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NFELO_GAMES_CSV_URL = (
    "https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv"
)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_FILENAME = "nfelo_games.csv"
CACHE_MAX_AGE = timedelta(days=1)
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {"User-Agent": "survivor-picker/0.1 (+https://github.com/greerreNFL/nfelo)"}

# nfelo's abbreviation -> nflverse's abbreviation, for franchises whose
# abbreviation changed after a relocation. See module docstring caveat.
TEAM_ALIASES = {
    "OAK": "LV",
    "LAR": "LA",
}

OUTPUT_COLUMNS = ["game_id", "season", "week", "team", "opponent", "is_home", "elo_win_probability"]


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / CACHE_FILENAME


def _is_cache_fresh(path: Path, max_age: timedelta = CACHE_MAX_AGE) -> bool:
    if not path.exists():
        return False
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age < max_age


def _download_nfelo_games_csv(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = _cache_path(cache_dir)
    response = requests.get(NFELO_GAMES_CSV_URL, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def _load_raw_nfelo_games(cache_dir: Path = DEFAULT_CACHE_DIR, force_refresh: bool = False) -> pd.DataFrame:
    """Return the raw nfelo games table, downloading/refreshing the cache as needed."""
    dest = _cache_path(cache_dir)
    if force_refresh or not _is_cache_fresh(dest):
        try:
            _download_nfelo_games_csv(cache_dir)
        except requests.RequestException:
            if dest.exists():
                logger.warning(
                    "Failed to refresh nfelo_games.csv (network error); using stale cache at %s",
                    dest,
                )
            else:
                raise
    return pd.read_csv(dest, low_memory=False)


@dataclass(frozen=True)
class _ParsedGameId:
    season: int
    week: int
    away_team: str
    home_team: str


def _parse_game_id(raw_game_id: str) -> Optional[_ParsedGameId]:
    """Parse nfelo's "YYYY_WW_AWAY_HOME" game_id, normalizing team abbreviations."""
    parts = str(raw_game_id).split("_")
    if len(parts) != 4:
        return None
    season_str, week_str, away, home = parts
    try:
        season = int(season_str)
        week = int(week_str)
    except ValueError:
        return None
    return _ParsedGameId(
        season=season,
        week=week,
        away_team=TEAM_ALIASES.get(away, away),
        home_team=TEAM_ALIASES.get(home, home),
    )


def _clean_nfelo_games(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape the raw one-row-per-game nfelo table into a long, one-row-per-team table.

    Keyed by (season, week, team) with a reconstructed, nflverse-compatible
    `game_id`, so it can be joined against `nflverse_client.load_games()`
    output on either `game_id` or `season`/`week`/`team`.
    """
    required = {"game_id", "nfelo_home_probability_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"nfelo_games.csv is missing expected columns: {sorted(missing)}")

    home_prob_col = "nfelo_home_probability_close"
    home_prob_fallback_col = "nfelo_home_probability_open"
    has_fallback = home_prob_fallback_col in df.columns

    rows = []
    for _, raw in df.iterrows():
        parsed = _parse_game_id(raw["game_id"])
        if parsed is None:
            continue

        home_prob = raw.get(home_prob_col)
        if pd.isna(home_prob) and has_fallback:
            home_prob = raw.get(home_prob_fallback_col)
        if pd.isna(home_prob):
            continue
        home_prob = float(home_prob)

        game_id = f"{parsed.season}_{parsed.week:02d}_{parsed.away_team}_{parsed.home_team}"
        rows.append(
            {
                "game_id": game_id,
                "season": parsed.season,
                "week": parsed.week,
                "team": parsed.home_team,
                "opponent": parsed.away_team,
                "is_home": True,
                "elo_win_probability": home_prob,
            }
        )
        rows.append(
            {
                "game_id": game_id,
                "season": parsed.season,
                "week": parsed.week,
                "team": parsed.away_team,
                "opponent": parsed.home_team,
                "is_home": False,
                "elo_win_probability": 1.0 - home_prob,
            }
        )

    games = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    games["season"] = games["season"].astype("Int64")
    games["week"] = games["week"].astype("Int64")
    return games.reset_index(drop=True)


def load_nfelo_games(
    season: Optional[int] = None,
    week: Optional[int] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load nfelo Elo win probabilities, one row per team per game.

    Args:
        season: If given, restrict results to this season.
        week: If given, restrict results to this week number.
        cache_dir: Directory to store the cached CSV in.
        force_refresh: Force a re-download even if the cache is fresh.

    Returns:
        DataFrame with columns: game_id, season, week, team, opponent,
        is_home, elo_win_probability. `game_id` and team abbreviations are
        normalized to match `data.nflverse_client.load_games()`'s
        conventions (see module docstring for the relocation-alias
        caveat).
    """
    raw = _load_raw_nfelo_games(cache_dir=cache_dir, force_refresh=force_refresh)
    games = _clean_nfelo_games(raw)

    if season is not None:
        games = games[games["season"] == season]
    if week is not None:
        games = games[games["week"] == week]

    return games.reset_index(drop=True)


def get_team_elo_win_probability(elo_games: pd.DataFrame, game_id: str, team: str) -> Optional[float]:
    """Look up `team`'s nfelo win probability for `game_id` in a table from `load_nfelo_games()`.

    Returns None if no matching row exists (e.g. the game hasn't been
    rated yet, or -- for pre-2016 St. Louis Rams games -- the relocation-
    alias caveat above means it never will).
    """
    matches = elo_games[(elo_games["game_id"] == game_id) & (elo_games["team"] == team)]
    if matches.empty:
        return None
    return float(matches.iloc[0]["elo_win_probability"])
