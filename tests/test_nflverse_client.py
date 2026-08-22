import pandas as pd
import pytest

from data import nflverse_client as nc

SAMPLE_CSV_COLUMNS = nc.OUTPUT_COLUMNS + ["game_type", "weekday"]


def _sample_df() -> pd.DataFrame:
    rows = [
        {
            "game_id": "2023_01_DET_KC",
            "season": 2023,
            "week": 1,
            "gameday": "2023-09-07",
            "away_team": "DET",
            "home_team": "KC",
            "away_score": 21,
            "home_score": 20,
            "result": -1,
            "spread_line": -5.5,
            "away_moneyline": 220,
            "home_moneyline": -270,
            "game_type": "REG",
            "weekday": "Thursday",
        },
        {
            # Future/unplayed game: scores and result are null.
            "game_id": "2024_01_BAL_KC",
            "season": 2024,
            "week": 1,
            "gameday": "2024-09-05",
            "away_team": "BAL",
            "home_team": "KC",
            "away_score": None,
            "home_score": None,
            "result": None,
            "spread_line": -2.5,
            "away_moneyline": 130,
            "home_moneyline": -155,
            "game_type": "REG",
            "weekday": "Thursday",
        },
    ]
    return pd.DataFrame(rows, columns=SAMPLE_CSV_COLUMNS)


@pytest.fixture
def cached_csv(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dest = cache_dir / nc.CACHE_FILENAME
    _sample_df().to_csv(dest, index=False)

    def _fail_download(*_args, **_kwargs):
        raise AssertionError("network download should not be triggered by a fresh cache")

    monkeypatch.setattr(nc, "_download_games_csv", _fail_download)
    return cache_dir


def test_load_games_returns_expected_columns(cached_csv):
    games = nc.load_games(cache_dir=cached_csv)
    assert list(games.columns) == nc.OUTPUT_COLUMNS
    assert len(games) == 2


def test_load_games_handles_null_scores_without_erroring(cached_csv):
    games = nc.load_games(cache_dir=cached_csv)
    future_game = games[games["game_id"] == "2024_01_BAL_KC"].iloc[0]
    assert pd.isna(future_game["away_score"])
    assert pd.isna(future_game["home_score"])
    assert pd.isna(future_game["result"])


def test_load_games_filters_by_season_and_week(cached_csv):
    games = nc.load_games(season=2023, cache_dir=cached_csv)
    assert len(games) == 1
    assert games.iloc[0]["game_id"] == "2023_01_DET_KC"

    games = nc.get_week_games(2024, 1, cache_dir=cached_csv)
    assert len(games) == 1
    assert games.iloc[0]["game_id"] == "2024_01_BAL_KC"


def test_get_available_seasons(cached_csv):
    seasons = nc.get_available_seasons(cache_dir=cached_csv)
    assert seasons == [2023, 2024]


def test_stale_cache_triggers_refresh(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dest = cache_dir / nc.CACHE_FILENAME
    _sample_df().to_csv(dest, index=False)

    import os
    import time

    stale_time = time.time() - nc.CACHE_MAX_AGE.total_seconds() - 3600
    os.utime(dest, (stale_time, stale_time))

    called = {"count": 0}

    def _fake_download(cache_dir):
        called["count"] += 1
        return dest

    monkeypatch.setattr(nc, "_download_games_csv", _fake_download)
    nc.load_games(cache_dir=cache_dir)
    assert called["count"] == 1


def test_force_refresh_triggers_download_even_when_fresh(cached_csv, monkeypatch):
    called = {"count": 0}

    def _fake_download(cache_dir):
        called["count"] += 1
        return cached_csv / nc.CACHE_FILENAME

    monkeypatch.setattr(nc, "_download_games_csv", _fake_download)
    nc.load_games(cache_dir=cached_csv, force_refresh=True)
    assert called["count"] == 1
