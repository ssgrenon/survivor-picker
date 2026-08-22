import pandas as pd
import pytest

from data import nfelo_client as nec

RAW_COLUMNS = ["game_id", "nfelo_home_probability_close", "nfelo_home_probability_open"]


def _sample_raw_df() -> pd.DataFrame:
    rows = [
        {
            "game_id": "2023_01_DET_KC",
            "nfelo_home_probability_close": 0.62,
            "nfelo_home_probability_open": 0.60,
        },
        {
            # Relocated-franchise abbreviations: nfelo's stable OAK/LAR should
            # normalize to nflverse's LV/LA.
            "game_id": "2023_02_OAK_LAR",
            "nfelo_home_probability_close": 0.55,
            "nfelo_home_probability_open": 0.53,
        },
        {
            # No close probability -- falls back to the open probability.
            "game_id": "2023_03_SEA_SF",
            "nfelo_home_probability_close": None,
            "nfelo_home_probability_open": 0.70,
        },
        {
            # Neither probability available -- dropped entirely.
            "game_id": "2023_04_MIA_NE",
            "nfelo_home_probability_close": None,
            "nfelo_home_probability_open": None,
        },
        {
            # Malformed game_id -- dropped entirely.
            "game_id": "not-a-valid-game-id",
            "nfelo_home_probability_close": 0.5,
            "nfelo_home_probability_open": 0.5,
        },
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


@pytest.fixture
def cached_csv(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dest = cache_dir / nec.CACHE_FILENAME
    _sample_raw_df().to_csv(dest, index=False)

    def _fail_download(*_args, **_kwargs):
        raise AssertionError("network download should not be triggered by a fresh cache")

    monkeypatch.setattr(nec, "_download_nfelo_games_csv", _fail_download)
    return cache_dir


def test_load_nfelo_games_returns_two_rows_per_game(cached_csv):
    games = nec.load_nfelo_games(cache_dir=cached_csv)
    assert list(games.columns) == nec.OUTPUT_COLUMNS
    # 3 usable games (one dropped for missing probabilities, one for a
    # malformed game_id) * 2 rows (home + away) each.
    assert len(games) == 6


def test_load_nfelo_games_normalizes_relocated_team_abbreviations(cached_csv):
    games = nec.load_nfelo_games(cache_dir=cached_csv)
    row = games[(games["season"] == 2023) & (games["week"] == 2)]
    assert set(row["team"]) == {"LV", "LA"}
    assert set(row["opponent"]) == {"LV", "LA"}
    assert set(row["game_id"]) == {"2023_02_LV_LA"}


def test_load_nfelo_games_home_away_probabilities_sum_to_one(cached_csv):
    games = nec.load_nfelo_games(cache_dir=cached_csv)
    row = games[games["game_id"] == "2023_01_DET_KC"]
    assert len(row) == 2
    assert row["elo_win_probability"].sum() == pytest.approx(1.0)
    home_row = row[row["is_home"]].iloc[0]
    assert home_row["team"] == "KC"
    assert home_row["elo_win_probability"] == pytest.approx(0.62)


def test_load_nfelo_games_falls_back_to_open_probability_when_close_missing(cached_csv):
    games = nec.load_nfelo_games(cache_dir=cached_csv)
    row = games[games["game_id"] == "2023_03_SEA_SF"]
    home_row = row[row["is_home"]].iloc[0]
    assert home_row["team"] == "SF"
    assert home_row["elo_win_probability"] == pytest.approx(0.70)


def test_load_nfelo_games_drops_games_with_no_usable_probability(cached_csv):
    games = nec.load_nfelo_games(cache_dir=cached_csv)
    assert "2023_04_MIA_NE" not in set(games["game_id"])


def test_load_nfelo_games_filters_by_season_and_week(cached_csv):
    games = nec.load_nfelo_games(season=2023, week=1, cache_dir=cached_csv)
    assert len(games) == 2
    assert set(games["game_id"]) == {"2023_01_DET_KC"}


def test_get_team_elo_win_probability_returns_expected_value(cached_csv):
    games = nec.load_nfelo_games(cache_dir=cached_csv)
    assert nec.get_team_elo_win_probability(games, "2023_01_DET_KC", "KC") == pytest.approx(0.62)
    assert nec.get_team_elo_win_probability(games, "2023_01_DET_KC", "DET") == pytest.approx(0.38)


def test_get_team_elo_win_probability_returns_none_when_missing(cached_csv):
    games = nec.load_nfelo_games(cache_dir=cached_csv)
    assert nec.get_team_elo_win_probability(games, "2099_01_XXX_YYY", "XXX") is None


def test_stale_cache_triggers_refresh(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dest = cache_dir / nec.CACHE_FILENAME
    _sample_raw_df().to_csv(dest, index=False)

    import os
    import time

    stale_time = time.time() - nec.CACHE_MAX_AGE.total_seconds() - 3600
    os.utime(dest, (stale_time, stale_time))

    called = {"count": 0}

    def _fake_download(cache_dir):
        called["count"] += 1
        return dest

    monkeypatch.setattr(nec, "_download_nfelo_games_csv", _fake_download)
    nec.load_nfelo_games(cache_dir=cache_dir)
    assert called["count"] == 1
