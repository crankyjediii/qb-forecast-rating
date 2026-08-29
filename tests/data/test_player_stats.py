"""Tests for weekly player-stat ingestion."""

from pathlib import Path

import nflreadpy as nfl
import polars as pl
import pytest

from qb_forecast_rating.data import player_stats


def sample_player_stats(season: int = 2024) -> pl.DataFrame:
    """Create a minimal valid weekly player-stat fixture."""
    return pl.DataFrame(
        {
            "player_id": ["00-001", "00-002"],
            "player_display_name": ["Example QB", "Example RB"],
            "position": ["QB", "RB"],
            "season": [season, season],
            "week": [1, 1],
            "season_type": ["REG", "REG"],
            "team": ["BUF", "BUF"],
            "opponent_team": ["ARI", "ARI"],
            "completions": [20, 0],
            "attempts": [30, 0],
            "passing_yards": [250, 0],
            "passing_tds": [2, 0],
            "passing_interceptions": [1, 0],
        }
    )


def test_validate_player_stats_accepts_valid_data() -> None:
    player_stats.validate_player_stats(sample_player_stats(), 2024)


def test_validate_player_stats_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="are empty"):
        player_stats.validate_player_stats(
            sample_player_stats().head(0),
            2024,
        )


def test_validate_player_stats_rejects_missing_columns() -> None:
    data = sample_player_stats().drop("attempts")

    with pytest.raises(ValueError, match="attempts"):
        player_stats.validate_player_stats(data, 2024)


def test_validate_player_stats_rejects_unexpected_season() -> None:
    with pytest.raises(ValueError, match="expected only season 2024"):
        player_stats.validate_player_stats(
            sample_player_stats(2023),
            2024,
        )


def test_load_player_stats_configures_loads_and_validates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = sample_player_stats()
    configured_paths: list[Path] = []
    requests: list[tuple[list[int], str]] = []

    def fake_configure(cache_dir: Path) -> None:
        configured_paths.append(cache_dir)

    def fake_load(
        seasons: list[int],
        summary_level: str,
    ) -> pl.DataFrame:
        requests.append((seasons, summary_level))
        return expected

    monkeypatch.setattr(
        player_stats,
        "configure_nflreadpy",
        fake_configure,
    )
    monkeypatch.setattr(nfl, "load_player_stats", fake_load)

    result = player_stats.load_player_stats(2024, tmp_path)

    assert result.equals(expected)
    assert configured_paths == [tmp_path]
    assert requests == [([2024], "week")]


def test_raw_player_stats_path_is_deterministic(
    tmp_path: Path,
) -> None:
    result = player_stats.raw_player_stats_path(2024, tmp_path)

    assert result == (tmp_path / "player_stats" / "player_stats_weekly_2024.parquet")


def test_ingest_player_stats_writes_compressed_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = sample_player_stats()
    cache_dir = tmp_path / "cache"
    raw_dir = tmp_path / "raw"

    def fake_load(season: int, cache_path: Path) -> pl.DataFrame:
        assert season == 2024
        assert cache_path == cache_dir
        return expected

    monkeypatch.setattr(
        player_stats,
        "load_player_stats",
        fake_load,
    )

    output_path = player_stats.ingest_player_stats(
        2024,
        cache_dir,
        raw_dir,
    )

    assert output_path == (
        raw_dir / "player_stats" / "player_stats_weekly_2024.parquet"
    )
    assert output_path.exists()
    assert pl.read_parquet(output_path).equals(expected)
