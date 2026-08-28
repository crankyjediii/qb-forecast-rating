"""Tests for play-by-play ingestion."""

from pathlib import Path

import nflreadpy as nfl
import polars as pl
import pytest

from qb_forecast_rating.data import pbp


def sample_pbp(season: int = 2024) -> pl.DataFrame:
    """Create a minimal valid play-by-play fixture."""
    return pl.DataFrame(
        {
            "season": [season, season],
            "week": [1, 1],
            "game_id": ["2024_01_ARI_BUF", "2024_01_ARI_BUF"],
            "play_id": [40.0, 61.0],
            "passer_player_id": ["00-001", None],
            "passer_player_name": ["Example QB", None],
            "rusher_player_id": [None, "00-001"],
            "rusher_player_name": [None, "Example QB"],
            "qb_epa": [0.25, 0.40],
            "epa": [0.25, 0.40],
            "cpoe": [3.5, None],
            "play_type": ["pass", "run"],
            "qb_dropback": [1.0, 1.0],
            "qb_scramble": [0.0, 1.0],
            "sack": [0.0, 0.0],
        }
    )


def test_validate_pbp_accepts_valid_data() -> None:
    pbp.validate_pbp(sample_pbp(), 2024)


def test_validate_pbp_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="is empty"):
        pbp.validate_pbp(sample_pbp().head(0), 2024)


def test_validate_pbp_rejects_missing_columns() -> None:
    data = sample_pbp().drop("qb_epa")

    with pytest.raises(ValueError, match="qb_epa"):
        pbp.validate_pbp(data, 2024)


def test_validate_pbp_rejects_unexpected_season() -> None:
    with pytest.raises(ValueError, match="expected only season 2024"):
        pbp.validate_pbp(sample_pbp(2023), 2024)


def test_configure_nflreadpy_uses_filesystem_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded: dict[str, object] = {}

    def fake_update_config(**kwargs: object) -> None:
        recorded.update(kwargs)

    monkeypatch.setattr(pbp, "update_config", fake_update_config)

    pbp.configure_nflreadpy(tmp_path)

    assert recorded == {
        "cache_mode": "filesystem",
        "cache_dir": tmp_path,
        "verbose": False,
    }


def test_load_pbp_configures_loads_and_validates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = sample_pbp()
    configured_paths: list[Path] = []
    requested_seasons: list[list[int]] = []

    def fake_configure(cache_dir: Path) -> None:
        configured_paths.append(cache_dir)

    def fake_load(seasons: list[int]) -> pl.DataFrame:
        requested_seasons.append(seasons)
        return expected

    monkeypatch.setattr(pbp, "configure_nflreadpy", fake_configure)
    monkeypatch.setattr(nfl, "load_pbp", fake_load)

    result = pbp.load_pbp(2024, tmp_path)

    assert result.equals(expected)
    assert configured_paths == [tmp_path]
    assert requested_seasons == [[2024]]


def test_ingest_pbp_writes_compressed_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = sample_pbp()
    cache_dir = tmp_path / "cache"
    raw_dir = tmp_path / "raw"

    def fake_load(season: int, cache_path: Path) -> pl.DataFrame:
        assert season == 2024
        assert cache_path == cache_dir
        return expected

    monkeypatch.setattr(pbp, "load_pbp", fake_load)

    output_path = pbp.ingest_pbp(2024, cache_dir, raw_dir)

    assert output_path == raw_dir / "pbp" / "pbp_2024.parquet"
    assert output_path.exists()
    assert pl.read_parquet(output_path).equals(expected)
