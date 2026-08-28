"""Tests for quarterback-game metric aggregation."""

from pathlib import Path

import polars as pl
import pytest

from qb_forecast_rating.data import qb_games


def sample_game_source() -> pl.DataFrame:
    """Create actions for two quarterback-game rows."""
    return pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "season_type": ["REG"] * 4,
            "week": [1, 1, 1, 1],
            "game_id": [
                "2024_01_ARI_BUF",
                "2024_01_ARI_BUF",
                "2024_01_ARI_BUF",
                "2024_01_BAL_KC",
            ],
            "game_date": ["2024-09-08"] * 3 + ["2024-09-05"],
            "posteam": ["ARI", "ARI", "ARI", "BAL"],
            "defteam": ["BUF", "BUF", "BUF", "KC"],
            "qb_id": ["00-001", "00-001", "00-001", "00-002"],
            "qb_name": ["Example QB", "Example QB", "Example QB", "Other QB"],
            "qb_epa": [0.25, -0.50, 0.40, -0.10],
            "cpoe": [3.5, None, None, -2.0],
            "sack": [0.0, 1.0, 0.0, 0.0],
            "qb_scramble": [0.0, 0.0, 1.0, 0.0],
            "success": [1.0, 0.0, 1.0, 0.0],
            "action_type": ["pass", "sack", "scramble", "pass"],
        }
    )


def test_build_qb_games_aggregates_metrics() -> None:
    games = qb_games.build_qb_games(sample_game_source())
    first = games.filter(pl.col("qb_id") == "00-001").row(0, named=True)

    assert games.height == 2
    assert first["dropbacks"] == 3
    assert first["total_qb_epa"] == pytest.approx(0.15)
    assert first["epa_per_dropback"] == pytest.approx(0.05)
    assert first["cpoe_plays"] == 1
    assert first["cpoe"] == pytest.approx(3.5)
    assert first["pass_plays"] == 1
    assert first["sacks"] == 1
    assert first["scrambles"] == 1
    assert first["sack_rate"] == pytest.approx(1 / 3)
    assert first["scramble_rate"] == pytest.approx(1 / 3)
    assert first["success_rate"] == pytest.approx(2 / 3)


def test_validate_game_source_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="QB game source is empty"):
        qb_games.validate_game_source(sample_game_source().head(0))


def test_validate_game_source_rejects_missing_columns() -> None:
    data = sample_game_source().drop("qb_epa")

    with pytest.raises(ValueError, match="qb_epa"):
        qb_games.validate_game_source(data)


def test_validate_game_source_rejects_unexpected_nulls() -> None:
    data = sample_game_source().with_columns(
        pl.lit(None, dtype=pl.String).alias("qb_name")
    )

    with pytest.raises(ValueError, match="unexpected missing values"):
        qb_games.validate_game_source(data)


def test_process_qb_games_requires_action_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="processed QB action"):
        qb_games.process_qb_games(2024, processed_dir=tmp_path)


def test_process_qb_games_rejects_unexpected_season(tmp_path: Path) -> None:
    input_path = tmp_path / "qb_actions_2024.parquet"
    data = sample_game_source().with_columns(pl.lit(2023).alias("season"))
    data.write_parquet(input_path)

    with pytest.raises(ValueError, match="expected only season 2024"):
        qb_games.process_qb_games(2024, processed_dir=tmp_path)


def test_process_qb_games_writes_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "qb_actions_2024.parquet"
    sample_game_source().write_parquet(input_path)

    output_path = qb_games.process_qb_games(
        2024,
        processed_dir=tmp_path,
    )
    saved = pl.read_parquet(output_path)

    assert output_path == tmp_path / "qb_games_2024.parquet"
    assert output_path.exists()
    assert saved.height == 2
    assert saved.get_column("dropbacks").to_list() == [3, 1]
