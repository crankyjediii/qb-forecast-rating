"""Tests for leakage-safe quarterback benchmark metrics."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from qb_forecast_rating.data.player_stats import raw_player_stats_path
from qb_forecast_rating.features.benchmarks import (
    benchmark_dataset_path,
    build_benchmark_dataset,
    passer_rating_expr,
    process_benchmark_dataset,
    validate_benchmark_sources,
)
from qb_forecast_rating.features.forecast import forecast_dataset_path


def sample_forecast(season: int = 2024) -> pl.DataFrame:
    """Create three ordered forecast rows for one quarterback."""
    return pl.DataFrame(
        {
            "season": [season, season, season],
            "season_type": ["REG", "REG", "REG"],
            "week": [1, 2, 3],
            "game_id": [
                f"{season}_01_ARI_BUF",
                f"{season}_02_BUF_MIA",
                f"{season}_03_JAX_BUF",
            ],
            "game_date": [
                date(season, 9, 1),
                date(season, 9, 8),
                date(season, 9, 15),
            ],
            "qb_id": ["00-001", "00-001", "00-001"],
            "posteam": ["BUF", "BUF", "BUF"],
        }
    )


def sample_player_stats(season: int = 2024) -> pl.DataFrame:
    """Create passing totals for the first two forecast games."""
    return pl.DataFrame(
        {
            "season": [season, season],
            "season_type": ["REG", "REG"],
            "week": [1, 2],
            "player_id": ["00-001", "00-001"],
            "team": ["BUF", "BUF"],
            "completions": [20, 10],
            "attempts": [30, 20],
            "passing_yards": [250, 100],
            "passing_tds": [2, 0],
            "passing_interceptions": [1, 2],
        }
    )


def test_passer_rating_formula_caps_components() -> None:
    data = pl.DataFrame(
        {
            "completions": [20, 0, 0],
            "attempts": [20, 10, 0],
            "yards": [250, 0, 0],
            "touchdowns": [4, 0, 0],
            "interceptions": [0, 10, 0],
        }
    )

    ratings = data.with_columns(
        passer_rating_expr(
            "completions",
            "attempts",
            "yards",
            "touchdowns",
            "interceptions",
        ).alias("rating")
    ).get_column("rating")

    assert ratings[0] == pytest.approx(158.333333)
    assert ratings[1] == pytest.approx(0.0)
    assert ratings[2] is None


def test_build_benchmark_dataset_uses_only_prior_games() -> None:
    result = build_benchmark_dataset(
        sample_forecast(),
        sample_player_stats(),
    ).sort("week")

    assert result.height == 3
    assert result.get_column("prior_passer_rating")[0] is None
    assert result.get_column("current_passer_rating")[0] == pytest.approx(100.694444)
    assert result.get_column("prior_passer_rating")[1] == pytest.approx(
        result.get_column("current_passer_rating")[0]
    )
    assert result.get_column("prior_passer_rating")[2] == pytest.approx(69.583333)
    assert result.get_column("current_passer_rating")[2] is None
    assert result.get_column("pass_attempts").to_list() == [30, 20, 0]
    assert result.get_column("prior_pass_attempts").to_list() == [
        0,
        30,
        50,
    ]


def test_validate_rejects_empty_forecast_source() -> None:
    with pytest.raises(ValueError, match="forecast source is empty"):
        validate_benchmark_sources(
            sample_forecast().head(0),
            sample_player_stats(),
        )


def test_validate_rejects_empty_player_stat_source() -> None:
    with pytest.raises(ValueError, match="player-stat source is empty"):
        validate_benchmark_sources(
            sample_forecast(),
            sample_player_stats().head(0),
        )


def test_validate_rejects_missing_forecast_columns() -> None:
    with pytest.raises(ValueError, match="game_date"):
        validate_benchmark_sources(
            sample_forecast().drop("game_date"),
            sample_player_stats(),
        )


def test_validate_rejects_missing_player_columns() -> None:
    with pytest.raises(ValueError, match="attempts"):
        validate_benchmark_sources(
            sample_forecast(),
            sample_player_stats().drop("attempts"),
        )


def test_validate_rejects_duplicate_qb_games() -> None:
    duplicate = pl.concat([sample_forecast(), sample_forecast().head(1)])

    with pytest.raises(ValueError, match="duplicate QB-games"):
        validate_benchmark_sources(
            duplicate,
            sample_player_stats(),
        )


def test_validate_rejects_duplicate_player_rows() -> None:
    duplicate = pl.concat([sample_player_stats(), sample_player_stats().head(1)])

    with pytest.raises(ValueError, match="duplicate rows"):
        validate_benchmark_sources(
            sample_forecast(),
            duplicate,
        )


def test_validate_rejects_missing_passing_values() -> None:
    invalid = sample_player_stats().with_columns(
        pl.lit(None).cast(pl.Int64).alias("completions")
    )

    with pytest.raises(ValueError, match="unexpected missing values"):
        validate_benchmark_sources(
            sample_forecast(),
            invalid,
        )


def test_process_rejects_missing_forecast_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="forecast dataset does not exist",
    ):
        process_benchmark_dataset(
            2024,
            tmp_path / "raw",
            tmp_path / "features",
        )


def test_process_rejects_missing_player_stats_file(
    tmp_path: Path,
) -> None:
    features_dir = tmp_path / "features"
    path = forecast_dataset_path(2024, features_dir)
    path.parent.mkdir(parents=True)
    sample_forecast().write_parquet(path)

    with pytest.raises(
        FileNotFoundError,
        match="weekly player statistics do not exist",
    ):
        process_benchmark_dataset(
            2024,
            tmp_path / "raw",
            features_dir,
        )


def test_process_writes_benchmark_dataset(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    forecast_path = forecast_dataset_path(2024, features_dir)
    stats_path = raw_player_stats_path(2024, raw_dir)

    forecast_path.parent.mkdir(parents=True)
    stats_path.parent.mkdir(parents=True)
    sample_forecast().write_parquet(forecast_path)
    sample_player_stats().write_parquet(stats_path)

    output_path = process_benchmark_dataset(
        2024,
        raw_dir,
        features_dir,
    )

    expected = build_benchmark_dataset(
        sample_forecast(),
        sample_player_stats(),
    )
    actual = pl.read_parquet(output_path)

    assert output_path == benchmark_dataset_path(2024, features_dir)
    assert actual.equals(expected)


def test_process_rejects_unexpected_season(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    forecast_path = forecast_dataset_path(2024, features_dir)
    stats_path = raw_player_stats_path(2024, raw_dir)

    forecast_path.parent.mkdir(parents=True)
    stats_path.parent.mkdir(parents=True)
    sample_forecast(2023).write_parquet(forecast_path)
    sample_player_stats(2023).write_parquet(stats_path)

    with pytest.raises(ValueError, match="expected only season 2024"):
        process_benchmark_dataset(
            2024,
            raw_dir,
            features_dir,
        )
