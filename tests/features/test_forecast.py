"""Tests for leakage-safe quarterback forecast features."""

from pathlib import Path

import polars as pl
import pytest

from qb_forecast_rating.features import forecast


def sample_forecast_source() -> pl.DataFrame:
    """Create four games for one QB and one game for another."""
    return pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024, 2024],
            "season_type": ["REG"] * 5,
            "week": [3, 1, 1, 4, 2],
            "game_id": [
                "2024_03_A",
                "2024_01_B",
                "2024_01_A",
                "2024_04_A",
                "2024_02_A",
            ],
            "game_date": [
                "2024-09-22",
                "2024-09-08",
                "2024-09-08",
                "2024-09-29",
                "2024-09-15",
            ],
            "posteam": ["AAA", "BBB", "AAA", "AAA", "AAA"],
            "defteam": ["CCC", "DDD", "EEE", "FFF", "GGG"],
            "qb_id": ["00-A", "00-B", "00-A", "00-A", "00-A"],
            "qb_name": ["QB A", "QB B", "QB A", "QB A", "QB A"],
            "dropbacks": [30, 12, 10, 40, 20],
            "total_qb_epa": [-3.0, 0.0, 1.0, 12.0, 4.0],
            "epa_per_dropback": [-0.1, 0.0, 0.1, 0.3, 0.2],
            "cpoe_plays": [25, 0, 8, 35, 16],
            "cpoe": [-2.0, None, 2.0, 6.0, 4.0],
            "sacks": [3, 0, 1, 4, 2],
            "scrambles": [3, 1, 1, 4, 2],
            "success_rate": [0.4, 0.5, 0.5, 0.7, 0.6],
        }
    )


def row_for_game(
    dataset: pl.DataFrame,
    game_id: str,
) -> dict[str, object]:
    """Return one forecast row by game identifier."""
    return dataset.filter(pl.col("game_id") == game_id).row(0, named=True)


def test_build_forecast_dataset_creates_weighted_prior_features() -> None:
    dataset = forecast.build_forecast_dataset(sample_forecast_source())
    game_four = row_for_game(dataset, "2024_04_A")

    assert dataset.height == 5
    assert game_four["target_epa_per_dropback"] == pytest.approx(0.3)
    assert game_four["target_dropbacks"] == 40
    assert game_four["prior_games"] == 3
    assert game_four["prior_dropbacks"] == 60
    assert game_four["prior_cpoe_plays"] == 49
    assert game_four["prior_epa_per_dropback"] == pytest.approx(2 / 60)
    assert game_four["prior_cpoe"] == pytest.approx(30 / 49)
    assert game_four["prior_sack_rate"] == pytest.approx(6 / 60)
    assert game_four["prior_scramble_rate"] == pytest.approx(6 / 60)
    assert game_four["prior_success_rate"] == pytest.approx(29 / 60)
    assert game_four["last_game_epa_per_dropback"] == pytest.approx(-0.1)
    assert game_four["rolling_3_dropbacks"] == 60
    assert game_four["rolling_3_cpoe_plays"] == 49
    assert game_four["rolling_3_epa_per_dropback"] == pytest.approx(2 / 60)
    assert game_four["rolling_3_cpoe"] == pytest.approx(30 / 49)
    assert game_four["rolling_3_sack_rate"] == pytest.approx(6 / 60)
    assert game_four["rolling_3_scramble_rate"] == pytest.approx(6 / 60)
    assert game_four["rolling_3_success_rate"] == pytest.approx(29 / 60)


def test_first_game_has_no_prior_performance_features() -> None:
    dataset = forecast.build_forecast_dataset(sample_forecast_source())
    first_game = row_for_game(dataset, "2024_01_A")
    other_qb_first_game = row_for_game(dataset, "2024_01_B")

    assert first_game["prior_games"] == 0
    assert first_game["prior_dropbacks"] == 0
    assert first_game["prior_epa_per_dropback"] is None
    assert first_game["prior_cpoe"] is None
    assert first_game["last_game_epa_per_dropback"] is None
    assert first_game["rolling_3_epa_per_dropback"] is None
    assert other_qb_first_game["prior_games"] == 0
    assert other_qb_first_game["prior_epa_per_dropback"] is None


def test_current_game_epa_does_not_leak_into_prior_features() -> None:
    source = sample_forecast_source()
    original = row_for_game(
        forecast.build_forecast_dataset(source),
        "2024_04_A",
    )

    modified_source = source.with_columns(
        pl.when(pl.col("game_id") == "2024_04_A")
        .then(pl.lit(99.0))
        .otherwise(pl.col("total_qb_epa"))
        .alias("total_qb_epa"),
        pl.when(pl.col("game_id") == "2024_04_A")
        .then(pl.lit(9.9))
        .otherwise(pl.col("epa_per_dropback"))
        .alias("epa_per_dropback"),
    )
    modified = row_for_game(
        forecast.build_forecast_dataset(modified_source),
        "2024_04_A",
    )

    assert modified["target_epa_per_dropback"] == pytest.approx(9.9)
    assert modified["prior_epa_per_dropback"] == pytest.approx(
        original["prior_epa_per_dropback"]
    )
    assert modified["rolling_3_epa_per_dropback"] == pytest.approx(
        original["rolling_3_epa_per_dropback"]
    )


def test_validate_forecast_source_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="forecast source is empty"):
        forecast.validate_forecast_source(sample_forecast_source().head(0))


def test_validate_forecast_source_rejects_missing_columns() -> None:
    data = sample_forecast_source().drop("total_qb_epa")

    with pytest.raises(ValueError, match="total_qb_epa"):
        forecast.validate_forecast_source(data)


def test_validate_forecast_source_rejects_unexpected_nulls() -> None:
    data = sample_forecast_source().with_columns(
        pl.lit(None, dtype=pl.String).alias("qb_name")
    )

    with pytest.raises(ValueError, match="unexpected missing values"):
        forecast.validate_forecast_source(data)


def test_process_forecast_dataset_requires_game_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="QB game metrics"):
        forecast.process_forecast_dataset(
            2024,
            processed_dir=tmp_path / "processed",
            features_dir=tmp_path / "features",
        )


def test_process_forecast_dataset_rejects_unexpected_season(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    input_path = processed_dir / "qb_games_2024.parquet"
    data = sample_forecast_source().with_columns(pl.lit(2023).alias("season"))
    data.write_parquet(input_path)

    with pytest.raises(ValueError, match="expected only season 2024"):
        forecast.process_forecast_dataset(
            2024,
            processed_dir=processed_dir,
            features_dir=tmp_path / "features",
        )


def test_process_forecast_dataset_writes_parquet(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    processed_dir.mkdir()
    input_path = processed_dir / "qb_games_2024.parquet"
    sample_forecast_source().write_parquet(input_path)

    output_path = forecast.process_forecast_dataset(
        2024,
        processed_dir=processed_dir,
        features_dir=features_dir,
    )
    saved = pl.read_parquet(output_path)

    assert output_path == features_dir / "qb_forecast_2024.parquet"
    assert output_path.exists()
    assert saved.height == 5
    assert saved.width == 27
