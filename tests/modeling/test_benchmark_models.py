"""Tests for calibrated quarterback benchmark models."""

from datetime import date

import polars as pl
import pytest

from qb_forecast_rating.modeling.benchmarks import (
    fit_calibrated_benchmark,
    fit_passer_rating_benchmark,
)


def sample_benchmark_data() -> pl.DataFrame:
    """Create a benchmark dataset with an exact linear relationship."""
    weeks = list(range(10, 18))
    ratings = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0]
    targets = [-0.2 + 0.004 * rating for rating in ratings]
    row_count = len(weeks)

    return pl.DataFrame(
        {
            "season": [2024] * row_count,
            "week": weeks,
            "game_date": [date(2024, 9, index + 1) for index in range(row_count)],
            "game_id": [f"2024_{week:02d}_TEST" for week in weeks],
            "qb_id": ["00-001"] * row_count,
            "prior_dropbacks": [100] * row_count,
            "target_dropbacks": [30] * row_count,
            "target_epa_per_dropback": targets,
            "prior_epa_per_dropback": [0.1] * row_count,
            "prior_cpoe": [2.0] * row_count,
            "prior_sack_rate": [0.05] * row_count,
            "prior_scramble_rate": [0.08] * row_count,
            "rolling_3_epa_per_dropback": [0.12] * row_count,
            "rolling_3_cpoe": [2.5] * row_count,
            "rolling_3_sack_rate": [0.04] * row_count,
            "prior_passer_rating": ratings,
        }
    )


def test_fit_passer_rating_benchmark_calibrates_on_training_data() -> None:
    run = fit_passer_rating_benchmark(sample_benchmark_data(), 14)
    evaluation = run.evaluation

    assert evaluation.name == "passer_rating"
    assert evaluation.feature_column == "prior_passer_rating"
    assert evaluation.train_rows == 5
    assert evaluation.test_rows == 3
    assert evaluation.train_end_week == 14
    assert evaluation.calibration_slope == pytest.approx(0.004)
    assert evaluation.calibration_intercept == pytest.approx(-0.2)
    assert evaluation.metrics.rmse == pytest.approx(0.0, abs=1e-12)
    assert evaluation.metrics.mae == pytest.approx(0.0, abs=1e-12)
    assert evaluation.metrics.r2 == pytest.approx(1.0)


def test_fit_calibrated_benchmark_preserves_custom_name() -> None:
    run = fit_calibrated_benchmark(
        data=sample_benchmark_data(),
        feature_column="prior_passer_rating",
        name="custom_rating",
        train_end_week=14,
    )

    assert run.evaluation.name == "custom_rating"


def test_fit_calibrated_benchmark_rejects_missing_feature() -> None:
    data = sample_benchmark_data().drop("prior_passer_rating")

    with pytest.raises(ValueError, match="prior_passer_rating"):
        fit_passer_rating_benchmark(data)


def test_fit_calibrated_benchmark_rejects_missing_eligible_values() -> None:
    data = sample_benchmark_data().with_columns(
        pl.when(pl.col("week") == 12)
        .then(None)
        .otherwise(pl.col("prior_passer_rating"))
        .alias("prior_passer_rating")
    )

    with pytest.raises(ValueError, match="contain missing"):
        fit_passer_rating_benchmark(data)
