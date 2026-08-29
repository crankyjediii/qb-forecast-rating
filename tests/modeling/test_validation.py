"""Tests for expanding-window quarterback validation."""

from datetime import date
from math import isfinite

import polars as pl
import pytest

from qb_forecast_rating.modeling.validation import (
    PREDICTION_COLUMNS,
    prepare_validation_data,
    walk_forward_validate,
)


def sample_validation_data() -> pl.DataFrame:
    """Create seven weeks of eligible data for two quarterbacks."""
    rows: list[dict[str, object]] = []

    for week in range(2, 9):
        for qb_number in range(2):
            prior_epa = 0.02 * week + 0.01 * qb_number
            prior_cpoe = float(week - 4) + 0.5 * qb_number
            prior_sack_rate = 0.03 + 0.002 * week
            prior_scramble_rate = 0.05 + 0.01 * qb_number
            rolling_epa = prior_epa + 0.015
            rolling_cpoe = prior_cpoe + 0.4
            rolling_sack_rate = prior_sack_rate - 0.003
            passer_rating = 70.0 + 3.0 * week + 2.0 * qb_number
            target = (
                0.3 * prior_epa
                + 0.002 * prior_cpoe
                - 1.5 * prior_sack_rate
                + 0.001 * passer_rating
                + 0.005 * qb_number
            )

            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "game_date": date(2024, 9, week),
                    "game_id": f"2024_{week:02d}_QB{qb_number}",
                    "qb_id": f"00-00{qb_number + 1}",
                    "qb_name": f"Quarterback {qb_number + 1}",
                    "posteam": f"T{qb_number + 1}",
                    "prior_dropbacks": 60 + 10 * week,
                    "target_dropbacks": 25 + qb_number,
                    "target_epa_per_dropback": target,
                    "prior_epa_per_dropback": prior_epa,
                    "prior_cpoe": prior_cpoe,
                    "prior_sack_rate": prior_sack_rate,
                    "prior_scramble_rate": prior_scramble_rate,
                    "rolling_3_epa_per_dropback": rolling_epa,
                    "rolling_3_cpoe": rolling_cpoe,
                    "rolling_3_sack_rate": rolling_sack_rate,
                    "prior_passer_rating": passer_rating,
                }
            )

    return pl.DataFrame(rows)


def test_walk_forward_validate_builds_expanding_folds() -> None:
    result = walk_forward_validate(
        sample_validation_data(),
        first_test_week=6,
    )

    assert [fold.test_week for fold in result.folds] == [6, 7, 8]
    assert [fold.train_rows for fold in result.folds] == [8, 10, 12]
    assert [fold.test_rows for fold in result.folds] == [2, 2, 2]
    assert result.predictions.height == 6
    assert set(result.metrics) == set(PREDICTION_COLUMNS)

    for metrics in result.metrics.values():
        assert isfinite(metrics.rmse)
        assert isfinite(metrics.mae)
        assert isfinite(metrics.r2)


def test_walk_forward_predictions_do_not_use_future_outcomes() -> None:
    original = sample_validation_data()
    changed = original.with_columns(
        pl.when(pl.col("week") == 8)
        .then(pl.col("target_epa_per_dropback") + 10.0)
        .otherwise(pl.col("target_epa_per_dropback"))
        .alias("target_epa_per_dropback")
    )

    original_result = walk_forward_validate(original, 6)
    changed_result = walk_forward_validate(changed, 6)
    prediction_columns = list(PREDICTION_COLUMNS.values())

    assert original_result.predictions.select(prediction_columns).equals(
        changed_result.predictions.select(prediction_columns)
    )


def test_prepare_validation_data_rejects_missing_columns() -> None:
    data = sample_validation_data().drop("qb_name")

    with pytest.raises(ValueError, match="qb_name"):
        prepare_validation_data(data)


def test_prepare_validation_data_rejects_missing_values() -> None:
    data = sample_validation_data().with_columns(
        pl.when(pl.col("week") == 6)
        .then(None)
        .otherwise(pl.col("prior_passer_rating"))
        .alias("prior_passer_rating")
    )

    with pytest.raises(ValueError, match="contain missing values"):
        prepare_validation_data(data)


def test_prepare_validation_data_rejects_multiple_seasons() -> None:
    data = sample_validation_data().with_columns(
        pl.when(pl.col("qb_id") == "00-002")
        .then(2023)
        .otherwise(pl.col("season"))
        .alias("season")
    )

    with pytest.raises(ValueError, match="exactly one season"):
        prepare_validation_data(data)


def test_walk_forward_rejects_nonpositive_first_week() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        walk_forward_validate(sample_validation_data(), 0)


def test_walk_forward_rejects_missing_requested_weeks() -> None:
    with pytest.raises(ValueError, match="no requested test weeks"):
        walk_forward_validate(sample_validation_data(), 99)


def test_walk_forward_rejects_fold_without_prior_training() -> None:
    with pytest.raises(ValueError, match="no prior training rows"):
        walk_forward_validate(sample_validation_data(), 2)
