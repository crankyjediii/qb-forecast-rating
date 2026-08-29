"""Tests for the chronological weighted linear baseline."""

import numpy as np
import polars as pl
import pytest

from qb_forecast_rating.modeling import baseline


def sample_model_source() -> pl.DataFrame:
    """Create deterministic model rows spanning training and test weeks."""
    rows: list[dict[str, object]] = []

    for index in range(13):
        week = index + 5
        prior_epa = -0.10 + 0.02 * index
        prior_cpoe = -3.0 + float((index * 2) % 7)
        prior_sack_rate = 0.02 + 0.005 * (index % 4)
        prior_scramble_rate = 0.03 + 0.01 * ((index * 3) % 5)
        rolling_epa = prior_epa + 0.03 * ((index % 3) - 1)
        rolling_cpoe = prior_cpoe + float((index % 3) - 1)
        rolling_sack_rate = prior_sack_rate + 0.004 * ((index % 3) - 1)

        target = (
            0.12
            + 0.35 * prior_epa
            + 0.005 * prior_cpoe
            - 0.80 * prior_sack_rate
            + 0.20 * prior_scramble_rate
            + 0.15 * rolling_epa
            + 0.002 * rolling_cpoe
            - 0.30 * rolling_sack_rate
        )

        rows.append(
            {
                "season": 2024,
                "week": week,
                "game_date": f"2024-10-{index + 1:02d}",
                "game_id": f"game-{index:02d}",
                "qb_id": f"QB-{index % 4}",
                "prior_dropbacks": 60 + index * 5,
                "target_dropbacks": 25 + index,
                "target_epa_per_dropback": target,
                "prior_epa_per_dropback": prior_epa,
                "prior_cpoe": prior_cpoe,
                "prior_sack_rate": prior_sack_rate,
                "prior_scramble_rate": prior_scramble_rate,
                "rolling_3_epa_per_dropback": rolling_epa,
                "rolling_3_cpoe": rolling_cpoe,
                "rolling_3_sack_rate": rolling_sack_rate,
            }
        )

    return pl.DataFrame(rows)


def test_prepare_model_data_filters_and_sorts() -> None:
    source = sample_model_source()
    ineligible = source.head(1).with_columns(
        pl.lit("ineligible").alias("game_id"),
        pl.lit(10, dtype=pl.Int64).alias("prior_dropbacks"),
    )
    combined = pl.concat([source, ineligible])

    prepared = baseline.prepare_model_data(combined)

    assert prepared.height == source.height
    assert prepared.get_column("week").to_list() == sorted(
        source.get_column("week").to_list()
    )
    assert "ineligible" not in prepared.get_column("game_id").to_list()


def test_prepare_model_data_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="model source is empty"):
        baseline.prepare_model_data(sample_model_source().head(0))


def test_prepare_model_data_rejects_missing_columns() -> None:
    source = sample_model_source().drop("prior_cpoe")

    with pytest.raises(ValueError, match="prior_cpoe"):
        baseline.prepare_model_data(source)


def test_prepare_model_data_rejects_no_eligible_rows() -> None:
    source = sample_model_source().with_columns(pl.lit(0).alias("prior_dropbacks"))

    with pytest.raises(ValueError, match="no eligible rows"):
        baseline.prepare_model_data(source)


def test_prepare_model_data_rejects_null_eligible_features() -> None:
    source = sample_model_source().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("prior_cpoe")
    )

    with pytest.raises(ValueError, match="contain missing values"):
        baseline.prepare_model_data(source)


def test_chronological_split_uses_later_weeks_for_testing() -> None:
    train, test = baseline.chronological_split(
        sample_model_source(),
        train_end_week=14,
    )

    assert train.height == 10
    assert test.height == 3
    assert train.get_column("week").max() == 14
    assert test.get_column("week").min() == 15


def test_chronological_split_rejects_empty_training_set() -> None:
    with pytest.raises(ValueError, match="empty training set"):
        baseline.chronological_split(
            sample_model_source(),
            train_end_week=0,
        )


def test_chronological_split_rejects_empty_test_set() -> None:
    with pytest.raises(ValueError, match="empty test set"):
        baseline.chronological_split(
            sample_model_source(),
            train_end_week=99,
        )


def test_model_arrays_have_expected_shapes() -> None:
    features, target, weights = baseline.model_arrays(sample_model_source())

    assert features.shape == (13, len(baseline.FEATURE_COLUMNS))
    assert target.shape == (13,)
    assert weights.shape == (13,)
    assert features.dtype == np.float64
    assert target.dtype == np.float64
    assert weights.dtype == np.float64


def test_score_predictions_uses_weights() -> None:
    target = np.array([0.0, 1.0], dtype=np.float64)
    predictions = np.array([0.0, 0.0], dtype=np.float64)
    weights = np.array([1.0, 3.0], dtype=np.float64)

    metrics = baseline.score_predictions(target, predictions, weights)

    assert metrics.rmse == pytest.approx(np.sqrt(3 / 4))
    assert metrics.mae == pytest.approx(3 / 4)


def test_fit_baseline_returns_chronological_evaluation() -> None:
    run = baseline.fit_baseline(
        sample_model_source(),
        train_end_week=14,
    )
    evaluation = run.evaluation

    assert evaluation.train_rows == 10
    assert evaluation.test_rows == 3
    assert evaluation.train_end_week == 14
    assert set(evaluation.metrics) == {
        "league_mean",
        "prior_epa",
        "linear_regression",
    }
    assert set(evaluation.coefficients) == set(baseline.FEATURE_COLUMNS)
    assert np.isfinite(evaluation.intercept)
    assert (
        evaluation.metrics["linear_regression"].rmse
        < evaluation.metrics["league_mean"].rmse
    )
