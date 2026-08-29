"""Tests for paired cluster-bootstrap forecast comparisons."""

import polars as pl
import pytest

from qb_forecast_rating.modeling.comparison import compare_forecasts


def sample_predictions() -> pl.DataFrame:
    """Create paired predictions across four QB clusters."""
    rows: list[dict[str, object]] = []

    for qb_number in range(4):
        for game_number in range(3):
            actual = 0.05 * qb_number + 0.02 * game_number
            rows.append(
                {
                    "qb_id": f"QB{qb_number}",
                    "target_epa_per_dropback": actual,
                    "target_dropbacks": 20 + game_number,
                    "candidate_prediction": actual,
                    "reference_prediction": actual + 0.2,
                }
            )

    return pl.DataFrame(rows)


def test_compare_forecasts_detects_better_candidate() -> None:
    result = compare_forecasts(
        predictions=sample_predictions(),
        candidate_column="candidate_prediction",
        reference_column="reference_prediction",
        bootstrap_replicates=200,
        random_seed=7,
    )

    assert result.candidate_column == "candidate_prediction"
    assert result.reference_column == "reference_prediction"
    assert result.qb_clusters == 4
    assert result.bootstrap_replicates == 200
    assert result.rmse.difference == pytest.approx(-0.2)
    assert result.rmse.confidence_low == pytest.approx(-0.2)
    assert result.rmse.confidence_high == pytest.approx(-0.2)
    assert result.rmse.candidate_win_probability == pytest.approx(1.0)
    assert result.mae.difference == pytest.approx(-0.2)
    assert result.mae.confidence_low == pytest.approx(-0.2)
    assert result.mae.confidence_high == pytest.approx(-0.2)
    assert result.mae.candidate_win_probability == pytest.approx(1.0)


def test_compare_forecasts_is_reproducible() -> None:
    first = compare_forecasts(
        sample_predictions(),
        "candidate_prediction",
        "reference_prediction",
        bootstrap_replicates=50,
        random_seed=11,
    )
    second = compare_forecasts(
        sample_predictions(),
        "candidate_prediction",
        "reference_prediction",
        bootstrap_replicates=50,
        random_seed=11,
    )

    assert first == second


def test_compare_forecasts_rejects_nonpositive_replicates() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compare_forecasts(
            sample_predictions(),
            "candidate_prediction",
            "reference_prediction",
            bootstrap_replicates=0,
        )


def test_compare_forecasts_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source is empty"):
        compare_forecasts(
            sample_predictions().head(0),
            "candidate_prediction",
            "reference_prediction",
        )


def test_compare_forecasts_rejects_missing_columns() -> None:
    data = sample_predictions().drop("reference_prediction")

    with pytest.raises(ValueError, match="reference_prediction"):
        compare_forecasts(
            data,
            "candidate_prediction",
            "reference_prediction",
        )


def test_compare_forecasts_rejects_missing_values() -> None:
    data = sample_predictions().with_columns(
        pl.when(pl.col("qb_id") == "QB0")
        .then(None)
        .otherwise(pl.col("candidate_prediction"))
        .alias("candidate_prediction")
    )

    with pytest.raises(ValueError, match="contains missing values"):
        compare_forecasts(
            data,
            "candidate_prediction",
            "reference_prediction",
        )


def test_compare_forecasts_rejects_nonpositive_weights() -> None:
    data = sample_predictions().with_columns(
        pl.when(pl.col("qb_id") == "QB0")
        .then(0)
        .otherwise(pl.col("target_dropbacks"))
        .alias("target_dropbacks")
    )

    with pytest.raises(ValueError, match="weights must be positive"):
        compare_forecasts(
            data,
            "candidate_prediction",
            "reference_prediction",
        )


def test_compare_forecasts_requires_multiple_clusters() -> None:
    data = sample_predictions().filter(pl.col("qb_id") == "QB0")

    with pytest.raises(ValueError, match="at least two QB clusters"):
        compare_forecasts(
            data,
            "candidate_prediction",
            "reference_prediction",
        )
