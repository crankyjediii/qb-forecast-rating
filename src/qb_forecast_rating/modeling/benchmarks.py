"""Fit calibrated external benchmarks for future quarterback EPA."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from qb_forecast_rating.modeling.baseline import (
    DEFAULT_TRAIN_END_WEEK,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    RegressionMetrics,
    chronological_split,
    prepare_model_data,
    score_predictions,
)

PASSER_RATING_COLUMN = "prior_passer_rating"
PASSER_RATING_NAME = "passer_rating"


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """Holdout results for one calibrated benchmark metric."""

    name: str
    feature_column: str
    train_rows: int
    test_rows: int
    train_end_week: int
    metrics: RegressionMetrics
    calibration_slope: float
    calibration_intercept: float


@dataclass(frozen=True)
class BenchmarkRun:
    """A fitted benchmark calibration and its evaluation."""

    model: LinearRegression
    evaluation: BenchmarkEvaluation


def fit_calibrated_benchmark(
    data: pl.DataFrame,
    feature_column: str,
    name: str,
    train_end_week: int = DEFAULT_TRAIN_END_WEEK,
) -> BenchmarkRun:
    """Calibrate one pregame metric to EPA using training data only."""
    if feature_column not in data.columns:
        raise ValueError(
            f"benchmark source is missing required column: {feature_column}"
        )

    eligible = prepare_model_data(data)
    if eligible.get_column(feature_column).null_count() > 0:
        raise ValueError(f"eligible benchmark rows contain missing {feature_column}")

    train, test = chronological_split(eligible, train_end_week)

    x_train = np.asarray(
        train.select(feature_column).to_numpy(),
        dtype=np.float64,
    )
    y_train = np.asarray(
        train.get_column(TARGET_COLUMN).to_numpy(),
        dtype=np.float64,
    )
    w_train = np.asarray(
        train.get_column(WEIGHT_COLUMN).to_numpy(),
        dtype=np.float64,
    )
    x_test = np.asarray(
        test.select(feature_column).to_numpy(),
        dtype=np.float64,
    )
    y_test = np.asarray(
        test.get_column(TARGET_COLUMN).to_numpy(),
        dtype=np.float64,
    )
    w_test = np.asarray(
        test.get_column(WEIGHT_COLUMN).to_numpy(),
        dtype=np.float64,
    )

    model = LinearRegression()
    model.fit(x_train, y_train, sample_weight=w_train)
    predictions = np.asarray(model.predict(x_test), dtype=np.float64)

    evaluation = BenchmarkEvaluation(
        name=name,
        feature_column=feature_column,
        train_rows=train.height,
        test_rows=test.height,
        train_end_week=train_end_week,
        metrics=score_predictions(y_test, predictions, w_test),
        calibration_slope=float(model.coef_[0]),
        calibration_intercept=float(model.intercept_),
    )
    return BenchmarkRun(model=model, evaluation=evaluation)


def fit_passer_rating_benchmark(
    data: pl.DataFrame,
    train_end_week: int = DEFAULT_TRAIN_END_WEEK,
) -> BenchmarkRun:
    """Fit the official NFL passer-rating benchmark."""
    return fit_calibrated_benchmark(
        data=data,
        feature_column=PASSER_RATING_COLUMN,
        name=PASSER_RATING_NAME,
        train_end_week=train_end_week,
    )
