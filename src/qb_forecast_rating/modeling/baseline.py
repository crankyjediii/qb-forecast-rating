"""Fit and evaluate the chronological weighted linear baseline."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

FloatArray = NDArray[np.float64]

FEATURE_COLUMNS = (
    "prior_epa_per_dropback",
    "prior_cpoe",
    "prior_sack_rate",
    "prior_scramble_rate",
    "rolling_3_epa_per_dropback",
    "rolling_3_cpoe",
    "rolling_3_sack_rate",
)

TARGET_COLUMN = "target_epa_per_dropback"
WEIGHT_COLUMN = "target_dropbacks"
MIN_PRIOR_DROPBACKS = 50
MIN_TARGET_DROPBACKS = 10
DEFAULT_TRAIN_END_WEEK = 14

REQUIRED_MODEL_COLUMNS = frozenset(
    {
        *FEATURE_COLUMNS,
        "season",
        "week",
        "game_date",
        "game_id",
        "qb_id",
        "prior_dropbacks",
        TARGET_COLUMN,
        WEIGHT_COLUMN,
    }
)


@dataclass(frozen=True)
class RegressionMetrics:
    """Weighted holdout metrics for one prediction method."""

    rmse: float
    mae: float
    r2: float


@dataclass(frozen=True)
class BaselineEvaluation:
    """Model metadata and holdout results."""

    train_rows: int
    test_rows: int
    train_end_week: int
    league_mean: float
    metrics: dict[str, RegressionMetrics]
    coefficients: dict[str, float]
    intercept: float


@dataclass(frozen=True)
class BaselineRun:
    """A fitted estimator and its chronological evaluation."""

    model: LinearRegression
    evaluation: BaselineEvaluation


def prepare_model_data(data: pl.DataFrame) -> pl.DataFrame:
    """Select eligible, complete rows for model fitting and evaluation."""
    if data.is_empty():
        raise ValueError("model source is empty")

    missing_columns = REQUIRED_MODEL_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"model source is missing required columns: {missing}")

    eligible = data.filter(
        (pl.col("prior_dropbacks") >= MIN_PRIOR_DROPBACKS)
        & (pl.col(WEIGHT_COLUMN) >= MIN_TARGET_DROPBACKS)
    ).sort(["season", "week", "game_date", "game_id", "qb_id"])

    if eligible.is_empty():
        raise ValueError("model source has no eligible rows")

    null_counts = eligible.select(sorted(REQUIRED_MODEL_COLUMNS)).null_count()
    if any(null_counts.row(0)):
        raise ValueError("eligible model rows contain missing values")

    return eligible


def chronological_split(
    data: pl.DataFrame,
    train_end_week: int = DEFAULT_TRAIN_END_WEEK,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split earlier weeks for training and later weeks for evaluation."""
    train = data.filter(pl.col("week") <= train_end_week)
    test = data.filter(pl.col("week") > train_end_week)

    if train.is_empty():
        raise ValueError("chronological split produced an empty training set")
    if test.is_empty():
        raise ValueError("chronological split produced an empty test set")

    return train, test


def model_arrays(
    data: pl.DataFrame,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert selected Polars columns to NumPy model arrays."""
    features = np.asarray(
        data.select(list(FEATURE_COLUMNS)).to_numpy(),
        dtype=np.float64,
    )
    target = np.asarray(
        data.get_column(TARGET_COLUMN).to_numpy(),
        dtype=np.float64,
    )
    weights = np.asarray(
        data.get_column(WEIGHT_COLUMN).to_numpy(),
        dtype=np.float64,
    )
    return features, target, weights


def score_predictions(
    target: FloatArray,
    predictions: FloatArray,
    weights: FloatArray,
) -> RegressionMetrics:
    """Calculate weighted regression metrics."""
    return RegressionMetrics(
        rmse=float(
            root_mean_squared_error(
                target,
                predictions,
                sample_weight=weights,
            )
        ),
        mae=float(
            mean_absolute_error(
                target,
                predictions,
                sample_weight=weights,
            )
        ),
        r2=float(
            r2_score(
                target,
                predictions,
                sample_weight=weights,
            )
        ),
    )


def fit_baseline(
    data: pl.DataFrame,
    train_end_week: int = DEFAULT_TRAIN_END_WEEK,
) -> BaselineRun:
    """Fit and evaluate the weighted chronological linear baseline."""
    eligible = prepare_model_data(data)
    train, test = chronological_split(eligible, train_end_week)

    x_train, y_train, w_train = model_arrays(train)
    x_test, y_test, w_test = model_arrays(test)

    model = LinearRegression()
    model.fit(x_train, y_train, sample_weight=w_train)

    regression_predictions = np.asarray(
        model.predict(x_test),
        dtype=np.float64,
    )
    league_mean = float(np.average(y_train, weights=w_train))
    league_predictions = np.full_like(
        y_test,
        league_mean,
        dtype=np.float64,
    )
    prior_predictions = np.asarray(
        test.get_column("prior_epa_per_dropback").to_numpy(),
        dtype=np.float64,
    )

    metrics = {
        "league_mean": score_predictions(
            y_test,
            league_predictions,
            w_test,
        ),
        "prior_epa": score_predictions(
            y_test,
            prior_predictions,
            w_test,
        ),
        "linear_regression": score_predictions(
            y_test,
            regression_predictions,
            w_test,
        ),
    }

    coefficients = {
        name: float(value)
        for name, value in zip(
            FEATURE_COLUMNS,
            model.coef_,
            strict=True,
        )
    }

    evaluation = BaselineEvaluation(
        train_rows=train.height,
        test_rows=test.height,
        train_end_week=train_end_week,
        league_mean=league_mean,
        metrics=metrics,
        coefficients=coefficients,
        intercept=float(model.intercept_),
    )

    return BaselineRun(model=model, evaluation=evaluation)
