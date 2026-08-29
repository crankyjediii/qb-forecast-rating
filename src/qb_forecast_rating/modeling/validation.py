"""Run expanding-window validation for quarterback forecasts."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from qb_forecast_rating.modeling.baseline import (
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    RegressionMetrics,
    model_arrays,
    prepare_model_data,
    score_predictions,
)
from qb_forecast_rating.modeling.benchmarks import PASSER_RATING_COLUMN

DEFAULT_FIRST_TEST_WEEK = 6

PREDICTION_ID_COLUMNS = (
    "season",
    "week",
    "game_id",
    "game_date",
    "qb_id",
    "qb_name",
    "posteam",
)
REQUIRED_VALIDATION_COLUMNS = frozenset(
    {
        *PREDICTION_ID_COLUMNS,
        PASSER_RATING_COLUMN,
    }
)

PREDICTION_COLUMNS = {
    "league_mean": "league_mean_prediction",
    "prior_epa": "prior_epa_prediction",
    "passer_rating": "passer_rating_prediction",
    "linear_regression": "linear_regression_prediction",
}


@dataclass(frozen=True)
class WalkForwardFold:
    """Metadata for one expanding-window evaluation fold."""

    test_week: int
    train_rows: int
    test_rows: int


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample predictions, fold metadata, and aggregate metrics."""

    folds: tuple[WalkForwardFold, ...]
    predictions: pl.DataFrame
    metrics: dict[str, RegressionMetrics]


def prepare_validation_data(data: pl.DataFrame) -> pl.DataFrame:
    """Validate and select eligible walk-forward observations."""
    missing_columns = REQUIRED_VALIDATION_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"validation source is missing required columns: {missing}")

    eligible = prepare_model_data(data)

    null_counts = eligible.select(sorted(REQUIRED_VALIDATION_COLUMNS)).null_count()
    if any(null_counts.row(0)):
        raise ValueError("eligible validation rows contain missing values")

    observed_seasons = set(eligible.get_column("season").unique().to_list())
    if len(observed_seasons) != 1:
        raise ValueError("walk-forward validation requires exactly one season")

    return eligible


def walk_forward_validate(
    data: pl.DataFrame,
    first_test_week: int = DEFAULT_FIRST_TEST_WEEK,
) -> WalkForwardResult:
    """Generate expanding-window out-of-sample weekly predictions."""
    if first_test_week <= 0:
        raise ValueError("first test week must be positive")

    eligible = prepare_validation_data(data)
    test_weeks = (
        eligible.filter(pl.col("week") >= first_test_week)
        .get_column("week")
        .unique()
        .sort()
        .to_list()
    )
    if not test_weeks:
        raise ValueError("validation source has no requested test weeks")

    folds: list[WalkForwardFold] = []
    prediction_frames: list[pl.DataFrame] = []

    for test_week_value in test_weeks:
        test_week = int(test_week_value)
        train = eligible.filter(pl.col("week") < test_week)
        test = eligible.filter(pl.col("week") == test_week)

        if train.is_empty():
            raise ValueError(f"test week {test_week} has no prior training rows")

        x_train, y_train, w_train = model_arrays(train)
        x_test, _, _ = model_arrays(test)

        regression_model = LinearRegression()
        regression_model.fit(
            x_train,
            y_train,
            sample_weight=w_train,
        )
        regression_predictions = np.asarray(
            regression_model.predict(x_test),
            dtype=np.float64,
        )

        passer_model = LinearRegression()
        passer_model.fit(
            train.select(PASSER_RATING_COLUMN).to_numpy(),
            y_train,
            sample_weight=w_train,
        )
        passer_predictions = np.asarray(
            passer_model.predict(test.select(PASSER_RATING_COLUMN).to_numpy()),
            dtype=np.float64,
        )

        league_mean = float(np.average(y_train, weights=w_train))
        league_predictions = np.full(
            test.height,
            league_mean,
            dtype=np.float64,
        )
        prior_predictions = np.asarray(
            test.get_column("prior_epa_per_dropback").to_numpy(),
            dtype=np.float64,
        )

        fold_predictions = test.select(
            *PREDICTION_ID_COLUMNS,
            TARGET_COLUMN,
            WEIGHT_COLUMN,
        ).with_columns(
            pl.Series(
                PREDICTION_COLUMNS["league_mean"],
                league_predictions,
            ),
            pl.Series(
                PREDICTION_COLUMNS["prior_epa"],
                prior_predictions,
            ),
            pl.Series(
                PREDICTION_COLUMNS["passer_rating"],
                passer_predictions,
            ),
            pl.Series(
                PREDICTION_COLUMNS["linear_regression"],
                regression_predictions,
            ),
        )
        prediction_frames.append(fold_predictions)
        folds.append(
            WalkForwardFold(
                test_week=test_week,
                train_rows=train.height,
                test_rows=test.height,
            )
        )

    predictions = pl.concat(prediction_frames, how="vertical")
    actual = np.asarray(
        predictions.get_column(TARGET_COLUMN).to_numpy(),
        dtype=np.float64,
    )
    weights = np.asarray(
        predictions.get_column(WEIGHT_COLUMN).to_numpy(),
        dtype=np.float64,
    )
    metrics = {
        name: score_predictions(
            actual,
            np.asarray(
                predictions.get_column(column).to_numpy(),
                dtype=np.float64,
            ),
            weights,
        )
        for name, column in PREDICTION_COLUMNS.items()
    }

    return WalkForwardResult(
        folds=tuple(folds),
        predictions=predictions,
        metrics=metrics,
    )
