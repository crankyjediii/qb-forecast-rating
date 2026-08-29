"""Fit standardized ridge models with nested temporal selection."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qb_forecast_rating.modeling.baseline import (
    FEATURE_COLUMNS,
    FloatArray,
    model_arrays,
    prepare_model_data,
    score_predictions,
)

DEFAULT_RIDGE_ALPHAS = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
)
DEFAULT_MINIMUM_INNER_ROWS = 30
DEFAULT_MINIMUM_INNER_FOLDS = 2


@dataclass(frozen=True)
class FittedRidgeModel:
    """A weighted feature scaler and fitted ridge estimator."""

    scaler: StandardScaler
    model: Ridge

    def predict(self, data: pl.DataFrame) -> FloatArray:
        """Predict EPA from the configured baseline features."""
        features = np.asarray(
            data.select(list(FEATURE_COLUMNS)).to_numpy(),
            dtype=np.float64,
        )
        scaled_features = self.scaler.transform(features)
        return np.asarray(
            self.model.predict(scaled_features),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RidgeSelection:
    """Training-only temporal selection results for ridge alpha."""

    selected_alpha: float
    inner_folds: int
    rmse_by_alpha: dict[float, float]


def fit_ridge_model(
    data: pl.DataFrame,
    alpha: float,
) -> FittedRidgeModel:
    """Fit a weighted standardized ridge regression."""
    if alpha <= 0:
        raise ValueError("ridge alpha must be positive")

    features, target, weights = model_arrays(data)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        features,
        sample_weight=weights,
    )

    model = Ridge(alpha=alpha)
    model.fit(
        scaled_features,
        target,
        sample_weight=weights,
    )
    return FittedRidgeModel(scaler=scaler, model=model)


def select_ridge_alpha(
    data: pl.DataFrame,
    alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    minimum_inner_rows: int = DEFAULT_MINIMUM_INNER_ROWS,
    minimum_inner_folds: int = DEFAULT_MINIMUM_INNER_FOLDS,
) -> RidgeSelection:
    """Select alpha using expanding folds inside outer training data."""
    if not alphas:
        raise ValueError("ridge alpha grid must not be empty")
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("ridge alpha grid values must be positive")
    if minimum_inner_rows <= 0:
        raise ValueError("minimum inner rows must be positive")
    if minimum_inner_folds <= 0:
        raise ValueError("minimum inner folds must be positive")

    eligible = prepare_model_data(data)
    weeks = eligible.get_column("week").unique().sort().to_list()
    inner_folds: list[tuple[pl.DataFrame, pl.DataFrame]] = []

    for validation_week_value in weeks:
        validation_week = int(validation_week_value)
        inner_train = eligible.filter(pl.col("week") < validation_week)
        inner_test = eligible.filter(pl.col("week") == validation_week)

        if inner_train.height >= minimum_inner_rows and not inner_test.is_empty():
            inner_folds.append((inner_train, inner_test))

    if len(inner_folds) < minimum_inner_folds:
        raise ValueError("ridge selection has insufficient inner temporal folds")

    rmse_by_alpha: dict[float, float] = {}

    for alpha in alphas:
        actual_chunks: list[FloatArray] = []
        prediction_chunks: list[FloatArray] = []
        weight_chunks: list[FloatArray] = []

        for inner_train, inner_test in inner_folds:
            model = fit_ridge_model(inner_train, alpha)
            predictions = model.predict(inner_test)
            _, actual, weights = model_arrays(inner_test)

            actual_chunks.append(actual)
            prediction_chunks.append(predictions)
            weight_chunks.append(weights)

        actual = np.concatenate(actual_chunks)
        predictions = np.concatenate(prediction_chunks)
        weights = np.concatenate(weight_chunks)
        rmse_by_alpha[alpha] = score_predictions(
            actual,
            predictions,
            weights,
        ).rmse

    selected_alpha = min(
        rmse_by_alpha,
        key=lambda alpha: (rmse_by_alpha[alpha], alpha),
    )
    return RidgeSelection(
        selected_alpha=selected_alpha,
        inner_folds=len(inner_folds),
        rmse_by_alpha=rmse_by_alpha,
    )
