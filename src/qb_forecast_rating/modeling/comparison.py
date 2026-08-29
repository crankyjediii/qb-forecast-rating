"""Compare paired quarterback forecasts with cluster bootstrap intervals."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from qb_forecast_rating.modeling.baseline import (
    TARGET_COLUMN,
    WEIGHT_COLUMN,
)

DEFAULT_BOOTSTRAP_REPLICATES = 5000
DEFAULT_RANDOM_SEED = 2024
CLUSTER_COLUMN = "qb_id"


@dataclass(frozen=True)
class BootstrapDifference:
    """Bootstrap estimate where negative values favor the candidate."""

    difference: float
    confidence_low: float
    confidence_high: float
    candidate_win_probability: float


@dataclass(frozen=True)
class PairedForecastComparison:
    """Paired RMSE and MAE comparisons for two forecast columns."""

    candidate_column: str
    reference_column: str
    qb_clusters: int
    bootstrap_replicates: int
    rmse: BootstrapDifference
    mae: BootstrapDifference


def loss_differences(
    actual: np.ndarray,
    weights: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float]:
    """Return candidate-minus-reference RMSE and MAE differences."""
    candidate_errors = actual - candidate
    reference_errors = actual - reference

    candidate_rmse = float(np.sqrt(np.average(candidate_errors**2, weights=weights)))
    reference_rmse = float(np.sqrt(np.average(reference_errors**2, weights=weights)))
    candidate_mae = float(np.average(np.abs(candidate_errors), weights=weights))
    reference_mae = float(np.average(np.abs(reference_errors), weights=weights))

    return (
        candidate_rmse - reference_rmse,
        candidate_mae - reference_mae,
    )


def compare_forecasts(
    predictions: pl.DataFrame,
    candidate_column: str,
    reference_column: str,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> PairedForecastComparison:
    """Compare paired forecast errors by resampling QB clusters."""
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if predictions.is_empty():
        raise ValueError("forecast comparison source is empty")

    required_columns = {
        CLUSTER_COLUMN,
        TARGET_COLUMN,
        WEIGHT_COLUMN,
        candidate_column,
        reference_column,
    }
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"forecast comparison is missing required columns: {missing}")

    null_counts = predictions.select(sorted(required_columns)).null_count()
    if any(null_counts.row(0)):
        raise ValueError("forecast comparison contains missing values")

    if predictions.filter(pl.col(WEIGHT_COLUMN) <= 0).height > 0:
        raise ValueError("forecast comparison weights must be positive")

    qb_ids = predictions.get_column(CLUSTER_COLUMN).to_numpy()
    unique_qbs = predictions.get_column(CLUSTER_COLUMN).unique().to_list()
    if len(unique_qbs) < 2:
        raise ValueError("forecast comparison requires at least two QB clusters")

    actual = predictions.get_column(TARGET_COLUMN).to_numpy().astype(float)
    weights = predictions.get_column(WEIGHT_COLUMN).to_numpy().astype(float)
    candidate = predictions.get_column(candidate_column).to_numpy().astype(float)
    reference = predictions.get_column(reference_column).to_numpy().astype(float)

    point_rmse, point_mae = loss_differences(
        actual,
        weights,
        candidate,
        reference,
    )

    cluster_rows = [np.flatnonzero(qb_ids == qb_id) for qb_id in unique_qbs]
    rng = np.random.default_rng(random_seed)
    rmse_differences = np.empty(bootstrap_replicates, dtype=float)
    mae_differences = np.empty(bootstrap_replicates, dtype=float)

    for replicate in range(bootstrap_replicates):
        sampled_clusters = rng.integers(
            0,
            len(cluster_rows),
            size=len(cluster_rows),
        )
        sampled_rows = np.concatenate(
            [cluster_rows[index] for index in sampled_clusters]
        )
        (
            rmse_differences[replicate],
            mae_differences[replicate],
        ) = loss_differences(
            actual[sampled_rows],
            weights[sampled_rows],
            candidate[sampled_rows],
            reference[sampled_rows],
        )

    rmse_interval = np.quantile(
        rmse_differences,
        [0.025, 0.975],
    )
    mae_interval = np.quantile(
        mae_differences,
        [0.025, 0.975],
    )

    return PairedForecastComparison(
        candidate_column=candidate_column,
        reference_column=reference_column,
        qb_clusters=len(unique_qbs),
        bootstrap_replicates=bootstrap_replicates,
        rmse=BootstrapDifference(
            difference=point_rmse,
            confidence_low=float(rmse_interval[0]),
            confidence_high=float(rmse_interval[1]),
            candidate_win_probability=float(np.mean(rmse_differences < 0)),
        ),
        mae=BootstrapDifference(
            difference=point_mae,
            confidence_low=float(mae_interval[0]),
            confidence_high=float(mae_interval[1]),
            candidate_win_probability=float(np.mean(mae_differences < 0)),
        ),
    )
