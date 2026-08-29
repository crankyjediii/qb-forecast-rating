"""Run cluster-robust inference on the training period."""

from dataclasses import dataclass

import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from qb_forecast_rating.modeling.baseline import (
    DEFAULT_TRAIN_END_WEEK,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    chronological_split,
    prepare_model_data,
)

INFERENCE_FEATURES = (
    "prior_epa_per_dropback",
    "prior_cpoe",
    "prior_sack_rate",
    "prior_scramble_rate",
)

INTERCEPT_NAME = "intercept"


@dataclass(frozen=True)
class CoefficientTest:
    """One cluster-robust coefficient test."""

    estimate: float
    standard_error: float
    p_value: float
    confidence_low: float
    confidence_high: float
    vif: float | None


@dataclass(frozen=True)
class InferenceEvaluation:
    """Training-only regression diagnostics and coefficient tests."""

    train_rows: int
    qb_clusters: int
    train_end_week: int
    weighted_r2: float
    model_f_p_value: float
    coefficients: dict[str, CoefficientTest]


def fit_inference(
    data: pl.DataFrame,
    train_end_week: int = DEFAULT_TRAIN_END_WEEK,
) -> InferenceEvaluation:
    """Fit weighted regression with QB-clustered standard errors."""
    eligible = prepare_model_data(data)
    train, _ = chronological_split(eligible, train_end_week)

    feature_matrix = np.asarray(
        train.select(list(INFERENCE_FEATURES)).to_numpy(),
        dtype=np.float64,
    )
    design_matrix = np.asarray(
        sm.add_constant(feature_matrix, has_constant="add"),
        dtype=np.float64,
    )
    target = np.asarray(
        train.get_column(TARGET_COLUMN).to_numpy(),
        dtype=np.float64,
    )
    weights = np.asarray(
        train.get_column(WEIGHT_COLUMN).to_numpy(),
        dtype=np.float64,
    )
    groups = train.get_column("qb_id").to_numpy()
    qb_clusters = train.get_column("qb_id").n_unique()

    if qb_clusters < 2:
        raise ValueError("cluster-robust inference requires at least two QBs")

    result = sm.WLS(
        target,
        design_matrix,
        weights=weights,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups},
    )

    vifs = {
        feature: float(variance_inflation_factor(design_matrix, index))
        for index, feature in enumerate(INFERENCE_FEATURES, start=1)
    }

    intervals = np.asarray(result.conf_int(), dtype=np.float64)
    names = (INTERCEPT_NAME, *INFERENCE_FEATURES)
    coefficients: dict[str, CoefficientTest] = {}

    for index, name in enumerate(names):
        coefficients[name] = CoefficientTest(
            estimate=float(result.params[index]),
            standard_error=float(result.bse[index]),
            p_value=float(result.pvalues[index]),
            confidence_low=float(intervals[index, 0]),
            confidence_high=float(intervals[index, 1]),
            vif=vifs.get(name),
        )

    return InferenceEvaluation(
        train_rows=train.height,
        qb_clusters=qb_clusters,
        train_end_week=train_end_week,
        weighted_r2=float(result.rsquared),
        model_f_p_value=float(result.f_pvalue),
        coefficients=coefficients,
    )
