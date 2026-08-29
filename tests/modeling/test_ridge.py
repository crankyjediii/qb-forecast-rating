"""Tests for nested temporal ridge selection."""

from datetime import date
from math import isfinite

import numpy as np
import polars as pl
import pytest

from qb_forecast_rating.modeling.ridge import (
    DEFAULT_RIDGE_ALPHAS,
    fit_ridge_model,
    select_ridge_alpha,
)


def sample_ridge_data() -> pl.DataFrame:
    """Create six weeks of eligible, noisy training observations."""
    rng = np.random.default_rng(2024)
    rows: list[dict[str, object]] = []

    for week in range(2, 8):
        for qb_number in range(12):
            prior_epa = float(rng.normal(0.1, 0.12))
            prior_cpoe = float(20.0 * prior_epa + rng.normal(0.0, 1.0))
            prior_sack_rate = float(rng.uniform(0.02, 0.12))
            prior_scramble_rate = float(rng.uniform(0.01, 0.15))
            rolling_epa = float(prior_epa + rng.normal(0.0, 0.06))
            rolling_cpoe = float(prior_cpoe + rng.normal(0.0, 1.5))
            rolling_sack_rate = float(
                np.clip(
                    prior_sack_rate + rng.normal(0.0, 0.015),
                    0.0,
                    0.25,
                )
            )
            target = float(
                0.08
                + 0.25 * prior_epa
                + 0.003 * prior_cpoe
                - 1.0 * prior_sack_rate
                + rng.normal(0.0, 0.18)
            )

            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "game_date": date(2024, 9, week),
                    "game_id": f"2024_{week:02d}_{qb_number:02d}",
                    "qb_id": f"QB{qb_number:02d}",
                    "prior_dropbacks": 80 + week,
                    "target_dropbacks": 20 + qb_number,
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


def test_fit_ridge_model_returns_finite_predictions() -> None:
    data = sample_ridge_data()
    train = data.filter(pl.col("week") < 7)
    test = data.filter(pl.col("week") == 7)

    model = fit_ridge_model(train, alpha=1.0)
    predictions = model.predict(test)

    assert predictions.shape == (12,)
    assert np.isfinite(predictions).all()


def test_select_ridge_alpha_uses_inner_temporal_folds() -> None:
    selection = select_ridge_alpha(sample_ridge_data())

    assert selection.selected_alpha in DEFAULT_RIDGE_ALPHAS
    assert selection.inner_folds == 3
    assert set(selection.rmse_by_alpha) == set(DEFAULT_RIDGE_ALPHAS)
    assert all(isfinite(rmse) for rmse in selection.rmse_by_alpha.values())


def test_select_ridge_alpha_is_reproducible() -> None:
    data = sample_ridge_data()

    first = select_ridge_alpha(data)
    second = select_ridge_alpha(data)

    assert first == second


def test_fit_ridge_model_rejects_nonpositive_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        fit_ridge_model(sample_ridge_data(), 0.0)


def test_select_ridge_alpha_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        select_ridge_alpha(sample_ridge_data(), alphas=())


def test_select_ridge_alpha_rejects_nonpositive_grid_value() -> None:
    with pytest.raises(ValueError, match="values must be positive"):
        select_ridge_alpha(
            sample_ridge_data(),
            alphas=(0.1, 0.0),
        )


def test_select_ridge_alpha_rejects_nonpositive_minimum_rows() -> None:
    with pytest.raises(ValueError, match="inner rows must be positive"):
        select_ridge_alpha(
            sample_ridge_data(),
            minimum_inner_rows=0,
        )


def test_select_ridge_alpha_rejects_nonpositive_minimum_folds() -> None:
    with pytest.raises(ValueError, match="inner folds must be positive"):
        select_ridge_alpha(
            sample_ridge_data(),
            minimum_inner_folds=0,
        )


def test_select_ridge_alpha_requires_enough_inner_folds() -> None:
    with pytest.raises(ValueError, match="insufficient inner"):
        select_ridge_alpha(
            sample_ridge_data(),
            minimum_inner_rows=1000,
        )
