"""Tests for cluster-robust training-period inference."""

import numpy as np
import polars as pl
import pytest

from qb_forecast_rating.modeling import inference


def sample_inference_source() -> pl.DataFrame:
    """Create deterministic rows across training and test weeks."""
    rows: list[dict[str, object]] = []

    rng = np.random.default_rng(2024)

    for index in range(24):
        week = 5 + index // 2
        prior_epa = float(rng.normal(0.05, 0.15))
        prior_cpoe = float(rng.normal(1.0, 5.0))
        prior_sack_rate = float(rng.uniform(0.02, 0.12))
        prior_scramble_rate = float(rng.uniform(0.01, 0.16))
        rolling_epa = float(0.6 * prior_epa + rng.normal(0.0, 0.08))
        rolling_cpoe = float(0.6 * prior_cpoe + rng.normal(0.0, 2.0))
        rolling_sack_rate = float(0.6 * prior_sack_rate + rng.uniform(0.01, 0.06))

        target = float(
            0.14
            + 0.30 * prior_epa
            + 0.006 * prior_cpoe
            - 1.10 * prior_sack_rate
            + 0.35 * prior_scramble_rate
            + rng.normal(0.0, 0.03)
        )

        rows.append(
            {
                "season": 2024,
                "week": week,
                "game_date": f"2024-10-{index + 1:02d}",
                "game_id": f"game-{index:02d}",
                "qb_id": f"QB-{index % 6}",
                "prior_dropbacks": 60 + index * 3,
                "target_dropbacks": 20 + index % 15,
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


def test_fit_inference_returns_clustered_diagnostics() -> None:
    evaluation = inference.fit_inference(
        sample_inference_source(),
        train_end_week=14,
    )

    assert evaluation.train_rows == 20
    assert evaluation.qb_clusters == 6
    assert evaluation.train_end_week == 14
    assert np.isfinite(evaluation.weighted_r2)
    assert 0.0 <= evaluation.model_f_p_value <= 1.0
    assert set(evaluation.coefficients) == {
        inference.INTERCEPT_NAME,
        *inference.INFERENCE_FEATURES,
    }

    intercept = evaluation.coefficients[inference.INTERCEPT_NAME]
    assert intercept.vif is None

    for feature in inference.INFERENCE_FEATURES:
        coefficient = evaluation.coefficients[feature]

        assert coefficient.vif is not None
        assert coefficient.vif >= 1.0
        assert 0.0 <= coefficient.p_value <= 1.0
        assert coefficient.confidence_low <= coefficient.estimate
        assert coefficient.estimate <= coefficient.confidence_high


def test_fit_inference_does_not_use_test_outcomes() -> None:
    source = sample_inference_source()
    original = inference.fit_inference(source, train_end_week=14)

    modified = source.with_columns(
        pl.when(pl.col("week") > 14)
        .then(pl.lit(99.0))
        .otherwise(pl.col("target_epa_per_dropback"))
        .alias("target_epa_per_dropback")
    )
    changed = inference.fit_inference(modified, train_end_week=14)

    assert changed.weighted_r2 == pytest.approx(original.weighted_r2)
    assert changed.model_f_p_value == pytest.approx(original.model_f_p_value)

    for name in original.coefficients:
        assert changed.coefficients[name].estimate == pytest.approx(
            original.coefficients[name].estimate
        )
        assert changed.coefficients[name].p_value == pytest.approx(
            original.coefficients[name].p_value
        )


def test_fit_inference_requires_multiple_qb_clusters() -> None:
    source = sample_inference_source().with_columns(pl.lit("only-qb").alias("qb_id"))

    with pytest.raises(ValueError, match="at least two QBs"):
        inference.fit_inference(source, train_end_week=14)
