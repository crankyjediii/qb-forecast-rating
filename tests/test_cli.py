"""Tests for the command-line interface."""

from pathlib import Path

import polars as pl
import pytest
from sklearn.linear_model import LinearRegression

from qb_forecast_rating.cli import build_parser, main
from qb_forecast_rating.modeling.baseline import (
    BaselineEvaluation,
    BaselineRun,
    RegressionMetrics,
)
from qb_forecast_rating.modeling.benchmarks import (
    BenchmarkEvaluation,
    BenchmarkRun,
)
from qb_forecast_rating.modeling.comparison import (
    BootstrapDifference,
    PairedForecastComparison,
)
from qb_forecast_rating.modeling.inference import (
    CoefficientTest,
    InferenceEvaluation,
)
from qb_forecast_rating.modeling.validation import (
    WalkForwardFold,
    WalkForwardResult,
)


def test_parser_uses_project_name() -> None:
    parser = build_parser()

    assert parser.prog == "qb-forecast-rating"


def test_parser_accepts_ingest_pbp_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["ingest-pbp", "--season", "2024"])

    assert args.command == "ingest-pbp"
    assert args.season == 2024


def test_parser_accepts_build_qb_actions_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["build-qb-actions", "--season", "2024"])

    assert args.command == "build-qb-actions"
    assert args.season == 2024


def test_main_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Forecast future NFL quarterback EPA per action." in captured.out
    assert "ingest-pbp" in captured.out
    assert "build-qb-actions" in captured.out
    assert "build-qb-games" in captured.out
    assert "build-forecast-data" in captured.out


def test_main_ingests_pbp(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "pbp_2024.parquet"
    requested_seasons: list[int] = []

    def fake_ingest(season: int) -> Path:
        requested_seasons.append(season)
        return output_path

    monkeypatch.setattr("qb_forecast_rating.cli.ingest_pbp", fake_ingest)

    exit_code = main(["ingest-pbp", "--season", "2024"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert requested_seasons == [2024]
    assert "Saved 2024 play-by-play data" in captured.out
    assert str(output_path) in captured.out


def test_main_builds_qb_actions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "qb_actions_2024.parquet"
    requested_seasons: list[int] = []

    def fake_process(season: int) -> Path:
        requested_seasons.append(season)
        return output_path

    monkeypatch.setattr(
        "qb_forecast_rating.cli.process_qb_actions",
        fake_process,
    )

    exit_code = main(["build-qb-actions", "--season", "2024"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert requested_seasons == [2024]
    assert "Saved 2024 QB actions" in captured.out
    assert str(output_path) in captured.out


def test_parser_accepts_build_qb_games_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["build-qb-games", "--season", "2024"])

    assert args.command == "build-qb-games"
    assert args.season == 2024


def test_main_builds_qb_games(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "qb_games_2024.parquet"
    requested_seasons: list[int] = []

    def fake_process(season: int) -> Path:
        requested_seasons.append(season)
        return output_path

    monkeypatch.setattr(
        "qb_forecast_rating.cli.process_qb_games",
        fake_process,
    )

    exit_code = main(["build-qb-games", "--season", "2024"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert requested_seasons == [2024]
    assert "Saved 2024 QB game metrics" in captured.out
    assert str(output_path) in captured.out


def test_parser_accepts_build_forecast_data_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["build-forecast-data", "--season", "2024"])

    assert args.command == "build-forecast-data"
    assert args.season == 2024


def test_main_builds_forecast_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "qb_forecast_2024.parquet"
    requested_seasons: list[int] = []

    def fake_process(season: int) -> Path:
        requested_seasons.append(season)
        return output_path

    monkeypatch.setattr(
        "qb_forecast_rating.cli.process_forecast_dataset",
        fake_process,
    )

    exit_code = main(["build-forecast-data", "--season", "2024"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert requested_seasons == [2024]
    assert "Saved 2024 forecast data" in captured.out
    assert str(output_path) in captured.out


def test_parser_accepts_ingest_player_stats_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["ingest-player-stats", "--season", "2024"])

    assert args.command == "ingest-player-stats"
    assert args.season == 2024


def test_main_ingests_player_stats(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "player_stats_weekly_2024.parquet"
    requested_seasons: list[int] = []

    def fake_ingest(season: int) -> Path:
        requested_seasons.append(season)
        return output_path

    monkeypatch.setattr(
        "qb_forecast_rating.cli.ingest_player_stats",
        fake_ingest,
    )

    exit_code = main(["ingest-player-stats", "--season", "2024"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert requested_seasons == [2024]
    assert "Saved 2024 weekly player stats" in captured.out
    assert str(output_path) in captured.out


def test_parser_accepts_build_benchmark_data_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["build-benchmark-data", "--season", "2024"])

    assert args.command == "build-benchmark-data"
    assert args.season == 2024


def test_main_builds_benchmark_data(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "qb_benchmarks_2024.parquet"
    requested_seasons: list[int] = []

    def fake_process(season: int) -> Path:
        requested_seasons.append(season)
        return output_path

    monkeypatch.setattr(
        "qb_forecast_rating.cli.process_benchmark_dataset",
        fake_process,
    )

    exit_code = main(["build-benchmark-data", "--season", "2024"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert requested_seasons == [2024]
    assert "Saved 2024 benchmark data" in captured.out
    assert str(output_path) in captured.out


def test_parser_accepts_evaluate_baseline_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["evaluate-baseline", "--season", "2024"])

    assert args.command == "evaluate-baseline"
    assert args.season == 2024
    assert args.train_end_week == 14


def test_main_evaluates_baseline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_data = pl.DataFrame({"placeholder": [1]})
    requested_paths: list[Path] = []
    fit_calls: list[tuple[str, int]] = []

    baseline_evaluation = BaselineEvaluation(
        train_rows=100,
        test_rows=25,
        train_end_week=13,
        league_mean=0.1,
        metrics={
            "linear_regression": RegressionMetrics(
                rmse=0.3,
                mae=0.2,
                r2=0.1,
            )
        },
        coefficients={"prior_sack_rate": -1.5},
        intercept=0.15,
    )
    baseline_run = BaselineRun(
        model=LinearRegression(),
        evaluation=baseline_evaluation,
    )
    benchmark_evaluation = BenchmarkEvaluation(
        name="passer_rating",
        feature_column="prior_passer_rating",
        train_rows=100,
        test_rows=25,
        train_end_week=13,
        metrics=RegressionMetrics(
            rmse=0.32,
            mae=0.24,
            r2=0.08,
        ),
        calibration_slope=0.0035,
        calibration_intercept=-0.22,
    )
    benchmark_run = BenchmarkRun(
        model=LinearRegression(),
        evaluation=benchmark_evaluation,
    )
    inference_evaluation = InferenceEvaluation(
        train_rows=100,
        qb_clusters=20,
        train_end_week=13,
        weighted_r2=0.05,
        model_f_p_value=0.00001,
        coefficients={
            "intercept": CoefficientTest(
                estimate=0.15,
                standard_error=0.05,
                p_value=0.003,
                confidence_low=0.05,
                confidence_high=0.25,
                vif=None,
            ),
            "prior_sack_rate": CoefficientTest(
                estimate=-1.5,
                standard_error=0.4,
                p_value=0.01,
                confidence_low=-2.3,
                confidence_high=-0.7,
                vif=1.6,
            ),
        },
    )

    def fake_read_parquet(path: Path) -> pl.DataFrame:
        requested_paths.append(path)
        return source_data

    def fake_fit_baseline(
        data: pl.DataFrame,
        train_end_week: int,
    ) -> BaselineRun:
        assert data is source_data
        fit_calls.append(("baseline", train_end_week))
        return baseline_run

    def fake_fit_benchmark(
        data: pl.DataFrame,
        train_end_week: int,
    ) -> BenchmarkRun:
        assert data is source_data
        fit_calls.append(("benchmark", train_end_week))
        return benchmark_run

    def fake_fit_inference(
        data: pl.DataFrame,
        train_end_week: int,
    ) -> InferenceEvaluation:
        assert data is source_data
        fit_calls.append(("inference", train_end_week))
        return inference_evaluation

    monkeypatch.setattr(
        "qb_forecast_rating.cli.pl.read_parquet",
        fake_read_parquet,
    )
    monkeypatch.setattr(
        "qb_forecast_rating.cli.fit_baseline",
        fake_fit_baseline,
    )
    monkeypatch.setattr(
        "qb_forecast_rating.cli.fit_passer_rating_benchmark",
        fake_fit_benchmark,
    )
    monkeypatch.setattr(
        "qb_forecast_rating.cli.fit_inference",
        fake_fit_inference,
    )

    exit_code = main(
        [
            "evaluate-baseline",
            "--season",
            "2024",
            "--train-end-week",
            "13",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert requested_paths == [Path("data/features/qb_benchmarks_2024.parquet")]
    assert fit_calls == [
        ("baseline", 13),
        ("benchmark", 13),
        ("inference", 13),
    ]
    assert "Season: 2024" in captured.out
    assert "Predictive evaluation" in captured.out
    assert "linear_regression: RMSE=0.3000 MAE=0.2000 R2=0.1000" in captured.out
    assert "Official passer-rating benchmark" in captured.out
    assert "passer_rating: RMSE=0.3200 MAE=0.2400 R2=0.0800" in captured.out
    assert "Calibration:" in captured.out
    assert "Statistical inference" in captured.out
    assert "QB clusters: 20" in captured.out
    assert "prior_sack_rate: estimate=-1.5000" in captured.out
    assert "VIF=-" in captured.out
    assert "VIF=1.60" in captured.out


def test_parser_accepts_validate_model_command() -> None:
    parser = build_parser()

    args = parser.parse_args(["validate-model", "--season", "2024"])

    assert args.command == "validate-model"
    assert args.season == 2024
    assert args.first_test_week == 6
    assert args.bootstrap_replicates == 5000


def test_main_validates_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_data = pl.DataFrame({"source": [1]})
    prediction_data = pl.DataFrame({"prediction": [1.0]})
    requested_paths: list[Path] = []
    validation_calls: list[int] = []
    comparison_calls: list[tuple[str, str, int]] = []

    validation = WalkForwardResult(
        folds=(
            WalkForwardFold(
                test_week=8,
                train_rows=136,
                test_rows=31,
            ),
            WalkForwardFold(
                test_week=9,
                train_rows=167,
                test_rows=30,
            ),
        ),
        predictions=prediction_data,
        metrics={
            "passer_rating": RegressionMetrics(
                rmse=0.29,
                mae=0.22,
                r2=0.03,
            ),
            "linear_regression": RegressionMetrics(
                rmse=0.28,
                mae=0.21,
                r2=0.05,
            ),
        },
    )
    comparison = PairedForecastComparison(
        candidate_column="linear_regression_prediction",
        reference_column="passer_rating_prediction",
        qb_clusters=40,
        bootstrap_replicates=100,
        rmse=BootstrapDifference(
            difference=-0.01,
            confidence_low=-0.02,
            confidence_high=0.01,
            candidate_win_probability=0.8,
        ),
        mae=BootstrapDifference(
            difference=-0.01,
            confidence_low=-0.02,
            confidence_high=0.01,
            candidate_win_probability=0.75,
        ),
    )

    def fake_read_parquet(path: Path) -> pl.DataFrame:
        requested_paths.append(path)
        return source_data

    def fake_validate(
        data: pl.DataFrame,
        first_test_week: int,
    ) -> WalkForwardResult:
        assert data is source_data
        validation_calls.append(first_test_week)
        return validation

    def fake_compare(
        predictions: pl.DataFrame,
        candidate_column: str,
        reference_column: str,
        bootstrap_replicates: int,
    ) -> PairedForecastComparison:
        assert predictions is prediction_data
        comparison_calls.append(
            (
                candidate_column,
                reference_column,
                bootstrap_replicates,
            )
        )
        return comparison

    monkeypatch.setattr(
        "qb_forecast_rating.cli.pl.read_parquet",
        fake_read_parquet,
    )
    monkeypatch.setattr(
        "qb_forecast_rating.cli.walk_forward_validate",
        fake_validate,
    )
    monkeypatch.setattr(
        "qb_forecast_rating.cli.compare_forecasts",
        fake_compare,
    )

    exit_code = main(
        [
            "validate-model",
            "--season",
            "2024",
            "--first-test-week",
            "8",
            "--bootstrap-replicates",
            "100",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert requested_paths == [Path("data/features/qb_benchmarks_2024.parquet")]
    assert validation_calls == [8]
    assert comparison_calls == [
        (
            "linear_regression_prediction",
            "passer_rating_prediction",
            100,
        )
    ]
    assert "Season: 2024" in captured.out
    assert "Walk-forward validation" in captured.out
    assert "Folds: 2" in captured.out
    assert "First test week: 8" in captured.out
    assert "Last test week: 9" in captured.out
    assert "Out-of-sample rows: 1" in captured.out
    assert "linear_regression: RMSE=0.2800" in captured.out
    assert "Regression minus passer rating" in captured.out
    assert "RMSE difference: -0.0100" in captured.out
    assert "P(regression wins)=0.800" in captured.out
    assert "Bootstrap replicates: 100" in captured.out
