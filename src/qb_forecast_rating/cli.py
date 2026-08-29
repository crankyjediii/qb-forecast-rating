"""Command-line interface for QB Forecast Rating."""

from argparse import ArgumentParser
from collections.abc import Sequence
from importlib.metadata import version

import polars as pl

from qb_forecast_rating.data.pbp import ingest_pbp
from qb_forecast_rating.data.player_stats import ingest_player_stats
from qb_forecast_rating.data.qb_actions import process_qb_actions
from qb_forecast_rating.data.qb_games import process_qb_games
from qb_forecast_rating.features.benchmarks import (
    benchmark_dataset_path,
    process_benchmark_dataset,
)
from qb_forecast_rating.features.forecast import process_forecast_dataset
from qb_forecast_rating.modeling.baseline import (
    BaselineEvaluation,
    fit_baseline,
)
from qb_forecast_rating.modeling.benchmarks import (
    BenchmarkEvaluation,
    fit_passer_rating_benchmark,
)
from qb_forecast_rating.modeling.comparison import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    PairedForecastComparison,
    compare_forecasts,
)
from qb_forecast_rating.modeling.inference import (
    InferenceEvaluation,
    fit_inference,
)
from qb_forecast_rating.modeling.validation import (
    DEFAULT_FIRST_TEST_WEEK,
    PREDICTION_COLUMNS,
    WalkForwardResult,
    walk_forward_validate,
)

DISTRIBUTION_NAME = "qb-forecast-rating"
DEFAULT_TRAIN_END_WEEK = 14


def add_season_argument(parser: ArgumentParser) -> None:
    """Add the shared required season option to a command parser."""
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season to process, such as 2024.",
    )


def print_baseline_evaluation(evaluation: BaselineEvaluation) -> None:
    """Print chronological holdout metrics."""
    print("Predictive evaluation")
    print(f"Train rows: {evaluation.train_rows}")
    print(f"Test rows: {evaluation.test_rows}")
    print(f"Train through week: {evaluation.train_end_week}")
    print()

    for name, metrics in evaluation.metrics.items():
        print(
            f"{name}: RMSE={metrics.rmse:.4f} MAE={metrics.mae:.4f} R2={metrics.r2:.4f}"
        )


def print_benchmark_evaluation(
    evaluation: BenchmarkEvaluation,
) -> None:
    """Print calibrated official benchmark results."""
    metrics = evaluation.metrics

    print()
    print("Official passer-rating benchmark")
    print(
        f"{evaluation.name}: "
        f"RMSE={metrics.rmse:.4f} "
        f"MAE={metrics.mae:.4f} "
        f"R2={metrics.r2:.4f}"
    )
    print(
        "Calibration: "
        f"EPA = {evaluation.calibration_intercept:+.4f} "
        f"{evaluation.calibration_slope:+.6f} * rating"
    )


def print_inference_evaluation(evaluation: InferenceEvaluation) -> None:
    """Print training-only regression inference diagnostics."""
    print()
    print("Statistical inference")
    print(f"Training rows: {evaluation.train_rows}")
    print(f"QB clusters: {evaluation.qb_clusters}")
    print(f"Weighted R2: {evaluation.weighted_r2:.4f}")
    print(f"Model F-test p-value: {evaluation.model_f_p_value:.6f}")
    print()

    for name, coefficient in evaluation.coefficients.items():
        vif = "-" if coefficient.vif is None else f"{coefficient.vif:.2f}"
        print(
            f"{name}: "
            f"estimate={coefficient.estimate:+.4f} "
            f"SE={coefficient.standard_error:.4f} "
            f"p={coefficient.p_value:.4f} "
            f"95% CI=[{coefficient.confidence_low:+.4f}, "
            f"{coefficient.confidence_high:+.4f}] "
            f"VIF={vif}"
        )


def print_walk_forward_result(result: WalkForwardResult) -> None:
    """Print expanding-window out-of-sample metrics."""
    print("Walk-forward validation")
    print(f"Folds: {len(result.folds)}")
    print(f"First test week: {result.folds[0].test_week}")
    print(f"Last test week: {result.folds[-1].test_week}")
    print(f"Out-of-sample rows: {result.predictions.height}")
    print()

    for name, metrics in result.metrics.items():
        print(
            f"{name}: RMSE={metrics.rmse:.4f} MAE={metrics.mae:.4f} R2={metrics.r2:.4f}"
        )

    alpha_counts: dict[float, int] = {}
    for fold in result.folds:
        alpha_counts[fold.ridge_alpha] = alpha_counts.get(fold.ridge_alpha, 0) + 1

    print()
    print("Selected ridge alphas")
    for alpha, count in sorted(alpha_counts.items()):
        print(f"{alpha:g}: {count} folds")


def print_forecast_comparison(
    comparison: PairedForecastComparison,
) -> None:
    """Print paired bootstrap differences and uncertainty."""
    print()
    print("Nested ridge minus passer rating")
    print("Negative differences favor nested ridge.")
    print(
        f"RMSE difference: {comparison.rmse.difference:+.4f} "
        f"95% CI=[{comparison.rmse.confidence_low:+.4f}, "
        f"{comparison.rmse.confidence_high:+.4f}] "
        f"P(ridge wins)={comparison.rmse.candidate_win_probability:.3f}"
    )
    print(
        f"MAE difference: {comparison.mae.difference:+.4f} "
        f"95% CI=[{comparison.mae.confidence_low:+.4f}, "
        f"{comparison.mae.confidence_high:+.4f}] "
        f"P(ridge wins)={comparison.mae.candidate_win_probability:.3f}"
    )
    print(f"QB clusters: {comparison.qb_clusters}")
    print(f"Bootstrap replicates: {comparison.bootstrap_replicates}")


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(
        prog=DISTRIBUTION_NAME,
        description="Forecast future NFL quarterback EPA per action.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version(DISTRIBUTION_NAME)}",
    )

    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest-pbp",
        help="Download and persist one season of play-by-play data.",
    )
    add_season_argument(ingest_parser)

    player_stats_parser = subparsers.add_parser(
        "ingest-player-stats",
        help="Download and persist weekly player statistics.",
    )
    add_season_argument(player_stats_parser)

    actions_parser = subparsers.add_parser(
        "build-qb-actions",
        help="Build the processed quarterback dropback table.",
    )
    add_season_argument(actions_parser)

    games_parser = subparsers.add_parser(
        "build-qb-games",
        help="Build game-level quarterback metrics.",
    )
    add_season_argument(games_parser)

    forecast_parser = subparsers.add_parser(
        "build-forecast-data",
        help="Build leakage-safe quarterback forecast features.",
    )
    add_season_argument(forecast_parser)

    benchmark_parser = subparsers.add_parser(
        "build-benchmark-data",
        help="Build leakage-safe official benchmark metrics.",
    )
    add_season_argument(benchmark_parser)

    baseline_parser = subparsers.add_parser(
        "evaluate-baseline",
        help="Evaluate the chronological regression baseline.",
    )
    add_season_argument(baseline_parser)
    baseline_parser.add_argument(
        "--train-end-week",
        type=int,
        default=DEFAULT_TRAIN_END_WEEK,
        help="Last week included in model training (default: 14).",
    )

    validation_parser = subparsers.add_parser(
        "validate-model",
        help="Run expanding-window validation and paired comparison.",
    )
    add_season_argument(validation_parser)
    validation_parser.add_argument(
        "--first-test-week",
        type=int,
        default=DEFAULT_FIRST_TEST_WEEK,
        help="First week evaluated out of sample (default: 6).",
    )
    validation_parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
        help="QB-cluster bootstrap replicates (default: 5000).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest-pbp":
        season = int(args.season)
        output_path = ingest_pbp(season)
        print(f"Saved {season} play-by-play data to {output_path}")
        return 0

    if args.command == "ingest-player-stats":
        season = int(args.season)
        output_path = ingest_player_stats(season)
        print(f"Saved {season} weekly player stats to {output_path}")
        return 0

    if args.command == "build-qb-actions":
        season = int(args.season)
        output_path = process_qb_actions(season)
        print(f"Saved {season} QB actions to {output_path}")
        return 0

    if args.command == "build-qb-games":
        season = int(args.season)
        output_path = process_qb_games(season)
        print(f"Saved {season} QB game metrics to {output_path}")
        return 0

    if args.command == "build-forecast-data":
        season = int(args.season)
        output_path = process_forecast_dataset(season)
        print(f"Saved {season} forecast data to {output_path}")
        return 0

    if args.command == "build-benchmark-data":
        season = int(args.season)
        output_path = process_benchmark_dataset(season)
        print(f"Saved {season} benchmark data to {output_path}")
        return 0

    if args.command == "evaluate-baseline":
        season = int(args.season)
        train_end_week = int(args.train_end_week)
        input_path = benchmark_dataset_path(season)
        data = pl.read_parquet(input_path)

        baseline_run = fit_baseline(data, train_end_week)
        benchmark_run = fit_passer_rating_benchmark(
            data,
            train_end_week,
        )
        inference = fit_inference(data, train_end_week)

        print(f"Season: {season}")
        print(f"Source: {input_path}")
        print_baseline_evaluation(baseline_run.evaluation)
        print_benchmark_evaluation(benchmark_run.evaluation)
        print_inference_evaluation(inference)
        return 0

    if args.command == "validate-model":
        season = int(args.season)
        first_test_week = int(args.first_test_week)
        bootstrap_replicates = int(args.bootstrap_replicates)
        input_path = benchmark_dataset_path(season)
        data = pl.read_parquet(input_path)

        validation = walk_forward_validate(
            data,
            first_test_week,
        )
        comparison = compare_forecasts(
            predictions=validation.predictions,
            candidate_column=PREDICTION_COLUMNS["ridge_regression"],
            reference_column=PREDICTION_COLUMNS["passer_rating"],
            bootstrap_replicates=bootstrap_replicates,
        )

        print(f"Season: {season}")
        print(f"Source: {input_path}")
        print_walk_forward_result(validation)
        print_forecast_comparison(comparison)
        return 0

    parser.print_help()
    return 0
