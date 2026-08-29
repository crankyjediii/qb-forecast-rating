"""Command-line interface for QB Forecast Rating."""

from argparse import ArgumentParser
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import polars as pl

from qb_forecast_rating.data.pbp import ingest_pbp
from qb_forecast_rating.data.qb_actions import process_qb_actions
from qb_forecast_rating.data.qb_games import process_qb_games
from qb_forecast_rating.features.forecast import process_forecast_dataset
from qb_forecast_rating.modeling.baseline import (
    BaselineEvaluation,
    fit_baseline,
)
from qb_forecast_rating.modeling.inference import (
    InferenceEvaluation,
    fit_inference,
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

    if args.command == "evaluate-baseline":
        season = int(args.season)
        train_end_week = int(args.train_end_week)
        input_path = Path("data/features") / f"qb_forecast_{season}.parquet"
        data = pl.read_parquet(input_path)

        baseline_run = fit_baseline(data, train_end_week)
        inference = fit_inference(data, train_end_week)

        print(f"Season: {season}")
        print(f"Source: {input_path}")
        print_baseline_evaluation(baseline_run.evaluation)
        print_inference_evaluation(inference)
        return 0

    parser.print_help()
    return 0
