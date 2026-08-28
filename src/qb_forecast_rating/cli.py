"""Command-line interface for QB Forecast Rating."""

from argparse import ArgumentParser
from collections.abc import Sequence
from importlib.metadata import version

from qb_forecast_rating.data.pbp import ingest_pbp
from qb_forecast_rating.data.qb_actions import process_qb_actions

DISTRIBUTION_NAME = "qb-forecast-rating"


def add_season_argument(parser: ArgumentParser) -> None:
    """Add the shared required season option to a command parser."""
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season to process, such as 2024.",
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

    parser.print_help()
    return 0
