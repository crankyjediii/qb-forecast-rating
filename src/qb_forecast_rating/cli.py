"""Command-line interface for QB Forecast Rating."""

from argparse import ArgumentParser
from collections.abc import Sequence
from importlib.metadata import version

DISTRIBUTION_NAME = "qb-forecast-rating"


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line application."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
