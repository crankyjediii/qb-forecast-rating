"""Tests for the command-line interface."""

import pytest

from qb_forecast_rating.cli import build_parser, main


def test_parser_uses_project_name() -> None:
    """The CLI should expose the public project name."""
    parser = build_parser()

    assert parser.prog == "qb-forecast-rating"


def test_main_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Running without arguments should explain the CLI."""
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Forecast future NFL quarterback EPA per action." in captured.out
