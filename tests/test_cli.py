"""Tests for the command-line interface."""

from pathlib import Path

import pytest

from qb_forecast_rating.cli import build_parser, main


def test_parser_uses_project_name() -> None:
    """The CLI should expose the public project name."""
    parser = build_parser()

    assert parser.prog == "qb-forecast-rating"


def test_parser_accepts_ingest_pbp_command() -> None:
    """The ingestion command should parse a required season."""
    parser = build_parser()

    args = parser.parse_args(["ingest-pbp", "--season", "2024"])

    assert args.command == "ingest-pbp"
    assert args.season == 2024


def test_main_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Running without arguments should explain the CLI."""
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Forecast future NFL quarterback EPA per action." in captured.out
    assert "ingest-pbp" in captured.out


def test_main_ingests_pbp(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The ingestion command should call the pipeline and report its output."""
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
