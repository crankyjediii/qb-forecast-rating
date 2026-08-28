"""Tests for the command-line interface."""

from pathlib import Path

import pytest

from qb_forecast_rating.cli import build_parser, main


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
