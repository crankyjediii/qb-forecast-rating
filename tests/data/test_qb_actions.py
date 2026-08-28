"""Tests for the processed quarterback action table."""

from pathlib import Path

import polars as pl
import pytest

from qb_forecast_rating.data import qb_actions


def sample_action_source() -> pl.DataFrame:
    """Create pass, sack, scramble, and non-dropback examples."""
    return pl.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 1],
            "game_id": ["2024_01_ARI_BUF"] * 4,
            "play_id": [40.0, 61.0, 83.0, 108.0],
            "passer_player_id": ["00-001", "00-001", None, None],
            "passer_player_name": ["Example QB", "Example QB", None, None],
            "rusher_player_id": [None, None, "00-001", "00-002"],
            "rusher_player_name": [None, None, "Example QB", "Example RB"],
            "qb_epa": [0.25, -0.50, 0.40, 0.10],
            "epa": [0.25, -0.50, 0.40, 0.10],
            "cpoe": [3.5, None, None, None],
            "play_type": ["pass", "pass", "run", "run"],
            "qb_dropback": [1.0, 1.0, 1.0, 0.0],
            "qb_scramble": [0.0, 0.0, 1.0, 0.0],
            "sack": [0.0, 1.0, 0.0, 0.0],
        }
    )


def test_build_qb_actions_filters_and_classifies_dropbacks() -> None:
    actions = qb_actions.build_qb_actions(sample_action_source())

    assert actions.height == 3
    assert actions.get_column("qb_id").to_list() == ["00-001"] * 3
    assert actions.get_column("qb_name").to_list() == ["Example QB"] * 3
    assert actions.get_column("action_type").to_list() == [
        "pass",
        "sack",
        "scramble",
    ]


def test_validate_action_source_rejects_missing_columns() -> None:
    data = sample_action_source().drop("qb_dropback")

    with pytest.raises(ValueError, match="qb_dropback"):
        qb_actions.validate_action_source(data)


def test_build_qb_actions_rejects_empty_result() -> None:
    data = sample_action_source().with_columns(pl.lit(0.0).alias("qb_dropback"))

    with pytest.raises(ValueError, match="QB action table is empty"):
        qb_actions.build_qb_actions(data)


def test_build_qb_actions_rejects_missing_identity() -> None:
    data = sample_action_source().with_columns(
        pl.lit(None, dtype=pl.String).alias("passer_player_id"),
        pl.lit(None, dtype=pl.String).alias("passer_player_name"),
        pl.lit(None, dtype=pl.String).alias("rusher_player_id"),
        pl.lit(None, dtype=pl.String).alias("rusher_player_name"),
    )

    with pytest.raises(ValueError, match="missing identity or EPA"):
        qb_actions.build_qb_actions(data)


def test_process_qb_actions_requires_raw_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="raw play-by-play"):
        qb_actions.process_qb_actions(
            2024,
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
        )


def test_process_qb_actions_writes_parquet(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    input_path = raw_dir / "pbp" / "pbp_2024.parquet"
    input_path.parent.mkdir(parents=True)
    sample_action_source().write_parquet(input_path)

    output_path = qb_actions.process_qb_actions(
        2024,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    saved = pl.read_parquet(output_path)

    assert output_path == processed_dir / "qb_actions_2024.parquet"
    assert output_path.exists()
    assert saved.height == 3
    assert saved.get_column("action_type").to_list() == [
        "pass",
        "sack",
        "scramble",
    ]
