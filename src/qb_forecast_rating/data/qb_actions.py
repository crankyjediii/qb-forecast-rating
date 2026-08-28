"""Build the processed quarterback dropback table."""

from pathlib import Path

import polars as pl

from qb_forecast_rating.data.pbp import (
    DEFAULT_RAW_DIR,
    raw_pbp_path,
    validate_pbp,
)

DEFAULT_PROCESSED_DIR = Path("data/processed")

REQUIRED_ACTION_COLUMNS = frozenset(
    {
        "game_id",
        "play_id",
        "qb_dropback",
        "qb_scramble",
        "sack",
        "qb_epa",
        "passer_player_id",
        "passer_player_name",
        "rusher_player_id",
        "rusher_player_name",
    }
)


def validate_action_source(data: pl.DataFrame) -> None:
    """Validate columns required to identify and classify QB dropbacks."""
    missing_columns = REQUIRED_ACTION_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"QB action source is missing required columns: {missing}")


def build_qb_actions(data: pl.DataFrame) -> pl.DataFrame:
    """Filter dropbacks and assign a consistent QB identity and action type."""
    validate_action_source(data)

    actions = (
        data.filter(pl.col("qb_dropback") == 1)
        .with_columns(
            pl.coalesce("passer_player_id", "rusher_player_id").alias("qb_id"),
            pl.coalesce("passer_player_name", "rusher_player_name").alias("qb_name"),
            pl.when(pl.col("sack") == 1)
            .then(pl.lit("sack"))
            .when(pl.col("qb_scramble") == 1)
            .then(pl.lit("scramble"))
            .otherwise(pl.lit("pass"))
            .alias("action_type"),
        )
        .sort(["game_id", "play_id"])
    )

    if actions.is_empty():
        raise ValueError("QB action table is empty")

    key_null_counts = actions.select(
        "qb_id",
        "qb_name",
        "qb_epa",
    ).null_count()

    if any(key_null_counts.row(0)):
        raise ValueError("QB action table contains missing identity or EPA values")

    return actions


def qb_actions_path(
    season: int,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Return the deterministic processed Parquet path for a season."""
    return processed_dir / f"qb_actions_{season}.parquet"


def process_qb_actions(
    season: int,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Transform raw play-by-play into a processed QB action table."""
    input_path = raw_pbp_path(season, raw_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"raw play-by-play file does not exist: {input_path}")

    data = pl.read_parquet(input_path)
    validate_pbp(data, season)
    actions = build_qb_actions(data)

    output_path = qb_actions_path(season, processed_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actions.write_parquet(output_path, compression="zstd")
    return output_path
