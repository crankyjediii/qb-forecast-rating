"""Aggregate quarterback actions into one row per quarterback-game."""

from pathlib import Path

import polars as pl

from qb_forecast_rating.data.qb_actions import (
    DEFAULT_PROCESSED_DIR,
    qb_actions_path,
)

GAME_KEYS = [
    "season",
    "season_type",
    "week",
    "game_id",
    "game_date",
    "posteam",
    "defteam",
    "qb_id",
    "qb_name",
]

REQUIRED_GAME_SOURCE_COLUMNS = frozenset(
    {
        *GAME_KEYS,
        "qb_epa",
        "cpoe",
        "sack",
        "qb_scramble",
        "success",
        "action_type",
    }
)

NON_NULL_GAME_SOURCE_COLUMNS = REQUIRED_GAME_SOURCE_COLUMNS.difference({"cpoe"})


def validate_game_source(data: pl.DataFrame) -> None:
    """Validate the processed actions required for game aggregation."""
    if data.is_empty():
        raise ValueError("QB game source is empty")

    missing_columns = REQUIRED_GAME_SOURCE_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"QB game source is missing required columns: {missing}")

    null_counts = data.select(sorted(NON_NULL_GAME_SOURCE_COLUMNS)).null_count()

    if any(null_counts.row(0)):
        raise ValueError("QB game source contains unexpected missing values")


def build_qb_games(data: pl.DataFrame) -> pl.DataFrame:
    """Aggregate quarterback actions into game-level metrics."""
    validate_game_source(data)

    return (
        data.group_by(GAME_KEYS)
        .agg(
            pl.len().alias("dropbacks"),
            pl.col("qb_epa").sum().alias("total_qb_epa"),
            pl.col("qb_epa").mean().alias("epa_per_dropback"),
            pl.col("cpoe").count().alias("cpoe_plays"),
            pl.col("cpoe").mean().alias("cpoe"),
            (pl.col("action_type") == "pass").sum().cast(pl.Int64).alias("pass_plays"),
            pl.col("sack").sum().cast(pl.Int64).alias("sacks"),
            pl.col("qb_scramble").sum().cast(pl.Int64).alias("scrambles"),
            pl.col("sack").mean().alias("sack_rate"),
            pl.col("qb_scramble").mean().alias("scramble_rate"),
            pl.col("success").mean().alias("success_rate"),
        )
        .sort(["season", "week", "game_id", "qb_id"])
    )


def qb_games_path(
    season: int,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Return the deterministic game-metrics Parquet path."""
    return processed_dir / f"qb_games_{season}.parquet"


def process_qb_games(
    season: int,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Path:
    """Build and persist one season of quarterback-game metrics."""
    input_path = qb_actions_path(season, processed_dir)
    if not input_path.exists():
        raise FileNotFoundError(
            f"processed QB action file does not exist: {input_path}"
        )

    actions = pl.read_parquet(input_path)
    observed_seasons = set(actions.get_column("season").drop_nulls().unique().to_list())
    if observed_seasons != {season}:
        raise ValueError(
            f"expected only season {season}, found {sorted(observed_seasons)}"
        )

    games = build_qb_games(actions)
    output_path = qb_games_path(season, processed_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    games.write_parquet(output_path, compression="zstd")
    return output_path
