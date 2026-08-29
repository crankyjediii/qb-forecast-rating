"""Download, validate, and persist nflverse weekly player statistics."""

from pathlib import Path
from typing import cast

import nflreadpy as nfl
import polars as pl

from qb_forecast_rating.data.pbp import (
    DEFAULT_CACHE_DIR,
    DEFAULT_RAW_DIR,
    configure_nflreadpy,
)

REQUIRED_PLAYER_STATS_COLUMNS = frozenset(
    {
        "player_id",
        "player_display_name",
        "position",
        "season",
        "week",
        "season_type",
        "team",
        "opponent_team",
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
    }
)


def validate_player_stats(data: pl.DataFrame, season: int) -> None:
    """Validate the weekly player-stat schema and season coverage."""
    if data.is_empty():
        raise ValueError(f"player statistics for {season} are empty")

    missing_columns = REQUIRED_PLAYER_STATS_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"player statistics are missing required columns: {missing}")

    observed_seasons = set(data.get_column("season").drop_nulls().unique().to_list())
    if observed_seasons != {season}:
        raise ValueError(
            f"expected only season {season}, found {sorted(observed_seasons)}"
        )


def load_player_stats(
    season: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pl.DataFrame:
    """Load and validate one season of weekly nflverse player statistics."""
    configure_nflreadpy(cache_dir)
    data = cast(
        pl.DataFrame,
        nfl.load_player_stats([season], summary_level="week"),
    )
    validate_player_stats(data, season)
    return data


def raw_player_stats_path(
    season: int,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> Path:
    """Return the deterministic raw weekly player-stat Parquet path."""
    return raw_dir / "player_stats" / f"player_stats_weekly_{season}.parquet"


def ingest_player_stats(
    season: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> Path:
    """Download, validate, and persist weekly player stats as Parquet."""
    data = load_player_stats(season, cache_dir)
    output_path = raw_player_stats_path(season, raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_path, compression="zstd")
    return output_path
