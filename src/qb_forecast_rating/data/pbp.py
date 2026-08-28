"""Download, validate, and persist nflverse play-by-play data."""

from pathlib import Path
from typing import cast

import nflreadpy as nfl
import polars as pl
from nflreadpy.config import update_config

DEFAULT_CACHE_DIR = Path(".cache/nflreadpy")
DEFAULT_RAW_DIR = Path("data/raw")

REQUIRED_PBP_COLUMNS = frozenset(
    {
        "season",
        "week",
        "game_id",
        "play_id",
        "passer_player_id",
        "passer_player_name",
        "rusher_player_id",
        "rusher_player_name",
        "qb_epa",
        "epa",
        "cpoe",
        "play_type",
        "qb_dropback",
        "qb_scramble",
        "sack",
    }
)


def configure_nflreadpy(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Configure nflreadpy to reuse a local filesystem cache."""
    update_config(
        cache_mode="filesystem",
        cache_dir=cache_dir,
        verbose=False,
    )


def validate_pbp(data: pl.DataFrame, season: int) -> None:
    """Validate the minimum schema and season coverage required downstream."""
    if data.is_empty():
        raise ValueError(f"play-by-play data for {season} is empty")

    missing_columns = REQUIRED_PBP_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"play-by-play data is missing required columns: {missing}")

    observed_seasons = set(data.get_column("season").drop_nulls().unique().to_list())
    if observed_seasons != {season}:
        raise ValueError(
            f"expected only season {season}, found {sorted(observed_seasons)}"
        )


def load_pbp(
    season: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pl.DataFrame:
    """Load and validate one season of nflverse play-by-play data."""
    configure_nflreadpy(cache_dir)
    data = cast(pl.DataFrame, nfl.load_pbp([season]))
    validate_pbp(data, season)
    return data


def raw_pbp_path(
    season: int,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> Path:
    """Return the deterministic raw Parquet path for a season."""
    return raw_dir / "pbp" / f"pbp_{season}.parquet"


def ingest_pbp(
    season: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> Path:
    """Download, validate, and persist one season as compressed Parquet."""
    data = load_pbp(season, cache_dir)
    output_path = raw_pbp_path(season, raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(output_path, compression="zstd")
    return output_path
