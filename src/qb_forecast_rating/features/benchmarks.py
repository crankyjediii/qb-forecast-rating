"""Build leakage-safe benchmark quarterback metrics."""

from pathlib import Path

import polars as pl

from qb_forecast_rating.data.pbp import DEFAULT_RAW_DIR
from qb_forecast_rating.data.player_stats import raw_player_stats_path
from qb_forecast_rating.features.forecast import (
    DEFAULT_FEATURES_DIR,
    forecast_dataset_path,
)

PASSER_COMPONENT_MAX = 2.375
QB_SEASON_KEYS = ["season", "qb_id"]

FORECAST_JOIN_KEYS = [
    "season",
    "season_type",
    "week",
    "qb_id",
    "posteam",
]
PLAYER_STATS_JOIN_KEYS = [
    "season",
    "season_type",
    "week",
    "player_id",
    "team",
]

REQUIRED_BENCHMARK_FORECAST_COLUMNS = frozenset(
    {
        *FORECAST_JOIN_KEYS,
        "game_id",
        "game_date",
    }
)
REQUIRED_BENCHMARK_PLAYER_COLUMNS = frozenset(
    {
        *PLAYER_STATS_JOIN_KEYS,
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
    }
)

PASSING_COLUMN_MAP = {
    "completions": "pass_completions",
    "attempts": "pass_attempts",
    "passing_yards": "pass_yards",
    "passing_tds": "pass_touchdowns",
    "passing_interceptions": "pass_interceptions",
}
PASSING_COUNT_COLUMNS = list(PASSING_COLUMN_MAP.values())
PRIOR_PASSING_COUNT_COLUMNS = [f"prior_{column}" for column in PASSING_COUNT_COLUMNS]


def validate_benchmark_sources(
    forecast_data: pl.DataFrame,
    player_stats: pl.DataFrame,
) -> None:
    """Validate forecast and weekly-stat sources before joining."""
    if forecast_data.is_empty():
        raise ValueError("benchmark forecast source is empty")
    if player_stats.is_empty():
        raise ValueError("benchmark player-stat source is empty")

    missing_forecast = REQUIRED_BENCHMARK_FORECAST_COLUMNS.difference(
        forecast_data.columns
    )
    if missing_forecast:
        missing = ", ".join(sorted(missing_forecast))
        raise ValueError(
            f"benchmark forecast source is missing required columns: {missing}"
        )

    missing_player = REQUIRED_BENCHMARK_PLAYER_COLUMNS.difference(player_stats.columns)
    if missing_player:
        missing = ", ".join(sorted(missing_player))
        raise ValueError(
            f"benchmark player-stat source is missing required columns: {missing}"
        )

    forecast_keys = forecast_data.select("season", "game_id", "qb_id")
    if forecast_keys.is_duplicated().any():
        raise ValueError("benchmark forecast source contains duplicate QB-games")

    passing_stats = player_stats.filter(pl.col("attempts") > 0)
    player_keys = passing_stats.select(PLAYER_STATS_JOIN_KEYS)
    if player_keys.is_duplicated().any():
        raise ValueError("benchmark player-stat source contains duplicate rows")

    null_counts = passing_stats.select(list(PASSING_COLUMN_MAP)).null_count()
    if any(null_counts.row(0)):
        raise ValueError("passing statistics contain unexpected missing values")


def passer_rating_expr(
    completions: str,
    attempts: str,
    yards: str,
    touchdowns: str,
    interceptions: str,
) -> pl.Expr:
    """Return the official NFL passer-rating formula as a Polars expression."""
    attempt_count = pl.col(attempts).cast(pl.Float64)

    completion_component = ((pl.col(completions) / attempt_count - 0.3) * 5.0).clip(
        0.0, PASSER_COMPONENT_MAX
    )
    yard_component = ((pl.col(yards) / attempt_count - 3.0) * 0.25).clip(
        0.0, PASSER_COMPONENT_MAX
    )
    touchdown_component = (pl.col(touchdowns) / attempt_count * 20.0).clip(
        0.0, PASSER_COMPONENT_MAX
    )
    interception_component = (
        PASSER_COMPONENT_MAX - pl.col(interceptions) / attempt_count * 25.0
    ).clip(0.0, PASSER_COMPONENT_MAX)

    rating = (
        (
            completion_component
            + yard_component
            + touchdown_component
            + interception_component
        )
        / 6.0
        * 100.0
    )

    return pl.when(attempt_count > 0).then(rating)


def build_benchmark_dataset(
    forecast_data: pl.DataFrame,
    player_stats: pl.DataFrame,
) -> pl.DataFrame:
    """Add current and strictly pregame NFL passer-rating metrics."""
    validate_benchmark_sources(forecast_data, player_stats)

    passing_stats = player_stats.filter(pl.col("attempts") > 0).select(
        *PLAYER_STATS_JOIN_KEYS,
        *[
            pl.col(source).cast(pl.Int64).alias(output)
            for source, output in PASSING_COLUMN_MAP.items()
        ],
    )

    joined = forecast_data.join(
        passing_stats,
        left_on=FORECAST_JOIN_KEYS,
        right_on=PLAYER_STATS_JOIN_KEYS,
        how="left",
    )

    ordered = (
        joined.with_columns(
            *[
                pl.col(column).fill_null(0).cast(pl.Int64)
                for column in PASSING_COUNT_COLUMNS
            ]
        )
        .sort(["season", "qb_id", "game_date", "game_id"])
        .with_columns(
            *[
                (pl.col(column).cum_sum().over(QB_SEASON_KEYS) - pl.col(column)).alias(
                    f"prior_{column}"
                )
                for column in PASSING_COUNT_COLUMNS
            ]
        )
        .with_columns(
            passer_rating_expr(
                "pass_completions",
                "pass_attempts",
                "pass_yards",
                "pass_touchdowns",
                "pass_interceptions",
            ).alias("current_passer_rating"),
            passer_rating_expr(
                "prior_pass_completions",
                "prior_pass_attempts",
                "prior_pass_yards",
                "prior_pass_touchdowns",
                "prior_pass_interceptions",
            ).alias("prior_passer_rating"),
        )
    )

    return ordered.select(
        *forecast_data.columns,
        *PASSING_COUNT_COLUMNS,
        *PRIOR_PASSING_COUNT_COLUMNS,
        "current_passer_rating",
        "prior_passer_rating",
    )


def benchmark_dataset_path(
    season: int,
    features_dir: Path = DEFAULT_FEATURES_DIR,
) -> Path:
    """Return the deterministic benchmark-dataset Parquet path."""
    return features_dir / f"qb_benchmarks_{season}.parquet"


def process_benchmark_dataset(
    season: int,
    raw_dir: Path = DEFAULT_RAW_DIR,
    features_dir: Path = DEFAULT_FEATURES_DIR,
) -> Path:
    """Build and persist one season of quarterback benchmark features."""
    forecast_path = forecast_dataset_path(season, features_dir)
    stats_path = raw_player_stats_path(season, raw_dir)

    if not forecast_path.exists():
        raise FileNotFoundError(f"forecast dataset does not exist: {forecast_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"weekly player statistics do not exist: {stats_path}")

    forecast_data = pl.read_parquet(forecast_path)
    player_stats = pl.read_parquet(stats_path)
    dataset = build_benchmark_dataset(forecast_data, player_stats)

    observed_seasons = set(dataset.get_column("season").drop_nulls().unique().to_list())
    if observed_seasons != {season}:
        raise ValueError(
            f"expected only season {season}, found {sorted(observed_seasons)}"
        )

    output_path = benchmark_dataset_path(season, features_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(output_path, compression="zstd")
    return output_path
