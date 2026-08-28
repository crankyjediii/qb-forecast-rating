"""Build leakage-safe quarterback forecasting features."""

from pathlib import Path

import polars as pl

from qb_forecast_rating.data.qb_actions import DEFAULT_PROCESSED_DIR
from qb_forecast_rating.data.qb_games import qb_games_path

DEFAULT_FEATURES_DIR = Path("data/features")
QB_SEASON_KEYS = ["season", "qb_id"]
ROLLING_WINDOW = 3

METADATA_COLUMNS = [
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

REQUIRED_FORECAST_COLUMNS = frozenset(
    {
        *METADATA_COLUMNS,
        "dropbacks",
        "total_qb_epa",
        "epa_per_dropback",
        "cpoe_plays",
        "cpoe",
        "sacks",
        "scrambles",
        "success_rate",
    }
)

NON_NULL_FORECAST_COLUMNS = REQUIRED_FORECAST_COLUMNS.difference({"cpoe"})


def validate_forecast_source(data: pl.DataFrame) -> None:
    """Validate game metrics required for leakage-safe features."""
    if data.is_empty():
        raise ValueError("forecast source is empty")

    missing_columns = REQUIRED_FORECAST_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"forecast source is missing required columns: {missing}")

    null_counts = data.select(sorted(NON_NULL_FORECAST_COLUMNS)).null_count()
    if any(null_counts.row(0)):
        raise ValueError("forecast source contains unexpected missing values")


def build_forecast_dataset(data: pl.DataFrame) -> pl.DataFrame:
    """Create pregame features and current-game prediction targets."""
    validate_forecast_source(data)

    ordered = (
        data.sort(["season", "qb_id", "game_date", "game_id"])
        .with_columns(
            (pl.col("cpoe").fill_null(0.0) * pl.col("cpoe_plays")).alias(
                "_game_cpoe_total"
            ),
            (pl.col("success_rate") * pl.col("dropbacks")).alias("_game_successes"),
        )
        .with_columns(
            (pl.col("game_id").cum_count().over(QB_SEASON_KEYS) - 1).alias(
                "prior_games"
            ),
            (
                pl.col("dropbacks").cum_sum().over(QB_SEASON_KEYS) - pl.col("dropbacks")
            ).alias("prior_dropbacks"),
            (
                pl.col("total_qb_epa").cum_sum().over(QB_SEASON_KEYS)
                - pl.col("total_qb_epa")
            ).alias("_prior_total_epa"),
            (
                pl.col("cpoe_plays").cum_sum().over(QB_SEASON_KEYS)
                - pl.col("cpoe_plays")
            ).alias("prior_cpoe_plays"),
            (
                pl.col("_game_cpoe_total").cum_sum().over(QB_SEASON_KEYS)
                - pl.col("_game_cpoe_total")
            ).alias("_prior_cpoe_total"),
            (pl.col("sacks").cum_sum().over(QB_SEASON_KEYS) - pl.col("sacks")).alias(
                "_prior_sacks"
            ),
            (
                pl.col("scrambles").cum_sum().over(QB_SEASON_KEYS) - pl.col("scrambles")
            ).alias("_prior_scrambles"),
            (
                pl.col("_game_successes").cum_sum().over(QB_SEASON_KEYS)
                - pl.col("_game_successes")
            ).alias("_prior_successes"),
            pl.col("epa_per_dropback")
            .shift(1)
            .over(QB_SEASON_KEYS)
            .alias("last_game_epa_per_dropback"),
            pl.col("dropbacks")
            .shift(1)
            .rolling_sum(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(QB_SEASON_KEYS)
            .alias("rolling_3_dropbacks"),
            pl.col("total_qb_epa")
            .shift(1)
            .rolling_sum(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(QB_SEASON_KEYS)
            .alias("_rolling_3_total_epa"),
            pl.col("cpoe_plays")
            .shift(1)
            .rolling_sum(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(QB_SEASON_KEYS)
            .alias("rolling_3_cpoe_plays"),
            pl.col("_game_cpoe_total")
            .shift(1)
            .rolling_sum(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(QB_SEASON_KEYS)
            .alias("_rolling_3_cpoe_total"),
            pl.col("sacks")
            .shift(1)
            .rolling_sum(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(QB_SEASON_KEYS)
            .alias("_rolling_3_sacks"),
            pl.col("scrambles")
            .shift(1)
            .rolling_sum(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(QB_SEASON_KEYS)
            .alias("_rolling_3_scrambles"),
            pl.col("_game_successes")
            .shift(1)
            .rolling_sum(
                window_size=ROLLING_WINDOW,
                min_samples=1,
            )
            .over(QB_SEASON_KEYS)
            .alias("_rolling_3_successes"),
        )
        .with_columns(
            pl.when(pl.col("prior_dropbacks") > 0)
            .then(pl.col("_prior_total_epa") / pl.col("prior_dropbacks"))
            .alias("prior_epa_per_dropback"),
            pl.when(pl.col("prior_cpoe_plays") > 0)
            .then(pl.col("_prior_cpoe_total") / pl.col("prior_cpoe_plays"))
            .alias("prior_cpoe"),
            pl.when(pl.col("prior_dropbacks") > 0)
            .then(pl.col("_prior_sacks") / pl.col("prior_dropbacks"))
            .alias("prior_sack_rate"),
            pl.when(pl.col("prior_dropbacks") > 0)
            .then(pl.col("_prior_scrambles") / pl.col("prior_dropbacks"))
            .alias("prior_scramble_rate"),
            pl.when(pl.col("prior_dropbacks") > 0)
            .then(pl.col("_prior_successes") / pl.col("prior_dropbacks"))
            .alias("prior_success_rate"),
            pl.when(pl.col("rolling_3_dropbacks") > 0)
            .then(pl.col("_rolling_3_total_epa") / pl.col("rolling_3_dropbacks"))
            .alias("rolling_3_epa_per_dropback"),
            pl.when(pl.col("rolling_3_cpoe_plays") > 0)
            .then(pl.col("_rolling_3_cpoe_total") / pl.col("rolling_3_cpoe_plays"))
            .alias("rolling_3_cpoe"),
            pl.when(pl.col("rolling_3_dropbacks") > 0)
            .then(pl.col("_rolling_3_sacks") / pl.col("rolling_3_dropbacks"))
            .alias("rolling_3_sack_rate"),
            pl.when(pl.col("rolling_3_dropbacks") > 0)
            .then(pl.col("_rolling_3_scrambles") / pl.col("rolling_3_dropbacks"))
            .alias("rolling_3_scramble_rate"),
            pl.when(pl.col("rolling_3_dropbacks") > 0)
            .then(pl.col("_rolling_3_successes") / pl.col("rolling_3_dropbacks"))
            .alias("rolling_3_success_rate"),
        )
    )

    return ordered.select(
        *METADATA_COLUMNS,
        pl.col("epa_per_dropback").alias("target_epa_per_dropback"),
        pl.col("dropbacks").alias("target_dropbacks"),
        "prior_games",
        "prior_dropbacks",
        "prior_cpoe_plays",
        "prior_epa_per_dropback",
        "prior_cpoe",
        "prior_sack_rate",
        "prior_scramble_rate",
        "prior_success_rate",
        "last_game_epa_per_dropback",
        "rolling_3_dropbacks",
        "rolling_3_cpoe_plays",
        "rolling_3_epa_per_dropback",
        "rolling_3_cpoe",
        "rolling_3_sack_rate",
        "rolling_3_scramble_rate",
        "rolling_3_success_rate",
    )


def forecast_dataset_path(
    season: int,
    features_dir: Path = DEFAULT_FEATURES_DIR,
) -> Path:
    """Return the deterministic forecast-dataset Parquet path."""
    return features_dir / f"qb_forecast_{season}.parquet"


def process_forecast_dataset(
    season: int,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    features_dir: Path = DEFAULT_FEATURES_DIR,
) -> Path:
    """Build and persist one season of leakage-safe forecast features."""
    input_path = qb_games_path(season, processed_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"QB game metrics file does not exist: {input_path}")

    games = pl.read_parquet(input_path)
    dataset = build_forecast_dataset(games)

    observed_seasons = set(dataset.get_column("season").drop_nulls().unique().to_list())
    if observed_seasons != {season}:
        raise ValueError(
            f"expected only season {season}, found {sorted(observed_seasons)}"
        )

    output_path = forecast_dataset_path(season, features_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(output_path, compression="zstd")
    return output_path
