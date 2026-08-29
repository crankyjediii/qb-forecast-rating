# QB Forecast Rating

[![CI](https://github.com/crankyjediii/qb-forecast-rating/actions/workflows/ci.yml/badge.svg)](https://github.com/crankyjediii/qb-forecast-rating/actions/workflows/ci.yml)

A reproducible NFL quarterback rating project designed to forecast future EPA per dropback using leakage-safe features, chronological validation, and regression modeling.

The project uses 2024 nflverse data to answer:

> Can a regression-based quarterback metric predict future performance better than traditional NFL passer rating?

## Current result

The nested ridge model produced the best point estimates in the frozen 2024 development analysis, but its advantage over passer rating was small and statistically uncertain.

| Method | RMSE | MAE | Weighted R-squared |
|---|---:|---:|---:|
| League mean | 0.2979 | 0.2300 | -0.0106 |
| Prior EPA per dropback | 0.3030 | 0.2333 | -0.0456 |
| NFL passer rating | 0.2917 | 0.2247 | 0.0313 |
| Ordinary linear regression | 0.2943 | 0.2272 | 0.0136 |
| **Nested ridge regression** | **0.2906** | **0.2243** | **0.0381** |

The paired QB-cluster bootstrap comparison between ridge regression and passer rating found:

- RMSE difference: **-0.0010**
- RMSE 95% confidence interval: **[-0.0048, +0.0027]**
- MAE difference: **-0.0004**
- MAE 95% confidence interval: **[-0.0042, +0.0033]**

Negative differences favor ridge regression. Both intervals include zero, so the analysis does not establish that ridge is truly better.

A future untouched season is required for confirmation.

## Terminology

NFL passer rating and ESPN Total QBR are different metrics.

- **NFL passer rating** is the traditional public formula based on completions, attempts, yards, touchdowns, and interceptions.
- **Total QBR** is a proprietary ESPN metric.

This repository currently benchmarks against official NFL passer rating because it can be reproduced directly from nflverse data. ESPN Total QBR is not currently included.

## What this project demonstrates

### Data engineering

- Downloads play-by-play and weekly player statistics with `nflreadpy`
- Uses a local filesystem cache to avoid unnecessary downloads
- Validates schemas and season coverage
- Persists compressed Parquet datasets
- Separates raw, processed, feature, and modeling layers

### Statistics

- Fits weighted regression models
- Uses quarterback-clustered standard errors
- Reports coefficient confidence intervals and p-values
- Measures multicollinearity with variance inflation factors
- Uses a paired QB-cluster bootstrap for model comparisons

### Data science

- Creates leakage-safe pregame features
- Uses chronological and expanding-window validation
- Tunes ridge penalties only inside each outer training window
- Compares against league-mean, prior-EPA, and passer-rating benchmarks
- Preserves an honest distinction between development and confirmation

### Data analysis

- Exposes the pipeline through a tested CLI
- Produces interpretable comparison tables
- Documents limitations instead of overstating results
- Provides a foundation for an interactive dashboard and written report

## Data flow

```text

nflverse play-by-play

        |

        v

raw play-by-play Parquet

        |

        v

quarterback actions

        |

        v

quarterback-game metrics

        |

        v

leakage-safe forecast features

        |

        +-----------------------------+

        |                             |

        v                             v

weekly player statistics       EPA/CPOE features

        |                             |

        v                             |

NFL passer rating benchmark <--------+

        |

        v

chronological and walk-forward validation

        |

        v

OLS, nested ridge, and paired bootstrap comparison

```

## Modeling target

The prediction target is current-game quarterback EPA per dropback.

Each model row represents one quarterback entering one game. Predictor values are calculated only from games that occurred earlier in time.

Rows are eligible when:

- the quarterback has at least 50 prior dropbacks
- the target game contains at least 10 dropbacks
- every required modeling feature is present

Target-game dropbacks are used as sample weights so larger quarterback performances receive more influence than brief appearances.

## Model features

The primary regression uses:

- season-to-date EPA per dropback
- season-to-date CPOE
- season-to-date sack rate
- season-to-date scramble rate
- rolling three-game EPA per dropback
- rolling three-game CPOE
- rolling three-game sack rate

The nested ridge model standardizes these features before fitting.

## Validation design

### Chronological holdout

The initial baseline trains through week 14 and evaluates weeks 15 through 22.

This provides a simple late-season holdout but depends on one split.

### Expanding-window validation

The primary evaluation uses 17 weekly outer folds:

- first test week: 6
- final test week: 22
- out-of-sample QB-games: 390
- QB clusters: 48

For each test week, models train only on eligible observations from earlier weeks.

### Nested ridge selection

Within every outer training window, ridge regression selects its penalty using inner expanding-window folds.

The frozen penalty grid is:

```text

0.01

0.1

1

10

100

1,000

10,000

100,000

1,000,000

```

The final 2024 folds selected:

- alpha 1,000: 4 folds
- alpha 10,000: 13 folds

Larger available penalties were not selected, resolving the earlier grid-boundary issue.

### Paired uncertainty

Forecasts are compared using a paired cluster bootstrap that resamples entire quarterbacks.

This keeps repeated games from the same quarterback together and produces confidence intervals for candidate-minus-reference RMSE and MAE differences.

## Project structure

- `src/qb_forecast_rating/`
  - `cli.py`: command-line interface
  - `data/`: ingestion and processed datasets
    - `pbp.py`
    - `player_stats.py`
    - `qb_actions.py`
    - `qb_games.py`
  - `features/`: leakage-safe feature construction
    - `benchmarks.py`
    - `forecast.py`
  - `modeling/`: estimation, inference, and validation
    - `baseline.py`
    - `benchmarks.py`
    - `comparison.py`
    - `inference.py`
    - `ridge.py`
    - `validation.py`
- `tests/`
  - `data/`
  - `features/`
  - `modeling/`
  - `test_cli.py`

## Requirements

- Python 3.13
- `uv`
- Git

Core libraries:

- `nflreadpy`
- Polars
- NumPy
- scikit-learn
- statsmodels

## Installation

Clone the repository:

```powershell

git clone https://github.com/crankyjediii/qb-forecast-rating.git

cd qb-forecast-rating

```

Create the environment and install locked dependencies:

```powershell

uv sync

```

Verify the CLI:

```powershell

uv run qb-forecast-rating --help

```

## Reproduce the 2024 pipeline

Download and cache the source data:

```powershell

uv run qb-forecast-rating ingest-pbp --season 2024

uv run qb-forecast-rating ingest-player-stats --season 2024

```

Build the processed and feature datasets:

```powershell

uv run qb-forecast-rating build-qb-actions --season 2024

uv run qb-forecast-rating build-qb-games --season 2024

uv run qb-forecast-rating build-forecast-data --season 2024

uv run qb-forecast-rating build-benchmark-data --season 2024

```

Run the simple chronological baseline:

```powershell

uv run qb-forecast-rating evaluate-baseline --season 2024

```

Run the primary expanding-window evaluation:

```powershell

uv run qb-forecast-rating validate-model --season 2024

```

The validation command accepts optional controls:

```powershell

uv run qb-forecast-rating validate-model `

  --season 2024 `

  --first-test-week 6 `

  --bootstrap-replicates 5000

```

## Quality checks

Run the complete local quality gate:

```powershell

uv run ruff format --check .

uv run ruff check .

uv run mypy src tests

uv run pytest -W error --cov=qb_forecast_rating --cov-report=term-missing

```

The current suite contains 110 tests with 100% statement and branch coverage.

GitHub Actions runs formatting, linting, typing, and tests for every pull request.

## Generated data

Downloaded and derived datasets are intentionally excluded from Git.

Local outputs include:

```text

.cache/nflreadpy/

data/raw/pbp/pbp_2024.parquet

data/raw/player_stats/player_stats_weekly_2024.parquet

data/processed/qb_actions_2024.parquet

data/processed/qb_games_2024.parquet

data/features/qb_forecast_2024.parquet

data/features/qb_benchmarks_2024.parquet

```

These files can be reproduced using the CLI commands above.

## Limitations

- The current analysis uses one development season.
- The ridge penalty grid was expanded after a boundary diagnostic on 2024 data.
- The final 2024 advantage over passer rating is not statistically conclusive.
- Team, opponent, offensive line, receiver, weather, and injury context are not yet modeled.
- The project benchmarks NFL passer rating, not proprietary ESPN Total QBR.
- Game-level EPA remains noisy and only partially predictable.
- The current metric forecasts quarterback outcomes; it does not isolate quarterback talent from every surrounding factor.
- No causal interpretation should be assigned to regression coefficients.

## Roadmap

- Freeze the current specification and test it on an untouched season
- Add opponent and supporting-cast context
- Export reproducible evaluation artifacts
- Build the interactive dashboard
- Add quarterback ranking and trend views
- Publish the full written methodology and results report
- Evaluate whether a reproducible ESPN QBR source can be added
- Add a repository license

## Data attribution

NFL data is loaded from the nflverse ecosystem through `nflreadpy`.

This project is unofficial and is not affiliated with the NFL, ESPN, or nflverse.
