<div align="center">

# ⚡ OTE-CR Day-Ahead Electricity Price Prediction

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**Production-grade machine learning system forecasting hourly spot electricity prices on the Czech Day-Ahead market (OTE-CR) using gradient-boosted trees, multi-hub meteorological data, and renewable generation dynamics.**

[Overview](#overview) • [Architecture](#architecture) • [Key Engineering Decisions](#key-engineering-decisions) • [Quickstart](#quickstart) • [Evaluation](#evaluation--benchmark-results) • [Project Layout](#project-layout)

</div>

---

## Overview

Wholesale electricity prices in the Czech bidding zone (single national market coupled via SDAC / 4M MC) exhibit extreme non-linearities and intraday volatility:
- **Midday solar cannibalization ("Duck Curve")**: High solar PV penetration during summer midday hours drastically depresses spot prices, frequently driving them negative.
- **Morning & evening demand ramps**: Industrial activity and domestic heating/cooling create sharp price spikes.
- **Weather dependency**: Wind generation in northern Europe and regional irradiance drive substantial price differentials across delivery hours.

This project delivers an **end-to-end, reproducible ML forecasting system** designed for:
1. **BESS (Battery Energy Storage System) Arbitrage**: Identifying optimal charge/discharge windows with $>70\%$ directional price movement accuracy.
2. **Demand-Response Optimization**: Scheduling flexible commercial and industrial loads outside peak-price blocks.
3. **Retail Dynamic Tariffs**: Predicting tomorrow's spot tariffs before market gate closure.

---

## Architecture

```mermaid
flowchart TD
    subgraph Data Sources
        OTE["OTE-CR API / JSON<br/>(Hourly Spot Prices EUR/MWh)"]
        METEO["Open-Meteo Multi-Hub<br/>(Praha, Brno, Ostrava, Plzeň)"]
        GEN["Energy-Charts / ENTSO-E<br/>(Solar & Wind Fleet Output)"]
    end

    subgraph Data Engineering Pipeline
        INGEST["Ingestion Service<br/>(Incremental, Retry & Backoff)"]
        OTE --> INGEST
        METEO --> INGEST
        GEN --> INGEST
        
        DB[("Storage Layer<br/>PostgreSQL + TimescaleDB<br/>or Zero-Conf SQLite")]
        INGEST --> DB
        
        FE["Feature Engineering Pipeline<br/>(Cyclic Time, Autoregressive Lags, Holidays)"]
        DB --> FE
        FE --> FACT[("fact_hourly_market_features<br/>(51,000+ Verified Hours)")]
    end

    subgraph Machine Learning & Analytics
        FACT --> TRAIN["Model Training (XGBoost)<br/>(Walk-Forward Time-Series CV)"]
        TRAIN --> ARTIFACT["Model Artifact (joblib)"]
        
        FORECAST_W["Weather Forecast (D+1)"] --> PREDICT["Day-Ahead Inference Service"]
        ARTIFACT --> PREDICT
        PREDICT --> PREDS[("model_predictions (D+1 Forecast)")]
    end

    subgraph Serving & Observability
        PREDS --> EVAL["Backtesting & Evaluation Engine<br/>(MAE, RMSE, Directional Accuracy)"]
        FACT --> EVAL
        EVAL --> REPORTS["Static Publication Reports<br/>(reports/figures)"]
        EVAL & PREDS & FACT --> DASHBOARD["Interactive Streamlit Dashboard<br/>(Real-Time KPIs & SHAP Explainability)"]
    end
```

---

## Key Engineering Decisions

| Problem | Naive Solution | Senior Engineering Approach |
|---|---|---|
| **Lookahead Leakage** | Using same-day actual solar/wind generation as features. | **Real generation output is unknown at Day-Ahead gate closure (12:00 CET).** We store actual generation strictly for monitoring, but train the pricing model exclusively on exogenous numerical weather forecasts (irradiance, wind speed, temperature) and autoregressive price lags ($t-24$, $t-168$). |
| **National Pricing vs. Weather Location** | Querying weather solely for Prague. | **Czech Republic is a single copper-plate bidding zone.** We poll multi-point coordinates across the 4 major distribution hubs (**Praha, Brno, Ostrava, Plzeň**) and compute spatial averages, reflecting the distributed footprint of solar PV in Moravia and wind turbines in northern regions. |
| **Timezone & DST Shifts** | Naive local time strings (`Europe/Prague`). | **All pipeline timestamps are strictly normalized to UTC-aware datetimes.** This naturally accommodates 23-hour (spring) and 25-hour (autumn) clock-change days without index collisions or missing rows. |
| **Market Harmonization (Oct 2025)** | Breaking on 15-minute interval rollout. | The ingestion client dynamically detects 15-minute vs. 60-minute auction payloads, accurately re-sampling down to uniform 1-hour intervals with true weighted pricing. |
| **Storage Portability** | Hard dependency on running a Docker database daemon. | **Dual-engine abstraction (`src/database/db_manager.py`)**: Seamless zero-configuration fallback to local SQLite for instant developer setup and unit tests, alongside production PostgreSQL + TimescaleDB support via `DATABASE_URL`. |

---

## Quickstart

### 1. Prerequisites & Installation

```bash
# Clone the repository
git clone https://github.com/wendzel42/Energy_price_prediction.git
cd Energy_price_prediction

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (or: make install)
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

Execute the end-to-end pipeline: historical ingestion (2021–present, ~51,000 hours), feature generation, XGBoost training with walk-forward CV, tomorrow's price forecast, hold-out backtest, and figure generation:

```bash
# Full automated run (or: make pipeline)
python scripts/run_pipeline.py --mode full --start-date 2021-01-01
```

Individual stages can be executed independently:
```bash
python scripts/run_pipeline.py --mode ingest     # Poll OTE-CR, Open-Meteo, Energy-Charts
python scripts/run_pipeline.py --mode transform  # Compute lags, cyclical features, imputations
python scripts/run_pipeline.py --mode train      # Train XGBoost regressor with CV
python scripts/run_pipeline.py --mode predict    # Generate D+1 hourly price forecast
python scripts/run_pipeline.py --mode evaluate   # Evaluate hold-out actuals vs predictions
python scripts/run_pipeline.py --mode plots      # Render publication PNG figures
```

### 3. Launch the Interactive Dashboard

```bash
# Or: make dashboard
streamlit run src/dashboard/app.py
```
The dashboard provides:
1. **Actual vs. Predicted**: Interactive date range slider, MAE/RMSE/Directional Accuracy metrics, and worst-error inspection table.
2. **Tomorrow's Forecast**: 24-hour price curve highlighting the top 6 cheapest charging hours vs. top 6 most expensive peak hours.
3. **SHAP Feature Attribution**: TreeExplainer beeswarm plot quantifying the marginal impact of weather, lags, and calendar factors.

---

## Evaluation & Benchmark Results

The model is evaluated using a strict **14-day trailing out-of-sample hold-out window** (336 consecutive hours) where the model is trained exclusively on prior data:

| Metric | Hold-out (Last 14 Days) | 5-Fold Walk-Forward CV (Full 5.8y Dataset) | Business Significance |
|---|---|---|---|
| **MAE** | **34.85 EUR/MWh** | **21.33 EUR/MWh** | Mean price error magnitude |
| **RMSE** | **44.84 EUR/MWh** | **31.39 EUR/MWh** | Penalizes severe outlier errors |
| **MAPE** | **15.92%** | **17.8%** | Relative percentage accuracy |
| **Directional Accuracy** | **70.15%** | **71.4%** | Critical for battery charge/discharge trading signals |

### Visual Artifacts

All figures are automatically rendered to [`reports/figures/`](reports/figures/):

| Figure | Description |
|---|---|
| ![Actual vs Predicted](reports/figures/actual_vs_predicted.png) | **Actual vs. Predicted Prices**: 14-day out-of-sample backtest tracking peak hours and weekend troughs. |
| ![Forecast](reports/figures/forecast_next_day.png) | **Day-Ahead Forecast**: Hourly curve for the upcoming delivery day, color-coded by percentile blocks. |
| ![SHAP Summary](reports/figures/shap_summary.png) | **SHAP Beeswarm Analysis**: High solar irradiance pushes spot prices down; autoregressive lags ($t-24$) provide baseline level anchors. |
| ![Feature Importance](reports/figures/feature_importance.png) | **Feature Importance**: Autoregressive signals ($t-24$, $t-168$) and solar irradiance dominate predictive power. |
| ![Error Distribution](reports/figures/error_distribution.png) | **Residual Distribution**: Gaussian-centered prediction error curve centered at zero. |

Detailed evaluation documentation: [`reports/evaluation_results.md`](reports/evaluation_results.md).

---

## Project Layout

```
Energy_price_prediction/
├── config/
│   └── settings.py              # Central env-driven configuration
├── data/
│   ├── raw/                     # Cached payloads (git-ignored)
│   └── energy.db                # SQLite store for local dev (git-ignored)
├── notebooks/
│   └── 01_market_and_weather_eda.ipynb # EDA on duck curves & weather correlation
├── reports/
│   ├── evaluation_results.md    # In-depth backtest analysis & metrics
│   └── figures/                 # Generated vector/PNG visual artifacts
├── scripts/
│   ├── run_pipeline.py          # Unified CLI pipeline orchestrator
│   └── generate_results_plots.py# Publication figure generator
├── src/
│   ├── database/
│   │   ├── schema.sql           # PostgreSQL + TimescaleDB DDL
│   │   ├── db_manager.py        # Dual-engine connection & session scope
│   │   └── upsert.py            # Idempotent SQLite / Postgres upsert helpers
│   ├── ingestion/
│   │   ├── ote_client.py        # OTE-CR day-ahead auction client
│   │   ├── weather_client.py    # Open-Meteo multi-hub weather client
│   │   ├── energy_charts_client.py # Token-free CZ renewable generation client
│   │   └── entsoe_client.py     # ENTSO-E REST client with token support
│   ├── transformation/
│   │   └── feature_engineering.py # Cyclic encodings, lags, holiday detection
│   ├── models/
│   │   ├── train.py             # Walk-forward CV & XGBoost training
│   │   └── predict.py           # Day-ahead inference & D+1 feature builder
│   ├── evaluation/
│   │   ├── metrics.py           # MAE, RMSE, MAPE, Directional Accuracy
│   │   ├── backtest.py          # Backtesting engine
│   │   └── historical.py        # Out-of-sample historical prediction generator
│   └── dashboard/
│       └── app.py               # Interactive Streamlit application
├── tests/
│   ├── test_database.py         # DB initialization & transactional integrity
│   ├── test_ingestion.py        # Client parsing & error handling
│   ├── test_energy_charts.py    # Renewable power aggregation tests
│   ├── test_feature_engineering.py # Cyclic bounds & lag alignment
│   ├── test_models.py           # Model training & CV temporal ordering
│   ├── test_evaluation.py       # Metrics verification & backtest logic
│   └── test_upsert_and_ote_json.py # OTE JSON parser & idempotency tests
├── .github/workflows/
│   └── daily_pipeline.yml       # Scheduled GitHub Actions runner
├── .env.example                 # Configuration environment template
├── .gitignore                   # Production gitignore
├── .pre-commit-config.yaml      # Code quality & formatting hooks
├── Makefile                     # Developer command runner
├── pyproject.toml               # Modern PEP 517/621 packaging & linter configs
├── requirements.txt             # Pinned production requirements
└── docker-compose.yml           # PostgreSQL 16 + TimescaleDB container setup
```

---

## Production Deployment & Scheduling

### Automated Daily Workflow

The repository includes a GitHub Actions workflow ([`.github/workflows/daily_pipeline.yml`](.github/workflows/daily_pipeline.yml)) orchestrating the production timeline:
- **11:00 UTC**: Queries live weather forecasts and generates the D+1 day-ahead prediction before market clearing.
- **14:00 UTC**: Ingests published OTE spot prices (released ~13:00 UTC), records actuals, executes automated backtesting, and refreshes report figures.

### Running with PostgreSQL + TimescaleDB

To use PostgreSQL with TimescaleDB instead of SQLite:
```bash
# 1. Start the database container
make docker-up

# 2. Export connection string
export DATABASE_URL="postgresql+psycopg2://energy_user:energy_pass@localhost:5432/energy_db"

# 3. Run pipeline against TimescaleDB
python scripts/run_pipeline.py --mode full
```

---

## Testing & Quality Assurance

The test suite covers unit and integration tests across all layers with **100% passing tests** (no external network dependencies required during testing):

```bash
# Run pytest (or: make test)
pytest -v

# Run lint checks (or: make lint)
ruff check .
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Developed by **[Lukas W](https://github.com/wendzel42)**.
