# Energy Price Prediction (CZ day-ahead market)

Forecasts hourly OTE-CR day-ahead electricity prices from weather (Open-Meteo, four Czech
cities), renewable generation (ENTSO-E) and calendar features, with a proper backtesting
workflow and a Streamlit dashboard.

![Actual vs predicted](reports/figures/actual_vs_predicted.png)

Full results and all graphs: [reports/evaluation_results.md](reports/evaluation_results.md).

## Architecture

```mermaid
flowchart LR
    OTE[OTE-CR JSON] --> ING[src/ingestion]
    METEO[Open-Meteo] --> ING
    ENTSOE[ENTSO-E] --> ING
    ING --> RAW[(raw_* tables)]
    RAW --> FE[src/transformation]
    FE --> FACT[(fact_hourly_market_features)]
    FACT --> TRAIN[src/models/train.py XGBoost]
    TRAIN --> PRED[src/models/predict.py]
    PRED --> MP[(model_predictions)]
    MP --> EVAL[src/evaluation]
    RAW --> EVAL
    EVAL --> DASH[Streamlit dashboard]
```

Storage is PostgreSQL + TimescaleDB (`docker compose up -d`) in production and SQLite
(default, zero setup) for local work and tests. Select with `DATABASE_URL`.

## Project layout

```
config/            settings (env-driven)
src/database/      schema.sql, db_manager.py, upsert.py
src/ingestion/     ote_client.py, weather_client.py, entsoe_client.py
src/transformation feature_engineering.py (lags, sin/cos time, holidays, imputation)
src/models/        train.py (walk-forward CV + XGBoost), predict.py
src/evaluation/    metrics.py, backtest.py, historical.py (hold-out backtest)
src/dashboard/     app.py (Streamlit)
scripts/           run_pipeline.py, generate_results_plots.py
reports/           evaluation_results.md, figures/
tests/             pytest suite
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --mode full --start-date 2025-11-01   # ingest→train→predict→evaluate→plots
streamlit run src/dashboard/app.py
pytest
```

Pipeline stages (`--mode`): `ingest`, `transform`, `train`, `predict`, `evaluate`, `plots`, `full`.

Optional: `ENTSOE_API_TOKEN` (free at transparency.entsoe.eu), `DATABASE_URL`
(`postgresql+psycopg2://energy_user:energy_pass@localhost:5432/energy_db`).

## Design notes

- **Real prices.** OTE's public JSON endpoint is used; quarter-hour prices (since Oct 2025)
  are averaged to hours and placed on the true UTC axis, so 23/25-hour DST days are correct.
  Unpublished days return no rows – nothing is fabricated.
- **Multi-city weather.** Prices are national, so weather is averaged over Praha, Brno,
  Ostrava and Plzeň.
- **No leakage.** Targets/lags are never imputed; CV and the hold-out backtest train strictly
  on the past.
- **Daily operation.** Run `predict` ~11:00 UTC for tomorrow, `ingest` + `evaluate` after OTE
  publishes (~13:00 UTC) – schedule with cron / GitHub Actions / Airflow.

## Known limitations

- Without `ENTSOE_API_TOKEN`, solar/wind generation is **synthetic** (clearly logged); those
  features are then uninformative. Set the token for real data.
- The hold-out window includes a price-level shift the model under-predicts; see the report.
- Intraday market and the dbt layer from the original design are not implemented yet.
