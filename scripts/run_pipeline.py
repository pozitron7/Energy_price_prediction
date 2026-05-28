"""
scripts/run_pipeline.py
=======================
Daily pipeline orchestrator for the Energy Price Prediction system.

Usage::

    # Full historical ingest + train + predict
    python scripts/run_pipeline.py --mode full

    # Daily ingest only (call via cron / GitHub Actions)
    python scripts/run_pipeline.py --mode ingest

    # Train model on existing data
    python scripts/run_pipeline.py --mode train

    # Generate Day-Ahead prediction for tomorrow
    python scripts/run_pipeline.py --mode predict

    # Evaluate model against actuals
    python scripts/run_pipeline.py --mode evaluate

    # Generate static result plots
    python scripts/run_pipeline.py --mode plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import text

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DATABASE_URL,
    ENTSOE_API_TOKEN,
    FIGURES_DIR,
    INITIAL_HISTORY_DAYS,
    MODEL_PATH,
    MODEL_VERSION,
    WEATHER_LOCATIONS,
)
from src.database.db_manager import get_engine, init_db
from src.evaluation.backtest import Backtester
from src.ingestion.entsoe_client import ENTSOEClient
from src.ingestion.ote_client import OTEClient
from src.ingestion.weather_client import WeatherClient
from src.models.predict import PricePredictor
from src.models.train import ModelTrainer
from src.transformation.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _table_count(engine, table_name: str) -> int:
    """Return the row count of *table_name*, or 0 on error."""
    try:
        with engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)
    except Exception:
        return 0


def ingest(engine, start_date: date, end_date: date) -> None:
    """Ingest raw data from all three sources."""
    logger.info("=== INGEST: %s → %s ===", start_date, end_date)

    # 1. OTE prices
    ote = OTEClient()
    df_prices = ote.fetch_historical(start_date, end_date)
    n = ote.save_to_db(df_prices, engine)
    logger.info("OTE: saved %d price records.", n)

    # 2. Weather (multi-location)
    weather = WeatherClient()
    df_weather = weather.fetch_all_locations(WEATHER_LOCATIONS, start_date, end_date)
    n = weather.save_to_db(df_weather, engine)
    logger.info("Weather: saved %d weather records.", n)

    # 3. Generation: ENTSO-E (if token) -> Energy-Charts (token-free) -> synthetic
    from src.ingestion.energy_charts_client import EnergyChartsClient

    if ENTSOE_API_TOKEN:
        gen_client = ENTSOEClient(api_token=ENTSOE_API_TOKEN)
        df_gen = gen_client.fetch_generation(start_date, end_date)
        source = "ENTSO-E"
    else:
        gen_client = EnergyChartsClient()
        df_gen = gen_client.fetch_generation(start_date, end_date)
        source = "Energy-Charts"
        if df_gen.empty:
            logger.warning("Energy-Charts returned no data; falling back to SYNTHETIC generation.")
            gen_client = ENTSOEClient(api_token="")
            df_gen = gen_client.fetch_generation(start_date, end_date)
            source = "synthetic"
    n = gen_client.save_to_db(df_gen, engine)
    logger.info("Generation (%s): saved %d records.", source, n)


def transform(engine) -> None:
    """Build / refresh the feature fact table."""
    if _table_count(engine, "raw_energy_prices") == 0:
        logger.warning("raw_energy_prices is empty. Ingesting last 30 days first...")
        ingest(engine, date.today() - timedelta(days=30), date.today())

    logger.info("=== TRANSFORM: building feature table ===")
    fe = FeatureEngineer()
    df = fe.build_feature_table(engine)
    n = fe.save_to_db(df, engine)
    logger.info("Feature table: %d rows written.", n)


def train(engine) -> dict:
    """Train the XGBoost model."""
    if _table_count(engine, "fact_hourly_market_features") == 0:
        logger.warning("fact_hourly_market_features is empty. Running transform first...")
        transform(engine)

    logger.info("=== TRAIN: XGBoost model v%s ===", MODEL_VERSION)
    trainer = ModelTrainer(model_path=MODEL_PATH, model_version=MODEL_VERSION)
    metrics = trainer.run(engine)
    logger.info("Training metrics: %s", metrics)
    return metrics


def predict(engine) -> None:
    """Generate Day-Ahead predictions for tomorrow."""
    if _table_count(engine, "raw_energy_prices") == 0:
        logger.warning("raw_energy_prices is empty. Ingesting last 30 days first...")
        ingest(engine, date.today() - timedelta(days=30), date.today())
        transform(engine)

    if not MODEL_PATH.exists():
        logger.warning("Model file not found at %s. Triggering training first...", MODEL_PATH)
        train(engine)

    logger.info("=== PREDICT: Day-Ahead for tomorrow ===")
    tomorrow = date.today() + timedelta(days=1)

    # Fetch weather forecast for tomorrow (all locations)
    weather = WeatherClient()
    forecast_dfs = []
    for loc in WEATHER_LOCATIONS:
        df_fc = weather.fetch_forecast(loc, days_ahead=3)
        forecast_dfs.append(df_fc)

    import pandas as pd
    forecast_df = pd.concat(forecast_dfs, ignore_index=True)

    # Load recent prices for lag features
    with engine.connect() as conn:
        last_prices = pd.read_sql(
            text("SELECT timestamp_utc, price_eur_mwh FROM raw_energy_prices "
                 "ORDER BY timestamp_utc DESC LIMIT 200"),
            conn,
        )

    predictor = PricePredictor(model_path=MODEL_PATH, model_version=MODEL_VERSION)
    predictions = predictor.run(engine, forecast_df, last_prices)
    logger.info("Saved %d predictions for %s.", len(predictions), tomorrow)


def evaluate(engine) -> dict:
    """Run backtesting evaluation (hold-out predictions + metrics)."""
    if _table_count(engine, "fact_hourly_market_features") == 0:
        logger.warning("fact_hourly_market_features is empty. Running transform first...")
        transform(engine)

    from src.evaluation.historical import generate_historical_predictions

    logger.info("=== EVALUATE: backtesting model predictions ===")
    generate_historical_predictions(engine, MODEL_VERSION, holdout_days=14)
    backtester = Backtester(engine=engine, model_version=MODEL_VERSION)
    metrics = backtester.evaluate_and_store(engine)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def plots(engine) -> None:
    """Generate static result plots."""
    if not MODEL_PATH.exists():
        logger.warning("Model file not found at %s. Triggering training first...", MODEL_PATH)
        train(engine)

    if _table_count(engine, "model_predictions") == 0:
        logger.warning("model_predictions is empty. Running evaluate and predict first...")
        evaluate(engine)
        predict(engine)

    logger.info("=== PLOTS: generating figures to %s ===", FIGURES_DIR)
    from scripts.generate_results_plots import generate_all_plots
    paths = generate_all_plots(engine, MODEL_PATH, FIGURES_DIR)
    logger.info("Generated %d figures: %s", len(paths), paths)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Energy Price Prediction pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "ingest", "transform", "train", "predict", "evaluate", "plots"],
        default="full",
        help="Pipeline stage to run (default: full)",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=date.today() - timedelta(days=INITIAL_HISTORY_DAYS),
        help="Start date for historical ingest (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=date.today(),
        help="End date for historical ingest (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    engine = get_engine(DATABASE_URL)
    init_db(engine)

    if args.mode in ("full", "ingest"):
        ingest(engine, args.start_date, args.end_date)

    if args.mode in ("full", "transform"):
        transform(engine)

    if args.mode in ("full", "train"):
        train(engine)

    if args.mode in ("full", "predict"):
        predict(engine)

    if args.mode in ("full", "evaluate"):
        evaluate(engine)

    if args.mode in ("full", "plots"):
        plots(engine)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
