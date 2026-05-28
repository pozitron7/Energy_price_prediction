"""
src/evaluation/historical.py
============================
Generate out-of-sample *historical* predictions for backtesting.

In production, ``model_predictions`` fills up one day at a time.  To get a
meaningful actual-vs-predicted comparison right away, this module trains a
model strictly on data *before* a cut-off and predicts the hold-out window
that follows – so no future information leaks into the predictions.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from src.database.upsert import sqlite_upsert
from src.models.train import FEATURE_COLS, ModelTrainer

logger = logging.getLogger(__name__)


def generate_historical_predictions(
    engine: Engine,
    model_version: str,
    holdout_days: int = 14,
) -> int:
    """Train on data before a cut-off and store predictions for the hold-out.

    Args:
        engine: Engine connected to the project database.
        model_version: Version label stored with the predictions.
        holdout_days: Length of the trailing hold-out window in days.

    Returns:
        Number of prediction rows written.

    Raises:
        ValueError: If there is not enough data for a train/hold-out split.
    """
    trainer = ModelTrainer(model_path=Path("unused.joblib"), model_version=model_version)
    df = trainer.load_training_data(engine)  # UTC DatetimeIndex

    cutoff = df.index.max() - timedelta(days=holdout_days)
    train_df, test_df = df[df.index <= cutoff], df[df.index > cutoff]
    if len(train_df) < 200 or test_df.empty:
        # Gracefully adjust hold-out window when available history is shorter
        if len(df) >= 120:
            effective_days = max(1, min(holdout_days, (len(df) - 100) // 24))
            cutoff = df.index.max() - timedelta(days=effective_days)
            train_df, test_df = df[df.index <= cutoff], df[df.index > cutoff]
        if len(train_df) < 50 or test_df.empty:
            raise ValueError("Not enough data for a historical backtest split.")

    model = trainer.train(train_df)
    preds = model.predict(test_df[FEATURE_COLS].fillna(0))

    out = pd.DataFrame(
        {
            "target_timestamp_utc": test_df.index,
            "predicted_price_eur": preds.astype(float),
            "model_version": model_version,
        }
    )
    n = sqlite_upsert(out, "model_predictions", engine) if engine.dialect.name == "sqlite" else _pg_write(out, engine)
    logger.info("Stored %d historical predictions (%s → %s).", n, test_df.index.min(), test_df.index.max())
    return n


def _pg_write(df: pd.DataFrame, engine: Engine) -> int:
    """Upsert predictions on PostgreSQL."""
    from sqlalchemy import text

    stmt = text(
        "INSERT INTO model_predictions (target_timestamp_utc, predicted_price_eur, model_version) "
        "VALUES (:target_timestamp_utc, :predicted_price_eur, :model_version) "
        "ON CONFLICT (target_timestamp_utc, model_version) DO UPDATE "
        "SET predicted_price_eur = EXCLUDED.predicted_price_eur"
    )
    with engine.begin() as conn:
        conn.execute(stmt, df.to_dict(orient="records"))
    return len(df)
