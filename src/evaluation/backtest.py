"""
src/evaluation/backtest.py
==========================
Backtesting engine for energy price prediction models.

The :class:`Backtester` class compares stored model predictions against actual
OTE day-ahead prices retrieved from the project database, computes evaluation
metrics, and optionally persists results back to the ``evaluation_results``
table for trend analysis.

Typical usage::

    from src.database.db_manager import get_engine
    from src.evaluation.backtest import Backtester

    engine = get_engine()
    bt = Backtester(engine, model_version="1.0.0")

    # One-shot evaluation with DB persistence
    metrics = bt.evaluate_and_store(engine, start_date=date(2024, 1, 1))
    print(metrics)

    # Rolling window metrics
    rolling_df = bt.rolling_evaluation(window_days=7)
    print(rolling_df.tail())
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy.engine import Engine
from sqlalchemy import text

from src.evaluation import metrics as m

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------

# Fetch actuals from the fact table (already joined / transformed)
_SQL_ACTUALS = """
SELECT
    timestamp_utc,
    price_eur_mwh
FROM fact_hourly_market_features
WHERE price_eur_mwh IS NOT NULL
  {where_clauses}
ORDER BY timestamp_utc
"""

# Fallback: fetch actuals directly from raw_energy_prices
_SQL_ACTUALS_RAW = """
SELECT
    timestamp_utc,
    price_eur_mwh
FROM raw_energy_prices
WHERE price_eur_mwh IS NOT NULL
  {where_clauses}
ORDER BY timestamp_utc
"""

# Fetch predictions for a specific model version
_SQL_PREDICTIONS = """
SELECT
    target_timestamp_utc AS timestamp_utc,
    predicted_price_eur
FROM model_predictions
WHERE model_version = :model_version
  {where_clauses}
ORDER BY timestamp_utc
"""

# Insert one row into evaluation_results
_SQL_INSERT_EVAL = """
INSERT INTO evaluation_results
    (evaluated_at, period_start, period_end, model_version,
     mae_eur_mwh, rmse_eur_mwh, mape_pct, directional_acc, n_samples)
VALUES
    (:evaluated_at, :period_start, :period_end, :model_version,
     :mae_eur_mwh, :rmse_eur_mwh, :mape_pct, :directional_acc, :n_samples)
"""


class Backtester:
    """Compares stored model predictions against actual OTE prices.

    The backtester loads both actuals and predictions from the project
    database, merges them on ``timestamp_utc``, and computes a set of
    standard regression and directional-accuracy metrics.

    Attributes:
        engine: SQLAlchemy engine connected to the project database.
        model_version: Identifier of the model version to evaluate,
            matching the ``model_version`` column in ``model_predictions``.

    Example::

        bt = Backtester(engine, model_version="1.0.0")
        actuals_df, predictions_df = bt.load_data(
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 30),
        )
        result = bt.evaluate(actuals_df, predictions_df)
    """

    def __init__(self, engine: Engine, model_version: str) -> None:
        """Initialise the Backtester.

        Args:
            engine: Connected SQLAlchemy engine (PostgreSQL or SQLite).
            model_version: Model version string to filter predictions, e.g.
                ``"1.0.0"``.
        """
        self.engine: Engine = engine
        self.model_version: str = model_version

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load actuals and predictions from the database.

        Queries ``fact_hourly_market_features`` for observed prices and
        ``model_predictions`` for the configured model version.  Both
        DataFrames are filtered to the optional date range.

        Args:
            start_date: Inclusive lower bound on ``timestamp_utc`` (UTC date).
                If ``None``, no lower bound is applied.
            end_date: Inclusive upper bound on ``timestamp_utc`` (UTC date).
                If ``None``, no upper bound is applied.

        Returns:
            A tuple ``(actuals_df, predictions_df)`` where:

            - ``actuals_df`` has columns ``["timestamp_utc", "price_eur_mwh"]``
            - ``predictions_df`` has columns
              ``["timestamp_utc", "predicted_price_eur"]``

            Both DataFrames are sorted by ``timestamp_utc`` ascending.

        Raises:
            RuntimeError: If the database query fails.
        """
        where_parts_a: list[str] = []
        where_parts_p: list[str] = []
        params: dict[str, Any] = {"model_version": self.model_version}

        if start_date is not None:
            start_str = start_date.isoformat()
            where_parts_a.append("AND timestamp_utc >= :start_date")
            where_parts_p.append("AND timestamp_utc >= :start_date")
            params["start_date"] = start_str

        if end_date is not None:
            # Include the full end day by going up to the end of that day
            end_str = f"{end_date.isoformat()}T23:59:59"
            where_parts_a.append("AND timestamp_utc <= :end_date")
            where_parts_p.append("AND timestamp_utc <= :end_date")
            params["end_date"] = end_str

        sql_a = _SQL_ACTUALS.format(where_clauses="\n  ".join(where_parts_a))
        sql_p = _SQL_PREDICTIONS.format(where_clauses="\n  ".join(where_parts_p))

        with self.engine.connect() as conn:
            logger.debug("Loading actuals from DB …")
            actuals_df = pd.read_sql(text(sql_a), conn, params=params)
            if actuals_df.empty:
                sql_a_raw = _SQL_ACTUALS_RAW.format(where_clauses="\n  ".join(where_parts_a))
                actuals_df = pd.read_sql(text(sql_a_raw), conn, params=params)

            logger.debug("Loading predictions from DB (version=%s) …", self.model_version)
            predictions_df = pd.read_sql(text(sql_p), conn, params=params)

        logger.info(
            "Loaded %d actuals and %d predictions.",
            len(actuals_df),
            len(predictions_df),
        )
        return actuals_df, predictions_df

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        actuals_df: pd.DataFrame | None = None,
        predictions_df: pd.DataFrame | None = None,
    ) -> dict[str, float | int]:
        """Merge actuals and predictions, then compute all metrics.

        An inner join on ``timestamp_utc`` is performed so only hours with
        both an actual price and a prediction are evaluated.  Hours present
        in only one of the DataFrames are silently dropped.

        If either ``actuals_df`` or ``predictions_df`` is not provided, they
        are loaded automatically via :meth:`load_data`.

        Args:
            actuals_df: DataFrame with columns ``["timestamp_utc",
                "price_eur_mwh"]`` as returned by :meth:`load_data`.
            predictions_df: DataFrame with columns ``["timestamp_utc",
                "predicted_price_eur"]`` as returned by :meth:`load_data`.

        Returns:
            A dictionary containing:

            - ``"mae"``                  – Mean Absolute Error (EUR/MWh).
            - ``"rmse"``                 – Root Mean Squared Error (EUR/MWh).
            - ``"mape"``                 – MAPE (%).
            - ``"directional_accuracy"`` – Direction accuracy in ``[0, 1]``.
            - ``"n_samples"``            – Number of matched hours.
            - ``"period_start"``         – ISO-8601 string of earliest matched timestamp.
            - ``"period_end"``           – ISO-8601 string of latest matched timestamp.

        Raises:
            ValueError: If the merged DataFrame is empty (no overlapping
                timestamps).
        """
        if actuals_df is None or predictions_df is None:
            actuals_df, predictions_df = self.load_data()
        merged = pd.merge(
            actuals_df,
            predictions_df,
            on="timestamp_utc",
            how="inner",
        ).sort_values("timestamp_utc")

        if merged.empty:
            raise ValueError(
                "No overlapping timestamps between actuals and predictions. "
                "Check date range or model_version."
            )

        actual_arr = merged["price_eur_mwh"].to_numpy()
        predicted_arr = merged["predicted_price_eur"].to_numpy()

        result = m.compute_all_metrics(actual_arr, predicted_arr)
        result["period_start"] = str(merged["timestamp_utc"].iloc[0])
        result["period_end"] = str(merged["timestamp_utc"].iloc[-1])

        logger.info(
            "Evaluation complete: MAE=%.4f, RMSE=%.4f, MAPE=%.2f%%, DA=%.4f (n=%d)",
            result["mae"],
            result["rmse"],
            result["mape"],
            result["directional_accuracy"],
            result["n_samples"],
        )
        return result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def evaluate_and_store(
        self,
        engine: Engine,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, float | int]:
        """Run the full load → evaluate → persist pipeline.

        Loads actuals and predictions for the given date range, computes
        all metrics, and inserts one summary row into ``evaluation_results``.

        Args:
            engine: SQLAlchemy engine used for the INSERT.  Can be the same
                engine stored in ``self.engine``; it is accepted as an
                explicit argument to match the interface used by orchestration
                scripts.
            start_date: Inclusive start date filter.  ``None`` means no lower
                bound.
            end_date: Inclusive end date filter.  ``None`` means no upper
                bound.

        Returns:
            The metrics dictionary returned by :meth:`evaluate`, augmented
            with the keys ``"period_start"`` and ``"period_end"``.

        Raises:
            ValueError: If no overlapping data is found.
            RuntimeError: If the database write fails.
        """
        actuals_df, predictions_df = self.load_data(start_date, end_date)
        result = self.evaluate(actuals_df, predictions_df)

        now_utc = datetime.now(tz=timezone.utc).isoformat()
        row = {
            "evaluated_at": now_utc,
            "period_start": result["period_start"],
            "period_end": result["period_end"],
            "model_version": self.model_version,
            "mae_eur_mwh": result["mae"],
            "rmse_eur_mwh": result["rmse"],
            "mape_pct": result["mape"],
            "directional_acc": result["directional_accuracy"],
            "n_samples": result["n_samples"],
        }

        with engine.begin() as conn:
            conn.execute(text(_SQL_INSERT_EVAL), row)

        logger.info("Evaluation results stored in 'evaluation_results' table.")
        return result

    # ------------------------------------------------------------------
    # Rolling evaluation
    # ------------------------------------------------------------------

    def rolling_evaluation(self, window_days: int = 7) -> pd.DataFrame:
        """Compute rolling MAE and RMSE over a sliding window.

        For each unique calendar day present in the prediction data the
        method computes MAE and RMSE using the preceding *window_days*
        days (inclusive).  This gives a smoothed view of model performance
        degradation or improvement over time.

        Algorithm:
            1. Load all available actuals and predictions (no date filter).
            2. Merge on ``timestamp_utc``.
            3. For each day *d* in the merged data, select all rows where
               ``timestamp_utc`` falls in ``[d - window_days + 1 days, d]``
               and compute metrics over that window.

        Args:
            window_days: Number of calendar days in the rolling window.
                Defaults to ``7`` (one week).

        Returns:
            A ``pd.DataFrame`` with one row per calendar day containing:

            - ``"date"``       – The calendar day (``datetime.date``).
            - ``"mae"``        – Rolling MAE for the window ending on that day.
            - ``"rmse"``       – Rolling RMSE for the window ending on that day.
            - ``"n_samples"``  – Number of hours in the window.

            Days with fewer than 2 samples in the window are assigned
            ``NaN`` for metric values.

        Raises:
            ValueError: If no data is available after the merge.
        """
        actuals_df, predictions_df = self.load_data()
        merged = pd.merge(
            actuals_df,
            predictions_df,
            on="timestamp_utc",
            how="inner",
        ).sort_values("timestamp_utc")

        if merged.empty:
            raise ValueError("No overlapping data found for rolling evaluation.")

        # Parse timestamp column to datetime for date arithmetic
        merged["ts"] = pd.to_datetime(merged["timestamp_utc"], utc=True)
        merged["date"] = merged["ts"].dt.date

        window = pd.Timedelta(days=window_days)
        rows: list[dict[str, Any]] = []

        for day in sorted(merged["date"].unique()):
            day_ts = pd.Timestamp(day, tz="UTC")
            window_start = day_ts - window + pd.Timedelta(days=1)
            mask = (merged["ts"] >= window_start) & (merged["ts"] <= day_ts + pd.Timedelta(hours=23, minutes=59))
            subset = merged.loc[mask]

            n = len(subset)
            if n < 2:
                rows.append({"date": day, "mae": float("nan"), "rmse": float("nan"), "n_samples": n})
                continue

            a = subset["price_eur_mwh"].to_numpy()
            p = subset["predicted_price_eur"].to_numpy()
            rows.append(
                {
                    "date": day,
                    "mae": m.mae(a, p),
                    "rmse": m.rmse(a, p),
                    "n_samples": n,
                }
            )

        result_df = pd.DataFrame(rows)
        logger.info(
            "Rolling evaluation complete: %d days, window=%d days.",
            len(result_df),
            window_days,
        )
        return result_df
