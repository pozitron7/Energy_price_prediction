"""XGBoost model training pipeline for the Energy Price Prediction project.

This module implements walk-forward cross-validation and full training of an
``XGBRegressor`` on the hourly feature table produced by
:mod:`src.transformation.feature_engineering`.

Typical usage
-------------
>>> from pathlib import Path
>>> from sqlalchemy import create_engine
>>> from src.models.train import ModelTrainer
>>>
>>> engine = create_engine("postgresql+psycopg2://user:pw@localhost/eprice")
>>> trainer = ModelTrainer(model_path=Path("models/xgb_v1.joblib"), model_version="v1")
>>> metrics = trainer.run(engine)
>>> print(metrics)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL: str = "price_eur_mwh"

#: Complete list of feature column names consumed by the model.
#: The order here is preserved when constructing the feature matrix.
FEATURE_COLS: list[str] = [
    # Cyclic calendar encodings
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # Autoregressive lag features
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    # Averaged weather features
    "avg_temperature_c",
    "avg_cloud_cover_pct",
    "avg_wind_speed",
    "avg_irradiance",
    # Raw calendar features
    "hour_of_day",
    "day_of_week",
    "month",
    "is_holiday",
    # NOTE: actual solar/wind generation (fact table columns) is deliberately NOT
    # a feature: it is unknown the day before delivery, so using it would leak
    # future information into training/backtests. Weather forecasts act as the
    # day-ahead proxy for renewable output.
]

#: Default XGBoost hyperparameters.
_XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

_FACT_TABLE = "fact_hourly_market_features"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error, ignoring zero targets.

    Args:
        y_true: Array of actual values.
        y_pred: Array of predicted values.

    Returns:
        MAPE as a percentage (0–100 scale).
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE regression metrics.

    Args:
        y_true: Array of actual target values.
        y_pred: Array of model predictions.

    Returns:
        Dictionary with keys ``mae``, ``rmse``, and ``mape``.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = _mape(y_true, y_pred)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


# ---------------------------------------------------------------------------
# Patch missing import (math needed by helper)
# ---------------------------------------------------------------------------
import math  # noqa: E402  – placed here to keep the module-level block clean


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ModelTrainer:
    """Orchestrates loading, cross-validating, training, and saving the model.

    Attributes:
        model_path: Filesystem path where the serialised model is persisted.
        model_version: Human-readable version string embedded in artefact names
            and logged for traceability.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        model_version: str = "1.0.0",
    ) -> None:
        """Initialise the trainer.

        Args:
            model_path: ``Path`` to the ``.joblib`` file that will be written
                after training. If None, defaults to config.settings.MODEL_PATH.
            model_version: Short version label, e.g. ``"v1"`` or ``"1.0.0"``.
        """
        if model_path is None:
            from config.settings import MODEL_PATH
            self.model_path = Path(MODEL_PATH)
        else:
            self.model_path = Path(model_path)
        self.model_version: str = model_version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_training_data(self, engine: Engine) -> pd.DataFrame:
        """Load training data from ``fact_hourly_market_features``.

        Rows where ``price_eur_mwh`` (the target) is ``NULL`` / ``NaN`` are
        dropped immediately, as they cannot contribute to supervised learning.

        Args:
            engine: SQLAlchemy ``Engine`` connected to the project database.

        Returns:
            DataFrame indexed by ``timestamp_utc`` (UTC-aware) with all
            feature columns and the target column.

        Raises:
            ValueError: If the loaded table is empty after dropping NaN
                targets.
        """
        cols_to_load = ", ".join(f'"{c}"' for c in ["timestamp_utc", TARGET_COL] + FEATURE_COLS)
        query = text(
            f"SELECT {cols_to_load} FROM {_FACT_TABLE} ORDER BY timestamp_utc"
        )

        with engine.connect() as conn:
            df = pd.read_sql(query, conn, parse_dates=["timestamp_utc"])

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.set_index("timestamp_utc").sort_index()

        before = len(df)
        df = df.dropna(subset=[TARGET_COL])
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with NaN target during load.", dropped)

        if df.empty:
            raise ValueError(
                "Training dataset is empty after dropping NaN targets. "
                "Ensure fact_hourly_market_features is populated."
            )

        logger.info("Loaded %d training rows.", len(df))
        return df

    def time_series_cv(
        self, df: pd.DataFrame, n_splits: int = 5
    ) -> list[dict[str, Any]]:
        """Walk-forward time-series cross-validation.

        Splits the data into ``n_splits`` folds.  In each fold, the model is
        trained on all data strictly before the validation window; the
        validation window uses the chronologically subsequent slice.  This
        mirrors real-world usage where the model never sees future data.

        Args:
            df: Training DataFrame indexed by a chronologically sorted UTC
                ``DatetimeIndex``.  Must contain :data:`FEATURE_COLS` and
                :data:`TARGET_COL`.
            n_splits: Number of walk-forward folds (default: 5).

        Returns:
            List of dicts, one per fold, each containing:
            ``fold``, ``train_size``, ``val_size``, ``mae``, ``rmse``,
            ``mape``, ``val_start``, ``val_end``.
        """
        logger.info("Starting %d-fold walk-forward CV …", n_splits)
        n = len(df)
        # Minimum training window = 70 % of data; validation folds cover remaining 30 %
        min_train = int(n * 0.7)
        fold_size = (n - min_train) // n_splits

        results: list[dict[str, Any]] = []

        for fold in range(n_splits):
            train_end = min_train + fold * fold_size
            val_start = train_end
            val_end = val_start + fold_size

            if val_end > n:
                break

            train_df = df.iloc[:train_end]
            val_df = df.iloc[val_start:val_end]

            X_train = train_df[FEATURE_COLS].fillna(0)
            y_train = train_df[TARGET_COL]
            X_val = val_df[FEATURE_COLS].fillna(0)
            y_val = val_df[TARGET_COL]

            model = xgb.XGBRegressor(**_XGB_DEFAULTS)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)
            metrics = _compute_metrics(y_val.values, y_pred)

            if "timestamp_utc" in train_df.columns:
                train_timestamps = list(train_df["timestamp_utc"])
                val_timestamps = list(val_df["timestamp_utc"])
            else:
                train_timestamps = list(train_df.index)
                val_timestamps = list(val_df.index)

            val_start_val = val_df.index[0].isoformat() if hasattr(val_df.index[0], "isoformat") else str(val_df.index[0])
            val_end_val = val_df.index[-1].isoformat() if hasattr(val_df.index[-1], "isoformat") else str(val_df.index[-1])

            fold_result: dict[str, Any] = {
                "fold": fold + 1,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "val_start": val_start_val,
                "val_end": val_end_val,
                "train": train_df,
                "val": val_df,
                "train_timestamps": train_timestamps,
                "val_timestamps": val_timestamps,
                **metrics,
            }
            results.append(fold_result)
            logger.info(
                "Fold %d/%d – MAE: %.3f, RMSE: %.3f, MAPE: %.2f%%",
                fold + 1,
                n_splits,
                metrics["mae"],
                metrics["rmse"],
                metrics["mape"],
            )

        return results

    def train(self, df: pd.DataFrame) -> xgb.XGBRegressor:
        """Train the final XGBoost model on the full dataset.

        Uses the last 20 % of the (chronologically sorted) data as a
        held-out validation set for early stopping.  The best iteration is
        retained automatically by XGBoost.

        Args:
            df: Full training DataFrame with :data:`FEATURE_COLS` and
                :data:`TARGET_COL`.  Rows with NaN features are zero-filled
                so XGBoost never receives ``NaN`` inputs.

        Returns:
            Fitted :class:`xgboost.XGBRegressor` instance.
        """
        split = int(len(df) * 0.8)
        train_df = df.iloc[:split]
        val_df = df.iloc[split:]

        X_train = train_df[FEATURE_COLS].fillna(0)
        y_train = train_df[TARGET_COL]
        X_val = val_df[FEATURE_COLS].fillna(0)
        y_val = val_df[TARGET_COL]

        params = {**_XGB_DEFAULTS, "early_stopping_rounds": 50}
        model = xgb.XGBRegressor(**params)

        logger.info(
            "Training XGBoost on %d rows (val set: %d rows) …",
            len(train_df),
            len(val_df),
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        logger.info("Training complete. Best iteration: %d.", model.best_iteration)

        # ------------------------------------------------------------------
        # Feature importances – top 10
        # ------------------------------------------------------------------
        importance_scores: np.ndarray = model.feature_importances_
        importance_df = (
            pd.DataFrame({"feature": FEATURE_COLS, "importance": importance_scores})
            .sort_values("importance", ascending=False)
            .head(10)
        )
        logger.info("Top 10 feature importances:\n%s", importance_df.to_string(index=False))
        print("\n=== Top 10 Feature Importances ===")
        print(importance_df.to_string(index=False))
        print("=" * 34)

        return model

    def save_model(self, model: xgb.XGBRegressor) -> Path:
        """Serialise the trained model to disk using ``joblib``.

        Creates parent directories if they do not exist.

        Args:
            model: Fitted :class:`xgboost.XGBRegressor` to persist.

        Returns:
            Absolute :class:`~pathlib.Path` to the saved ``.joblib`` file.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        artefact = {
            "model": model,
            "model_version": self.model_version,
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
        }
        joblib.dump(artefact, self.model_path)
        logger.info("Model saved to %s.", self.model_path.resolve())
        return self.model_path.resolve()

    def run(self, engine: Engine) -> dict[str, Any]:
        """End-to-end training pipeline: load → CV → train → save.

        Args:
            engine: SQLAlchemy ``Engine`` connected to the project database.

        Returns:
            Dictionary with keys:
            - ``"cv_results"``: list of per-fold metric dicts.
            - ``"cv_mean_mae"``: mean MAE across CV folds.
            - ``"cv_mean_rmse"``: mean RMSE across CV folds.
            - ``"model_path"``: absolute path string to the saved model.
            - ``"model_version"``: version label.
        """
        # 1. Load data
        df = self.load_training_data(engine)

        # 2. Walk-forward cross-validation
        cv_results = self.time_series_cv(df)
        if cv_results:
            mean_mae = round(float(np.mean([r["mae"] for r in cv_results])), 4)
            mean_rmse = round(float(np.mean([r["rmse"] for r in cv_results])), 4)
        else:
            mean_mae = float("nan")
            mean_rmse = float("nan")
        logger.info("CV summary – mean MAE: %.4f, mean RMSE: %.4f", mean_mae, mean_rmse)

        # 3. Train on full dataset
        model = self.train(df)

        # 4. Persist
        saved_path = self.save_model(model)

        return {
            "cv_results": cv_results,
            "cv_mean_mae": mean_mae,
            "cv_mean_rmse": mean_rmse,
            "model_path": str(saved_path),
            "model_version": self.model_version,
        }

