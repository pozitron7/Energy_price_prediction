"""Day-ahead price prediction service for the Energy Price Prediction project.

This module loads a trained XGBoost model from disk and produces 24-hour-ahead
electricity price forecasts.  It integrates with :mod:`src.transformation.feature_engineering`
for feature construction and persists predictions to the ``model_predictions``
database table.

Typical usage
-------------
>>> from pathlib import Path
>>> from sqlalchemy import create_engine
>>> from src.models.predict import PricePredictor
>>>
>>> engine = create_engine("postgresql+psycopg2://user:pw@localhost/eprice")
>>> predictor = PricePredictor(model_path=Path("models/xgb_v1.joblib"), model_version="v1")
>>> predictions = predictor.run(engine, weather_forecast_df, last_prices_df)
>>> print(predictions)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import Engine, text

from src.models.train import FEATURE_COLS
from src.transformation.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREDICTIONS_TABLE = "model_predictions"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PricePredictor:
    """Day-ahead electricity price prediction service.

    Loads a pre-trained model artefact from disk, constructs inference
    features from weather forecasts and recent price history, runs the model,
    and persists the resulting 24-hour price predictions to the database.

    Attributes:
        model_path: Path to the ``.joblib`` artefact produced by
            :meth:`src.models.train.ModelTrainer.save_model`.
        model_version: Version label used when writing predictions to the DB.
        _model: Loaded :class:`xgboost.XGBRegressor` instance (``None`` until
            :meth:`load_model` is called).
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        model_version: str = "1.0.0",
        *,
        model: Any = None,
    ) -> None:
        """Initialise the predictor.

        Args:
            model_path: Filesystem path to a ``.joblib`` model artefact.
                If None, defaults to config.settings.MODEL_PATH.
            model_version: Short version label, e.g. ``"v1"`` or ``"1.0.0"``.
            model: Pre-loaded model instance (e.g. for testing).
        """
        if model_path is None:
            from config.settings import MODEL_PATH
            self.model_path = Path(MODEL_PATH)
        else:
            self.model_path = Path(model_path)
        self.model_version: str = model_version
        self._model: Any = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the serialised model artefact from :attr:`model_path`.

        The joblib artefact is expected to be a dict with the keys produced by
        :meth:`src.models.train.ModelTrainer.save_model`:
        ``"model"``, ``"model_version"``, ``"feature_cols"``,
        ``"target_col"``.

        Raises:
            FileNotFoundError: If :attr:`model_path` does not exist on disk.
            KeyError: If the artefact dict is missing the ``"model"`` key.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artefact not found at '{self.model_path}'. "
                "Run ModelTrainer.run() to train and save the model first."
            )

        artefact: dict[str, Any] = joblib.load(self.model_path)
        self._model = artefact["model"]
        saved_version: str = artefact.get("model_version", "unknown")
        logger.info(
            "Loaded model version '%s' from '%s'.",
            saved_version,
            self.model_path,
        )

    def predict_next_day(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Run the model on inference features and return a predictions DataFrame.

        Args:
            feature_df: DataFrame produced by
                :meth:`src.transformation.feature_engineering.FeatureEngineer.build_inference_features`.
                Must be indexed by ``timestamp_utc`` (UTC-aware) and contain
                all columns in :data:`src.models.train.FEATURE_COLS`.

        Returns:
            DataFrame with two columns:

            - ``target_timestamp_utc``: UTC timestamp for each predicted hour.
            - ``predicted_price_eur``: Model's price forecast in EUR/MWh.

        Raises:
            RuntimeError: If :meth:`load_model` has not been called before
                this method.
            ValueError: If ``feature_df`` is empty or missing required feature
                columns.
        """
        if self._model is None:
            raise RuntimeError(
                "Model is not loaded. Call load_model() before predict_next_day()."
            )

        if feature_df.empty:
            raise ValueError("feature_df is empty – cannot generate predictions.")

        missing_cols = [c for c in FEATURE_COLS if c not in feature_df.columns]
        if missing_cols:
            raise ValueError(
                f"feature_df is missing required columns: {missing_cols}"
            )

        X = feature_df[FEATURE_COLS].fillna(0)  # XGBoost expects no NaNs
        raw_predictions: pd.Series = pd.Series(
            self._model.predict(X),
            index=feature_df.index,
            name="predicted_price_eur",
        )

        predictions_df = pd.DataFrame(
            {
                "target_timestamp_utc": raw_predictions.index,
                "predicted_price_eur": raw_predictions.values,
            }
        )

        logger.info(
            "Generated %d price predictions (%.2f – %.2f EUR/MWh).",
            len(predictions_df),
            predictions_df["predicted_price_eur"].min(),
            predictions_df["predicted_price_eur"].max(),
        )
        return predictions_df

    def save_predictions(self, predictions: pd.DataFrame, engine: Engine) -> int:
        """Upsert price predictions into the ``model_predictions`` table.

        Uses INSERT … ON CONFLICT (target_timestamp_utc, model_version) DO
        UPDATE for PostgreSQL.  Falls back to a simple ``to_sql`` append for
        SQLite (used in local dev / CI).

        Args:
            predictions: DataFrame returned by :meth:`predict_next_day`.
                Must contain ``target_timestamp_utc`` and
                ``predicted_price_eur`` columns.
            engine: SQLAlchemy ``Engine`` connected to the project database.

        Returns:
            Number of rows written / updated.
        """
        if predictions.empty:
            logger.warning("No predictions to save – DataFrame is empty.")
            return 0

        # Attach model version to each row
        df = predictions.copy()
        df["model_version"] = self.model_version

        dialect = engine.dialect.name
        rows_written: int = 0

        if dialect == "sqlite":
            # SQLite – simpler approach suitable for tests and local dev
            sqlite_records = df[
                ["target_timestamp_utc", "predicted_price_eur", "model_version"]
            ].copy()
            sqlite_records["target_timestamp_utc"] = pd.to_datetime(
                sqlite_records["target_timestamp_utc"], utc=True
            ).map(lambda t: t.isoformat())
            sqlite_records = sqlite_records.to_dict(orient="records")
            with engine.begin() as conn:
                conn.execute(
                    text(
                        f"INSERT OR REPLACE INTO {_PREDICTIONS_TABLE} "
                        "(target_timestamp_utc, predicted_price_eur, model_version) "
                        "VALUES (:target_timestamp_utc, :predicted_price_eur, :model_version)"
                    ),
                    sqlite_records,
                )
            rows_written = len(df)
            logger.info(
                "SQLite: appended %d rows to %s.", rows_written, _PREDICTIONS_TABLE
            )
            return rows_written

        # PostgreSQL – upsert
        upsert_sql = text(
            f"""
            INSERT INTO {_PREDICTIONS_TABLE}
                (target_timestamp_utc, predicted_price_eur, model_version)
            VALUES
                (:target_timestamp_utc, :predicted_price_eur, :model_version)
            ON CONFLICT (target_timestamp_utc, model_version) DO UPDATE
                SET predicted_price_eur = EXCLUDED.predicted_price_eur;
            """
        )

        records = df[
            ["target_timestamp_utc", "predicted_price_eur", "model_version"]
        ].to_dict(orient="records")

        with engine.begin() as conn:
            conn.execute(upsert_sql, records)

        rows_written = len(records)
        logger.info(
            "PostgreSQL: upserted %d prediction rows into %s.",
            rows_written,
            _PREDICTIONS_TABLE,
        )
        return rows_written

    def run(
        self,
        engine: Engine,
        weather_forecast_df: pd.DataFrame,
        last_prices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Full day-ahead prediction pipeline.

        Orchestrates:
        1. Feature construction via :class:`~src.transformation.feature_engineering.FeatureEngineer`.
        2. Model inference via :meth:`predict_next_day`.
        3. Persistence via :meth:`save_predictions`.

        If the model has not yet been loaded (i.e. :meth:`load_model` was not
        called externally), it is loaded automatically at the start of this
        method.

        Args:
            engine: SQLAlchemy ``Engine`` connected to the project database.
            weather_forecast_df: Forecast weather DataFrame required by
                :meth:`~src.transformation.feature_engineering.FeatureEngineer.build_inference_features`.
                Must cover at least the next 24 hours.
            last_prices_df: Recent actual price DataFrame required for lag
                feature computation.  Must cover at least the past 168 hours.

        Returns:
            Predictions DataFrame as returned by :meth:`predict_next_day`
            (columns: ``target_timestamp_utc``, ``predicted_price_eur``).
        """
        # ------------------------------------------------------------------
        # 1. Ensure model is loaded
        # ------------------------------------------------------------------
        if self._model is None:
            logger.info("Model not pre-loaded – loading now from '%s'.", self.model_path)
            self.load_model()

        # ------------------------------------------------------------------
        # 2. Build inference features
        # ------------------------------------------------------------------
        fe = FeatureEngineer()
        feature_df = fe.build_inference_features(
            weather_forecast_df=weather_forecast_df,
            last_prices_df=last_prices_df,
        )

        # Predict the first 24 hours *after* the latest known price, i.e. the
        # delivery day whose OTE prices are not published yet.
        if not feature_df.empty:
            last_known = pd.to_datetime(last_prices_df["timestamp_utc"], utc=True).max()
            idx = pd.to_datetime(feature_df.index, utc=True)
            feature_df = feature_df[idx > last_known].iloc[:24]

        # ------------------------------------------------------------------
        # 3. Run inference
        # ------------------------------------------------------------------
        predictions = self.predict_next_day(feature_df)

        # ------------------------------------------------------------------
        # 4. Persist
        # ------------------------------------------------------------------
        self.save_predictions(predictions, engine)

        return predictions
