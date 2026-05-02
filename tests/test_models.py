"""
tests/test_models.py
====================
Tests for :mod:`src.models.train` and :mod:`src.models.predict`.

All tests are self-contained: they generate synthetic DataFrames in-memory
and do not touch the file system or any database (except where explicitly
noted for save_predictions).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.database.db_manager import get_engine, init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_or_skip(module_path: str) -> Any:
    """Import *module_path* or skip the test if it cannot be imported.

    Args:
        module_path: Dotted module path.

    Returns:
        The imported module object.
    """
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Module not yet available: {exc}")


def _load_train_module() -> Any:
    """Load :mod:`src.models.train` or skip.

    Returns:
        The train module.
    """
    return _import_or_skip("src.models.train")


def _load_predict_module() -> Any:
    """Load :mod:`src.models.predict` or skip.

    Returns:
        The predict module.
    """
    return _import_or_skip("src.models.predict")


def _build_synthetic_df(
    n_rows: int,
    feature_cols: list[str],
    target_col: str,
    *,
    start: str = "2023-01-01",
    freq: str = "h",
) -> pd.DataFrame:
    """Generate a synthetic DataFrame for model training / evaluation.

    Args:
        n_rows: Number of rows to generate.
        feature_cols: List of feature column names to populate with random data.
        target_col: Name of the target column (filled with random positive floats).
        start: ISO-8601 datetime string for the first timestamp.
        freq: Pandas frequency alias (default ``"h"`` for hourly).

    Returns:
        A :class:`pandas.DataFrame` with a ``timestamp_utc`` index column,
        all *feature_cols* as float columns, and the *target_col*.
    """
    rng = np.random.default_rng(seed=42)
    timestamps = pd.date_range(start, periods=n_rows, freq=freq, tz="UTC")
    data: dict[str, Any] = {"timestamp_utc": timestamps}

    for col in feature_cols:
        data[col] = rng.uniform(0.0, 100.0, size=n_rows)

    # Target: realistic-looking positive electricity prices
    data[target_col] = rng.uniform(20.0, 200.0, size=n_rows)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Feature column tests
# ---------------------------------------------------------------------------


class TestFeatureCols:
    """Tests for the :data:`~src.models.train.FEATURE_COLS` constant."""

    def test_feature_cols_completeness(self) -> None:
        """FEATURE_COLS must be a non-empty list of strings.

        Imports the constant from :mod:`src.models.train` and asserts that it
        is a non-empty :class:`list` whose every element is a :class:`str`.
        """
        mod = _load_train_module()

        feature_cols = mod.FEATURE_COLS

        assert isinstance(feature_cols, list), (
            f"FEATURE_COLS should be a list, got {type(feature_cols).__name__}."
        )
        assert len(feature_cols) > 0, "FEATURE_COLS must not be empty."
        assert all(isinstance(col, str) for col in feature_cols), (
            "Every element of FEATURE_COLS must be a str."
        )


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


class TestModelTrainer:
    """Tests for :class:`src.models.train.ModelTrainer`."""

    def test_train_produces_model(self) -> None:
        """ModelTrainer.train() must return an object with a predict() method.

        Generates a 300-row synthetic DataFrame, trains the model, and confirms
        that the returned artefact exposes a callable ``predict`` attribute –
        the minimal interface required by the prediction pipeline.
        """
        mod = _load_train_module()

        feature_cols: list[str] = mod.FEATURE_COLS
        target_col: str = mod.TARGET_COL

        df = _build_synthetic_df(300, feature_cols, target_col)
        trainer = mod.ModelTrainer()
        model = trainer.train(df)

        assert model is not None, "train() returned None."
        assert callable(getattr(model, "predict", None)), (
            "The trained model must expose a callable predict() method."
        )

    def test_time_series_cv_no_data_leakage(self) -> None:
        """time_series_cv must ensure max(train_ts) < min(val_ts) in every fold.

        Runs time-series cross-validation with n_splits=3 on a 300-row dataset
        and iterates over each fold, verifying that the training window never
        extends beyond the validation window (no data leakage).
        """
        mod = _load_train_module()

        feature_cols: list[str] = mod.FEATURE_COLS
        target_col: str = mod.TARGET_COL

        df = _build_synthetic_df(300, feature_cols, target_col)
        trainer = mod.ModelTrainer()

        folds = trainer.time_series_cv(df, n_splits=3)

        assert folds, "time_series_cv returned no folds."

        for fold_idx, fold in enumerate(folds):
            # Accept both dict-style and tuple-style fold representation
            if isinstance(fold, dict):
                train_ts = fold.get("train_timestamps") or fold.get("train")["timestamp_utc"]
                val_ts = fold.get("val_timestamps") or fold.get("val")["timestamp_utc"]
            elif isinstance(fold, (tuple, list)) and len(fold) == 2:
                train_df, val_df = fold
                train_ts = train_df["timestamp_utc"]
                val_ts = val_df["timestamp_utc"]
            else:
                pytest.fail(
                    f"Unexpected fold format at index {fold_idx}: {type(fold)}"
                )

            max_train = pd.Timestamp(max(train_ts))
            min_val = pd.Timestamp(min(val_ts))

            assert max_train < min_val, (
                f"Fold {fold_idx}: data leakage detected – "
                f"max train timestamp ({max_train}) >= min val timestamp ({min_val})."
            )


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


class TestPricePredictor:
    """Tests for :class:`src.models.predict.PricePredictor`."""

    def _get_trained_predictor(self):
        """Return a :class:`PricePredictor` built from a minimal trained model.

        Returns:
            A :class:`~src.models.predict.PricePredictor` instance ready for
            inference.
        """
        train_mod = _load_train_module()
        predict_mod = _load_predict_module()

        feature_cols: list[str] = train_mod.FEATURE_COLS
        target_col: str = train_mod.TARGET_COL

        df = _build_synthetic_df(300, feature_cols, target_col)
        trainer = train_mod.ModelTrainer()
        model = trainer.train(df)

        predictor = predict_mod.PricePredictor(model=model)
        return predictor, feature_cols

    def test_predict_output_shape(self) -> None:
        """predict_next_day() must return exactly 24 rows with the right columns.

        Trains a minimal model, passes a 24-row feature DataFrame to
        :meth:`~PricePredictor.predict_next_day`, and verifies:

        * The output has exactly 24 rows.
        * The output contains columns ``target_timestamp_utc`` and
          ``predicted_price_eur``.
        """
        predictor, feature_cols = self._get_trained_predictor()

        feature_df = _build_synthetic_df(24, feature_cols, "price_eur_mwh")
        result = predictor.predict_next_day(feature_df)

        assert isinstance(result, pd.DataFrame), (
            "predict_next_day() must return a pandas DataFrame."
        )
        assert len(result) == 24, (
            f"Expected 24 rows in prediction output, got {len(result)}."
        )

        required_cols = {"target_timestamp_utc", "predicted_price_eur"}
        missing = required_cols - set(result.columns)
        assert not missing, (
            f"Prediction DataFrame is missing columns: {missing}."
        )
