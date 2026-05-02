"""
tests/test_evaluation.py
=========================
Tests for :mod:`src.evaluation.metrics` and :mod:`src.evaluation.backtest`.

All tests are self-contained: metric tests operate on NumPy arrays; the
Backtester integration test uses an in-memory SQLite database.
"""

from __future__ import annotations

import importlib
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.database.db_manager import get_engine, init_db
from sqlalchemy import text


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


@pytest.fixture()
def memory_engine():
    """Provide a fresh in-memory SQLite engine with all tables initialised.

    Yields:
        A :class:`~sqlalchemy.engine.Engine` pointed at ``sqlite:///:memory:``.
    """
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# Metric function tests
# ---------------------------------------------------------------------------


class TestMAE:
    """Tests for :func:`src.evaluation.metrics.mae`."""

    def test_mae_perfect_prediction(self) -> None:
        """mae(a, a) must equal 0.0 for any array a.

        Passes the same array as both actuals and predictions and asserts the
        result is exactly 0.0 (perfect forecast).
        """
        mod = _import_or_skip("src.evaluation.metrics")
        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = mod.mae(a, a)
        assert result == pytest.approx(0.0), (
            f"mae(a, a) should be 0.0, got {result}."
        )


class TestRMSE:
    """Tests for :func:`src.evaluation.metrics.rmse`."""

    def test_rmse_known_values(self) -> None:
        """rmse must match the hand-computed value for a known array pair.

        Actuals:     [1, 2, 3]
        Predictions: [2, 2, 2]
        Errors:      [1, 0, 1]   →  MSE = (1+0+1)/3 ≈ 0.6667  →  RMSE ≈ 0.8165
        """
        mod = _import_or_skip("src.evaluation.metrics")
        actuals = np.array([1.0, 2.0, 3.0])
        predictions = np.array([2.0, 2.0, 2.0])

        expected_rmse = math.sqrt(2.0 / 3.0)  # ≈ 0.8165
        result = mod.rmse(actuals, predictions)

        assert result == pytest.approx(expected_rmse, rel=1e-5), (
            f"Expected RMSE ≈ {expected_rmse:.6f}, got {result}."
        )


class TestMAPE:
    """Tests for :func:`src.evaluation.metrics.mape`."""

    def test_mape_zero_actuals(self) -> None:
        """mape must not raise ZeroDivisionError when actuals contain zeros.

        Passes an actuals array with a zero entry and verifies the function
        returns a finite (non-NaN, non-Inf) value, confirming epsilon-based
        zero-handling is in place.
        """
        mod = _import_or_skip("src.evaluation.metrics")
        actuals = np.array([0.0, 100.0, 200.0])
        predictions = np.array([10.0, 90.0, 210.0])

        result = mod.mape(actuals, predictions)

        assert math.isfinite(result), (
            f"mape() returned a non-finite value ({result}) when actuals contain 0."
        )
        assert result >= 0.0, "MAPE must be non-negative."


class TestDirectionalAccuracy:
    """Tests for :func:`src.evaluation.metrics.directional_accuracy`."""

    def test_directional_accuracy_all_correct(self) -> None:
        """directional_accuracy must return 1.0 when every direction is correct.

        Constructs actuals and predictions that always move in the same
        direction (both increase) and asserts the result is 1.0.
        """
        mod = _import_or_skip("src.evaluation.metrics")
        actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Predictions also always increase → all directions match
        predictions = np.array([12.0, 22.0, 32.0, 42.0, 52.0])

        result = mod.directional_accuracy(actuals, predictions)
        assert result == pytest.approx(1.0), (
            f"Expected directional_accuracy=1.0, got {result}."
        )

    def test_directional_accuracy_all_wrong(self) -> None:
        """directional_accuracy must return 0.0 when every direction is wrong.

        Constructs actuals that always increase while predictions always
        decrease (opposite direction every step), expecting a result of 0.0.
        """
        mod = _import_or_skip("src.evaluation.metrics")
        actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # Predictions decrease while actuals increase → all directions wrong
        predictions = np.array([50.0, 40.0, 30.0, 20.0, 10.0])

        result = mod.directional_accuracy(actuals, predictions)
        assert result == pytest.approx(0.0), (
            f"Expected directional_accuracy=0.0, got {result}."
        )


class TestComputeAllMetrics:
    """Tests for :func:`src.evaluation.metrics.compute_all_metrics`."""

    def test_compute_all_metrics_keys(self) -> None:
        """compute_all_metrics must return a dict with the four required keys.

        Passes a simple array pair and verifies the output dictionary contains
        exactly the keys: ``mae``, ``rmse``, ``mape``, ``directional_accuracy``.
        """
        mod = _import_or_skip("src.evaluation.metrics")
        actuals = np.array([10.0, 20.0, 30.0])
        predictions = np.array([11.0, 19.0, 31.0])

        result = mod.compute_all_metrics(actuals, predictions)

        assert isinstance(result, dict), (
            f"compute_all_metrics must return a dict, got {type(result).__name__}."
        )

        required_keys = {"mae", "rmse", "mape", "directional_accuracy"}
        missing_keys = required_keys - result.keys()
        assert not missing_keys, (
            f"compute_all_metrics result is missing keys: {missing_keys}."
        )


# ---------------------------------------------------------------------------
# Backtester integration test
# ---------------------------------------------------------------------------


class TestBacktester:
    """Integration test for :class:`src.evaluation.backtest.Backtester`."""

    _N: int = 48  # number of hours to synthesise

    def _populate_predictions_and_actuals(self, engine) -> None:
        """Insert synthetic predictions and matching actuals into the test DB.

        Inserts *_N* rows into ``model_predictions`` and matching price rows
        into ``raw_energy_prices`` so the Backtester has data to evaluate.

        Args:
            engine: Bound SQLAlchemy engine with tables already created.
        """
        start = datetime(2024, 3, 1, tzinfo=timezone.utc)
        import datetime as dt

        pred_rows = []
        price_rows = []
        for i in range(self._N):
            ts = start + dt.timedelta(hours=i)
            ts_str = ts.isoformat()
            predicted = 50.0 + (i % 30)
            actual = 48.0 + (i % 28)

            pred_rows.append(
                f"(lower(hex(randomblob(16))), '{ts_str}', {predicted}, 'test_v1')"
            )
            price_rows.append(
                f"('{ts_str}', 'day_ahead', {actual})"
            )

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT OR IGNORE INTO model_predictions "
                "(prediction_id, target_timestamp_utc, predicted_price_eur, model_version) VALUES "
                + ", ".join(pred_rows)
            ))
            conn.execute(text(
                "INSERT OR IGNORE INTO raw_energy_prices "
                "(timestamp_utc, market_type, price_eur_mwh) VALUES "
                + ", ".join(price_rows)
            ))

    def test_backtester_evaluate_with_synthetic_data(self, memory_engine) -> None:
        """Backtester.evaluate() must return a dict with expected keys and non-negative values.

        Populates the in-memory SQLite database with synthetic predictions and
        actuals, runs :meth:`~Backtester.evaluate`, and asserts:

        * The result is a :class:`dict`.
        * It contains the keys: ``mae``, ``rmse``, ``mape``,
          ``directional_accuracy``.
        * All metric values are non-negative real numbers.
        """
        mod = _import_or_skip("src.evaluation.backtest")

        self._populate_predictions_and_actuals(memory_engine)

        backtester = mod.Backtester(engine=memory_engine, model_version="test_v1")
        metrics = backtester.evaluate()

        assert isinstance(metrics, dict), (
            f"Backtester.evaluate() must return a dict, got {type(metrics).__name__}."
        )

        required_keys = {"mae", "rmse", "mape", "directional_accuracy"}
        missing_keys = required_keys - metrics.keys()
        assert not missing_keys, (
            f"evaluate() result is missing keys: {missing_keys}."
        )

        for key in required_keys:
            val = metrics[key]
            assert isinstance(val, (int, float)), (
                f"Metric '{key}' must be numeric, got {type(val).__name__}."
            )
            assert math.isfinite(float(val)), (
                f"Metric '{key}' must be finite, got {val}."
            )
            assert float(val) >= 0.0, (
                f"Metric '{key}' must be non-negative, got {val}."
            )
