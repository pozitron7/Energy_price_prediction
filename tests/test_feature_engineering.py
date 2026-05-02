"""
tests/test_feature_engineering.py
==================================
Tests for :mod:`src.transformation.feature_engineering`.

All tests use an in-memory SQLite database populated with minimal synthetic
data.  No live database or external API is required.
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
# Helper – lazy import guard
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def memory_engine():
    """Provide a fresh in-memory SQLite engine with all tables created.

    Yields:
        A :class:`~sqlalchemy.engine.Engine` pointed at ``sqlite:///:memory:``.
    """
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    yield engine
    engine.dispose()


# ---------------------------------------------------------------------------
# Cyclic feature tests (pure logic, no DB needed)
# ---------------------------------------------------------------------------


class TestCyclicFeatures:
    """Tests for the internal cyclic-encoding helper in FeatureEngineer."""

    def test_cyclic_features_range(self) -> None:
        """hour_sin and hour_cos values must be in the closed interval [-1, 1].

        Creates a DataFrame with one row per hour for a full 24-hour day,
        invokes the cyclic transform, and asserts that all encoded values lie
        within the expected trigonometric range.
        """
        mod = _import_or_skip("src.transformation.feature_engineering")

        timestamps = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
        df = pd.DataFrame({"timestamp_utc": timestamps})
        df["hour_of_day"] = df["timestamp_utc"].dt.hour

        engineer = mod.FeatureEngineer()

        # Support both a public method and the private _add_cyclic_features helper
        if hasattr(engineer, "_add_cyclic_features"):
            result = engineer._add_cyclic_features(df.copy())
        elif hasattr(engineer, "add_cyclic_features"):
            result = engineer.add_cyclic_features(df.copy())
        else:
            pytest.skip("No cyclic-feature method found on FeatureEngineer.")

        for col in ("hour_sin", "hour_cos"):
            assert col in result.columns, f"Column '{col}' not found in result."
            vals = result[col].dropna()
            assert (vals >= -1.0).all() and (vals <= 1.0).all(), (
                f"Column '{col}' contains values outside [-1, 1]: {vals.describe()}"
            )


# ---------------------------------------------------------------------------
# Lag feature tests (pure logic, no DB needed)
# ---------------------------------------------------------------------------


class TestLagFeatures:
    """Tests for lag-feature generation in FeatureEngineer."""

    def test_lag_features_correct(self) -> None:
        """price_lag_24h at row 24 must equal the price at row 0.

        Constructs a 48-row price series with distinct values, applies the lag
        transform, and checks the 24-hour lag alignment.
        """
        mod = _import_or_skip("src.transformation.feature_engineering")

        n = 48
        prices = np.arange(float(n))  # 0.0, 1.0, ..., 47.0 – all unique
        timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({"timestamp_utc": timestamps, "price_eur_mwh": prices})

        engineer = mod.FeatureEngineer()

        if hasattr(engineer, "_add_lag_features"):
            result = engineer._add_lag_features(df.copy())
        elif hasattr(engineer, "add_lag_features"):
            result = engineer.add_lag_features(df.copy())
        else:
            pytest.skip("No lag-feature method found on FeatureEngineer.")

        assert "price_lag_24h" in result.columns, (
            "'price_lag_24h' column not created."
        )

        # Row 24 lag should reference row 0's price (0.0)
        actual_lag = result["price_lag_24h"].iloc[24]
        expected_lag = prices[0]
        assert actual_lag == pytest.approx(expected_lag), (
            f"price_lag_24h at row 24 should be {expected_lag}, got {actual_lag}."
        )


# ---------------------------------------------------------------------------
# Integration test – build_feature_table
# ---------------------------------------------------------------------------


class TestBuildFeatureTable:
    """Integration test for :meth:`~FeatureEngineer.build_feature_table`."""

    # Minimum rows required for features that need lookback windows (168 h lag)
    _N_ROWS: int = 200

    def _populate_raw_prices(self, engine, n: int = _N_ROWS) -> None:
        """Insert *n* hourly rows into ``raw_energy_prices``.

        Args:
            engine: Bound SQLAlchemy engine.
            n: Number of rows to insert.
        """
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        for i in range(n):
            ts = start + __import__("datetime").timedelta(hours=i)
            rows.append(
                f"('{ts.isoformat()}', 'day_ahead', {40.0 + (i % 20)})"
            )

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT OR IGNORE INTO raw_energy_prices "
                "(timestamp_utc, market_type, price_eur_mwh) VALUES "
                + ", ".join(rows)
            ))

    def _populate_raw_weather(self, engine, n: int = _N_ROWS) -> None:
        """Insert *n* hourly rows into ``raw_weather_data``.

        Args:
            engine: Bound SQLAlchemy engine.
            n: Number of rows to insert.
        """
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        for i in range(n):
            ts = start + __import__("datetime").timedelta(hours=i)
            rows.append(
                f"('{ts.isoformat()}', 'Praha', 50.0755, 14.4378, "
                f"{5.0 + (i % 15)}, {30 + (i % 50)}, {3.0 + (i % 10)}, "
                f"{100.0 + (i % 200)}, 0)"
            )

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT OR IGNORE INTO raw_weather_data "
                "(timestamp_utc, location_name, latitude, longitude, "
                "temperature_c, cloud_cover_pct, wind_speed_10m, "
                "direct_normal_irradiance, is_forecast) VALUES "
                + ", ".join(rows)
            ))

    def _populate_raw_generation(self, engine, n: int = _N_ROWS) -> None:
        """Insert synthetic generation rows into ``raw_generation_data``.

        Args:
            engine: Bound SQLAlchemy engine.
            n: Number of rows to insert (per production type).
        """
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        for prod_type in ("solar", "wind_onshore"):
            for i in range(n):
                ts = start + __import__("datetime").timedelta(hours=i)
                rows.append(
                    f"('{ts.isoformat()}', '{prod_type}', {500.0 + (i % 300)})"
                )

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT OR IGNORE INTO raw_generation_data "
                "(timestamp_utc, production_type, actual_power_mw) VALUES "
                + ", ".join(rows)
            ))

    def test_build_feature_table_integration(self, memory_engine) -> None:
        """build_feature_table must return a DataFrame with required columns and no NaN prices.

        Populates all three raw tables with at least 200 hourly rows, calls
        :meth:`~FeatureEngineer.build_feature_table`, and asserts:

        * The required feature columns are all present.
        * ``price_eur_mwh`` contains no ``NaN`` values.
        """
        mod = _import_or_skip("src.transformation.feature_engineering")

        self._populate_raw_prices(memory_engine)
        self._populate_raw_weather(memory_engine)
        self._populate_raw_generation(memory_engine)

        engineer = mod.FeatureEngineer()
        df = engineer.build_feature_table(memory_engine)

        assert isinstance(df, pd.DataFrame), "build_feature_table must return a DataFrame."
        assert len(df) > 0, "Returned DataFrame is empty."

        required_columns = {
            "price_eur_mwh",
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "hour_of_day",
            "month",
            "is_holiday",
        }
        missing = required_columns - set(df.columns)
        assert not missing, f"Missing columns in feature table: {missing}"

        nan_count = df["price_eur_mwh"].isna().sum()
        assert nan_count == 0, (
            f"price_eur_mwh has {nan_count} NaN values – expected 0."
        )


# ---------------------------------------------------------------------------
# Holiday detection test
# ---------------------------------------------------------------------------


class TestHolidaysDetection:
    """Tests for Czech public holiday detection in FeatureEngineer."""

    def test_holidays_detection(self) -> None:
        """New Year's Day 2024-01-01 must be marked as is_holiday=True.

        Constructs a single-row DataFrame for 2024-01-01 (a known Czech public
        holiday) and verifies that :meth:`~FeatureEngineer._add_holiday_flag`
        (or equivalent) sets ``is_holiday`` to ``True`` (or ``1``).
        """
        mod = _import_or_skip("src.transformation.feature_engineering")

        holiday_ts = pd.Timestamp("2024-01-01", tz="UTC")
        df = pd.DataFrame({"timestamp_utc": [holiday_ts]})

        engineer = mod.FeatureEngineer()

        if hasattr(engineer, "_add_holiday_flag"):
            result = engineer._add_holiday_flag(df.copy())
        elif hasattr(engineer, "add_holiday_flag"):
            result = engineer.add_holiday_flag(df.copy())
        else:
            pytest.skip("No holiday-flag method found on FeatureEngineer.")

        assert "is_holiday" in result.columns, (
            "'is_holiday' column not created."
        )

        flag = result["is_holiday"].iloc[0]
        assert bool(flag), (
            f"2024-01-01 (New Year's Day) should be is_holiday=True, got {flag}."
        )
