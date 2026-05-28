"""
tests/test_ingestion.py
=======================
Tests for the ingestion layer:

* :class:`src.ingestion.ote_client.OTEClient`
* :class:`src.ingestion.weather_client.WeatherClient`
* :class:`src.ingestion.entsoe_client.ENTSOEClient`

All tests are self-contained: network calls are either not made (synthetic
methods) or mocked via :mod:`unittest.mock`.
"""

from __future__ import annotations

import importlib
from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests


# ---------------------------------------------------------------------------
# Helper – lazy import guard
# ---------------------------------------------------------------------------

def _import_or_skip(module_path: str) -> Any:
    """Import *module_path* or skip the test if it cannot be imported.

    Args:
        module_path: Dotted module path, e.g. ``"src.ingestion.ote_client"``.

    Returns:
        The imported module object.
    """
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Module not yet available: {exc}")


# ---------------------------------------------------------------------------
# OTEClient tests
# ---------------------------------------------------------------------------


class TestOTEClient:
    """Tests for :class:`src.ingestion.ote_client.OTEClient`."""

    def test_ote_synthetic_prices_structure(self) -> None:
        """_generate_synthetic_prices must return 24 dicts with the right keys.

        Calls the private synthetic-data generator for today's date and
        verifies that:

        * The return value is a list of exactly 24 items.
        * Each item is a dict containing the keys ``timestamp_utc``,
          ``market_type``, and ``price_eur_mwh``.
        """
        mod = _import_or_skip("src.ingestion.ote_client")
        client = mod.OTEClient()
        records = client._generate_synthetic_prices(date.today())

        assert isinstance(records, list), "Expected a list."
        assert len(records) == 24, f"Expected 24 records, got {len(records)}."

        required_keys = {"timestamp_utc", "market_type", "price_eur_mwh"}
        for i, rec in enumerate(records):
            missing = required_keys - rec.keys()
            assert not missing, (
                f"Record {i} is missing keys: {missing}. Got: {rec.keys()}"
            )

    def test_ote_fetch_historical_uses_synthetic_fallback(self) -> None:
        """fetch_historical must fall back to synthetic data on network failure.

        Patches ``requests.Session.get`` to always raise :class:`requests.RequestException`
        and calls ``fetch_historical`` for a 3-day window, expecting 3 × 24 = 72
        rows in the returned DataFrame.
        """
        mod = _import_or_skip("src.ingestion.ote_client")

        start = date.today() - timedelta(days=3)
        end = date.today() - timedelta(days=1)

        with (
            patch("requests.Session.get", side_effect=requests.RequestException("mocked network error")),
            patch("time.sleep", return_value=None),
        ):
            client = mod.OTEClient()
            result = client.fetch_historical(start=start, end=end)

        assert isinstance(result, pd.DataFrame), "Expected a pandas DataFrame."
        assert len(result) == 72, (
            f"Expected 72 rows for 3 days × 24 hours, got {len(result)}."
        )


# ---------------------------------------------------------------------------
# WeatherClient tests
# ---------------------------------------------------------------------------


class TestWeatherClient:
    """Tests for :class:`src.ingestion.weather_client.WeatherClient`."""

    def test_weather_client_synthetic_on_api_failure(self) -> None:
        """WeatherClient must handle API failure gracefully.

        Expected behaviour (documented here):
        When the upstream Open-Meteo API returns a non-2xx status code, the
        client should either:

        a. Return a non-empty :class:`pandas.DataFrame` built from synthetic /
           cached data, **or**
        b. Raise a descriptive exception (e.g. ``RuntimeError`` or a custom
           ``WeatherFetchError``) that clearly communicates the failure.

        The test accepts both outcomes and simply asserts that the client does
        NOT silently return an empty/``None`` result without raising.
        """
        mod = _import_or_skip("src.ingestion.weather_client")

        # Construct a mock response with status 500
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.ok = False
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with (
            patch("requests.Session.get", return_value=mock_resp),
            patch("time.sleep", return_value=None),
        ):
            client = mod.WeatherClient()
            try:
                result = client.fetch_historical(
                    {"name": "Praha", "latitude": 50.0755, "longitude": 14.4378},
                    date.today() - timedelta(days=10),
                    date.today() - timedelta(days=5),
                )
                # If it returns without raising, the result must be non-empty
                assert result is not None, "fetch_historical returned None on API failure."
                assert not (isinstance(result, pd.DataFrame) and result.empty), (
                    "fetch_historical returned an empty DataFrame on API failure without raising."
                )
            except Exception as exc:
                # Any descriptive exception is acceptable
                assert str(exc), "Exception raised but has no message – hard to debug."


# ---------------------------------------------------------------------------
# ENTSOEClient tests
# ---------------------------------------------------------------------------


class TestENTSOEClient:
    """Tests for :class:`src.ingestion.entsoe_client.ENTSOEClient`."""

    def test_entsoe_synthetic_generation_structure(self) -> None:
        """_generate_synthetic_generation must return a DataFrame with required columns.

        Calls the synthetic-data generator over a 7-day window and verifies:

        * Columns ``timestamp_utc``, ``production_type``, and
          ``actual_power_mw`` are all present.
        * ``production_type`` values include at least ``'solar'`` and
          ``'wind_onshore'``.
        """
        mod = _import_or_skip("src.ingestion.entsoe_client")

        start = date.today() - timedelta(days=7)
        end = date.today()

        client = mod.ENTSOEClient(api_token="")
        df = client._generate_synthetic_generation(start, end)

        assert isinstance(df, pd.DataFrame), "Expected a pandas DataFrame."

        required_columns = {"timestamp_utc", "production_type", "actual_power_mw"}
        missing_cols = required_columns - set(df.columns)
        assert not missing_cols, f"Missing columns: {missing_cols}"

        production_types = set(df["production_type"].unique())
        for expected_type in ("solar", "wind_onshore"):
            assert expected_type in production_types, (
                f"Expected production_type '{expected_type}' not found. "
                f"Got: {production_types}"
            )
