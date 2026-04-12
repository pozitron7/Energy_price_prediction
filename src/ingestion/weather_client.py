"""Open-Meteo weather data client for Czech energy price prediction.

This module provides the :class:`WeatherClient` class for fetching historical
and forecast weather data from the **Open-Meteo** API for multiple Czech city
locations.

Open-Meteo is a free, open-source weather API – no API key is required.
Full documentation: https://open-meteo.com/en/docs

Endpoints used
--------------
* **Historical archive**: ``https://archive-api.open-meteo.com/v1/archive``
* **Forecast**: ``https://api.open-meteo.com/v1/forecast``

Hourly variables fetched
------------------------
``temperature_2m``, ``cloudcover``, ``windspeed_10m``,
``direct_normal_irradiance``

All timestamps returned by the API are in UTC (``timezone=UTC`` query
parameter is passed explicitly).

Typical usage::

    from datetime import date
    from weather_client import WeatherClient

    LOCATIONS = [
        {"name": "Prague", "latitude": 50.0755, "longitude": 14.4378},
        {"name": "Brno",   "latitude": 49.1951, "longitude": 16.6068},
    ]

    client = WeatherClient()
    df = client.fetch_all_locations(LOCATIONS, date(2024, 1, 1), date(2024, 1, 31))
    print(df.head())
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Open-Meteo historical archive API endpoint.
HISTORICAL_URL: str = "https://archive-api.open-meteo.com/v1/archive"
#: Open-Meteo weather forecast API endpoint.
FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"

#: Hourly variable names to request from Open-Meteo.
_HOURLY_VARIABLES: list[str] = [
    "temperature_2m",
    "cloudcover",
    "windspeed_10m",
    "direct_normal_irradiance",
]

#: Mapping from Open-Meteo JSON keys to final DataFrame column names.
_COLUMN_MAP: dict[str, str] = {
    "temperature_2m": "temperature_c",
    "cloudcover": "cloud_cover_pct",
    "windspeed_10m": "wind_speed_10m",
    "direct_normal_irradiance": "direct_normal_irradiance",
}

#: Maximum number of HTTP request retries.
_MAX_RETRIES: int = 3
#: Initial back-off delay (seconds); doubles on each subsequent retry.
_INITIAL_BACKOFF: float = 1.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WeatherClient
# ---------------------------------------------------------------------------


class WeatherClient:
    """Fetches historical and forecast weather data from Open-Meteo.

    Handles retry logic with exponential back-off and provides a
    convenience method for fetching data for multiple locations at once.

    Args:
        session: Optional :class:`requests.Session`.  A new session is
            created automatically when *None* is given.
        timeout: HTTP request timeout in seconds (default ``60``).

    Example::

        client = WeatherClient()
        location = {"name": "Prague", "latitude": 50.0755, "longitude": 14.4378}
        df = client.fetch_historical(location, date(2024, 6, 1), date(2024, 6, 30))
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: int = 60,
    ) -> None:
        self._session: requests.Session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "EnergyPricePrediction/1.0 (research project)"}
        )
        self._timeout: int = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical(
        self,
        location: dict[str, Any],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch hourly historical weather for *location* over a date range.

        Calls the Open-Meteo **archive** endpoint and returns the result as a
        tidy :class:`~pandas.DataFrame` with UTC timestamps.

        Args:
            location: Dictionary with at minimum the keys:

                * ``"name"``      (:class:`str`) – human-readable location name.
                * ``"latitude"``  (:class:`float`) – WGS-84 latitude.
                * ``"longitude"`` (:class:`float`) – WGS-84 longitude.

            start_date: First day of the requested range (inclusive).
            end_date: Last day of the requested range (inclusive).

        Returns:
            DataFrame with columns:
            ``timestamp_utc``, ``location_name``, ``latitude``,
            ``longitude``, ``temperature_c``, ``cloud_cover_pct``,
            ``wind_speed_10m``, ``direct_normal_irradiance``,
            ``is_forecast`` (always ``False``).

        Raises:
            requests.HTTPError: If all retries are exhausted without a
                successful response.
        """
        # The archive API only serves data up to ~2 days ago; cap the range.
        latest_available = date.today() - timedelta(days=2)
        end_date = min(end_date, latest_available)
        if start_date > end_date:
            logger.warning("Requested range is beyond archive availability; nothing to fetch.")
            return _empty_weather_df()

        params = self._build_params(
            location=location,
            start=start_date,
            end=end_date,
            is_forecast=False,
        )
        data = self._get_with_retry(HISTORICAL_URL, params)
        return self._parse_response(
            data=data, location=location, is_forecast=False
        )

    def fetch_forecast(
        self,
        location: dict[str, Any],
        days_ahead: int = 2,
    ) -> pd.DataFrame:
        """Fetch hourly weather forecast for *location*.

        Calls the Open-Meteo **forecast** endpoint.  The number of forecast
        days is controlled by *days_ahead*.

        Args:
            location: Same structure as described in :meth:`fetch_historical`.
            days_ahead: Number of days ahead to fetch (1–16, default ``2``).

        Returns:
            DataFrame with the same columns as :meth:`fetch_historical`
            but ``is_forecast`` is ``True``.

        Raises:
            requests.HTTPError: If all retries are exhausted.
        """
        today = date.today()
        end = today + timedelta(days=days_ahead - 1)
        params = self._build_params(
            location=location,
            start=today,
            end=end,
            is_forecast=True,
        )
        # Forecast endpoint uses forecast_days rather than start/end
        params.pop("start_date", None)
        params.pop("end_date", None)
        params["forecast_days"] = days_ahead

        data = self._get_with_retry(FORECAST_URL, params)
        return self._parse_response(
            data=data, location=location, is_forecast=True
        )

    def fetch_all_locations(
        self,
        locations: list[dict[str, Any]],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch historical weather for multiple locations and concatenate.

        Calls :meth:`fetch_historical` for every entry in *locations* and
        vertically concatenates the results into a single DataFrame.

        Args:
            locations: List of location dictionaries (see
                :meth:`fetch_historical`).
            start_date: First day of the requested range (inclusive).
            end_date: Last day of the requested range (inclusive).

        Returns:
            Combined DataFrame for all locations, sorted by
            ``(location_name, timestamp_utc)``.

        Example::

            LOCATIONS = [
                {"name": "Prague", "latitude": 50.0755, "longitude": 14.4378},
                {"name": "Ostrava", "latitude": 49.8209, "longitude": 18.2625},
            ]
            df = client.fetch_all_locations(LOCATIONS, date(2024, 1, 1), date(2024, 1, 31))
        """
        frames: list[pd.DataFrame] = []
        for loc in locations:
            logger.debug(
                "Fetching historical weather for '%s' (%s → %s).",
                loc.get("name"),
                start_date,
                end_date,
            )
            df = self.fetch_historical(loc, start_date, end_date)
            frames.append(df)

        if not frames:
            return _empty_weather_df()

        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(
            ["location_name", "timestamp_utc"], inplace=True
        )
        combined.reset_index(drop=True, inplace=True)
        return combined

    def save_to_db(self, df: pd.DataFrame, engine: Engine) -> int:
        """Upsert weather records from *df* into the ``raw_weather_data`` table.

        Performs a PostgreSQL ``INSERT … ON CONFLICT DO UPDATE`` so that
        repeated ingestion runs remain idempotent.  The conflict target is
        ``(timestamp_utc, location_name)``.

        Args:
            df: DataFrame returned by any of the ``fetch_*`` methods.
            engine: Bound SQLAlchemy :class:`~sqlalchemy.engine.Engine`.

        Returns:
            Number of rows inserted or updated.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On any database error.
        """
        if df.empty:
            logger.debug("save_to_db called with empty DataFrame; skipping.")
            return 0

        if engine.dialect.name == "sqlite":
            from src.database.upsert import sqlite_upsert
            return sqlite_upsert(df, "raw_weather_data", engine)

        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy import MetaData, Table

        table_name = "raw_weather_data"
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)

        records = df.to_dict(orient="records")
        stmt = pg_insert(table).values(records)

        update_cols = {
            col.name: stmt.excluded[col.name]
            for col in table.columns
            if col.name not in {"timestamp_utc", "location_name", "is_forecast"}
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["timestamp_utc", "location_name", "is_forecast"],
            set_=update_cols,
        )

        with engine.begin() as conn:
            result = conn.execute(stmt)

        row_count: int = result.rowcount
        logger.debug("Upserted %d rows into %s.", row_count, table_name)
        return row_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_params(
        self,
        location: dict[str, Any],
        start: date,
        end: date,
        is_forecast: bool,
    ) -> dict[str, Any]:
        """Build the query-parameter dictionary for an Open-Meteo request.

        Args:
            location: Location dictionary (``name``, ``latitude``,
                ``longitude``).
            start: Start date of the range.
            end: End date of the range.
            is_forecast: Whether this is a forecast request.  Included as
                metadata but does not alter the parameters here (forecast
                callers strip ``start_date``/``end_date`` as needed).

        Returns:
            Dictionary of query parameters ready for :mod:`requests`.
        """
        return {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "hourly": ",".join(_HOURLY_VARIABLES),
            "timezone": "UTC",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
        }

    def _get_with_retry(
        self, url: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform a GET request with exponential back-off retry logic.

        Args:
            url: Fully qualified API endpoint URL.
            params: Query parameters to include in the request.

        Returns:
            Parsed JSON response body as a Python dictionary.

        Raises:
            requests.HTTPError: If the final retry attempt also fails.
        """
        backoff = _INITIAL_BACKOFF
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            logger.debug(
                "GET %s params=%s (attempt %d/%d)", url, params, attempt, _MAX_RETRIES
            )
            try:
                response = self._session.get(
                    url, params=params, timeout=self._timeout
                )
                response.raise_for_status()
                return response.json()  # type: ignore[return-value]
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2  # exponential back-off

        raise requests.HTTPError(
            f"All {_MAX_RETRIES} retries exhausted for {url}."
        ) from last_exc

    def _parse_response(
        self,
        data: dict[str, Any],
        location: dict[str, Any],
        is_forecast: bool,
    ) -> pd.DataFrame:
        """Convert a raw Open-Meteo JSON response into a tidy DataFrame.

        Args:
            data: Parsed JSON body returned by the Open-Meteo API.
            location: Location dictionary used for the request.
            is_forecast: Whether the records should be flagged as forecast
                data (``is_forecast`` column).

        Returns:
            Tidy DataFrame with schema documented in
            :meth:`fetch_historical`.
        """
        hourly: dict[str, list] = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            logger.warning(
                "Empty or unexpected Open-Meteo response for '%s'.",
                location.get("name"),
            )
            return _empty_weather_df()

        # Parse ISO-8601 timestamps (already UTC per timezone=UTC param)
        timestamps = pd.to_datetime(hourly["time"], utc=True)

        df = pd.DataFrame({"timestamp_utc": timestamps})
        for api_key, col_name in _COLUMN_MAP.items():
            df[col_name] = hourly.get(api_key)

        df["location_name"] = location.get("name", "")
        df["latitude"] = location.get("latitude")
        df["longitude"] = location.get("longitude")
        df["is_forecast"] = is_forecast

        # Reorder columns to canonical order
        df = df[
            [
                "timestamp_utc",
                "location_name",
                "latitude",
                "longitude",
                "temperature_c",
                "cloud_cover_pct",
                "wind_speed_10m",
                "direct_normal_irradiance",
                "is_forecast",
            ]
        ]
        return df


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _empty_weather_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical weather schema.

    Returns:
        Empty :class:`~pandas.DataFrame` with all expected columns and
        no rows.
    """
    return pd.DataFrame(
        columns=[
            "timestamp_utc",
            "location_name",
            "latitude",
            "longitude",
            "temperature_c",
            "cloud_cover_pct",
            "wind_speed_10m",
            "direct_normal_irradiance",
            "is_forecast",
        ]
    )
