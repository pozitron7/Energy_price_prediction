"""OTE-CR Day-Ahead electricity price client.

This module provides the :class:`OTEClient` class for downloading Czech
electricity Day-Ahead spot prices from the OTE-CR (Operátor trhu s
elektřinou) website.

OTE publishes hourly prices at the following URL pattern::

    https://www.ote-cr.cz/cs/kratkodobe-trhy/elektrina/denni-trh?date=YYYY-MM-DD

Because OTE does not expose a formal JSON REST API the client fetches the
HTML page and extracts prices from the embedded data table using
:mod:`re` and :mod:`html.parser` (via :mod:`html`).

Prices are published in CET/CEST (``Europe/Prague`` timezone) and are
converted to UTC before being returned to the caller.

Typical usage::

    from datetime import date
    from ote_client import OTEClient

    client = OTEClient()
    prices = client.fetch_day_ahead_prices(date(2024, 1, 15))
    df = client.fetch_historical(date(2024, 1, 1), date(2024, 1, 31))
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import insert as pg_insert

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Base URL for OTE-CR Day-Ahead market pages.
OTE_BASE_URL: str = (
    "https://www.ote-cr.cz/cs/kratkodobe-trhy/elektrina/denni-trh"
)

#: Public JSON endpoint used for chart data (real published prices).
OTE_CHART_URL: str = f"{OTE_BASE_URL}/@@chart-data"

#: Timezone used by OTE-CR for published prices.
_TZ_PRAGUE: ZoneInfo = ZoneInfo("Europe/Prague")
#: UTC timezone reference.
_TZ_UTC: ZoneInfo = ZoneInfo("UTC")

#: Market type label embedded in every returned record.
_MARKET_TYPE: str = "day_ahead"

#: Maximum number of HTTP request retries before falling back.
_MAX_RETRIES: int = 3
#: Base back-off delay (seconds) between retries.
_BACKOFF_SECONDS: float = 2.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OTEClient
# ---------------------------------------------------------------------------


class OTEClient:
    """Client for the OTE-CR Day-Ahead electricity market.

    Downloads hourly CZK/EUR spot prices from the OTE-CR website,
    converts timestamps to UTC, and persists data via SQLAlchemy.

    Args:
        session: Optional :class:`requests.Session` to use for HTTP calls.
            Useful for testing with a mock session.  If *None* a new
            session is created automatically.
        timeout: HTTP request timeout in seconds (default ``30``).

    Example::

        client = OTEClient()
        df = client.fetch_historical(date(2024, 3, 1), date(2024, 3, 31))
        print(df.head())
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: int = 30,
    ) -> None:
        self._session: requests.Session = session or requests.Session()
        self._session.headers.update(
            {
                "Accept-Language": "cs,en;q=0.9",
                "User-Agent": (
                    "EnergyPricePrediction/1.0 "
                    "(research project; contact: research@example.com)"
                ),
            }
        )
        self._timeout: int = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_day_ahead_prices(self, date: date) -> list[dict[str, Any]]:
        """Fetch hourly Day-Ahead prices for a single calendar date.

        Attempts to retrieve real prices from the OTE-CR website.  If the
        request fails after :data:`_MAX_RETRIES` attempts the method falls
        back to :meth:`_generate_synthetic_prices` and logs a warning.

        Args:
            date: The calendar date for which prices should be fetched.

        Returns:
            A list of hourly price records, each a :class:`dict` with keys:

            * ``timestamp_utc`` (:class:`~datetime.datetime`) – start of the
              hourly interval in UTC (timezone-aware).
            * ``market_type`` (:class:`str`) – always ``"day_ahead"``.
            * ``price_eur_mwh`` (:class:`float`) – price in EUR/MWh.

        Example::

            records = client.fetch_day_ahead_prices(date(2024, 6, 21))
            for r in records:
                print(r["timestamp_utc"], r["price_eur_mwh"])
        """
        # Preferred path: OTE's public JSON chart endpoint (real published prices).
        try:
            json_records = self._fetch_json_prices(date)
            if json_records:
                return json_records
            logger.warning("OTE has not published day-ahead prices for %s yet.", date)
            return []
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("OTE JSON endpoint failed for %s: %s; trying HTML fallback.", date, exc)

        url = OTE_BASE_URL
        params = {"date": date.strftime("%Y-%m-%d")}
        logger.debug("GET %s params=%s", url, params)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._session.get(
                    url, params=params, timeout=self._timeout
                )
                response.raise_for_status()
                records = self._parse_html_prices(response.text, date)
                if records:
                    logger.debug(
                        "Fetched %d price records for %s", len(records), date
                    )
                    return records
                logger.warning(
                    "Parsed 0 records from OTE for %s (attempt %d/%d); "
                    "page structure may have changed.",
                    date,
                    attempt,
                    _MAX_RETRIES,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "HTTP error fetching OTE prices for %s (attempt %d/%d): %s",
                    date,
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
            # Back-off before retry (except after final attempt)
            if attempt < _MAX_RETRIES:
                time.sleep(_BACKOFF_SECONDS * attempt)

        logger.warning(
            "All retries exhausted for %s; using synthetic prices.", date
        )
        return self._generate_synthetic_prices(date)

    def _fetch_json_prices(self, day: date) -> list[dict[str, Any]]:
        """Fetch real hourly prices from OTE's public JSON chart endpoint.

        The endpoint returns one price series per resolution (60-minute /
        15-minute since the Oct-2025 market change, or a single hourly series
        for older dates).  Points are placed on the real UTC time axis (so DST
        days with 23/25 hours are handled) and averaged to hourly resolution.

        Args:
            day: Delivery date (Europe/Prague).

        Returns:
            Hourly records; an empty list when OTE has no data for *day*.

        Raises:
            requests.RequestException: On network / HTTP errors.
            ValueError: If the payload is not valid JSON.
        """
        response = self._session.get(
            OTE_CHART_URL,
            params={"report_date": day.strftime("%Y-%m-%d")},
            timeout=self._timeout,
        )
        response.raise_for_status()
        lines = response.json().get("data", {}).get("dataLine", [])
        price_lines = [ln for ln in lines if str(ln.get("type")) == "1" and ln.get("point")]
        if not price_lines:
            return []
        # Prefer the 60-minute series, otherwise whatever price series exists.
        chosen = next((ln for ln in price_lines if "60min" in ln.get("title", "")), price_lines[0])
        points = chosen["point"]

        start_local = datetime(day.year, day.month, day.day, tzinfo=_TZ_PRAGUE)
        next_local = start_local + timedelta(days=1)
        # Compute the real day length in UTC (23 / 24 / 25 h around DST changes).
        start_utc = start_local.astimezone(_TZ_UTC)
        day_len = next_local.replace(tzinfo=_TZ_PRAGUE).astimezone(_TZ_UTC) - start_utc
        step = day_len / len(points)

        series = pd.Series(
            [float(p["y"]) for p in points],
            index=pd.DatetimeIndex([start_utc + i * step for i in range(len(points))]),
        )
        hourly = series.resample("h").mean().dropna()
        return [
            {"timestamp_utc": ts, "market_type": _MARKET_TYPE, "price_eur_mwh": round(float(v), 2)}
            for ts, v in hourly.items()
        ]

    def fetch_historical(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Fetch Day-Ahead prices for a date range.

        Iterates day by day from *start_date* to *end_date* (inclusive),
        calling :meth:`fetch_day_ahead_prices` for each day and concatenating
        the results into a single :class:`~pandas.DataFrame`.

        Args:
            start_date: First day of the desired range (inclusive).
            end_date: Last day of the desired range (inclusive).
            start: Alias for start_date.
            end: Alias for end_date.

        Returns:
            A :class:`~pandas.DataFrame` with columns:

            * ``timestamp_utc`` – timezone-aware :class:`~datetime.datetime`.
            * ``market_type`` – always ``"day_ahead"``.
            * ``price_eur_mwh`` – price in EUR/MWh as :class:`float`.

        Raises:
            ValueError: If *start_date* is after *end_date*.

        Example::

            df = client.fetch_historical(date(2024, 1, 1), date(2024, 1, 31))
            print(df.dtypes)
        """
        if start_date is None:
            start_date = start
        if end_date is None:
            end_date = end
        if start_date is None or end_date is None:
            raise ValueError("Both start_date (or start) and end_date (or end) must be provided.")
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be <= end_date ({end_date})."
            )

        all_records: list[dict[str, Any]] = []
        current = start_date
        while current <= end_date:
            records = self.fetch_day_ahead_prices(current)
            all_records.extend(records)
            current += timedelta(days=1)

        if not all_records:
            return pd.DataFrame(
                columns=["timestamp_utc", "market_type", "price_eur_mwh"]
            )

        df = pd.DataFrame(all_records)
        df.sort_values("timestamp_utc", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def save_to_db(self, df: pd.DataFrame, engine: Engine) -> int:
        """Upsert price records from *df* into the ``raw_energy_prices`` table.

        Uses a PostgreSQL ``INSERT … ON CONFLICT DO UPDATE`` (upsert) so that
        re-running the ingestion pipeline is idempotent.  The conflict target
        is the ``timestamp_utc`` column.

        Args:
            df: DataFrame returned by :meth:`fetch_historical` or
                :meth:`fetch_day_ahead_prices` (converted to DataFrame).
            engine: Bound SQLAlchemy :class:`~sqlalchemy.engine.Engine`.

        Returns:
            The number of rows inserted or updated.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On any database error.

        Example::

            from sqlalchemy import create_engine
            engine = create_engine("postgresql+psycopg2://user:pw@localhost/db")
            n = client.save_to_db(df, engine)
            print(f"{n} rows upserted.")
        """
        if df.empty:
            logger.debug("save_to_db called with empty DataFrame; skipping.")
            return 0

        if engine.dialect.name == "sqlite":
            from src.database.upsert import sqlite_upsert
            cols = [c for c in ("timestamp_utc", "market_type", "price_eur_mwh") if c in df.columns]
            return sqlite_upsert(df[cols], "raw_energy_prices", engine)

        records = df.to_dict(orient="records")
        table_name = "raw_energy_prices"

        with engine.begin() as conn:
            stmt = pg_insert(
                _get_reflected_table(table_name, engine)
            ).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=["timestamp_utc", "market_type"],
                set_={
                    "price_eur_mwh": stmt.excluded.price_eur_mwh,
                },
            )
            result = conn.execute(stmt)

        row_count: int = result.rowcount
        logger.debug("Upserted %d rows into %s.", row_count, table_name)
        return row_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_html_prices(
        self, html: str, target_date: date
    ) -> list[dict[str, Any]]:
        """Extract hourly prices from OTE-CR HTML response.

        Searches for numeric price values embedded in the HTML table that OTE
        publishes for each trading day.  The regex targets cells that contain
        decimal numbers representing EUR/MWh values.

        Args:
            html: Raw HTML text of the OTE-CR page.
            target_date: The calendar date the page corresponds to (used to
                build timezone-aware timestamps).

        Returns:
            A list of hourly records (same schema as
            :meth:`fetch_day_ahead_prices`).  Returns an empty list if no
            prices could be parsed.
        """
        # OTE-CR embeds price data in a <table> with rows containing:
        # <td class="...">HH</td><td ...>price</td>
        # We extract all numeric cells and attempt to reconstruct 24 values.
        price_pattern = re.compile(
            r"<td[^>]*>\s*(\d{1,3}(?:[.,]\d+)?)\s*</td>"
        )
        raw_values: list[str] = price_pattern.findall(html)

        # Filter to plausible EUR/MWh range: [-500, 4000]
        prices: list[float] = []
        for raw in raw_values:
            try:
                value = float(raw.replace(",", "."))
                if -500.0 <= value <= 4000.0:
                    prices.append(value)
            except ValueError:
                continue

        if len(prices) < 23:
            # Not enough data – page structure differs or no data for this day
            return []

        # Take the first 24 values (or 23 for DST fall-back days)
        prices = prices[:24]
        records: list[dict[str, Any]] = []
        for hour, price in enumerate(prices):
            # Build a naive datetime in Prague time, then convert to UTC
            naive_dt = datetime(
                target_date.year,
                target_date.month,
                target_date.day,
                hour,
                0,
                0,
            )
            local_dt = naive_dt.replace(tzinfo=_TZ_PRAGUE)
            utc_dt = local_dt.astimezone(_TZ_UTC)
            records.append(
                {
                    "timestamp_utc": utc_dt,
                    "market_type": _MARKET_TYPE,
                    "price_eur_mwh": price,
                }
            )
        return records

    def _generate_synthetic_prices(
        self, target_date: date
    ) -> list[dict[str, Any]]:
        """Generate realistic synthetic Day-Ahead prices for *target_date*.

        Uses a sine-wave model that mimics a typical daily price curve
        (low prices at night, peak in the morning and evening) with additive
        Gaussian noise.  This method is invoked as a fallback when live
        fetching fails after all retries.

        Args:
            target_date: The calendar date for which synthetic prices are
                generated.

        Returns:
            24 hourly records with the same schema as
            :meth:`fetch_day_ahead_prices`.  Prices are guaranteed to be
            >= 0 EUR/MWh.

        Note:
            Synthetic prices are **not** suitable for production use.  They
            exist solely to allow unit-testing and pipeline validation without
            live API access.
        """
        rng = np.random.default_rng(
            seed=int(target_date.strftime("%Y%m%d"))
        )
        hours = np.arange(24)

        # Primary wave: dual-peak shape (morning ~8h, evening ~19h)
        base_price = 60.0  # EUR/MWh baseline
        morning_peak = 25.0 * np.sin(np.pi * (hours - 5) / 10) * (
            (hours >= 5) & (hours <= 15)
        )
        evening_peak = 30.0 * np.sin(np.pi * (hours - 15) / 10) * (
            (hours >= 15) & (hours <= 23)
        )
        noise = rng.normal(0, 5, size=24)
        raw_prices = base_price + morning_peak + evening_peak + noise

        # Clamp to plausible range
        prices = np.clip(raw_prices, 0.0, 500.0)

        records: list[dict[str, Any]] = []
        for hour, price in enumerate(prices):
            naive_dt = datetime(
                target_date.year,
                target_date.month,
                target_date.day,
                hour,
                0,
                0,
            )
            local_dt = naive_dt.replace(tzinfo=_TZ_PRAGUE)
            utc_dt = local_dt.astimezone(_TZ_UTC)
            records.append(
                {
                    "timestamp_utc": utc_dt,
                    "market_type": f"{_MARKET_TYPE}_synthetic",
                    "price_eur_mwh": round(float(price), 2),
                }
            )
        return records


# ---------------------------------------------------------------------------
# Internal SQLAlchemy helper
# ---------------------------------------------------------------------------


def _get_reflected_table(table_name: str, engine: Engine):
    """Reflect *table_name* from the database into a SQLAlchemy Table object.

    Args:
        table_name: Name of the database table to reflect.
        engine: Active SQLAlchemy :class:`~sqlalchemy.engine.Engine`.

    Returns:
        A :class:`sqlalchemy.Table` object ready for use in DML statements.
    """
    from sqlalchemy import MetaData, Table  # local import to keep module lean

    metadata = MetaData()
    return Table(table_name, metadata, autoload_with=engine)
