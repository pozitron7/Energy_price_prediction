"""ENTSO-E Transparency Platform client for Czech renewable generation data.

This module provides the :class:`ENTSOEClient` class for downloading actual
generation-per-production-type data for the Czech Republic from the
**ENTSO-E Transparency Platform** REST API.

API overview
------------
* A free API token is required – register at
  https://transparency.entsoe.eu/ and set the token via the
  ``ENTSOE_API_TOKEN`` environment variable.
* Endpoint used (Actual Generation Per Production Type, document A75):

  .. code-block:: text

      https://web-api.tp.entsoe.eu/api
        ?documentType=A75
        &processType=A16
        &in_Domain=10YCZ-CEPS-----N
        &periodStart=<YYYYMMDDHHMM>
        &periodEnd=<YYYYMMDDHHMM>
        &securityToken=<token>

* Czech control area EIC code: ``10YCZ-CEPS-----N``
* XML responses are parsed with Python's built-in
  :mod:`xml.etree.ElementTree`.

PSR type codes mapped to human-readable names
----------------------------------------------
+--------+-------------------+
| Code   | Name              |
+========+===================+
| B16    | solar             |
+--------+-------------------+
| B18    | wind_offshore     |
+--------+-------------------+
| B19    | wind_onshore      |
+--------+-------------------+
| B01    | biomass           |
+--------+-------------------+

Typical usage::

    import os
    from datetime import date
    from entsoe_client import ENTSOEClient

    client = ENTSOEClient(api_token=os.environ["ENTSOE_API_TOKEN"])
    df = client.fetch_generation(date(2024, 6, 1), date(2024, 6, 30))
    print(df.head())
"""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: ENTSO-E Transparency Platform REST API base URL.
ENTSOE_API_URL: str = "https://web-api.tp.entsoe.eu/api"

#: Czech control area EIC code.
CZ_DOMAIN: str = "10YCZ-CEPS-----N"

#: Mapping of ENTSO-E PSR type codes to human-readable generation type names.
PSR_TYPE_MAP: dict[str, str] = {
    "B16": "solar",
    "B18": "wind_offshore",
    "B19": "wind_onshore",
    "B01": "biomass",
}

#: ENTSO-E XML namespace (GL_MarketDocument / A75 documents).
_NS: dict[str, str] = {
    "entsoe": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
}

#: Maximum number of HTTP retries before raising.
_MAX_RETRIES: int = 3
#: Initial back-off delay in seconds (doubles on each retry).
_INITIAL_BACKOFF: float = 2.0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ENTSOEClient
# ---------------------------------------------------------------------------


class ENTSOEClient:
    """Client for the ENTSO-E Transparency Platform REST API.

    Downloads actual generation-per-production-type XML data for the Czech
    Republic, parses it into tidy DataFrames, and persists it via SQLAlchemy.

    If *api_token* is an empty string or ``None``, the client automatically
    falls back to :meth:`_generate_synthetic_generation` and logs a
    ``WARNING``.

    Args:
        api_token: ENTSO-E API security token.  Typically loaded from the
            ``ENTSOE_API_TOKEN`` environment variable.  Pass ``""`` or
            ``None`` to force synthetic mode (e.g. for unit testing).
        session: Optional :class:`requests.Session`; a new one is created if
            *None*.
        timeout: HTTP request timeout in seconds (default ``60``).

    Example::

        import os
        client = ENTSOEClient(api_token=os.environ.get("ENTSOE_API_TOKEN", ""))
        df = client.fetch_generation(date(2024, 3, 1), date(2024, 3, 31))
    """

    def __init__(
        self,
        api_token: str | None = None,
        session: requests.Session | None = None,
        timeout: int = 60,
    ) -> None:
        self._api_token: str = api_token or os.environ.get(
            "ENTSOE_API_TOKEN", ""
        )
        self._session: requests.Session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "EnergyPricePrediction/1.0 (research project)"}
        )
        self._timeout: int = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_generation(
        self,
        start_date: date,
        end_date: date,
        psr_type: str | None = None,
    ) -> pd.DataFrame:
        """Fetch actual generation data for the Czech Republic.

        Requests ``documentType=A75`` (Actual Generation Per Production Type)
        from the ENTSO-E API for the CZ area between *start_date* and
        *end_date*.  If *psr_type* is supplied only that production type is
        requested; otherwise all available types are fetched.

        If the API token is absent or empty, or if all retries fail, the
        method falls back to :meth:`_generate_synthetic_generation`.

        Args:
            start_date: First day of the requested range (inclusive).
            end_date: Last day of the requested range (inclusive, the API
                query end is set to 23:59 of this day).
            psr_type: Optional ENTSO-E PSR type code (e.g. ``"B16"`` for
                solar).  When omitted all configured PSR types are fetched in
                a single request.

        Returns:
            DataFrame with columns:

            * ``timestamp_utc`` – timezone-aware :class:`~datetime.datetime`.
            * ``production_type`` – human-readable name (see
              :data:`PSR_TYPE_MAP`) or the raw code if unmapped.
            * ``actual_power_mw`` – generation in MW as :class:`float`.

        Raises:
            ValueError: If *start_date* is after *end_date*.

        Example::

            df = client.fetch_generation(date(2024, 6, 1), date(2024, 6, 30),
                                         psr_type="B16")
        """
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must be <= end_date ({end_date})."
            )

        if not self._api_token:
            logger.warning(
                "ENTSOE_API_TOKEN is not set. Using synthetic generation data."
            )
            return self._generate_synthetic_generation(start_date, end_date)

        params = self._build_params(start_date, end_date, psr_type)
        xml_text = self._get_with_retry(params)
        if xml_text is None:
            logger.warning(
                "All retries exhausted for ENTSO-E API (%s → %s). "
                "Falling back to synthetic generation.",
                start_date,
                end_date,
            )
            return self._generate_synthetic_generation(start_date, end_date)

        df = self._parse_xml(xml_text)
        if df.empty:
            logger.warning(
                "Parsed 0 generation records for %s → %s; "
                "using synthetic fallback.",
                start_date,
                end_date,
            )
            return self._generate_synthetic_generation(start_date, end_date)

        return df

    def save_to_db(self, df: pd.DataFrame, engine: Engine) -> int:
        """Upsert generation records into the ``raw_generation_data`` table.

        Uses PostgreSQL ``INSERT … ON CONFLICT DO UPDATE`` for idempotent
        ingestion.  The conflict target is
        ``(timestamp_utc, production_type)``.

        Args:
            df: DataFrame returned by :meth:`fetch_generation`.
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
            return sqlite_upsert(df, "raw_generation_data", engine)

        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy import MetaData, Table

        table_name = "raw_generation_data"
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)

        records = df.to_dict(orient="records")
        stmt = pg_insert(table).values(records)
        update_cols = {
            col.name: stmt.excluded[col.name]
            for col in table.columns
            if col.name not in {"timestamp_utc", "production_type"}
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["timestamp_utc", "production_type"],
            set_=update_cols,
        )

        with engine.begin() as conn:
            result = conn.execute(stmt)

        row_count: int = result.rowcount
        logger.debug("Upserted %d rows into %s.", row_count, table_name)
        return row_count

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------

    def _generate_synthetic_generation(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Generate synthetic solar and wind_onshore profiles for testing.

        * **Solar** follows a daylight Gaussian curve peaking at 12:00 UTC
          with amplitude proportional to summer/winter irradiance.
        * **Wind onshore** is modelled as mean-reverting random noise.

        Args:
            start_date: First day of the synthetic range (inclusive).
            end_date: Last day of the synthetic range (inclusive).

        Returns:
            DataFrame with columns ``timestamp_utc``, ``production_type``,
            ``actual_power_mw``.  Each day contains 24 records per production
            type.

        Note:
            Synthetic data is **not suitable for production** – use for
            pipeline validation and unit testing only.
        """
        rng = np.random.default_rng(
            seed=int(start_date.strftime("%Y%m%d"))
        )
        records: list[dict[str, Any]] = []

        current = start_date
        while current <= end_date:
            # Day-of-year governs seasonal solar amplitude [0.2 – 1.0]
            doy = current.timetuple().tm_yday
            solar_scale = 0.2 + 0.8 * np.sin(np.pi * (doy - 80) / 365) ** 2

            for hour in range(24):
                ts_utc = datetime(
                    current.year,
                    current.month,
                    current.day,
                    hour,
                    tzinfo=timezone.utc,
                )

                # Solar: Gaussian centred at 12:00 UTC, max ~4000 MW
                solar_mw = float(
                    4000.0
                    * solar_scale
                    * np.exp(-((hour - 12) ** 2) / (2 * 3.5**2))
                    + rng.normal(0, 50)
                )
                solar_mw = max(0.0, round(solar_mw, 1))

                # Wind onshore: ~1500 MW baseline with AR(1) noise
                wind_mw = float(
                    1500.0
                    + rng.normal(0, 300)
                    + 200.0 * np.sin(2 * np.pi * hour / 24)
                )
                wind_mw = max(0.0, round(wind_mw, 1))

                records.append(
                    {
                        "timestamp_utc": ts_utc,
                        "production_type": "solar",
                        "actual_power_mw": solar_mw,
                    }
                )
                records.append(
                    {
                        "timestamp_utc": ts_utc,
                        "production_type": "wind_onshore",
                        "actual_power_mw": wind_mw,
                    }
                )

            current += timedelta(days=1)

        df = pd.DataFrame(records)
        df.sort_values(
            ["production_type", "timestamp_utc"], inplace=True
        )
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_params(
        self,
        start_date: date,
        end_date: date,
        psr_type: str | None,
    ) -> dict[str, str]:
        """Build query parameters for the ENTSO-E REST API request.

        Args:
            start_date: Start of the period (mapped to ``00:00`` UTC).
            end_date: End of the period (mapped to ``23:59`` UTC, one
                resolution-interval past midnight to capture the last hour).
            psr_type: Optional PSR type code filter.  If *None* the parameter
                is omitted from the request.

        Returns:
            Dictionary of query parameters ready for :mod:`requests`.
        """
        params: dict[str, str] = {
            "documentType": "A75",
            "processType": "A16",
            "in_Domain": CZ_DOMAIN,
            "periodStart": start_date.strftime("%Y%m%d") + "0000",
            "periodEnd": end_date.strftime("%Y%m%d") + "2300",
            "securityToken": self._api_token,
        }
        if psr_type:
            params["psrType"] = psr_type
        return params

    def _get_with_retry(
        self, params: dict[str, str]
    ) -> str | None:
        """Perform a GET request to the ENTSO-E API with retry logic.

        Handles common HTTP error conditions:

        * ``401`` / ``403`` – bad or expired token; retrying is pointless so
          the method returns ``None`` immediately.
        * ``429`` – rate limited; waits twice as long before retrying.
        * Other HTTP errors – logged at WARNING and retried.

        Args:
            params: Query parameters (including ``securityToken``).

        Returns:
            Raw XML response body as a string, or ``None`` if all retries
            were exhausted or a non-retryable error occurred.
        """
        backoff = _INITIAL_BACKOFF
        for attempt in range(1, _MAX_RETRIES + 1):
            logger.debug(
                "GET %s (attempt %d/%d)", ENTSOE_API_URL, attempt, _MAX_RETRIES
            )
            try:
                response = self._session.get(
                    ENTSOE_API_URL, params=params, timeout=self._timeout
                )

                # Non-retryable authentication errors
                if response.status_code in {401, 403}:
                    logger.warning(
                        "ENTSO-E API returned HTTP %d – check your API token.",
                        response.status_code,
                    )
                    return None

                # Rate limiting – wait longer
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("Retry-After", backoff * 2)
                    )
                    logger.warning(
                        "ENTSO-E API rate limit hit (429). "
                        "Waiting %ds before retry.",
                        retry_after,
                    )
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.text

            except requests.RequestException as exc:
                logger.warning(
                    "ENTSO-E request error (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
            if attempt < _MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2

        return None

    def _parse_xml(self, xml_text: str) -> pd.DataFrame:
        """Parse an ENTSO-E A75 XML document into a tidy DataFrame.

        Navigates the ``GL_MarketDocument`` structure:
        ``TimeSeries → MktPSRType → psrType`` identifies the production type,
        and ``Period → timeInterval → start`` together with ``Point →
        position`` determines each UTC timestamp.

        Args:
            xml_text: Raw XML string returned by the ENTSO-E API.

        Returns:
            Tidy DataFrame with columns ``timestamp_utc``,
            ``production_type``, ``actual_power_mw``.
            Returns an empty DataFrame if parsing fails or yields no records.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("Failed to parse ENTSO-E XML response: %s", exc)
            return _empty_generation_df()

        records: list[dict[str, Any]] = []

        for ts_elem in root.findall("entsoe:TimeSeries", _NS):
            # --- Identify production type ---
            psr_code = ""
            psr_elem = ts_elem.find(
                "entsoe:MktPSRType/entsoe:psrType", _NS
            )
            if psr_elem is not None and psr_elem.text:
                psr_code = psr_elem.text.strip()
            production_type = PSR_TYPE_MAP.get(psr_code, psr_code)

            # --- Iterate over Periods ---
            for period_elem in ts_elem.findall("entsoe:Period", _NS):
                # Parse period start time
                start_str_elem = period_elem.find(
                    "entsoe:timeInterval/entsoe:start", _NS
                )
                if start_str_elem is None or not start_str_elem.text:
                    continue
                try:
                    period_start = datetime.fromisoformat(
                        start_str_elem.text.replace("Z", "+00:00")
                    )
                except ValueError:
                    logger.warning(
                        "Skipping period with unparseable start: %s",
                        start_str_elem.text,
                    )
                    continue

                # Determine resolution (default 60 minutes)
                res_elem = period_elem.find("entsoe:resolution", _NS)
                resolution_minutes = _parse_resolution(
                    res_elem.text if res_elem is not None else None
                )

                # Parse each measurement Point
                for point_elem in period_elem.findall("entsoe:Point", _NS):
                    pos_elem = point_elem.find("entsoe:position", _NS)
                    qty_elem = point_elem.find(
                        "entsoe:quantity", _NS
                    )
                    if pos_elem is None or qty_elem is None:
                        continue
                    try:
                        position = int(pos_elem.text)  # type: ignore[arg-type]
                        quantity = float(qty_elem.text)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue

                    ts_utc = period_start + timedelta(
                        minutes=resolution_minutes * (position - 1)
                    )
                    records.append(
                        {
                            "timestamp_utc": ts_utc,
                            "production_type": production_type,
                            "actual_power_mw": quantity,
                        }
                    )

        if not records:
            return _empty_generation_df()

        df = pd.DataFrame(records)
        df.sort_values(
            ["production_type", "timestamp_utc"], inplace=True
        )
        df.reset_index(drop=True, inplace=True)
        return df


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def _parse_resolution(resolution_str: str | None) -> int:
    """Convert an ENTSO-E ISO 8601 duration string to minutes.

    Only the resolutions commonly encountered in generation data are
    handled: ``PT15M`` (15 min), ``PT30M`` (30 min), ``PT60M`` / ``PT1H``
    (60 min).  Anything else defaults to 60 minutes.

    Args:
        resolution_str: Raw duration string from the XML (e.g. ``"PT60M"``).

    Returns:
        Resolution in minutes as an integer.
    """
    if not resolution_str:
        return 60
    mapping = {
        "PT15M": 15,
        "PT30M": 30,
        "PT60M": 60,
        "PT1H": 60,
    }
    return mapping.get(resolution_str.upper(), 60)


def _empty_generation_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical generation schema.

    Returns:
        Empty :class:`~pandas.DataFrame` with columns
        ``timestamp_utc``, ``production_type``, ``actual_power_mw``.
    """
    return pd.DataFrame(
        columns=["timestamp_utc", "production_type", "actual_power_mw"]
    )
