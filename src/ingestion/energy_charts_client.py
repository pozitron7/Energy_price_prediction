"""
src/ingestion/energy_charts_client.py
=====================================
Token-free source of Czech renewable generation: the Energy-Charts public API
(Fraunhofer ISE, https://api.energy-charts.info), which republishes
ENTSO-E / TSO data.

``/public_power?country=cz`` returns 15-minute net generation per production
type.  Solar and onshore wind are extracted and averaged to hourly values in
the same schema as :class:`src.ingestion.entsoe_client.ENTSOEClient`
(``timestamp_utc``, ``production_type``, ``actual_power_mw``), so the rest of
the pipeline is unchanged.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

API_URL = "https://api.energy-charts.info/public_power"

#: Energy-Charts series name -> internal production_type
_SERIES_MAP: dict[str, str] = {
    "Solar": "solar",
    "Wind onshore": "wind_onshore",
}

_MAX_RETRIES = 3
_CHUNK_DAYS = 365


class EnergyChartsClient:
    """Fetch hourly Czech solar / wind generation without an API token."""

    def __init__(self, country: str = "cz", timeout: int = 60) -> None:
        """Create a client.

        Args:
            country: Energy-Charts country code (``"cz"``).
            timeout: HTTP timeout in seconds.
        """
        self.country = country
        self.timeout = timeout

    def fetch_generation(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch hourly solar and wind generation for a date range.

        Args:
            start_date: First day (inclusive, UTC).
            end_date: Last day (inclusive, UTC).

        Returns:
            DataFrame with columns ``timestamp_utc``, ``production_type`` and
            ``actual_power_mw``.  Empty if nothing could be fetched.
        """
        frames: list[pd.DataFrame] = []
        current = start_date
        while current <= end_date:
            chunk_end = min(current + timedelta(days=_CHUNK_DAYS - 1), end_date)
            frame = self._fetch_chunk(current, chunk_end)
            if not frame.empty:
                frames.append(frame)
            current = chunk_end + timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=["timestamp_utc", "production_type", "actual_power_mw"])
        return pd.concat(frames, ignore_index=True).drop_duplicates(
            ["timestamp_utc", "production_type"]
        )

    def _fetch_chunk(self, start: date, end: date) -> pd.DataFrame:
        """Fetch one request window with retries and parse it."""
        params = {"country": self.country, "start": start.isoformat(), "end": end.isoformat()}
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(API_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return self._parse(resp.json())
            except (requests.RequestException, ValueError) as exc:
                logger.warning("Energy-Charts %s..%s attempt %d failed: %s", start, end, attempt, exc)
                time.sleep(2 * attempt)
        return pd.DataFrame(columns=["timestamp_utc", "production_type", "actual_power_mw"])

    @staticmethod
    def _parse(payload: dict) -> pd.DataFrame:
        """Convert an Energy-Charts payload to hourly long-format rows."""
        index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
        rows: list[pd.DataFrame] = []
        for series in payload["production_types"]:
            ptype = _SERIES_MAP.get(series["name"])
            if ptype is None:
                continue
            s = pd.Series(series["data"], index=index, dtype="float64").dropna()
            hourly = s.resample("h").mean().dropna()
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp_utc": hourly.index,
                        "production_type": ptype,
                        "actual_power_mw": hourly.clip(lower=0).round(2).values,
                    }
                )
            )
        if not rows:
            return pd.DataFrame(columns=["timestamp_utc", "production_type", "actual_power_mw"])
        return pd.concat(rows, ignore_index=True)

    def save_to_db(self, df: pd.DataFrame, engine: Engine) -> int:
        """Upsert into ``raw_generation_data`` (shared with the ENTSO-E client)."""
        from src.ingestion.entsoe_client import ENTSOEClient

        return ENTSOEClient(api_token="").save_to_db(df, engine)
