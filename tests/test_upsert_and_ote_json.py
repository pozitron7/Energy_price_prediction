"""Tests for the SQLite upsert helper and the OTE JSON price parser."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
from sqlalchemy import text

from src.database.db_manager import get_engine, init_db
from src.database.upsert import sqlite_upsert
from src.ingestion.ote_client import OTEClient


def _payload(n_points: int, title: str = "60min cena (EUR/MWh)") -> dict:
    """Build a minimal OTE chart-data payload with ``n_points`` prices."""
    points = [{"x": str(i + 1), "y": 100.0 + i} for i in range(n_points)]
    return {"data": {"dataLine": [{"title": title, "type": "1", "point": points}]}}


def _client_with(payload: dict) -> OTEClient:
    session = MagicMock()
    session.get.return_value.json.return_value = payload
    session.get.return_value.raise_for_status.return_value = None
    session.headers = {}
    return OTEClient(session=session)


def test_sqlite_upsert_is_idempotent() -> None:
    """Writing the same timestamps twice must not duplicate rows."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "market_type": "day_ahead",
            "price_eur_mwh": [1.0, 2.0, 3.0],
        }
    )
    sqlite_upsert(df, "raw_energy_prices", engine)
    df["price_eur_mwh"] = [9.0, 9.0, 9.0]
    sqlite_upsert(df, "raw_energy_prices", engine)
    with engine.connect() as conn:
        n, avg = conn.execute(text("SELECT COUNT(*), AVG(price_eur_mwh) FROM raw_energy_prices")).one()
    assert (n, avg) == (3, 9.0)


def test_ote_json_normal_day_has_24_hours() -> None:
    """A regular day yields 24 hourly records."""
    records = _client_with(_payload(24))._fetch_json_prices(date(2026, 10, 20))
    assert len(records) == 24


def test_ote_json_dst_days() -> None:
    """DST days have 23 (spring) and 25 (autumn) hours on the UTC axis."""
    assert len(_client_with(_payload(23))._fetch_json_prices(date(2024, 3, 31))) == 23
    assert len(_client_with(_payload(25))._fetch_json_prices(date(2026, 10, 25))) == 25


def test_ote_json_quarter_hours_are_averaged() -> None:
    """96 quarter-hour prices are averaged into 24 hourly prices."""
    records = _client_with(_payload(96))._fetch_json_prices(date(2026, 10, 20))
    assert len(records) == 24
    assert records[0]["price_eur_mwh"] == round((100 + 101 + 102 + 103) / 4, 2)


def test_ote_json_unpublished_day_is_empty() -> None:
    """A day without published prices yields no records (no fake data)."""
    assert _client_with({"data": {"dataLine": []}})._fetch_json_prices(date(2030, 1, 1)) == []
