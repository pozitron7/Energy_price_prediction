"""
src/database/upsert.py
======================
Dialect-agnostic helpers for idempotent DataFrame writes.

The ingestion clients use PostgreSQL ``INSERT ... ON CONFLICT`` upserts in
production.  SQLite (local dev / CI) cannot bind ``pandas.Timestamp`` or
``numpy`` scalars, so this module provides a small ``INSERT OR REPLACE``
fallback that normalises values to plain Python / ISO-8601 strings.
"""

from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def _to_sqlite_value(value: object) -> object:
    """Convert a pandas/numpy scalar to a type accepted by ``sqlite3``."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.isoformat()
    if hasattr(value, "item"):  # numpy scalar -> python scalar
        return value.item()
    return value


def sqlite_upsert(df: pd.DataFrame, table: str, engine: Engine) -> int:
    """Insert or replace all rows of *df* into *table* on a SQLite engine.

    Args:
        df: DataFrame whose columns match column names of *table*.
        table: Destination table name.
        engine: SQLAlchemy engine using the SQLite dialect.

    Returns:
        Number of rows written.
    """
    if df.empty:
        return 0
    cols = list(df.columns)
    stmt = text(
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join(':' + c for c in cols)})"
    )
    records = [
        {c: _to_sqlite_value(v) for c, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
    with engine.begin() as conn:
        conn.execute(stmt, records)
    logger.debug("SQLite upserted %d rows into %s.", len(records), table)
    return len(records)
