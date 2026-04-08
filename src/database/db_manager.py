"""
src/database/db_manager.py
==========================
Database connection manager with dual backend support:
  - PostgreSQL + TimescaleDB  (production – set DATABASE_URL env var)
  - SQLite                     (local dev / CI – zero configuration)

The module exposes a single ``get_engine()`` factory and a ``init_db()``
helper that creates all tables from the SQL schema (PostgreSQL) or via
SQLAlchemy Core (SQLite).
"""

from __future__ import annotations

import logging
import sqlite3
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite schema – mirrors schema.sql but uses SQLite-compatible syntax
# (no TIMESTAMPTZ, no SERIAL, no gen_random_uuid, no TimescaleDB calls)
# ---------------------------------------------------------------------------
_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS raw_energy_prices (
    timestamp_utc  TEXT    NOT NULL,
    market_type    TEXT    NOT NULL DEFAULT 'day_ahead',
    price_eur_mwh  REAL    NOT NULL,
    price_czk_mwh  REAL,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (timestamp_utc, market_type)
);

CREATE TABLE IF NOT EXISTS raw_weather_data (
    timestamp_utc            TEXT    NOT NULL,
    location_name            TEXT    NOT NULL,
    latitude                 REAL    NOT NULL,
    longitude                REAL    NOT NULL,
    temperature_c            REAL,
    cloud_cover_pct          INTEGER,
    wind_speed_10m           REAL,
    direct_normal_irradiance REAL,
    is_forecast              INTEGER NOT NULL DEFAULT 0,
    created_at               TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (timestamp_utc, location_name, is_forecast)
);

CREATE TABLE IF NOT EXISTS raw_generation_data (
    timestamp_utc    TEXT    NOT NULL,
    production_type  TEXT    NOT NULL,
    actual_power_mw  REAL    NOT NULL,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (timestamp_utc, production_type)
);

CREATE TABLE IF NOT EXISTS fact_hourly_market_features (
    timestamp_utc       TEXT    PRIMARY KEY,
    price_eur_mwh       REAL,
    solar_generation_mw REAL,
    wind_generation_mw  REAL,
    avg_temperature_c   REAL,
    avg_cloud_cover_pct INTEGER,
    avg_wind_speed      REAL,
    avg_irradiance      REAL,
    day_of_week         INTEGER,
    hour_of_day         INTEGER,
    month               INTEGER,
    is_holiday          INTEGER NOT NULL DEFAULT 0,
    price_lag_24h       REAL,
    price_lag_48h       REAL,
    price_lag_168h      REAL,
    hour_sin            REAL,
    hour_cos            REAL,
    dow_sin             REAL,
    dow_cos             REAL,
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id        TEXT    NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    created_at           TEXT    NOT NULL DEFAULT (datetime('now')),
    target_timestamp_utc TEXT    NOT NULL,
    predicted_price_eur  REAL    NOT NULL,
    model_version        TEXT    NOT NULL,
    PRIMARY KEY (target_timestamp_utc, model_version)
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    eval_id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    evaluated_at     TEXT     NOT NULL DEFAULT (datetime('now')),
    period_start     TEXT     NOT NULL,
    period_end       TEXT     NOT NULL,
    model_version    TEXT     NOT NULL,
    mae_eur_mwh      REAL,
    rmse_eur_mwh     REAL,
    mape_pct         REAL,
    directional_acc  REAL,
    n_samples        INTEGER
);
"""


def get_engine(database_url: str | None = None) -> Engine:
    """Return a SQLAlchemy ``Engine`` for the configured database.

    Args:
        database_url: Connection string.  Defaults to the value in
            ``config.settings.DATABASE_URL``.

    Returns:
        A SQLAlchemy Engine instance.
    """
    if database_url is None:
        from config.settings import DATABASE_URL  # lazy import avoids circular deps
        database_url = DATABASE_URL

    is_sqlite = database_url.startswith("sqlite")
    connect_args: dict = {}
    if is_sqlite:
        # SQLite: allow multiple threads to share the same connection
        connect_args = {"check_same_thread": False}
        # Ensure parent directory exists
        path = database_url.replace("sqlite:///", "")
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    engine = sa.create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        echo=False,
    )
    logger.info("Database engine created: %s", database_url.split("@")[-1])
    return engine


def init_db(engine: Engine) -> None:
    """Create all tables if they do not exist.

    For PostgreSQL, reads and executes ``src/database/schema.sql``.
    For SQLite, applies the embedded ``_SQLITE_DDL`` string.

    Args:
        engine: SQLAlchemy engine returned by :func:`get_engine`.
    """
    dialect = engine.dialect.name
    logger.info("Initialising database tables (%s backend)…", dialect)

    if dialect == "sqlite":
        _init_sqlite(engine)
    else:
        _init_postgres(engine)

    logger.info("Database initialisation complete.")


def _init_sqlite(engine: Engine) -> None:
    """Apply SQLite DDL statements."""
    with engine.begin() as conn:
        for statement in _SQLITE_DDL.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))


def _init_postgres(engine: Engine) -> None:
    """Apply the PostgreSQL/TimescaleDB schema from schema.sql."""
    schema_path = Path(__file__).parent / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql))


def get_session_factory(engine: Engine) -> sessionmaker:
    """Return a bound session factory.

    Args:
        engine: SQLAlchemy engine.

    Returns:
        A ``sessionmaker`` bound to *engine*.
    """
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def session_scope(engine: Engine) -> Generator[Session, None, None]:
    """Context manager that provides a transactional database session.

    Automatically commits on success or rolls back on exception.

    Args:
        engine: SQLAlchemy engine.

    Yields:
        An open SQLAlchemy ``Session``.

    Example::

        with session_scope(engine) as session:
            session.execute(...)
    """
    factory = get_session_factory(engine)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
