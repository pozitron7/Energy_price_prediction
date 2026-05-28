"""
tests/test_database.py
======================
Tests for :mod:`src.database.db_manager`.

All tests use an in-memory SQLite database so no external service is required.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy import inspect, text

from src.database.db_manager import get_engine, init_db, session_scope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_engine() -> sa.engine.Engine:
    """Return a fresh, in-memory SQLite engine for each test.

    Returns:
        An in-memory SQLite :class:`~sqlalchemy.engine.Engine`.
    """
    return get_engine("sqlite:///:memory:")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitDb:
    """Tests for :func:`src.database.db_manager.init_db`."""

    def test_init_db_sqlite_creates_all_tables(self) -> None:
        """init_db must create all 6 expected tables in an in-memory SQLite DB.

        Verifies that calling :func:`init_db` on a blank SQLite engine results
        in exactly the six domain tables defined in the project schema.
        """
        engine = _make_memory_engine()
        init_db(engine)

        existing_tables = set(inspect(engine).get_table_names())

        expected_tables = {
            "raw_energy_prices",
            "raw_weather_data",
            "raw_generation_data",
            "fact_hourly_market_features",
            "model_predictions",
            "evaluation_results",
        }

        assert expected_tables.issubset(existing_tables), (
            f"Missing tables: {expected_tables - existing_tables}"
        )


class TestSessionScope:
    """Tests for :func:`src.database.db_manager.session_scope`."""

    def test_session_scope_commit(self) -> None:
        """A successful block inside session_scope must persist the inserted row.

        Inserts a single row into ``raw_energy_prices`` and verifies it can be
        retrieved in a subsequent query, confirming that the commit path works.
        """
        engine = _make_memory_engine()
        init_db(engine)

        insert_sql = text(
            "INSERT INTO raw_energy_prices (timestamp_utc, market_type, price_eur_mwh) "
            "VALUES ('2024-06-01T00:00:00', 'day_ahead', 42.5)"
        )

        # Insert via session_scope
        with session_scope(engine) as session:
            session.execute(insert_sql)

        # Verify persistence in a separate connection
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT price_eur_mwh FROM raw_energy_prices WHERE timestamp_utc='2024-06-01T00:00:00'")
            ).fetchone()

        assert result is not None, "Row was not committed."
        assert result[0] == pytest.approx(42.5), (
            f"Expected price_eur_mwh=42.5, got {result[0]}"
        )

    def test_session_scope_rollback(self) -> None:
        """An exception inside session_scope must roll back the transaction.

        Forces a ``RuntimeError`` after the INSERT so the session must roll back,
        leaving the table empty.
        """
        engine = _make_memory_engine()
        init_db(engine)

        insert_sql = text(
            "INSERT INTO raw_energy_prices (timestamp_utc, market_type, price_eur_mwh) "
            "VALUES ('2024-06-02T00:00:00', 'day_ahead', 99.9)"
        )

        with pytest.raises(RuntimeError):
            with session_scope(engine) as session:
                session.execute(insert_sql)
                raise RuntimeError("Forced rollback")

        # Verify that nothing was persisted
        with engine.connect() as conn:
            count = conn.execute(
                text("SELECT COUNT(*) FROM raw_energy_prices WHERE timestamp_utc='2024-06-02T00:00:00'")
            ).scalar()

        assert count == 0, f"Expected 0 rows after rollback, found {count}."


class TestGetEngine:
    """Tests for :func:`src.database.db_manager.get_engine` robustness."""

    def test_get_engine_empty_string_fallback(self) -> None:
        """Empty string or whitespace must safely fall back to default SQLite engine."""
        engine = get_engine("")
        assert engine.dialect.name == "sqlite"

        engine_ws = get_engine("   ")
        assert engine_ws.dialect.name == "sqlite"
