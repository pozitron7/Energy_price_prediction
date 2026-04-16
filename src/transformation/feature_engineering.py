"""Feature engineering pipeline for the Energy Price Prediction project.

This module is responsible for joining the three raw tables
(``raw_energy_prices``, ``raw_weather_data``, ``raw_generation_data``) and
producing the ``fact_hourly_market_features`` table that is consumed by the
model training and inference pipelines.

Typical usage
-------------
>>> from sqlalchemy import create_engine
>>> from src.transformation.feature_engineering import FeatureEngineer
>>>
>>> engine = create_engine("postgresql+psycopg2://user:pw@localhost/eprice")
>>> fe = FeatureEngineer()
>>> df = fe.build_feature_table(engine, start_date=date(2024, 1, 1))
>>> rows_written = fe.save_to_db(df, engine)
"""

from __future__ import annotations

import logging
import math
from datetime import date, timezone
from typing import Optional

import holidays
import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column constants – kept in sync with schema.sql
# ---------------------------------------------------------------------------

# Columns written to / read from fact_hourly_market_features
FACT_TABLE = "fact_hourly_market_features"

FACT_COLUMNS: list[str] = [
    "timestamp_utc",
    "price_eur_mwh",
    "solar_generation_mw",
    "wind_generation_mw",
    "avg_temperature_c",
    "avg_cloud_cover_pct",
    "avg_wind_speed",
    "avg_irradiance",
    "day_of_week",
    "hour_of_day",
    "month",
    "is_holiday",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _cyclic_encode(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode a numeric series as a pair of sine / cosine features.

    Args:
        series: Numeric pandas Series to encode (e.g. hour-of-day values).
        period: The full cycle length (e.g. 24 for hours, 7 for days).

    Returns:
        A tuple ``(sin_series, cos_series)`` with values in [-1, 1].
    """
    angle = 2.0 * math.pi * series / period
    return np.sin(angle), np.cos(angle)


def _resample_to_hourly(df: pd.DataFrame, timestamp_col: str = "timestamp_utc") -> pd.DataFrame:
    """Resample a DataFrame to hourly UTC frequency using mean aggregation.

    Args:
        df: Input DataFrame.  Must have a column named ``timestamp_col`` that
            is already timezone-aware (UTC).
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame indexed by an hourly ``DatetimeIndex`` (UTC).
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.set_index(timestamp_col).sort_index()
    # Resample numeric columns to hourly means; non-numeric are dropped
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return df[numeric_cols].resample("h").mean()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class FeatureEngineer:
    """Orchestrates the end-to-end feature engineering pipeline.

    Loads raw tables from the database, aligns them to an hourly UTC grid,
    computes derived features (cyclic encodings, lag features, holiday flags,
    weather averages), handles missing values, and persists the result to
    ``fact_hourly_market_features``.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_feature_table(
        self,
        engine: Engine,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Build the complete hourly feature DataFrame from the raw DB tables.

        Steps
        -----
        1. Load ``raw_energy_prices``, ``raw_weather_data`` (historical only),
           and ``raw_generation_data`` from the database.
        2. Resample / align all sources to hourly UTC timestamps.
        3. Merge all sources on ``timestamp_utc``.
        4. Compute cyclic features, lag features, holiday flag, and calendar
           columns.
        5. Handle missing values with forward-fill (≤ 3 h) then linear
           interpolation for weather columns; drop rows where price is still
           NaN after imputation.
        6. Return a cleaned DataFrame whose columns match ``FACT_COLUMNS``.

        Args:
            engine: SQLAlchemy ``Engine`` connected to the project database.
            start_date: Optional inclusive lower bound for ``timestamp_utc``
                filtering.  When ``None`` the full history is loaded.
            end_date: Optional inclusive upper bound for ``timestamp_utc``
                filtering.  When ``None`` the full history is loaded.

        Returns:
            Cleaned DataFrame with columns defined by :data:`FACT_COLUMNS`
            and ``timestamp_utc`` as the index.

        Raises:
            ValueError: If the merged DataFrame is empty after filtering.
        """
        logger.info("Loading raw tables from database …")

        # ------------------------------------------------------------------
        # 1. Load raw tables
        # ------------------------------------------------------------------
        price_df = self._load_prices(engine, start_date, end_date)
        weather_df = self._load_weather(engine, start_date, end_date)
        gen_df = self._load_generation(engine, start_date, end_date)

        # ------------------------------------------------------------------
        # 2. Resample to hourly
        # ------------------------------------------------------------------
        price_hourly = self._resample_prices(price_df)
        weather_hourly = self._resample_weather(weather_df)
        gen_hourly = self._resample_generation(gen_df)

        # ------------------------------------------------------------------
        # 3. Merge on timestamp_utc
        # ------------------------------------------------------------------
        merged = (
            price_hourly
            .join(weather_hourly, how="outer")
            .join(gen_hourly, how="outer")
        )

        if merged.empty:
            raise ValueError(
                "Merged DataFrame is empty – check that raw tables contain "
                "data for the requested date range."
            )

        # ------------------------------------------------------------------
        # 4. Derived features
        # ------------------------------------------------------------------
        merged = self._add_calendar_features(merged)
        merged = self._add_lag_features(merged)
        merged = self._add_holiday_flag(merged)

        # ------------------------------------------------------------------
        # 5. Handle missing values
        # ------------------------------------------------------------------
        merged = self._impute(merged)

        # ------------------------------------------------------------------
        # 6. Select and order final columns (excluding timestamp_utc index)
        # ------------------------------------------------------------------
        output_cols = [c for c in FACT_COLUMNS if c != "timestamp_utc"]
        # Add any missing columns as NaN so the schema is always satisfied
        for col in output_cols:
            if col not in merged.columns:
                merged[col] = np.nan

        result = merged[output_cols].copy()
        result.index.name = "timestamp_utc"

        logger.info("Feature table built: %d rows, %d columns.", len(result), len(result.columns))
        return result

    def save_to_db(self, df: pd.DataFrame, engine: Engine) -> int:
        """Upsert the feature DataFrame into ``fact_hourly_market_features``.

        Uses an INSERT … ON CONFLICT (timestamp_utc) DO UPDATE strategy for
        PostgreSQL/TimescaleDB.  Falls back to a simple ``to_sql`` with
        ``if_exists='replace'`` when the dialect is SQLite (used in tests).

        Args:
            df: DataFrame returned by :meth:`build_feature_table`.  Its index
                must be named ``timestamp_utc``.
            engine: SQLAlchemy ``Engine`` connected to the project database.

        Returns:
            Number of rows written / updated.
        """
        df_reset = df.reset_index()  # timestamp_utc becomes a regular column

        dialect = engine.dialect.name
        rows_written: int = 0

        if dialect == "sqlite":
            # SQLite – INSERT OR REPLACE keeps the table schema / primary key
            from src.database.upsert import sqlite_upsert

            rows_written = sqlite_upsert(df_reset, FACT_TABLE, engine)
            logger.info("SQLite: wrote %d rows to %s.", rows_written, FACT_TABLE)
            return rows_written

        # PostgreSQL – chunked upsert
        chunk_size = 1_000
        total = 0
        cols = df_reset.columns.tolist()
        col_list = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join(f":{c}" for c in cols)
        update_set = ", ".join(
            f'"{c}" = EXCLUDED."{c}"' for c in cols if c != "timestamp_utc"
        )
        upsert_sql = text(
            f"""
            INSERT INTO {FACT_TABLE} ({col_list})
            VALUES ({placeholders})
            ON CONFLICT (timestamp_utc) DO UPDATE SET {update_set};
            """
        )

        with engine.begin() as conn:
            for start in range(0, len(df_reset), chunk_size):
                chunk = df_reset.iloc[start : start + chunk_size]
                records = chunk.to_dict(orient="records")
                conn.execute(upsert_sql, records)
                total += len(records)

        logger.info("PostgreSQL: upserted %d rows into %s.", total, FACT_TABLE)
        return total

    def build_inference_features(
        self,
        weather_forecast_df: pd.DataFrame,
        last_prices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build feature rows for the next 24 hours using forecast data.

        Constructs a feature DataFrame suitable for model inference by
        combining weather forecast data with lagged price history.  The
        resulting DataFrame has the same column layout as the output of
        :meth:`build_feature_table` (excluding ``price_eur_mwh``, which is
        the target to predict).

        Args:
            weather_forecast_df: DataFrame with forecast weather rows.  Must
                contain columns: ``timestamp_utc``, ``temperature_c``,
                ``cloud_cover_pct``, ``wind_speed_10m``,
                ``direct_normal_irradiance``.  Rows can cover multiple CZ
                locations – they will be averaged by timestamp.
            last_prices_df: DataFrame of recent actual prices used to compute
                lag features.  Must contain columns ``timestamp_utc`` and
                ``price_eur_mwh`` and cover at least the past 168 hours
                (7 days).

        Returns:
            DataFrame with one row per forecast hour, indexed by
            ``timestamp_utc``, ready to be passed to the model.
        """
        # ------------------------------------------------------------------
        # 1. Aggregate weather forecast across CZ locations
        # ------------------------------------------------------------------
        weather_forecast_df = weather_forecast_df.copy()
        weather_forecast_df["timestamp_utc"] = pd.to_datetime(
            weather_forecast_df["timestamp_utc"], utc=True
        )
        weather_avg = (
            weather_forecast_df.groupby("timestamp_utc")[
                ["temperature_c", "cloud_cover_pct", "wind_speed_10m", "direct_normal_irradiance"]
            ]
            .mean()
            .rename(
                columns={
                    "temperature_c": "avg_temperature_c",
                    "cloud_cover_pct": "avg_cloud_cover_pct",
                    "wind_speed_10m": "avg_wind_speed",
                    "direct_normal_irradiance": "avg_irradiance",
                }
            )
        )

        # ------------------------------------------------------------------
        # 2. Build lag price series from historical prices
        # ------------------------------------------------------------------
        last_prices_df = last_prices_df.copy()
        last_prices_df["timestamp_utc"] = pd.to_datetime(
            last_prices_df["timestamp_utc"], utc=True
        )
        price_series = (
            last_prices_df.set_index("timestamp_utc")["price_eur_mwh"]
            .sort_index()
        )

        # Extend index to include forecast timestamps (price = NaN for future)
        all_idx = price_series.index.union(weather_avg.index)
        price_full = price_series.reindex(all_idx)

        lag_df = pd.DataFrame(index=all_idx)
        lag_df["price_lag_24h"] = price_full.shift(24)
        lag_df["price_lag_48h"] = price_full.shift(48)
        lag_df["price_lag_168h"] = price_full.shift(168)

        # Keep only forecast timestamps
        lag_df = lag_df.loc[weather_avg.index]

        # ------------------------------------------------------------------
        # 3. Combine and add calendar / cyclic features
        # ------------------------------------------------------------------
        features = weather_avg.join(lag_df, how="left")
        features = self._add_calendar_features(features)
        features = self._add_holiday_flag(features)

        # Placeholder generation columns (not available in forecast)
        features["solar_generation_mw"] = np.nan
        features["wind_generation_mw"] = np.nan
        features["price_eur_mwh"] = np.nan  # target – unknown at inference

        # Ensure column order matches FACT_COLUMNS
        output_cols = [c for c in FACT_COLUMNS if c != "timestamp_utc"]
        for col in output_cols:
            if col not in features.columns:
                features[col] = np.nan

        features.index.name = "timestamp_utc"
        logger.info(
            "Inference features built: %d rows for next-day prediction.", len(features)
        )
        return features[output_cols]

    # ------------------------------------------------------------------
    # Private – data loading
    # ------------------------------------------------------------------

    def _load_prices(
        self,
        engine: Engine,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> pd.DataFrame:
        """Load ``raw_energy_prices`` from the database.

        Args:
            engine: SQLAlchemy ``Engine``.
            start_date: Optional inclusive start filter.
            end_date: Optional inclusive end filter.

        Returns:
            DataFrame with columns ``timestamp_utc``, ``price_eur_mwh``.
        """
        query = "SELECT timestamp_utc, price_eur_mwh FROM raw_energy_prices WHERE market_type = 'day_ahead'"
        params: dict = {}
        if start_date:
            query += " AND timestamp_utc >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND timestamp_utc <= :end_date"
            params["end_date"] = end_date
        query += " ORDER BY timestamp_utc"

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        logger.debug("Loaded %d price rows.", len(df))
        return df

    def _load_weather(
        self,
        engine: Engine,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> pd.DataFrame:
        """Load historical (non-forecast) ``raw_weather_data`` from the database.

        Only rows with ``is_forecast = FALSE`` are loaded, as these represent
        verified observations used for model training.

        Args:
            engine: SQLAlchemy ``Engine``.
            start_date: Optional inclusive start filter.
            end_date: Optional inclusive end filter.

        Returns:
            DataFrame with columns ``timestamp_utc``, ``location_name``,
            ``temperature_c``, ``cloud_cover_pct``, ``wind_speed_10m``,
            ``direct_normal_irradiance``.
        """
        query = (
            "SELECT timestamp_utc, location_name, temperature_c, cloud_cover_pct, "
            "wind_speed_10m, direct_normal_irradiance "
            "FROM raw_weather_data WHERE is_forecast = FALSE"
        )
        params: dict = {}
        if start_date:
            query += " AND timestamp_utc >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND timestamp_utc <= :end_date"
            params["end_date"] = end_date
        query += " ORDER BY timestamp_utc"

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        logger.debug("Loaded %d weather rows.", len(df))
        return df

    def _load_generation(
        self,
        engine: Engine,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> pd.DataFrame:
        """Load ``raw_generation_data`` from the database.

        Args:
            engine: SQLAlchemy ``Engine``.
            start_date: Optional inclusive start filter.
            end_date: Optional inclusive end filter.

        Returns:
            DataFrame with columns ``timestamp_utc``, ``production_type``,
            ``actual_power_mw``.
        """
        query = (
            "SELECT timestamp_utc, production_type, actual_power_mw "
            "FROM raw_generation_data"
        )
        params: dict = {}
        if start_date:
            query += " WHERE timestamp_utc >= :start_date"
            params["start_date"] = start_date
            if end_date:
                query += " AND timestamp_utc <= :end_date"
                params["end_date"] = end_date
        elif end_date:
            query += " WHERE timestamp_utc <= :end_date"
            params["end_date"] = end_date
        query += " ORDER BY timestamp_utc"

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        logger.debug("Loaded %d generation rows.", len(df))
        return df

    # ------------------------------------------------------------------
    # Private – resampling
    # ------------------------------------------------------------------

    def _resample_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample price data to hourly UTC frequency.

        Args:
            df: Raw price DataFrame with ``timestamp_utc`` and
                ``price_eur_mwh`` columns.

        Returns:
            DataFrame indexed by hourly ``DatetimeIndex`` (UTC) with column
            ``price_eur_mwh``.
        """
        df = df.copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.set_index("timestamp_utc").sort_index()
        return df[["price_eur_mwh"]].resample("h").mean()

    def _resample_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average weather across CZ locations and resample to hourly.

        Computes mean values across all ``location_name`` values at each
        timestamp, then resamples to hourly UTC.

        Args:
            df: Raw weather DataFrame with columns ``timestamp_utc``,
                ``location_name``, ``temperature_c``, ``cloud_cover_pct``,
                ``wind_speed_10m``, ``direct_normal_irradiance``.

        Returns:
            DataFrame indexed by hourly ``DatetimeIndex`` (UTC) with averaged
            weather columns renamed to ``avg_temperature_c``,
            ``avg_cloud_cover_pct``, ``avg_wind_speed``, ``avg_irradiance``.
        """
        df = df.copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

        # Average across all CZ locations per timestamp
        weather_avg = (
            df.groupby("timestamp_utc")[
                ["temperature_c", "cloud_cover_pct", "wind_speed_10m", "direct_normal_irradiance"]
            ]
            .mean()
            .rename(
                columns={
                    "temperature_c": "avg_temperature_c",
                    "cloud_cover_pct": "avg_cloud_cover_pct",
                    "wind_speed_10m": "avg_wind_speed",
                    "direct_normal_irradiance": "avg_irradiance",
                }
            )
            .sort_index()
        )

        return weather_avg.resample("h").mean()

    def _resample_generation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot generation data by production type and resample to hourly.

        Only ``solar`` and ``wind_onshore`` production types are extracted.
        All other types are ignored.

        Args:
            df: Raw generation DataFrame with columns ``timestamp_utc``,
                ``production_type``, ``actual_power_mw``.

        Returns:
            DataFrame indexed by hourly ``DatetimeIndex`` (UTC) with columns
            ``solar_generation_mw`` and ``wind_generation_mw``.
        """
        df = df.copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.set_index("timestamp_utc").sort_index()

        # Pivot so each production_type becomes a column
        pivoted = df.pivot_table(
            index="timestamp_utc",
            columns="production_type",
            values="actual_power_mw",
            aggfunc="sum",
        )

        # Rename and keep only the columns of interest
        rename_map = {
            "solar": "solar_generation_mw",
            "wind_onshore": "wind_generation_mw",
        }
        available = {k: v for k, v in rename_map.items() if k in pivoted.columns}
        pivoted = pivoted[list(available.keys())].rename(columns=available)

        # Add missing columns as NaN
        for target_col in rename_map.values():
            if target_col not in pivoted.columns:
                pivoted[target_col] = np.nan

        return pivoted.resample("h").mean()

    # ------------------------------------------------------------------
    # Private – feature computation
    # ------------------------------------------------------------------

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar and cyclic time features to a DataFrame.

        Computes ``hour_of_day``, ``day_of_week``, ``month``, ``hour_sin``,
        ``hour_cos``, ``dow_sin``, and ``dow_cos`` from the DatetimeIndex.

        Args:
            df: DataFrame with a timezone-aware (UTC) ``DatetimeIndex``.

        Returns:
            DataFrame with the new columns added in-place.
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex) and "timestamp_utc" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp_utc"], utc=True))
        idx = df.index

        df["hour_of_day"] = idx.hour.astype("int16")
        df["day_of_week"] = idx.dayofweek.astype("int16")  # 0=Mon … 6=Sun
        df["month"] = idx.month.astype("int16")

        hour_sin, hour_cos = _cyclic_encode(df["hour_of_day"], period=24)
        dow_sin, dow_cos = _cyclic_encode(df["day_of_week"], period=7)

        df["hour_sin"] = hour_sin
        df["hour_cos"] = hour_cos
        df["dow_sin"] = dow_sin
        df["dow_cos"] = dow_cos

        return df

    _add_cyclic_features = _add_calendar_features
    add_cyclic_features = _add_calendar_features

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute autoregressive (lagged) price features.

        Adds ``price_lag_24h``, ``price_lag_48h``, and ``price_lag_168h``
        columns using a 1-hour-resolution index assumption.

        Args:
            df: DataFrame with a ``price_eur_mwh`` column and an hourly UTC
                ``DatetimeIndex``.

        Returns:
            DataFrame with the lag columns added.
        """
        df = df.copy()
        df["price_lag_24h"] = df["price_eur_mwh"].shift(24)
        df["price_lag_48h"] = df["price_eur_mwh"].shift(48)
        df["price_lag_168h"] = df["price_eur_mwh"].shift(168)
        return df

    def _add_holiday_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a Czech public-holiday boolean flag.

        Uses the ``holidays`` library to determine whether each timestamp in
        the DataFrame is a Czech public holiday. Handles both DatetimeIndex
        and 'timestamp_utc' column.

        Args:
            df: DataFrame with a DatetimeIndex or 'timestamp_utc' column.

        Returns:
            DataFrame with ``is_holiday`` (bool) column added.
        """
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            years = list(df.index.year.unique())
            cz_holidays = holidays.country_holidays("CZ", years=years)
            df["is_holiday"] = [ts.date() in cz_holidays for ts in df.index]
        elif "timestamp_utc" in df.columns:
            ts_series = pd.to_datetime(df["timestamp_utc"], utc=True)
            years = list(ts_series.dt.year.unique())
            cz_holidays = holidays.country_holidays("CZ", years=years)
            df["is_holiday"] = [ts.date() in cz_holidays for ts in ts_series]
        else:
            df["is_holiday"] = False

        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature DataFrame.

        Strategy:
        - Forward-fill up to 3 consecutive hours for all columns.
        - Apply linear interpolation for weather columns (``avg_*``) to fill
          any remaining gaps.
        - Drop rows where ``price_eur_mwh`` is still NaN after imputation,
          as a row without a target is useless for training.

        Args:
            df: Merged feature DataFrame with potential NaN values.

        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()

        # Step 1 – forward fill up to 3 hours for feature columns only.
        # The target (price) and its lags are never imputed: a filled target
        # would be a fabricated "actual" price.
        no_fill = [c for c in ("price_eur_mwh", "price_lag_24h", "price_lag_48h", "price_lag_168h") if c in df.columns]
        fill_cols = [c for c in df.columns if c not in no_fill]
        df[fill_cols] = df[fill_cols].ffill(limit=3)

        # Step 2 – linear interpolation for weather columns
        weather_cols = ["avg_temperature_c", "avg_cloud_cover_pct", "avg_wind_speed", "avg_irradiance"]
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear", limit_direction="both")

        # Step 3 – drop rows with no target price
        before = len(df)
        df = df.dropna(subset=["price_eur_mwh"])
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows where price_eur_mwh was NaN after imputation.", dropped)

        return df
