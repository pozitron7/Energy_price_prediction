-- =============================================================================
--  Energy Price Prediction – Database Schema
--  Compatible with: PostgreSQL + TimescaleDB (production)
--                   SQLite           (local dev / CI – managed by db_manager.py)
--
--  TimescaleDB-specific commands are wrapped in DO blocks so the file can be
--  sourced against a plain PostgreSQL instance without errors.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 0. TimescaleDB extension (optional – skipped silently if not available)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS timescaledb;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'TimescaleDB extension not available – proceeding without it.';
END;
$$;

-- ---------------------------------------------------------------------------
-- 1. Raw energy prices  (OTE-CR day-ahead spot prices)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_energy_prices (
    timestamp_utc  TIMESTAMPTZ NOT NULL,
    market_type    VARCHAR(20)  NOT NULL DEFAULT 'day_ahead',  -- 'day_ahead' | 'intraday'
    price_eur_mwh  NUMERIC(10, 2) NOT NULL,
    price_czk_mwh  NUMERIC(10, 2),
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (timestamp_utc, market_type)
);

-- ---------------------------------------------------------------------------
-- 2. Raw weather data  (Open-Meteo – multi-location CZ)
--    is_forecast=TRUE  → forecast used as ML feature at prediction time
--    is_forecast=FALSE → verified historical observation (used for training)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_weather_data (
    timestamp_utc              TIMESTAMPTZ    NOT NULL,
    location_name              VARCHAR(50)    NOT NULL,          -- e.g. 'Praha'
    latitude                   NUMERIC(6, 4)  NOT NULL,
    longitude                  NUMERIC(6, 4)  NOT NULL,
    temperature_c              NUMERIC(5, 2),
    cloud_cover_pct            SMALLINT,
    wind_speed_10m             NUMERIC(6, 2),
    direct_normal_irradiance   NUMERIC(8, 2),                    -- DNI W/m² (solar)
    is_forecast                BOOLEAN        NOT NULL DEFAULT FALSE,
    created_at                 TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (timestamp_utc, location_name, is_forecast)
);

-- ---------------------------------------------------------------------------
-- 3. Raw generation data  (ENTSO-E – renewable generation by type)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_generation_data (
    timestamp_utc    TIMESTAMPTZ    NOT NULL,
    production_type  VARCHAR(50)    NOT NULL,   -- 'solar' | 'wind_onshore' | ...
    actual_power_mw  NUMERIC(10, 2) NOT NULL,
    created_at       TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    PRIMARY KEY (timestamp_utc, production_type)
);

-- ---------------------------------------------------------------------------
-- 4. Fact table – hourly market features  (joined, cleaned, ready for ML)
--    Built by feature_engineering.py from the three raw tables above.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_hourly_market_features (
    timestamp_utc       TIMESTAMPTZ    PRIMARY KEY,
    -- Target variable
    price_eur_mwh       NUMERIC(10, 2),
    -- Renewable generation
    solar_generation_mw NUMERIC(10, 2),
    wind_generation_mw  NUMERIC(10, 2),
    -- Weather (averaged across CZ locations)
    avg_temperature_c   NUMERIC(5, 2),
    avg_cloud_cover_pct SMALLINT,
    avg_wind_speed      NUMERIC(6, 2),
    avg_irradiance      NUMERIC(8, 2),
    -- Calendar features (raw – transformed in Python)
    day_of_week         SMALLINT,                                 -- 0=Mon … 6=Sun
    hour_of_day         SMALLINT,
    month               SMALLINT,
    is_holiday          BOOLEAN NOT NULL DEFAULT FALSE,
    -- Lagged price features (autoregressive)
    price_lag_24h       NUMERIC(10, 2),
    price_lag_48h       NUMERIC(10, 2),
    price_lag_168h      NUMERIC(10, 2),
    -- Cyclic encodings
    hour_sin            NUMERIC(8, 6),
    hour_cos            NUMERIC(8, 6),
    dow_sin             NUMERIC(8, 6),
    dow_cos             NUMERIC(8, 6),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- 5. Model predictions  (saved at ~11:00 UTC every day for D+1)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id        UUID           NOT NULL DEFAULT gen_random_uuid(),
    created_at           TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    target_timestamp_utc TIMESTAMPTZ    NOT NULL,
    predicted_price_eur  NUMERIC(10, 2) NOT NULL,
    model_version        VARCHAR(50)    NOT NULL,
    PRIMARY KEY (target_timestamp_utc, model_version)
);

-- ---------------------------------------------------------------------------
-- 6. Evaluation results  (daily backtest metrics)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evaluation_results (
    eval_id          SERIAL         PRIMARY KEY,
    evaluated_at     TIMESTAMPTZ    NOT NULL DEFAULT NOW(),
    period_start     TIMESTAMPTZ    NOT NULL,
    period_end       TIMESTAMPTZ    NOT NULL,
    model_version    VARCHAR(50)    NOT NULL,
    mae_eur_mwh      NUMERIC(8, 4),
    rmse_eur_mwh     NUMERIC(8, 4),
    mape_pct         NUMERIC(8, 4),
    directional_acc  NUMERIC(5, 4),      -- 0.0 – 1.0
    n_samples        INTEGER
);

-- ---------------------------------------------------------------------------
-- 7. TimescaleDB hypertables  (skip if extension unavailable)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    PERFORM create_hypertable('raw_energy_prices',         'timestamp_utc', if_not_exists => TRUE);
    PERFORM create_hypertable('raw_weather_data',          'timestamp_utc', if_not_exists => TRUE);
    PERFORM create_hypertable('raw_generation_data',       'timestamp_utc', if_not_exists => TRUE);
    PERFORM create_hypertable('fact_hourly_market_features','timestamp_utc', if_not_exists => TRUE);
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'create_hypertable skipped – TimescaleDB not available.';
END;
$$;
