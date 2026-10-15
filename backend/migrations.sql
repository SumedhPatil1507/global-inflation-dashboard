-- PostgreSQL migration for macroeconomic engine
-- Ensure TimescaleDB extension is available
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Table for macro indicators
CREATE TABLE IF NOT EXISTS macro_indicators (
    timestamp TIMESTAMPTZ NOT NULL,
    country_iso VARCHAR(3) NOT NULL,
    indicator_code VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION,
    PRIMARY KEY (timestamp, country_iso, indicator_code)
);

-- Index on timestamp for fast time‑range queries
CREATE INDEX IF NOT EXISTS idx_macro_indicators_timestamp ON macro_indicators USING btree (timestamp);
-- Index on country_iso + indicator_code for lookup
CREATE INDEX IF NOT EXISTS idx_macro_indicators_country_indicator ON macro_indicators USING btree (country_iso, indicator_code);

-- Table for market assets
CREATE TABLE IF NOT EXISTS market_assets (
    timestamp TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    PRIMARY KEY (timestamp, ticker)
);

-- Index on timestamp
CREATE INDEX IF NOT EXISTS idx_market_assets_timestamp ON market_assets USING btree (timestamp);
-- Index on ticker for symbol queries
CREATE INDEX IF NOT EXISTS idx_market_assets_ticker ON market_assets USING btree (ticker);

-- Convert to TimescaleDB hypertables (partitioned by timestamp)
SELECT create_hypertable('macro_indicators', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('market_assets', 'timestamp', if_not_exists => TRUE);

-- IIoT samples. Kafka coordinates make at-least-once ingestion idempotent.
CREATE TABLE IF NOT EXISTS sensor_readings (
    timestamp TIMESTAMPTZ NOT NULL,
    asset_id TEXT NOT NULL,
    sensor_type TEXT NOT NULL CHECK (sensor_type IN ('vibration', 'thermal', 'acoustic')),
    value DOUBLE PRECISION NOT NULL,
    unit TEXT NOT NULL DEFAULT '',
    quality TEXT NOT NULL DEFAULT 'good',
    kafka_topic TEXT NOT NULL,
    kafka_partition INTEGER NOT NULL,
    kafka_offset BIGINT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (timestamp, kafka_topic, kafka_partition, kafka_offset, sensor_type)
);
SELECT create_hypertable('sensor_readings', 'timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_asset_type_time
    ON sensor_readings (asset_id, sensor_type, timestamp DESC);

-- Five-minute point-in-time aggregates exposed through the Feast source.
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_feature_windows
WITH (timescaledb.continuous) AS
SELECT time_bucket(INTERVAL '5 minutes', timestamp) AS event_timestamp,
       asset_id,
       sensor_type,
       sqrt(avg(value * value)) AS rms,
       CASE WHEN var_pop(value) > 0 THEN
           (avg(power(value, 4)) - 4 * avg(value) * avg(power(value, 3))
            + 6 * power(avg(value), 2) * avg(power(value, 2))
            - 3 * power(avg(value), 4)) / power(var_pop(value), 2)
       END AS kurtosis,
       max(abs(value)) / NULLIF(sqrt(avg(value * value)), 0) AS crest_factor,
       max(ingested_at) AS created_timestamp
FROM sensor_readings
GROUP BY event_timestamp, asset_id, sensor_type;
DO $$
BEGIN
    PERFORM add_continuous_aggregate_policy(
        'sensor_feature_windows',
        start_offset => INTERVAL '30 days',
        end_offset => INTERVAL '5 minutes',
        schedule_interval => INTERVAL '5 minutes'
    );
EXCEPTION WHEN duplicate_object THEN
    NULL;
END $$;
-- Add missing audit columns if not exist
ALTER TABLE IF EXISTS security_audit_ledger
    ADD COLUMN IF NOT EXISTS user_id TEXT,
    ADD COLUMN IF NOT EXISTS role TEXT,
    ADD COLUMN IF NOT EXISTS ip_address TEXT,
    ADD COLUMN IF NOT EXISTS request_signature TEXT,
    ADD COLUMN IF NOT EXISTS query_time_ms DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS rows_affected INT;
