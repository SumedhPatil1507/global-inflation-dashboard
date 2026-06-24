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
-- Add missing audit columns if not exist
ALTER TABLE IF EXISTS security_audit_ledger
    ADD COLUMN IF NOT EXISTS user_id TEXT,
    ADD COLUMN IF NOT EXISTS role TEXT,
    ADD COLUMN IF NOT EXISTS ip_address TEXT,
    ADD COLUMN IF NOT EXISTS request_signature TEXT,
    ADD COLUMN IF NOT EXISTS query_time_ms DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS rows_affected INT;
