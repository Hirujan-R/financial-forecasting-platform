-- Bootstrap script for the local docker-compose Postgres.
-- Runs once on first init of an empty data directory (docker-entrypoint-initdb.d).
-- Creates the MLflow backend-store database and the application schema.

CREATE DATABASE mlflow;

CREATE TABLE IF NOT EXISTS stock_data (
    id SERIAL PRIMARY KEY,
    Ticker VARCHAR(10) NOT NULL,
    Date DATE NOT NULL,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    Volume BIGINT,
    UNIQUE(Ticker, Date)
);

CREATE TABLE IF NOT EXISTS spy_data (
    id SERIAL PRIMARY KEY,
    Ticker VARCHAR(10) NOT NULL,
    Date DATE NOT NULL UNIQUE,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    Volume BIGINT
);

CREATE TABLE IF NOT EXISTS vix_data (
    id SERIAL PRIMARY KEY,
    Ticker VARCHAR(10) NOT NULL,
    Date DATE NOT NULL UNIQUE,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    Volume BIGINT
);

CREATE TABLE IF NOT EXISTS prediction_logs (
    prediction_id UUID PRIMARY KEY,
    timestamp TIMESTAMP,
    ticker VARCHAR(10),
    prediction SMALLINT,
    probability DOUBLE PRECISION,
    model_name VARCHAR(100),
    model_version INTEGER,
    feature_pipeline VARCHAR(100),
    actual_outcome SMALLINT,
    correct BOOLEAN
);
