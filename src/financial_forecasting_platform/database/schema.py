from financial_forecasting_platform.database.connection import get_connection

def create_schemas():
    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_data (
        id SERIAL PRIMARY KEY,
        Ticker VARCHAR(10) NOT NULL,
        Date DATE NOT NULL,
        Open FLOAT,
        High FLOAT,
        Low FLOAT,
        Close FLOAT,
        Volume BIGINT,

        UNIQUE(ticker, date)
    );
    """)

    cursor.execute("""
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
    """)

    cursor.execute("""
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
                
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_logs(
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
    """)

    connection.commit()

    cursor.close()
    connection.close()

if __name__ == "__main__":
    create_schemas()