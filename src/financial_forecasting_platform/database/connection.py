import os

import psycopg
from dotenv import load_dotenv


def get_connection():
    load_dotenv()

    return psycopg.connect(
        host = os.getenv("DATABASE_HOST"),
        dbname = os.getenv("DATABASE_NAME"),
        user = os.getenv("DATABASE_USER"),
        password = os.getenv("DATABASE_PASSWORD"),
        port = os.getenv("DATABASE_PORT"),
        sslmode = os.getenv("DATABASE_SSLMODE", "require")
    )

