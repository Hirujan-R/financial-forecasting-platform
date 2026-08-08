import os

from dotenv import load_dotenv

load_dotenv()


def get_api_base_url() -> str:
    """Base URL of the FastAPI prediction service."""
    return os.getenv("DASHBOARD_API_URL", "http://127.0.0.1:8000").rstrip("/")


def get_api_timeout() -> int:
    return int(os.getenv("DASHBOARD_API_TIMEOUT", "60"))


def get_history_limit() -> int:
    return int(os.getenv("DASHBOARD_HISTORY_LIMIT", "50"))
