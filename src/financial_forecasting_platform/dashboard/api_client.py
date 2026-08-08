import requests

from .config import get_api_base_url, get_api_timeout


class ApiError(RuntimeError):
    """Raised when the prediction API returns an error."""


class ApiClient:
    """Thin HTTP client for the financial forecasting FastAPI service."""

    def __init__(self, base_url: str | None = None, timeout: int | None = None):
        self.base_url = (base_url or get_api_base_url()).rstrip("/")
        self.timeout = timeout or get_api_timeout()

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ApiError(
                f"Could not reach the prediction API at {self.base_url}: {exc}"
            ) from exc

        if response.status_code >= 400:  # noqa: PLR2004
            raise ApiError(
                f"API request to {path} failed with status "
                f"{response.status_code}: {response.text}"
            )

        return response.json()

    def health_check(self) -> dict:
        return self._get("/")

    def get_prediction(self, ticker: str) -> dict:
        return self._get(f"/prediction/{ticker}")

    def get_history(self, limit: int | None = None) -> list:
        params = None if limit is None else {"limit": limit}
        return self._get("/history", params=params)
