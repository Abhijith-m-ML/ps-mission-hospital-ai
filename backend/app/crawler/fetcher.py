from dataclasses import dataclass
from typing import Optional
import httpx

from app.core.logging_config import logger


@dataclass
class FetchResponse:
    """Encapsulates the HTTP fetch outcome and metadata."""
    url: str
    status_code: int
    html: Optional[str] = None
    content_type: str = ""
    error: Optional[str] = None
    final_url: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Returns True if the request succeeded with 2xx status and contains HTML."""
        return (
            self.error is None
            and 200 <= self.status_code < 300
            and self.html is not None
        )


class Fetcher:
    """HTTP client component responsible for safely retrieving web pages."""

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (compatible; HospitalAI-Bot/1.0; "
        "+https://hospital-ai.local/bot; healthcare information indexer)"
    )

    def __init__(
        self,
        timeout: float = 10.0,
        user_agent: Optional[str] = None,
        verify_ssl: bool = True,
        client: Optional[httpx.Client] = None,
    ):
        self.timeout = timeout
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self.verify_ssl = verify_ssl
        self._custom_client = client

    def fetch(self, url: str) -> FetchResponse:
        """Fetches HTML content from a target URL with redirect and error handling."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        # If a mock/custom client was injected (e.g. in tests), reuse it
        if self._custom_client is not None:
            return self._execute_request(self._custom_client, url, headers)

        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            verify=self.verify_ssl,
        ) as client:
            return self._execute_request(client, url, headers)

    def _execute_request(self, client: httpx.Client, url: str, headers: dict) -> FetchResponse:
        try:
            logger.info("Fetching: %s", url)
            response = client.get(url, headers=headers)
            
            content_type = response.headers.get("content-type", "").lower()
            final_url = str(response.url)

            # Check for non-HTML MIME types
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                logger.warning("Skipping non-HTML content-type [%s] at %s", content_type, url)
                return FetchResponse(
                    url=url,
                    status_code=response.status_code,
                    html=None,
                    content_type=content_type,
                    error=f"Unsupported content type: {content_type}",
                    final_url=final_url,
                )

            # Check HTTP status
            if response.status_code >= 400:
                logger.warning("HTTP %d error for %s", response.status_code, url)
                return FetchResponse(
                    url=url,
                    status_code=response.status_code,
                    html=None,
                    content_type=content_type,
                    error=f"HTTP {response.status_code} error",
                    final_url=final_url,
                )

            return FetchResponse(
                url=url,
                status_code=response.status_code,
                html=response.text,
                content_type=content_type,
                error=None,
                final_url=final_url,
            )

        except httpx.TimeoutException as exc:
            logger.warning("Request timed out for %s: %s", url, exc)
            return FetchResponse(
                url=url,
                status_code=408,
                error=f"Timeout: {exc}",
            )
        except httpx.ConnectError as exc:
            logger.warning("Connection failure for %s: %s", url, exc)
            return FetchResponse(
                url=url,
                status_code=503,
                error=f"Connection error: {exc}",
            )
        except httpx.RequestError as exc:
            logger.warning("Network request error for %s: %s", url, exc)
            return FetchResponse(
                url=url,
                status_code=500,
                error=f"Request error: {exc}",
            )
        except Exception as exc:
            logger.exception("Unexpected error fetching %s: %s", url, exc)
            return FetchResponse(
                url=url,
                status_code=500,
                error=f"Unexpected error: {exc}",
            )
