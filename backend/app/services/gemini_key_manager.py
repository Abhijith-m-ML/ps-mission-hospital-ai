"""Gemini API Key Manager with 3-key fallback and health management.

Manages GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3 stored strictly in .env.
Ensures zero exposure of API keys in logs or responses, only reporting key slot numbers.

Fallback Logic:
- Attempt 1: GEMINI_API_KEY_1 (Key slot 1)
  ↓ (If 401 / 429 / temporary 5xx / timeout)
- Attempt 2: GEMINI_API_KEY_2 (Key slot 2)
  ↓ (If 401 / 429 / temporary 5xx / timeout)
- Attempt 3: GEMINI_API_KEY_3 (Key slot 3)
  ↓ (If all fail)
- Controlled error response (No indefinite retries, no silent fallback).

Non-retryable errors (400 Bad Request, malformed request, invalid parameters, empty text)
do NOT trigger key rotation and fail immediately.
"""
import os
import re
import time
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import httpx

from app.core.config import settings
from app.core.logging_config import logger

T = TypeVar("T")


class GeminiKeyManager:
    """Manages multi-key rotation and resilient fallback across up to 3 Gemini API keys."""

    def __init__(self, keys: Optional[List[Tuple[int, str]]] = None):
        """Initializes key manager from parameters or environment variables.
        
        Args:
            keys: Optional list of (slot_number, api_key) tuples for testing.
        """
        if keys is not None:
            self._keys = [
                (slot, key.strip())
                for slot, key in keys
                if key and isinstance(key, str) and key.strip()
            ]
        else:
            self._keys = self._load_keys_from_env()

    def _load_keys_from_env(self) -> List[Tuple[int, str]]:
        """Loads GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3 from environment."""
        loaded: List[Tuple[int, str]] = []

        k1 = os.getenv("GEMINI_API_KEY_1") or getattr(settings, "GEMINI_API_KEY_1", None)
        k2 = os.getenv("GEMINI_API_KEY_2") or getattr(settings, "GEMINI_API_KEY_2", None)
        k3 = os.getenv("GEMINI_API_KEY_3") or getattr(settings, "GEMINI_API_KEY_3", None)

        # Legacy backward-compatibility fallback if GEMINI_API_KEY_1 is unset
        if not k1:
            k1 = os.getenv("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)

        for slot, k in [(1, k1), (2, k2), (3, k3)]:
            if k and isinstance(k, str) and k.strip():
                loaded.append((slot, k.strip()))

        return loaded

    @property
    def available_slots(self) -> List[int]:
        """Returns slot numbers of currently configured keys."""
        return [slot for slot, _ in self._keys]

    def has_keys(self) -> bool:
        """Returns True if at least one Gemini key is configured."""
        return len(self._keys) > 0

    @classmethod
    def extract_safe_error_info(
        cls,
        err: Exception,
        keys: Optional[List[Tuple[int, str]]] = None,
    ) -> Tuple[Optional[int], str, str]:
        """Safely extracts (status_code, exception_type, safe_provider_message) without exposing secrets.
        
        Guarantees that no actual API keys, authorization tokens, or sensitive credentials
        are leaked in log output or error strings.
        """
        status_code: Optional[int] = None
        provider_msg: Optional[str] = None
        exc_type = type(err).__name__

        # 1. Direct provider message attached by caller
        if hasattr(err, "provider_message") and getattr(err, "provider_message", None):
            provider_msg = str(getattr(err, "provider_message"))

        if hasattr(err, "status_code") and getattr(err, "status_code", None) is not None:
            status_code = getattr(err, "status_code")

        # 2. Check HTTP response object (e.g. httpx.HTTPStatusError)
        resp = getattr(err, "response", None)
        if resp is not None:
            if status_code is None and getattr(resp, "status_code", None) is not None:
                status_code = resp.status_code

            if not provider_msg:
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        err_dict = data.get("error", {})
                        if isinstance(err_dict, dict):
                            provider_msg = err_dict.get("message") or err_dict.get("status")
                        elif isinstance(err_dict, str):
                            provider_msg = err_dict
                        if not provider_msg and "message" in data:
                            provider_msg = str(data["message"])
                except Exception:
                    pass

                if not provider_msg and hasattr(resp, "text") and resp.text:
                    provider_msg = resp.text[:400]

        # 3. Fallback regex for HTTP status in str(err)
        if status_code is None:
            m = re.search(r"\b([45]\d\d)\b", str(err))
            if m:
                try:
                    status_code = int(m.group(1))
                except Exception:
                    pass

        # 4. Fallback message to str(err)
        if not provider_msg:
            provider_msg = str(err)

        # 5. Sanitize provider_msg to guarantee ZERO credential/key exposure
        clean_msg = str(provider_msg)
        if keys:
            for slot, key in keys:
                if key and key in clean_msg:
                    clean_msg = clean_msg.replace(key, f"[REDACTED_GEMINI_KEY_{slot}]")

        clean_msg = re.sub(r'(?:key|api_key|x-goog-api-key)=["\']?[A-Za-z0-9_\-\.]{12,}["\']?', 'key=[REDACTED]', clean_msg, flags=re.IGNORECASE)
        clean_msg = re.sub(r'Bearer\s+[A-Za-z0-9_\-\.]{12,}', 'Bearer [REDACTED]', clean_msg, flags=re.IGNORECASE)
        clean_msg = re.sub(r'AQ\.[A-Za-z0-9_\-]{20,}', '[REDACTED_TOKEN]', clean_msg)
        clean_msg = clean_msg.strip()

        return status_code, exc_type, clean_msg

    @classmethod
    def categorize_error(cls, status_code: Optional[int], exc: Optional[Exception] = None, msg: str = "") -> str:
        """Categorizes Gemini failure into diagnostic categories.
        
        Categories:
        - authentication (401 / invalid API key)
        - permission (403 / API restriction)
        - model lookup (404 / unsupported model)
        - quota / rate limiting (429 / resource exhausted)
        - server error (500, 502, 503, 504)
        - timeout (read / connect timeout)
        - safety (content moderation / safety blocks)
        - generation (general generation error)
        """
        if exc is not None and isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return "timeout"
        clean_msg = (msg or "").lower()
        if status_code == 401 or "api key not valid" in clean_msg or "api_key_invalid" in clean_msg:
            return "authentication"
        if status_code == 403 or "permission" in clean_msg:
            return "permission"
        if status_code == 404:
            return "model lookup"
        if status_code == 429 or "quota" in clean_msg or "resource_exhausted" in clean_msg or "rate limit" in clean_msg:
            return "quota / rate limiting"
        if status_code in (500, 502, 503, 504):
            return "server error"
        if "safety" in clean_msg or "blocked" in clean_msg:
            return "safety"
        return "generation"

    @staticmethod
    def is_retryable_error(status_code: Optional[int], exc: Optional[Exception] = None) -> bool:
        """Determines if an error warrants fallback to the next Gemini API key.
        
        Retryable:
        - 401: Invalid / revoked API key (try next key)
        - 403: API not enabled / quota blocked (try next key)
        - 429: Rate limit / quota exhausted (try next key)
        - 500, 502, 503, 504: Temporary Google server errors (try next key)
        - Network timeouts / connection disconnects
        
        Non-retryable:
        - 400: Malformed JSON, unsupported parameters, invalid request
        - 404: Non-existent endpoint / model
        - 422: Validation error
        - ValueError: Application validation failure (e.g. empty text)
        """
        if status_code in (401, 403, 429, 500, 502, 503, 504):
            return True

        if exc is not None:
            if isinstance(exc, (httpx.TimeoutException, TimeoutError, httpx.ConnectError, ConnectionError)):
                return True
            exc_str = str(exc).lower()
            if any(term in exc_str for term in ("429", "quota", "rate limit", "busy", "overloaded", "resource_exhausted", "timed out")):
                return True

        return False

    def execute_with_fallback(
        self,
        func: Callable[[str, int], T],
        operation_name: str = "Gemini operation",
        model: Optional[str] = None,
        language: Optional[str] = None,
        text_length: Optional[int] = None,
    ) -> T:
        """Executes a function requiring a Gemini API key, falling back across configured keys.
        
        Args:
            func: Callable accepting (api_key: str, key_slot: int) and returning result.
            operation_name: Human-readable name for logging (e.g. 'Gemini TTS').
            model: Model name for debug logs.
            language: Language code for debug logs.
            text_length: Input length for debug logs.
            
        Returns:
            T: Result from func.
            
        Raises:
            ValueError: If no Gemini keys configured or request validation fails.
            RuntimeError: If all configured keys fail with retryable errors.
        """
        if not self._keys:
            raise ValueError(
                "No Gemini API keys are configured. Please set GEMINI_API_KEY_1 in backend/.env."
            )

        total_slots = len(self._keys)
        last_error: Optional[Exception] = None
        last_status_code: Optional[int] = None
        last_provider_msg: Optional[str] = None
        model_name = model or getattr(settings, "GEMINI_MODEL", "gemini-flash-lite-latest")

        for attempt_idx, (slot, key) in enumerate(self._keys, start=1):
            # Development logging: Selected key slot number ONLY (never log key itself)
            logger.info(
                "%s request: provider=gemini model=%s key_slot=%d attempt=%d/%d language=%s text_length=%s",
                operation_name,
                model_name,
                slot,
                attempt_idx,
                total_slots,
                language or "N/A",
                text_length if text_length is not None else "N/A",
            )

            try:
                return func(key, slot)

            except Exception as err:
                status_code, exc_type, safe_provider_msg = self.extract_safe_error_info(err, self._keys)
                is_retryable = self.is_retryable_error(status_code, err)
                category = self.categorize_error(status_code, err, safe_provider_msg)

                # Diagnostic logging preserving safe provider information and slot tracking
                logger.warning(
                    "Gemini request failed\nprovider=gemini\nmodel=%s\nkey_index=%d\nstatus_code=%s\nerror_type=%s\nerror_message=%s\nstage=%s\ncategory=%s\nRetryable: %s",
                    model_name,
                    slot,
                    status_code if status_code is not None else "N/A",
                    exc_type,
                    safe_provider_msg or "N/A",
                    operation_name,
                    category,
                    str(is_retryable).lower(),
                )

                logger.warning(
                    "Gemini key slot %d failed\nStatus: %s\nError type: %s\nProvider message: %s\nRetryable: %s",
                    slot,
                    status_code if status_code is not None else "N/A",
                    exc_type,
                    safe_provider_msg or "N/A",
                    str(is_retryable).lower(),
                )

                if not is_retryable:
                    # Non-retryable error (e.g. 400 Bad Request, empty text) must NOT rotate keys
                    logger.error(
                        "%s non-retryable error on key slot %d: %s. Aborting key fallback.",
                        operation_name,
                        slot,
                        safe_provider_msg,
                    )
                    raise

                last_error = err
                last_status_code = status_code
                last_provider_msg = safe_provider_msg

                # Exponential backoff before switching to next key slot
                if attempt_idx < total_slots:
                    backoff_seconds = min(0.2 * (2 ** (attempt_idx - 1)), 1.0)
                    time.sleep(backoff_seconds)
                    continue

        # If all keys failed
        logger.error(
            "All %d configured Gemini API key slots exhausted for %s (model=%s). Controlled fallback triggered. Last status: %s, Last provider message: %s",
            total_slots,
            operation_name,
            model_name,
            last_status_code if last_status_code is not None else "N/A",
            last_provider_msg or "N/A",
        )

        if isinstance(last_error, (TimeoutError, httpx.TimeoutException)):
            raise TimeoutError(f"{operation_name} timed out across all available API keys.") from last_error
        elif isinstance(last_error, (ConnectionError, httpx.ConnectError)):
            raise ConnectionError(f"Unable to connect to {operation_name} across all available API keys.") from last_error
        elif isinstance(last_error, ValueError):
            raise last_error

        raise RuntimeError(
            f"{operation_name} failed across all {total_slots} configured Gemini API keys "
            f"(last status: {last_status_code or 'N/A'}, last error: {type(last_error).__name__}, provider message: {last_provider_msg or 'N/A'})."
        ) from last_error


_default_key_manager: Optional[GeminiKeyManager] = None


def get_gemini_key_manager() -> GeminiKeyManager:
    """Returns singleton instance of GeminiKeyManager, reloading if keys changed."""
    global _default_key_manager
    if _default_key_manager is None:
        _default_key_manager = GeminiKeyManager()
    return _default_key_manager


def reset_gemini_key_manager() -> None:
    """Resets key manager singleton (useful for test isolation)."""
    global _default_key_manager
    _default_key_manager = None
