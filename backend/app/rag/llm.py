from abc import ABC, abstractmethod
import time
from typing import Any, Optional

from app.core.config import settings
from app.core.logging_config import logger
from app.services.gemini_key_manager import GeminiKeyManager, get_gemini_key_manager


class BaseLLMProvider(ABC):
    """Abstract interface defining provider-independent LLM generation contract."""

    @abstractmethod
    def generate(self, system_instruction: str, user_message: str) -> str:
        """Generates a text completion given a system instruction and user message.
        
        Args:
            system_instruction: Factual role, grounding rules, and constraints.
            user_message: Grounded prompt containing retrieved context and patient question.
            
        Returns:
            str: Generated completion text.
        """
        pass


class OpenAILLMProvider(BaseLLMProvider):
    """Hosted cloud LLM implementation using official OpenAI SDK.
    
    Uses the modern OpenAI Responses API with graceful fallback to Chat Completions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_output_tokens or settings.LLM_MAX_OUTPUT_TOKENS
        self._client = client

    @property
    def client(self) -> Any:
        """Lazily initializes the official OpenAI client."""
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is missing. Please set OPENAI_API_KEY in backend/.env "
                "or configure it in your deployment environment."
            )

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully (model=%s).", self.model)
            return self._client
        except Exception as err:
            logger.error("Failed to initialize OpenAI client: %s", err)
            raise

    def generate(self, system_instruction: str, user_message: str) -> str:
        """Sends the grounded prompt to the hosted OpenAI API."""
        cli = self.client
        start_time = time.time()
        logger.info("Sending request to hosted OpenAI API (model=%s)...", self.model)

        # 1. Try modern OpenAI Responses API first
        if hasattr(cli, "responses") and callable(getattr(cli.responses, "create", None)):
            try:
                response = cli.responses.create(
                    model=self.model,
                    instructions=system_instruction,
                    input=user_message,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                duration = time.time() - start_time
                output_text = getattr(response, "output_text", None)
                if output_text is not None and isinstance(output_text, str) and output_text.strip():
                    logger.info("OpenAI Responses API succeeded in %.3fs.", duration)
                    return output_text.strip()
            except Exception as resp_err:
                logger.debug(
                    "OpenAI Responses API was not available or unsupported by model '%s': %s. "
                    "Falling back to Chat Completions API.",
                    self.model,
                    resp_err,
                )

        # 2. Robust fallback to Chat Completions API
        try:
            from openai import (
                APIConnectionError,
                APIError,
                APITimeoutError,
                AuthenticationError,
                RateLimitError,
            )
        except ImportError:
            APIConnectionError = APIError = APITimeoutError = AuthenticationError = RateLimitError = Exception

        try:
            response = cli.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            duration = time.time() - start_time
            logger.info("OpenAI Chat Completions API succeeded in %.3fs.", duration)

            choices = getattr(response, "choices", []) or []
            if not choices:
                raise ValueError("OpenAI returned an empty choices list.")

            first_choice = choices[0]
            message_obj = getattr(first_choice, "message", None)
            if not message_obj:
                raise ValueError("OpenAI choice missing message object.")

            content = getattr(message_obj, "content", "") or ""
            return content.strip()

        except AuthenticationError as err:
            logger.error("OpenAI authentication error: %s", err)
            raise ValueError(
                "Invalid OpenAI API key. Please check your OPENAI_API_KEY configuration."
            ) from err
        except RateLimitError as err:
            logger.warning("OpenAI rate limit error: %s", err)
            raise RuntimeError(
                "The AI service is currently experiencing high demand. Please try again shortly."
            ) from err
        except APITimeoutError as err:
            logger.warning("OpenAI request timed out: %s", err)
            raise TimeoutError(
                "The request to the AI service timed out. Please try again."
            ) from err
        except APIConnectionError as err:
            logger.error("OpenAI connection error: %s", err)
            raise ConnectionError(
                "Unable to connect to the hosted AI service. Please verify network access."
            ) from err
        except APIError as err:
            logger.error("OpenAI API returned an error: %s", err)
            raise RuntimeError(
                "The AI service encountered an internal error while processing the request."
            ) from err
        except Exception as err:
            logger.exception("Unexpected error communicating with OpenAI: %s", err)
            raise



class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini hosted LLM provider using the Generative Language REST API with multi-key fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        key_manager: Optional[Any] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        from app.services.gemini_key_manager import GeminiKeyManager, get_gemini_key_manager
        if api_key:
            self.key_manager = GeminiKeyManager(keys=[(1, api_key)])
        else:
            self.key_manager = key_manager or get_gemini_key_manager()

        # Default model for Gemini
        default_model = "gemini-flash-latest"
        cfg_model = getattr(settings, "GEMINI_MODEL", None) or settings.LLM_MODEL or ""
        self.model = model or (cfg_model if "gemini" in cfg_model.lower() else default_model)
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_output_tokens or settings.LLM_MAX_OUTPUT_TOKENS

    def generate(self, system_instruction: str, user_message: str) -> str:
        """Sends the grounded prompt to Google's hosted Gemini API with automatic key fallback."""
        import httpx

        payload_base = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {"parts": [{"text": user_message}]}
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

        def _perform_generate(api_key: str, key_slot: int) -> str:
            logger.info("Gemini key slot: %d", key_slot)
            start_time = time.time()
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:generateContent"
            )
            headers = {"x-goog-api-key": api_key}

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload_base)
                duration = time.time() - start_time

                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise ValueError("Gemini returned an empty candidates list.")

                    parts = candidates[0].get("content", {}).get("parts", [])
                    if not parts:
                        raise ValueError("Gemini candidate contains no content parts.")

                    content = parts[0].get("text", "") or ""
                    logger.info(
                        "Gemini API succeeded in %.3fs on key slot %d (model=%s).",
                        duration,
                        key_slot,
                        self.model,
                    )
                    return content.strip()

                # Non-200 responses: Safely extract provider error message
                provider_msg = None
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

                safe_provider_msg = str(provider_msg or "Unknown error from Gemini API")
                if api_key and api_key in safe_provider_msg:
                    safe_provider_msg = safe_provider_msg.replace(api_key, f"[REDACTED_GEMINI_KEY_{key_slot}]")

                is_retryable = GeminiKeyManager.is_retryable_error(resp.status_code)

                if resp.status_code == 400:
                    err = ValueError(f"Gemini API 400 Bad Request: {safe_provider_msg}")
                    err.status_code = 400
                    err.provider_message = safe_provider_msg
                    raise err

                err = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}: {safe_provider_msg}",
                    request=resp.request,
                    response=resp,
                )
                err.status_code = resp.status_code
                err.provider_message = safe_provider_msg
                raise err

        return self.key_manager.execute_with_fallback(
            func=_perform_generate,
            operation_name="Gemini Generation",
            model=self.model,
            text_length=len(user_message),
        )


class MockLLMProvider(BaseLLMProvider):
    """Deterministic mock provider for unit tests without network or API costs."""

    def __init__(self, response_text: str = "This is a mocked hospital answer."):
        self.response_text = response_text
        self.last_system_instruction: Optional[str] = None
        self.last_user_message: Optional[str] = None
        self.call_count: int = 0

    def generate(self, system_instruction: str, user_message: str) -> str:
        self.last_system_instruction = system_instruction
        self.last_user_message = user_message
        self.call_count += 1
        return self.response_text


class GroqLLMProvider(BaseLLMProvider):
    """Groq hosted LLM provider wrapping GroqService."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        client: Optional[Any] = None,
    ):
        from app.services.groq_service import GroqService
        self.service = GroqService(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            client=client,
        )
        self.model = self.service.model

    def generate(self, system_instruction: str, user_message: str) -> str:
        return self.service.generate_response(
            system_prompt=system_instruction,
            user_prompt=user_message,
        )


def get_llm_provider() -> BaseLLMProvider:
    """Factory creating the configured LLM provider according to settings.LLM_PROVIDER
    or auto-detected from key prefix.
    """
    provider_name = (settings.LLM_PROVIDER or "openai").lower()

    if provider_name == "groq":
        return GroqLLMProvider()
    elif provider_name == "gemini":
        return GeminiLLMProvider()
    elif provider_name == "openai":
        return OpenAILLMProvider()
    elif provider_name == "mock":
        return MockLLMProvider()

    # Auto-detect if user provided a Google Gemini key (starts with AQ. or AIza)
    active_key = settings.GEMINI_API_KEY or settings.OPENAI_API_KEY or ""
    if active_key.startswith(("AQ.", "AIza")):
        return GeminiLLMProvider()
    elif settings.GROQ_API_KEY:
        return GroqLLMProvider()
    elif settings.OPENAI_API_KEY:
        return OpenAILLMProvider()
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider_name}'. Supported providers: 'groq', 'openai', 'gemini', 'mock'."
        )

