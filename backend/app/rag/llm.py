from abc import ABC, abstractmethod
import time
from typing import Any, Optional

from app.core.config import settings
from app.core.logging_config import logger


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
    """Google Gemini hosted LLM provider using the Generative Language REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        raw_key = api_key or settings.GEMINI_API_KEY or settings.OPENAI_API_KEY
        self.api_key = raw_key.strip() if raw_key else None
        
        # Default model for Gemini
        default_model = "gemini-flash-lite-latest"
        cfg_model = settings.LLM_MODEL or ""
        self.model = model or (cfg_model if "gemini" in cfg_model.lower() else default_model)
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_output_tokens or settings.LLM_MAX_OUTPUT_TOKENS

    def generate(self, system_instruction: str, user_message: str) -> str:
        """Sends the grounded prompt to Google's hosted Gemini API with automatic fallback."""
        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Please set GEMINI_API_KEY in backend/.env "
                "or configure it in your deployment environment."
            )

        import httpx

        candidate_models = [self.model]
        if "gemini-flash-lite-latest" not in candidate_models:
            candidate_models.append("gemini-flash-lite-latest")

        last_error = None
        for current_model in candidate_models:
            start_time = time.time()
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{current_model}:generateContent?key={self.api_key}"
            )
            payload = {
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

            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, json=payload)
                    duration = time.time() - start_time

                    if resp.status_code == 400:
                        data = resp.json()
                        msg = data.get("error", {}).get("message", "Invalid request to Gemini API")
                        logger.error("Gemini API 400 Bad Request: %s", msg)
                        raise ValueError(f"Gemini API error: {msg}")
                    elif resp.status_code in (401, 403):
                        logger.error("Gemini authentication failure (status %d)", resp.status_code)
                        raise ValueError(
                            "Invalid Gemini API key. Please verify your GEMINI_API_KEY configuration."
                        )
                    elif resp.status_code in (429, 503):
                        logger.warning("Gemini model '%s' busy or overloaded (HTTP %d). Trying fallback...", current_model, resp.status_code)
                        last_error = RuntimeError(f"Gemini model {current_model} is temporarily unavailable (HTTP {resp.status_code}).")
                        continue
                    elif resp.status_code != 200:
                        logger.error("Gemini API error status %d: %s", resp.status_code, resp.text)
                        raise RuntimeError(
                            f"The AI service encountered an error (HTTP {resp.status_code})."
                        )

                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise ValueError("Gemini returned an empty candidates list.")

                    parts = candidates[0].get("content", {}).get("parts", [])
                    if not parts:
                        raise ValueError("Gemini candidate contains no content parts.")

                    content = parts[0].get("text", "") or ""
                    logger.info("Gemini API succeeded in %.3fs (model=%s).", duration, current_model)
                    return content.strip()

            except httpx.TimeoutException as err:
                logger.warning("Gemini model '%s' timed out. Trying fallback...", current_model)
                last_error = TimeoutError("The request to the AI service timed out.")
                continue
            except httpx.ConnectError as err:
                raise ConnectionError("Unable to connect to the hosted AI service. Please verify network access.") from err

        if last_error:
            raise last_error
        raise RuntimeError("Failed to generate response from Gemini.")


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


def get_llm_provider() -> BaseLLMProvider:
    """Factory creating the configured LLM provider according to settings.LLM_PROVIDER
    or auto-detected from key prefix.
    """
    provider_name = (settings.LLM_PROVIDER or "openai").lower()
    
    # Auto-detect if user provided a Google Gemini key (starts with AQ. or AIza)
    active_key = settings.GEMINI_API_KEY or settings.OPENAI_API_KEY or ""
    if provider_name == "gemini" or active_key.startswith(("AQ.", "AIza")):
        return GeminiLLMProvider()
    elif provider_name == "openai":
        return OpenAILLMProvider()
    elif provider_name == "mock":
        return MockLLMProvider()
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider_name}'. Supported providers: 'openai', 'gemini', 'mock'."
        )

