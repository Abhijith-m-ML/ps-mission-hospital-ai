"""Groq LLM Service for fast text inference using official Groq Python SDK.
Reads GROQ_API_KEY and GROQ_MODEL from environment variables.
"""
import os
import time
from typing import Any, Optional

from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from app.core.config import settings
from app.core.logging_config import logger


class GroqService:
    """Service wrapping official Groq Python SDK for chat completions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        client: Optional[Any] = None,
    ):
        if api_key is not None:
            raw_key = api_key
        else:
            raw_key = os.getenv("GROQ_API_KEY") or getattr(settings, "GROQ_API_KEY", None)
        self.api_key = raw_key.strip() if raw_key else None
        self.model = (
            model
            or os.getenv("GROQ_MODEL")
            or getattr(settings, "GROQ_MODEL", None)
            or "openai/gpt-oss-20b"
        )
        self.temperature = (
            temperature
            if temperature is not None
            else getattr(settings, "LLM_TEMPERATURE", 0.1)
        )
        self.max_tokens = max(
            max_tokens
            or getattr(settings, "LLM_MAX_OUTPUT_TOKENS", 1500),
            1500,
        )
        self._client = client

    @property
    def client(self) -> Groq:
        """Lazily instantiates the official Groq client."""
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ValueError(
                "Groq API key is missing. Please set GROQ_API_KEY in backend/.env "
                "or configure it in your deployment environment."
            )

        try:
            self._client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized successfully (model=%s).", self.model)
            return self._client
        except Exception as err:
            logger.error("Failed to initialize Groq client: %s", err)
            raise

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generates a text completion using the Groq Python SDK.
        
        Args:
            system_prompt: System instructions and grounding rules.
            user_prompt: Grounded context, history, and user question.
            
        Returns:
            str: Generated text response from the model.
        """
        cli = self.client
        start_time = time.time()
        logger.info("Sending request to Groq API (model=%s)...", self.model)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        extra_kwargs = {}
        if any(rm in self.model.lower() for rm in ("gpt-oss", "deepseek", "reasoning", "r1")):
            extra_kwargs["extra_body"] = {"reasoning_format": "hidden"}

        try:
            try:
                completion = cli.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **extra_kwargs,
                )
            except Exception as call_err:
                if extra_kwargs:
                    logger.debug("Retrying Groq call without extra_kwargs: %s", call_err)
                    completion = cli.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                else:
                    raise

            duration = time.time() - start_time
            logger.info("Groq completion succeeded in %.3fs (model=%s).", duration, self.model)

            choices = getattr(completion, "choices", []) or []
            if not choices:
                raise ValueError("Groq returned an empty choices list.")

            first_choice = choices[0]
            message_obj = getattr(first_choice, "message", None)
            if not message_obj:
                raise ValueError("Groq choice missing message object.")

            content = getattr(message_obj, "content", "") or ""
            if not content.strip():
                reasoning = getattr(message_obj, "reasoning", "") or ""
                if reasoning.strip():
                    logger.warning("Groq content was empty; falling back to reasoning buffer")
                    content = reasoning
            return content.strip()

        except AuthenticationError as err:
            logger.error("Groq authentication failure: invalid or unauthorized API key.")
            raise ValueError(
                "Invalid Groq API key. Please check your GROQ_API_KEY configuration."
            ) from err

        except RateLimitError as err:
            logger.warning("Groq rate limit encountered: %s", err)
            raise RuntimeError(
                "The AI service is currently experiencing high demand. Please try again shortly."
            ) from err

        except APITimeoutError as err:
            logger.warning("Groq request timed out: %s", err)
            raise TimeoutError(
                "The request to the AI service timed out. Please try again."
            ) from err

        except APIConnectionError as err:
            logger.error("Groq connection error: %s", err)
            raise ConnectionError(
                "Unable to connect to the hosted AI service. Please verify network access."
            ) from err

        except APIError as err:
            logger.error("Groq API returned an error: %s", err)
            raise RuntimeError(
                "The AI service encountered an error while processing the request."
            ) from err

        except Exception as err:
            logger.exception("Unexpected error communicating with Groq: %s", err)
            raise


_default_groq_service: Optional[GroqService] = None


def get_groq_service() -> GroqService:
    """Returns singleton instance of GroqService."""
    global _default_groq_service
    if _default_groq_service is None:
        _default_groq_service = GroqService()
    return _default_groq_service


def generate_response(system_prompt: str, user_prompt: str) -> str:
    """Module-level function to generate a completion using Groq.
    
    Args:
        system_prompt: System instruction and grounding rules.
        user_prompt: User prompt containing verified context and question.
        
    Returns:
        str: Only the generated text.
    """
    service = get_groq_service()
    return service.generate_response(system_prompt=system_prompt, user_prompt=user_prompt)
