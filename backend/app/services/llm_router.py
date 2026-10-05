"""LLM Router service routing inference requests based strictly on INPUT TYPE.

Routing Architecture:
- TEXT INPUT -> Groq (Default model: openai/gpt-oss-20b)
- VOICE INPUT -> Gemini (Existing STT -> transcript -> RAG -> Gemini)

Decision is strictly based on input modality, NOT language:
- Malayalam TEXT -> Groq
- English TEXT -> Groq
- Hindi TEXT -> Groq
- Malayalam VOICE -> Gemini
- English VOICE -> Gemini
- Hindi VOICE -> Gemini
"""
from typing import Any, Optional

from app.core.logging_config import logger


class LLMRouter:
    """Routes generation queries to either Groq or Gemini based on input_type."""

    def __init__(
        self,
        groq_service: Optional[Any] = None,
        gemini_provider: Optional[Any] = None,
    ):
        self._groq_service = groq_service
        self._gemini_provider = gemini_provider

    @property
    def groq_service(self) -> Any:
        """Lazily loads GroqService."""
        if self._groq_service is None:
            from app.services.groq_service import get_groq_service
            self._groq_service = get_groq_service()
        return self._groq_service

    @property
    def gemini_provider(self) -> Any:
        """Lazily loads GeminiLLMProvider."""
        if self._gemini_provider is None:
            from app.rag.llm import GeminiLLMProvider
            self._gemini_provider = GeminiLLMProvider()
        return self._gemini_provider

    def generate(
        self,
        input_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Routes prompt to appropriate LLM provider based on input modality.
        
        Args:
            input_type: "text" or "voice" (case-insensitive).
            system_prompt: Common hospital system instructions and grounding rules.
            user_prompt: Grounded context, conversation history, and user question.
            
        Returns:
            str: Generated completion text.
        """
        raw_type = (input_type or "text").strip().lower()
        query_len = len(user_prompt or "")

        if raw_type == "voice":
            norm_input_type = "voice"
            llm_provider = "gemini"
            model_name = getattr(self.gemini_provider, "model", "gemini-flash-latest")

            # Development-only debug logging
            logger.info("input_type=%s", norm_input_type)
            logger.info("llm_provider=%s", llm_provider)
            logger.info("model=%s", model_name)
            logger.info("query_length=%d", query_len)

            try:
                return self.gemini_provider.generate(
                    system_instruction=system_prompt,
                    user_message=user_prompt,
                )
            except Exception as err:
                logger.error("Gemini voice generation error (model=%s): %s", model_name, err)
                raise
        else:
            # Default to text -> Groq
            norm_input_type = "text"
            llm_provider = "groq"
            model_name = getattr(self.groq_service, "model", "openai/gpt-oss-20b")

            # Development-only debug logging
            logger.info("input_type=%s", norm_input_type)
            logger.info("llm_provider=%s", llm_provider)
            logger.info("model=%s", model_name)
            logger.info("query_length=%d", query_len)

            try:
                return self.groq_service.generate_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except Exception as err:
                logger.error("Groq text generation error (model=%s): %s", model_name, err)
                # CRITICAL: Do NOT silently switch text request to Gemini upon failure
                raise


_default_router: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    """Returns singleton LLMRouter instance."""
    global _default_router
    if _default_router is None:
        _default_router = LLMRouter()
    return _default_router


def generate(
    input_type: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Routes generation request to Groq (for text) or Gemini (for voice).
    
    Args:
        input_type: "text" or "voice".
        system_prompt: System prompt / instruction.
        user_prompt: User prompt / context.
        
    Returns:
        str: Generated completion text.
    """
    router = get_llm_router()
    return router.generate(
        input_type=input_type,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
