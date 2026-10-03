"""RAG and Hosted LLM Integration Package."""
from app.rag.llm import BaseLLMProvider, OpenAILLMProvider, MockLLMProvider, get_llm_provider
from app.rag.prompts import HOSPITAL_SYSTEM_INSTRUCTION, build_grounded_user_prompt
from app.rag.rag_service import RAGService, NO_CONTEXT_DEFAULT_REPLY

__all__ = [
    "BaseLLMProvider",
    "OpenAILLMProvider",
    "MockLLMProvider",
    "get_llm_provider",
    "HOSPITAL_SYSTEM_INSTRUCTION",
    "build_grounded_user_prompt",
    "RAGService",
    "NO_CONTEXT_DEFAULT_REPLY",
]
