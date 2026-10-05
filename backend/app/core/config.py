import json
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment or .env file."""

    PROJECT_NAME: str = "Hospital AI Assistant"
    ENVIRONMENT: str = "development"
    API_V1_PREFIX: str = "/api"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_PREVIEW_DIMS: int = 5
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "hospital-ai"
    PINECONE_NAMESPACE: str = "ps_mission_hospital"
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_UPSERT_BATCH_SIZE: int = 100
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_MIN_SCORE: float = 0.0
    RETRIEVAL_MAX_CONTEXT_CHUNKS: int = 5
    RETRIEVAL_MAX_CONTEXT_CHARACTERS: int = 12000
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_API_KEY_1: Optional[str] = None
    GEMINI_API_KEY_2: Optional[str] = None
    GEMINI_API_KEY_3: Optional[str] = None
    GEMINI_MODEL: str = "gemini-flash-lite-latest"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    LLM_MODEL: str = "gemini-flash-lite-latest"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_OUTPUT_TOKENS: int = 800
    MAX_HISTORY_MESSAGES: int = 10
    DEBUG_QUERY_PIPELINE: bool = False
    BACKEND_CORS_ORIGINS: Union[List[str], str] = [

        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, (list, tuple)):
            return list(v)
        return []


settings = Settings()
