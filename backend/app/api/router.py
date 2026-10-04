from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.crawler import router as crawler_router
from app.api.chunks import router as chunks_router
from app.api.embeddings import router as embeddings_router
from app.api.search import router as search_router
from app.api.index import router as index_router
from app.api.retrieval import router as retrieval_router
from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.api.structured import router as structured_router
from app.api.voice import router as voice_router
from app.api.query import router as query_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(crawler_router)
api_router.include_router(chunks_router)
api_router.include_router(embeddings_router)
api_router.include_router(search_router)
api_router.include_router(index_router)
api_router.include_router(retrieval_router)
api_router.include_router(chat_router)
api_router.include_router(extract_router)
api_router.include_router(structured_router)
api_router.include_router(voice_router)
api_router.include_router(query_router)





