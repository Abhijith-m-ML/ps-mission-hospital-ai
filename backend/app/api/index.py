from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from app.core.config import settings
from app.core.logging_config import logger
from app.models.schemas import (
    IndexPreviewRequest,
    IndexPreviewResponse,
    IndexStatsResponse,
)
from app.retrieval.vector_store import PineconeVectorStore
from app.services.indexing_service import IndexingService

router = APIRouter(prefix="/index", tags=["indexing"])


@router.post(
    "/preview",
    response_model=IndexPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Crawl/fetch, clean, chunk, embed, and index a webpage into Pinecone",
)
def preview_index(request: IndexPreviewRequest) -> IndexPreviewResponse:
    """Takes a webpage URL (defaulting to https://www.psmissionhospital.org/),
    extracts structured documents, chunks them, generates 384-dimensional embeddings,
    and batch-upserts the vector records into Pinecone.
    """
    try:
        service = IndexingService()
        result = service.index_from_url(
            url=request.url,
            namespace=request.namespace,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            max_chunks=request.max_chunks,
        )
        return IndexPreviewResponse(
            index_name=result["index_name"],
            namespace=result["namespace"],
            url=result["url"],
            title=result.get("title"),
            indexed_count=result["indexed_count"],
            failed_count=result.get("failed_count", 0),
            dimension=result["dimension"],
            duration_seconds=result.get("duration_seconds"),
        )
    except ValueError as err:
        logger.warning("Validation error in indexing preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        logger.exception("Unexpected error executing indexing preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pinecone indexing failed: {str(err)}",
        )


@router.get(
    "/stats",
    response_model=IndexStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Pinecone index and namespace vector statistics",
)
def get_stats(
    namespace: Optional[str] = Query(default=None, description="Optional namespace override"),
) -> IndexStatsResponse:
    """Returns vector counts and dimensions for the configured Pinecone index and namespace."""
    try:
        vector_store = PineconeVectorStore()
        stats = vector_store.get_index_stats(namespace=namespace)
        return IndexStatsResponse(
            index=stats["index"],
            namespace=stats["namespace"],
            dimension=stats["dimension"],
            total_vector_count=stats["total_vector_count"],
            namespace_vector_count=stats["namespace_vector_count"],
        )
    except ValueError as err:
        logger.warning("Configuration error in get_stats: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        logger.exception("Failed to retrieve Pinecone statistics: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve index stats: {str(err)}",
        )
