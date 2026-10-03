from fastapi import APIRouter, HTTPException, status
from app.core.logging_config import logger
from app.models.schemas import (
    EmbeddingTextPreviewRequest,
    EmbeddingTextPreviewResponse,
    EmbeddingChunkPreviewRequest,
    ChunkEmbeddingPreviewResponse,
)
from app.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "/preview",
    response_model=EmbeddingTextPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Preview raw text embedding vector and dimensionality",
)
def preview_text_embedding(
    request: EmbeddingTextPreviewRequest,
) -> EmbeddingTextPreviewResponse:
    """Computes embeddings for provided text(s) and returns dimensionality and preview vectors."""
    try:
        return EmbeddingService.preview_text_embeddings(request)
    except ValueError as err:
        logger.warning("Validation error in embedding preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        logger.exception("Unexpected error generating text embedding preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(err)}",
        )


@router.post(
    "/chunks/preview",
    response_model=ChunkEmbeddingPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Chunk and embed a target webpage or embed provided chunks",
)
def preview_chunk_embeddings(
    request: EmbeddingChunkPreviewRequest,
) -> ChunkEmbeddingPreviewResponse:
    """Takes a webpage URL (running the crawl->clean->chunk pipeline) or direct chunks,
    generates embeddings for each chunk, and returns sample results with preserved metadata.
    """
    try:
        return EmbeddingService.preview_chunk_embeddings(request)
    except ValueError as err:
        logger.warning("Validation error in chunk embedding preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        logger.exception("Unexpected error generating chunk embedding preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunk embedding preview failed: {str(err)}",
        )
