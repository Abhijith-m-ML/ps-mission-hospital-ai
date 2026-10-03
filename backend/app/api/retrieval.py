from fastapi import APIRouter, HTTPException, status
from app.core.logging_config import logger
from app.models.schemas import RetrievalPreviewRequest, RetrievalPreviewResponse
from app.retrieval.retriever import Retriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post(
    "/preview",
    response_model=RetrievalPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute formal retrieval pipeline: embed query, search Pinecone, filter, deduplicate, and assemble context",
)
def preview_retrieval(request: RetrievalPreviewRequest) -> RetrievalPreviewResponse:
    """Takes a natural-language hospital query, generates its 384-dimensional embedding,
    searches Pinecone for nearest-neighbor candidates, applies score threshold filtering,
    removes duplicate chunks, orders results by descending relevance, and formats structured
    context for future LLM ingestion.
    
    NOTE: Does NOT generate an LLM response or answer.
    """
    try:
        retriever = Retriever()
        response = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score,
            namespace=request.namespace,
            filter_dict=request.filter,
            max_context_chunks=request.max_context_chunks,
            max_context_characters=request.max_context_characters,
        )
        return response
    except ValueError as err:
        logger.warning("Validation error in retrieval preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        logger.exception("Unexpected error in retrieval preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval pipeline failed: {str(err)}",
        )
