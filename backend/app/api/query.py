"""Development debugging endpoint for Query Understanding and Rewriting."""
from fastapi import APIRouter, HTTPException, status
from app.core.logging_config import logger
from app.models.schemas import QueryPreviewRequest, QueryPreviewResponse
from app.query.resolver import QueryResolver

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "/preview",
    response_model=QueryPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect Query Normalization, Intent Classification, and Query Rewriting",
)
def preview_query(request: QueryPreviewRequest) -> QueryPreviewResponse:
    """Takes a raw query and optional history, returning the understood entities,
    detected intent, and rewritten retrieval query without calling search or LLM generation.
    """
    try:
        resolver = QueryResolver()
        result = resolver.resolve(query=request.message, history=request.history)
        return QueryPreviewResponse(
            original_query=result.original_query,
            raw_query=result.original_query,
            resolved_query=result.resolved_query,
            normalized_query=result.resolved_query,
            intent=result.intent,
            entities=result.entities,
            confidence=round(result.confidence, 4),
        )
    except ValueError as err:
        logger.warning("Validation error in query preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        logger.exception("Unexpected error in query preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query resolution failed: {str(err)}",
        )
