from fastapi import APIRouter, HTTPException, status
from app.core.logging_config import logger
from app.models.schemas import SearchPreviewRequest, SearchPreviewResponse
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "/preview",
    response_model=SearchPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Embed natural-language query and return top-K matching chunks from Pinecone",
)
def preview_search(request: SearchPreviewRequest) -> SearchPreviewResponse:
    """Takes a user query, computes its 384-dimensional embedding, executes nearest-neighbor
    vector similarity search in Pinecone, and returns ranked chunk results without any LLM.
    """
    try:
        service = SearchService()
        result = service.search_query(
            query=request.query,
            top_k=request.top_k,
            namespace=request.namespace,
            filter_dict=request.filter,
        )
        return SearchPreviewResponse(
            query=result["query"],
            index=result["index"],
            namespace=result["namespace"],
            top_k=result["top_k"],
            total_found=result["total_found"],
            search_duration_seconds=result.get("search_duration_seconds"),
            results=result["results"],
        )
    except ValueError as err:
        logger.warning("Validation error in search preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        logger.exception("Unexpected error executing search preview: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pinecone vector search failed: {str(err)}",
        )
