from fastapi import APIRouter, status
from app.models.schemas import ChunkPreviewRequest, ChunkPreviewResponse
from app.services.chunking_service import ChunkingService
from app.core.logging_config import logger

router = APIRouter(prefix="/chunks", tags=["Chunking"])


@router.post(
    "/preview",
    response_model=ChunkPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Preview document chunking for a target URL",
    description="Fetches, cleans, and chunks a webpage into structure-aware semantic blocks.",
)
def preview_chunks(payload: ChunkPreviewRequest) -> ChunkPreviewResponse:
    logger.info("Received chunks preview request for: %s", payload.url)
    return ChunkingService.preview_chunks(payload)
