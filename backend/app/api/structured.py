from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.core.logging_config import logger
from app.crawler.fetcher import Fetcher
from app.ingestion.structured_extractor import StructuredExtractor
from app.models.schemas import (
    StructuredPreviewRequest,
    StructuredPreviewResponse,
)

router = APIRouter(prefix="/structured", tags=["Structured Hospital Entities"])


@router.post(
    "/preview",
    response_model=StructuredPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect extracted structured department, doctor, and facility records before re-indexing",
)
def preview_structured_post(request: StructuredPreviewRequest) -> StructuredPreviewResponse:
    """Accepts a webpage URL (or optional raw HTML) and extracts deterministic
    Department, Doctor, and Facility models without modifying Pinecone.
    """
    return _process_structured_preview(request.url, request.html)


@router.get(
    "/preview",
    response_model=StructuredPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect extracted structured records for a URL via GET query parameter",
)
def preview_structured_get(
    url: str = Query(
        default="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
        description="Target hospital webpage URL to extract structured entities from",
    ),
) -> StructuredPreviewResponse:
    """GET convenience endpoint to inspect structured hospital entities for a given URL."""
    return _process_structured_preview(url, html=None)


def _process_structured_preview(url: str, html: Optional[str] = None) -> StructuredPreviewResponse:
    target_url = (url or "").strip()
    if not target_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A valid URL string is required.",
        )

    try:
        raw_html = html
        if not raw_html:
            fetcher = Fetcher()
            fetch_res = fetcher.fetch(target_url)
            if not fetch_res.is_success:
                raise ValueError(f"Failed to fetch webpage at {target_url}: {fetch_res.error}")
            raw_html = fetch_res.html or ""

        result = StructuredExtractor.extract(raw_html, target_url)

        return StructuredPreviewResponse(
            url=target_url,
            source_title=result.source_title,
            department=result.department,
            doctors=result.doctors,
            facilities=result.facilities,
            images=result.images,
            total_doctors=len(result.doctors),
            total_facilities=len(result.facilities),
            total_images=len(result.images),
        )

    except ValueError as val_err:
        logger.warning("Validation error in structured preview: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception("Failed to execute structured preview for %s: %s", target_url, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Structured extraction failed: {str(exc)}",
        )
