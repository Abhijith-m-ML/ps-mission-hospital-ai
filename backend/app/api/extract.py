from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.core.logging_config import logger
from app.crawler.fetcher import Fetcher
from app.ingestion.extractor import HospitalMetadataExtractor
from app.models.schemas import (
    DoctorItem,
    ExtractPreviewRequest,
    ExtractPreviewResponse,
    FacilityItem,
)

router = APIRouter(prefix="/extract", tags=["Metadata Extraction"])


@router.post(
    "/preview",
    response_model=ExtractPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect extracted structured doctor, facility, and image metadata before indexing",
)
def preview_extraction_post(request: ExtractPreviewRequest) -> ExtractPreviewResponse:
    """Takes a webpage URL (or optional raw HTML) and returns structured doctors,
    facilities, and image metadata extracted via deterministic DOM parsing.
    """
    return _process_preview(request.url, request.html)


@router.get(
    "/preview",
    response_model=ExtractPreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect extracted structured metadata for a URL via GET query parameter",
)
def preview_extraction_get(
    url: str = Query(
        default="https://www.psmissionhospital.org/doctors",
        description="Webpage URL to fetch and inspect",
    ),
) -> ExtractPreviewResponse:
    """GET convenience endpoint to inspect extracted structured hospital metadata."""
    return _process_preview(url, html=None)


def _process_preview(url: str, html: Optional[str] = None) -> ExtractPreviewResponse:
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

        extracted = HospitalMetadataExtractor.extract_all(raw_html, target_url)

        return ExtractPreviewResponse(
            url=target_url,
            title=extracted.get("title", ""),
            department=extracted.get("department", ""),
            total_doctors=len(extracted.get("doctors", [])),
            total_facilities=len(extracted.get("facilities", [])),
            total_images=len(extracted.get("images", [])),
            doctors=extracted.get("doctors", []),
            facilities=extracted.get("facilities", []),
            images=extracted.get("images", []),
        )

    except ValueError as val_err:
        logger.warning("Validation error in extraction preview: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception("Failed to execute extraction preview for %s: %s", target_url, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata extraction failed: {str(exc)}",
        )
