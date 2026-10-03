from fastapi import APIRouter, status
from app.models.schemas import CrawlRequest, CrawlResponse, PreviewRequest, PreviewResponse
from app.services.crawler_service import CrawlerService
from app.core.logging_config import logger

router = APIRouter(tags=["Crawler"])


@router.post(
    "/crawl",
    response_model=CrawlResponse,
    status_code=status.HTTP_200_OK,
    summary="Crawl a hospital website",
    description="Initiates a scoped breadth-first crawl starting from a hospital URL, bounded by max_pages and max_depth.",
)
def crawl_website(payload: CrawlRequest) -> CrawlResponse:
    logger.info("Received crawl request for: %s (max_pages=%d, max_depth=%d)", payload.start_url, payload.max_pages, payload.max_depth)
    return CrawlerService.run_crawl(payload)


@router.post(
    "/crawl/preview",
    response_model=PreviewResponse,
    status_code=status.HTTP_200_OK,
    summary="Preview cleaned document extraction",
    description="Fetches and extracts a cleaned document structure from a single webpage for inspection.",
)
def preview_document(payload: PreviewRequest) -> PreviewResponse:
    logger.info("Received preview document request for: %s", payload.url)
    return CrawlerService.preview_page(payload)
