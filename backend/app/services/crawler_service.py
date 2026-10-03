from app.crawler.crawler import HospitalCrawler, CrawlSummary
from app.crawler.fetcher import Fetcher, FetchResponse
from app.ingestion.document_parser import DocumentParser, CleanedDocument
from app.models.schemas import (
    CrawlRequest,
    CrawlResponse,
    CrawledPageItem,
    PreviewRequest,
    PreviewResponse,
)
from app.core.logging_config import logger


class CrawlerService:
    """Service layer coordinating website crawling and document extraction operations."""

    @staticmethod
    def run_crawl(request: CrawlRequest) -> CrawlResponse:
        crawler = HospitalCrawler(delay_seconds=0.2)
        summary: CrawlSummary = crawler.crawl(
            start_url=request.start_url,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
        )

        return CrawlResponse(
            start_url=summary.start_url,
            pages_discovered=summary.pages_discovered,
            pages_fetched=summary.pages_fetched,
            pages_failed=summary.pages_failed,
            pages=[
                CrawledPageItem(
                    url=p.url,
                    title=p.title,
                    status=p.status,
                    depth=p.depth,
                    error=p.error,
                    content=p.content,
                    metadata=p.metadata,
                )
                for p in summary.pages
            ],
        )

    @staticmethod
    def preview_page(request: PreviewRequest) -> PreviewResponse:
        """Fetches a single page and returns its cleaned document structure without crawling."""
        logger.info("Executing preview inspection for: %s", request.url)
        fetcher = Fetcher()
        fetch_result: FetchResponse = fetcher.fetch(request.url)

        if not fetch_result.is_success:
            logger.warning("Preview fetch failed for %s: %s", request.url, fetch_result.error)
            return PreviewResponse(
                url=request.url,
                title="Fetch Error",
                content="",
                metadata={
                    "status_code": fetch_result.status_code,
                    "error": fetch_result.error or "Failed to fetch webpage",
                },
            )

        parser = DocumentParser()
        cleaned_doc: CleanedDocument = parser.parse_document(
            html=fetch_result.html or "",
            url=request.url,
        )

        return PreviewResponse(
            url=cleaned_doc.url,
            title=cleaned_doc.title,
            content=cleaned_doc.content,
            metadata=cleaned_doc.metadata,
        )
