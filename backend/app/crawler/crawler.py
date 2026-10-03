import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from app.crawler.fetcher import Fetcher, FetchResponse
from app.crawler.parser import HTMLParser, ParsedPage
from app.crawler.url_manager import URLManager
from app.ingestion.document_parser import DocumentParser, CleanedDocument
from app.core.logging_config import logger


@dataclass
class CrawledPageInfo:
    """Metadata and extracted content for an individual visited page."""
    url: str
    title: str
    status: int
    depth: int = 0
    error: Optional[str] = None
    text_preview: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CrawlSummary:
    """Summary metrics and page collection returned after completing a crawl."""
    start_url: str
    pages_discovered: int
    pages_fetched: int
    pages_failed: int
    pages: List[CrawledPageInfo] = field(default_factory=list)


class HospitalCrawler:
    """Orchestrates breadth-first crawling across a single hospital domain."""

    def __init__(
        self,
        fetcher: Optional[Fetcher] = None,
        parser: Optional[HTMLParser] = None,
        document_parser: Optional[DocumentParser] = None,
        delay_seconds: float = 0.2,
    ):
        self.fetcher = fetcher or Fetcher()
        self.parser = parser or HTMLParser()
        self.document_parser = document_parser or DocumentParser()
        self.delay_seconds = delay_seconds

    def crawl(
        self,
        start_url: str,
        max_pages: int = 50,
        max_depth: int = 3,
    ) -> CrawlSummary:
        """Executes a breadth-first crawl starting at start_url until max_pages or max_depth is reached."""
        logger.info("==========================================")
        logger.info("Crawl started for target: %s", start_url)
        logger.info("Constraints: max_pages=%d, max_depth=%d, delay=%.2fs", max_pages, max_depth, self.delay_seconds)
        logger.info("==========================================")

        url_manager = URLManager(start_url=start_url)
        normalized_start = url_manager.normalize_url(start_url)

        if not normalized_start or not url_manager.is_same_domain(normalized_start):
            logger.error("Invalid or unsupported start URL: %s", start_url)
            return CrawlSummary(
                start_url=start_url,
                pages_discovered=0,
                pages_fetched=0,
                pages_failed=1,
                pages=[
                    CrawledPageInfo(
                        url=start_url,
                        title="Invalid URL",
                        status=400,
                        error="Start URL is invalid, non-HTTP, or cannot be parsed",
                    )
                ],
            )

        # FIFO queue for Breadth-First Search (BFS): (url, current_depth)
        queue = deque([(normalized_start, 0)])
        url_manager.mark_discovered(normalized_start)

        pages_result: List[CrawledPageInfo] = []
        pages_fetched = 0
        pages_failed = 0

        while queue and len(url_manager.visited_urls) < max_pages:
            current_url, current_depth = queue.popleft()

            # Skip if already visited (prevents duplicate requests)
            if url_manager.is_visited(current_url):
                logger.debug("Already visited: %s", current_url)
                continue

            # Respect depth limit
            if current_depth > max_depth:
                logger.debug("Skipping %s: depth %d exceeds max_depth %d", current_url, current_depth, max_depth)
                continue

            # Rate limiting polite delay between HTTP fetches
            if self.delay_seconds > 0 and (pages_fetched + pages_failed) > 0:
                time.sleep(self.delay_seconds)

            # 1. Fetch HTML
            fetch_result: FetchResponse = self.fetcher.fetch(current_url)

            # Check for failure
            if not fetch_result.is_success:
                pages_failed += 1
                url_manager.mark_visited(current_url)
                pages_result.append(
                    CrawledPageInfo(
                        url=current_url,
                        title="Failed Page",
                        status=fetch_result.status_code,
                        depth=current_depth,
                        error=fetch_result.error or "Fetch failed",
                    )
                )
                continue

            # 2. Parse HTML for link discovery
            parsed_page: ParsedPage = self.parser.parse(
                html=fetch_result.html or "",
                current_url=current_url,
                url_manager=url_manager,
            )

            # 3. Clean and extract structured Document
            cleaned_doc: CleanedDocument = self.document_parser.parse_document(
                html=fetch_result.html or "",
                url=current_url,
            )

            pages_fetched += 1
            url_manager.mark_visited(current_url)

            logger.info("Found %d links on %s", len(parsed_page.links), current_url)

            # Store page result with both metadata and clean content
            pages_result.append(
                CrawledPageInfo(
                    url=current_url,
                    title=cleaned_doc.title or parsed_page.title,
                    status=fetch_result.status_code,
                    depth=current_depth,
                    text_preview=cleaned_doc.content[:200] if cleaned_doc.content else None,
                    content=cleaned_doc.content,
                    metadata=cleaned_doc.metadata,
                )
            )

            # 4. Enqueue children links if within depth limit and under max_pages
            if current_depth < max_depth:
                for link in parsed_page.links:
                    # Enforce domain restriction and visited check
                    if url_manager.should_crawl(link):
                        if url_manager.mark_discovered(link):
                            queue.append((link, current_depth + 1))
                        else:
                            logger.debug("Already discovered: %s", link)

        pages_discovered = len(url_manager.discovered_urls)
        logger.info("==========================================")
        logger.info("Crawl completed for: %s", start_url)
        logger.info(
            "Summary: Discovered=%d, Fetched=%d, Failed=%d",
            pages_discovered,
            pages_fetched,
            pages_failed,
        )
        logger.info("==========================================")

        return CrawlSummary(
            start_url=start_url,
            pages_discovered=pages_discovered,
            pages_fetched=pages_fetched,
            pages_failed=pages_failed,
            pages=pages_result,
        )
