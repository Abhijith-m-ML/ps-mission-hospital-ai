from dataclasses import dataclass, field
from typing import List, Optional
from bs4 import BeautifulSoup

from app.crawler.url_manager import URLManager
from app.core.logging_config import logger


@dataclass
class ParsedPage:
    """Structured data extracted from an HTML page."""
    title: str
    links: List[str] = field(default_factory=list)
    text_content: str = ""


class HTMLParser:
    """Parses raw HTML, extracting metadata, cleaned text, and normalized hyperlinks."""

    # Tags to strip before extracting readable body text
    STRIP_TAGS = {"script", "style", "noscript", "svg", "header_extra", "nav_skip"}

    def parse(self, html: str, current_url: str, url_manager: URLManager) -> ParsedPage:
        """Parses HTML document to extract page title, plain text, and valid links."""
        if not html or not isinstance(html, str):
            return ParsedPage(title="Empty Document", links=[], text_content="")

        soup = BeautifulSoup(html, "html.parser")

        # 1. Extract Page Title
        title = "Untitled Page"
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip() or "Untitled Page"

        # 2. Extract Valid Normalized Links
        discovered_links: List[str] = []
        seen_on_page = set()

        for a_tag in soup.find_all("a", href=True):
            raw_href = a_tag["href"]
            normalized_url = url_manager.normalize_url(raw_href, current_page_url=current_url)

            if normalized_url and normalized_url not in seen_on_page:
                seen_on_page.add(normalized_url)
                discovered_links.append(normalized_url)

        # 3. Extract Clean Readable Body Text
        # Decompose non-content tags
        for element in soup(["script", "style", "noscript", "svg", "head"]):
            element.decompose()

        body = soup.find("body") or soup
        # Get readable text with spaces
        raw_text = body.get_text(separator=" ", strip=True)
        # Collapse multiple spaces and newlines
        clean_text = " ".join(raw_text.split())

        logger.debug("Parsed %s: Title='%s', Discovered %d links", current_url, title, len(discovered_links))

        return ParsedPage(
            title=title,
            links=discovered_links,
            text_content=clean_text,
        )
