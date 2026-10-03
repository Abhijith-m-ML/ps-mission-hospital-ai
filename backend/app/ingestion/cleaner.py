import re
from typing import Optional
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from app.core.logging_config import logger


class HTMLCleaner:
    """Cleans raw HTML by removing scripts, styles, boilerplate, and navigation,
    while preserving document structure (headings, paragraphs, lists, tables).
    """

    # Elements to unconditionally strip (non-content / code / media elements)
    STRIP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "link",
        "meta",
        "audio",
        "video",
        "object",
        "embed",
    }

    # Common boilerplate CSS class & ID fragments (ads, cookies, tracking, site navigation)
    BOILERPLATE_SELECTORS = [
        # Navigation
        "nav",
        "[role='navigation']",
        ".navbar",
        ".nav-menu",
        ".main-navigation",
        ".site-navigation",
        "#navigation",
        "#nav",
        "#menu",
        # Footers & Copyright
        "footer",
        "[role='contentinfo']",
        ".site-footer",
        ".page-footer",
        "#footer",
        # Cookies, Modals, Banners & Ads
        ".cookie-banner",
        ".cookie-consent",
        ".cookie-notice",
        "#cookie-notice",
        ".advertisement",
        ".ad-banner",
        ".adsbygoogle",
        ".popup-modal",
        "[role='alertdialog']",
        # Social share bars
        ".social-share",
        ".share-buttons",
    ]

    def clean_html(self, raw_html: str) -> str:
        """Parses raw HTML, removes non-content elements, and formats readable structured text."""
        if not raw_html or not isinstance(raw_html, str):
            return ""

        soup = BeautifulSoup(raw_html, "html.parser")

        # 1. Remove comments
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            comment.extract()

        # 2. Decompose script, style, and media tags
        for tag in soup.find_all(self.STRIP_TAGS):
            tag.decompose()

        # 3. Remove identifiable navigation, footers, and advertisement boilerplate
        # Guard: check that removing the container does not delete the main content area
        for selector in self.BOILERPLATE_SELECTORS:
            for element in soup.select(selector):
                # Safeguard: do not decompose if element contains the main article or primary <h1>
                if element.find("main") or element.find("article"):
                    continue
                # If element has a heading that looks like valuable page content, be conservative
                element.decompose()

        # 4. Convert HTML structure to text while preserving headings, paragraphs, lists, and tables
        structured_text = self._extract_structured_text(soup)

        # 5. Clean up whitespace
        clean_text = self._normalize_whitespace(structured_text)
        return clean_text

    def _extract_structured_text(self, soup: BeautifulSoup) -> str:
        """Recursively walks DOM elements, inserting newlines and markdown-like spacing."""
        # Replace <br> tags with explicit newlines
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Format headings: ensure clear paragraph spacing
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            h_text = h.get_text(strip=True)
            if h_text:
                h.replace_with(f"\n\n{h_text}\n\n")

        # Format paragraphs and blockquotes
        for p in soup.find_all(["p", "blockquote"]):
            p_text = p.get_text(strip=True)
            if p_text:
                p.replace_with(f"\n\n{p_text}\n\n")

        # Format unordered and ordered list items
        for li in soup.find_all("li"):
            li_text = li.get_text(strip=True)
            if li_text:
                li.replace_with(f"\n- {li_text}")

        for ul in soup.find_all(["ul", "ol"]):
            ul_text = ul.get_text()
            ul.replace_with(f"\n{ul_text}\n\n")

        # Format tables into readable row blocks
        for tr in soup.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]
            if cells:
                row_str = " | ".join(cells)
                tr.replace_with(f"\n{row_str}")

        for table in soup.find_all("table"):
            table_text = table.get_text()
            table.replace_with(f"\n\n{table_text}\n\n")

        # Get body or full soup text
        target = soup.body if soup.body else soup
        return target.get_text()

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalizes horizontal whitespace and collapses excessive empty lines."""
        if not text:
            return ""

        # Replace non-breaking spaces with standard spaces
        text = text.replace("\xa0", " ")

        # Split into lines and trim horizontal whitespace on each line
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

        # Recombine while collapsing 3+ consecutive newlines into 2 (paragraph break)
        result = []
        consecutive_empty = 0

        for line in lines:
            if not line:
                consecutive_empty += 1
                if consecutive_empty <= 1 and result:
                    result.append("")
            else:
                consecutive_empty = 0
                result.append(line)

        return "\n".join(result).strip()
