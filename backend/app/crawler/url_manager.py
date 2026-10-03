import posixpath
from typing import Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse

from app.core.logging_config import logger


class URLManager:
    """Manages URL normalization, filtering, domain-restriction, and duplicate tracking."""

    # Disallowed URL schemes (non-HTTP/HTTPS)
    UNSUPPORTED_SCHEMES = {
        "mailto", "tel", "javascript", "ftp", "file", "data", "sms", "callto", "whatsapp"
    }

    # Common non-HTML media, asset, and download file extensions to bypass
    ASSET_EXTENSIONS = {
        # Images
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tiff",
        # Audio / Video
        ".mp4", ".mov", ".avi", ".wmv", ".mkv", ".webm", ".mp3", ".wav", ".ogg", ".flac",
        # Documents & Archives
        ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar", ".exe", ".dmg", ".iso",
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        # Code & Styles
        ".css", ".js", ".json", ".xml", ".rss", ".map", ".woff", ".woff2", ".ttf", ".eot"
    }

    def __init__(self, start_url: str, allow_subdomains: bool = False):
        self.raw_start_url = start_url
        self.allow_subdomains = allow_subdomains

        parsed = urlparse(start_url)
        self.allowed_scheme = parsed.scheme.lower() if parsed.scheme else "https"
        self.allowed_netloc = self._extract_clean_netloc(parsed.netloc or start_url)
        
        # State tracking
        self.visited_urls: Set[str] = set()
        self.discovered_urls: Set[str] = set()

    @staticmethod
    def _extract_clean_netloc(netloc: str) -> str:
        """Strips port and converts to lower case."""
        netloc = netloc.lower().strip()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc

    def normalize_url(self, raw_url: str, current_page_url: Optional[str] = None) -> Optional[str]:
        """Normalizes a raw or relative URL into an absolute canonical URL.
        
        Returns None if URL is invalid, uses an unsupported scheme, or targets static assets.
        """
        if not raw_url or not isinstance(raw_url, str):
            return None

        trimmed = raw_url.strip()
        if not trimmed:
            return None

        # Resolve relative URLs
        reference_url = current_page_url or self.raw_start_url
        absolute_url = urljoin(reference_url, trimmed)

        try:
            parsed = urlparse(absolute_url)
        except Exception:
            return None

        scheme = parsed.scheme.lower()
        if scheme in self.UNSUPPORTED_SCHEMES or scheme not in {"http", "https"}:
            return None

        netloc = parsed.netloc.lower()
        if not netloc:
            return None

        # Strip standard default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Normalize path
        path = parsed.path or "/"
        # Normalize double slashes or relative components safely
        path = posixpath.normpath(path)
        if not path.startswith("/"):
            path = "/" + path

        # Preserve root slash, but strip trailing slash on other paths for consistency
        if path != "/" and path.endswith("/"):
            path = path[:-1]

        # Check for asset extensions
        _, ext = posixpath.splitext(path.lower())
        if ext in self.ASSET_EXTENSIONS:
            return None

        # Reconstruct without fragment (#section)
        normalized = urlunparse((
            scheme,
            netloc,
            path,
            parsed.params,
            parsed.query,
            ""  # Always strip fragment
        ))

        return normalized

    def is_same_domain(self, url: str) -> bool:
        """Checks if a URL belongs to the allowed hospital domain."""
        try:
            parsed = urlparse(url)
            netloc = self._extract_clean_netloc(parsed.netloc)
            if not netloc:
                return False

            # Exact match
            if netloc == self.allowed_netloc:
                return True

            # www-prefix equivalence: hospital.com <-> www.hospital.com
            if netloc.removeprefix("www.") == self.allowed_netloc.removeprefix("www."):
                return True

            # Optional subdomain handling
            if self.allow_subdomains and netloc.endswith("." + self.allowed_netloc.removeprefix("www.")):
                return True

            return False
        except Exception:
            return False

    def is_visited(self, url: str) -> bool:
        """Checks if the URL has already been fetched."""
        return url in self.visited_urls

    def is_discovered(self, url: str) -> bool:
        """Checks if the URL has already been queued or seen."""
        return url in self.discovered_urls

    def mark_discovered(self, url: str) -> bool:
        """Marks a URL as discovered. Returns True if first time seen, False otherwise."""
        if url in self.discovered_urls:
            return False
        self.discovered_urls.add(url)
        return True

    def mark_visited(self, url: str) -> None:
        """Marks a URL as visited."""
        self.visited_urls.add(url)

    def should_crawl(self, url: Optional[str]) -> bool:
        """Determines whether a URL is eligible for crawling."""
        if not url:
            return False

        if not self.is_same_domain(url):
            logger.info("Skipping external URL: %s", url)
            return False

        if self.is_visited(url):
            logger.debug("Already visited: %s", url)
            return False

        return True
