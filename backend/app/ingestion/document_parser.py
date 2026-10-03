import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

from app.ingestion.cleaner import HTMLCleaner
from app.ingestion.extractor import HospitalMetadataExtractor
from app.core.logging_config import logger


@dataclass
class DocumentMetadata:
    """Standardized metadata attached to an ingested document."""
    source: str = "website"
    content_type: str = "webpage"
    canonical_url: Optional[str] = None
    description: Optional[str] = None
    headings: List[str] = field(default_factory=list)
    char_count: int = 0
    word_count: int = 0
    department: Optional[str] = None
    doctors: List[Dict[str, Any]] = field(default_factory=list)
    facilities: List[Dict[str, Any]] = field(default_factory=list)
    images: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CleanedDocument:
    """Structured representation of an extracted and cleaned web page."""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
        }


class DocumentParser:
    """Extracts structured document entities, metadata, and clean text from HTML."""

    def __init__(
        self,
        cleaner: Optional[HTMLCleaner] = None,
        extractor: Optional[HospitalMetadataExtractor] = None,
    ):
        self.cleaner = cleaner or HTMLCleaner()
        self.extractor = extractor or HospitalMetadataExtractor()

    def parse_document(self, html: str, url: str) -> CleanedDocument:
        """Transforms raw HTML into a CleanedDocument with structured content and metadata."""
        logger.info("Extraction started for URL: %s", url)

        if not html or not isinstance(html, str):
            logger.warning("Empty or non-string HTML provided for: %s", url)
            return CleanedDocument(
                url=url,
                title="Untitled Document",
                content="",
                metadata=DocumentMetadata(source="website", content_type="webpage").to_dict(),
            )

        try:
            soup = BeautifulSoup(html, "html.parser")

            # 1. Extract Title
            title = self._extract_title(soup)

            # 2. Extract Canonical URL
            canonical_url = self._extract_canonical_url(soup)

            # 3. Extract Meta Description
            description = self._extract_description(soup)

            # 4. Extract Headings hierarchy before cleaning
            headings = self._extract_headings(soup)

            # 5. Clean and structure the body text
            content = self.cleaner.clean_html(html)

            char_count = len(content)
            word_count = len(content.split())

            # 6. Extract structured entities (doctors, facilities, images, department)
            extracted_entities = self.extractor.extract_all(html, url)
            doctors_data = [
                d.model_dump() if hasattr(d, "model_dump") else d
                for d in extracted_entities.get("doctors", [])
            ]
            facilities_data = [
                f.model_dump() if hasattr(f, "model_dump") else f
                for f in extracted_entities.get("facilities", [])
            ]
            images_data = extracted_entities.get("images", [])
            department_data = extracted_entities.get("department", "")

            logger.info(
                "Content extracted for %s | Title: '%s' | Length: %d chars, %d words | Doctors: %d, Facilities: %d",
                url,
                title,
                char_count,
                word_count,
                len(doctors_data),
                len(facilities_data),
            )

            metadata = DocumentMetadata(
                source="website",
                content_type="webpage",
                canonical_url=canonical_url,
                description=description,
                headings=headings,
                char_count=char_count,
                word_count=word_count,
                department=department_data,
                doctors=doctors_data,
                facilities=facilities_data,
                images=images_data,
            )

            return CleanedDocument(
                url=url,
                title=title,
                content=content,
                metadata=metadata.to_dict(),
            )

        except Exception as exc:
            logger.exception("Extraction failure for %s: %s", url, exc)
            return CleanedDocument(
                url=url,
                title="Extraction Error",
                content="",
                metadata={
                    "source": "website",
                    "content_type": "webpage",
                    "error": str(exc),
                },
            )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        """Extracts the best document title from <title> or <h1>."""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
            if title:
                return title

        h1_tag = soup.find("h1")
        if h1_tag:
            lines = [line.strip() for line in h1_tag.get_text(separator="\n").splitlines() if line.strip()]
            if lines:
                return lines[0]

        return "Untitled Document"

    @staticmethod
    def _extract_canonical_url(soup: BeautifulSoup) -> Optional[str]:
        """Extracts rel='canonical' href if specified."""
        canonical_tag = soup.find("link", rel=lambda r: r and "canonical" in r)
        if canonical_tag and canonical_tag.get("href"):
            return canonical_tag["href"].strip()
        return None

    @staticmethod
    def _extract_description(soup: BeautifulSoup) -> Optional[str]:
        """Extracts page meta description or og:description."""
        meta_desc = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"].strip()

        og_desc = soup.find("meta", property=re.compile(r"^og:description$", re.I))
        if og_desc and og_desc.get("content"):
            return og_desc["content"].strip()

        return None

    @staticmethod
    def _extract_headings(soup: BeautifulSoup) -> List[str]:
        """Collects all heading texts in document order."""
        headings = []
        for h in soup.find_all(["h1", "h2", "h3", "h4"]):
            text = h.get_text(strip=True)
            if text and text not in headings:
                headings.append(text)
        return headings
