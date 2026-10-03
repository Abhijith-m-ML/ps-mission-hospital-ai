from app.ingestion.document_parser import DocumentParser, CleanedDocument
from tests.fixtures import (
    HOSPITAL_PAGE_FIXTURE,
    MALFORMED_HTML_FIXTURE,
    EMPTY_HTML_FIXTURE,
)


def test_document_parser_metadata_and_structure():
    parser = DocumentParser()
    doc: CleanedDocument = parser.parse_document(
        html=HOSPITAL_PAGE_FIXTURE,
        url="https://stjude-hospital.org/departments/cardiology",
    )

    assert doc.url == "https://stjude-hospital.org/departments/cardiology"
    assert doc.title == "St. Jude Hospital - Cardiology Department"
    assert "Cardiology Department" in doc.content

    # Metadata validation
    assert doc.metadata["source"] == "website"
    assert doc.metadata["content_type"] == "webpage"
    assert doc.metadata["canonical_url"] == "https://stjude-hospital.org/departments/cardiology"
    assert "cardiovascular diagnosis" in doc.metadata["description"]

    # Headings list check
    headings = doc.metadata["headings"]
    assert "Cardiology Department" in headings
    assert "Clinical Services" in headings
    assert "Visiting Hours" in headings
    assert "Weekly Clinic Schedule" in headings

    # Character and word counts
    assert doc.metadata["char_count"] > 100
    assert doc.metadata["word_count"] > 20


def test_document_parser_fallback_title():
    parser = DocumentParser()
    html_without_title = "<html><body><h1>Fallback Heading Title</h1><p>Body text</p></body></html>"
    doc = parser.parse_document(html_without_title, "https://hospital.org/page")

    assert doc.title == "Fallback Heading Title"


def test_document_parser_malformed_html():
    parser = DocumentParser()
    doc = parser.parse_document(MALFORMED_HTML_FIXTURE, "https://hospital.org/malformed")

    # Parser should not raise error and successfully extract text
    assert doc.title == "Unclosed Heading"
    assert "First paragraph without closing tag" in doc.content
    assert "Second Heading" in doc.content
    assert "Second paragraph" in doc.content


def test_document_parser_empty_html():
    parser = DocumentParser()

    # Empty string
    doc_empty = parser.parse_document("", "https://hospital.org/empty")
    assert doc_empty.content == ""
    assert doc_empty.metadata["char_count"] == 0

    # HTML with only empty tags / scripts
    doc_tags_only = parser.parse_document(EMPTY_HTML_FIXTURE, "https://hospital.org/tags-only")
    assert doc_tags_only.content == ""
    assert doc_tags_only.metadata["char_count"] == 0
