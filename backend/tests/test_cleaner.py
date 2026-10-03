from app.ingestion.cleaner import HTMLCleaner
from tests.fixtures import (
    HOSPITAL_PAGE_FIXTURE,
    SCRIPT_AND_STYLE_FIXTURE,
    WHITESPACE_FIXTURE,
)


def test_script_and_style_removal():
    cleaner = HTMLCleaner()
    cleaned = cleaner.clean_html(SCRIPT_AND_STYLE_FIXTURE)

    # Scripts and styles must be stripped
    assert "alertBox" not in cleaned
    assert "inline written code" not in cleaned
    assert "color: red" not in cleaned
    assert "line-height" not in cleaned

    # Core content must be preserved
    assert "Clean Title" in cleaned
    assert "Legitimate patient instructions." in cleaned


def test_navigation_and_footer_removal():
    cleaner = HTMLCleaner()
    cleaned = cleaner.clean_html(HOSPITAL_PAGE_FIXTURE)

    # Navigation items must not be in the cleaned text
    assert "Patient Portal" not in cleaned
    assert "Doctors" not in cleaned

    # Footer content and copyright must be removed
    assert "All rights reserved" not in cleaned
    assert "Privacy Policy" not in cleaned

    # Cookie consent banner must be removed
    assert "We use cookies" not in cleaned

    # Advertisement must be removed
    assert "Sponsored health insurance" not in cleaned


def test_structure_preservation_headings_and_paragraphs():
    cleaner = HTMLCleaner()
    sample_html = """
    <h1>Cardiology Department</h1>
    <p>Our cardiology department provides comprehensive cardiovascular treatments.</p>
    <h2>Visiting Hours</h2>
    <p>Monday-Saturday: 9 AM - 5 PM</p>
    """
    cleaned = cleaner.clean_html(sample_html)

    # Preserves clean multiline paragraph separation (not flattened into one line)
    expected_segments = [
        "Cardiology Department",
        "Our cardiology department provides comprehensive cardiovascular treatments.",
        "Visiting Hours",
        "Monday-Saturday: 9 AM - 5 PM",
    ]

    for segment in expected_segments:
        assert segment in cleaned

    # Headings and paragraphs should be separated by line breaks
    assert "\n" in cleaned
    # Ensure they are not joined on a single run-on sentence
    assert "Cardiology Department Our cardiology department" not in cleaned


def test_list_and_table_preservation():
    cleaner = HTMLCleaner()
    cleaned = cleaner.clean_html(HOSPITAL_PAGE_FIXTURE)

    # List items should be formatted with bullets
    assert "- Non-invasive diagnostic electrophysiology" in cleaned
    assert "- Cardiac catheterization and stent placement" in cleaned

    # Table rows should be formatted with cell separators
    assert "Mon - Fri | 08:00 - 18:00" in cleaned
    assert "Saturday | 09:00 - 14:00" in cleaned


def test_whitespace_cleaning():
    cleaner = HTMLCleaner()
    cleaned = cleaner.clean_html(WHITESPACE_FIXTURE)

    # Headings trimmed of surrounding irregular whitespace
    assert "Spaced Heading" in cleaned
    assert "   Spaced    Heading   " not in cleaned

    # Lines collapsed of extra internal spaces
    assert "Line one with irregular spaces." in cleaned
    assert "Line two after many blanks." in cleaned

    # No run of 3 or more newlines
    assert "\n\n\n" not in cleaned
