import pytest
from typing import Dict
from app.crawler.crawler import HospitalCrawler
from app.crawler.fetcher import Fetcher, FetchResponse


class MockFetcher(Fetcher):
    """Simulated fetcher providing in-memory HTML pages for test scenarios."""

    def __init__(self, pages: Dict[str, str]):
        super().__init__()
        self.pages = pages
        self.call_count = 0

    def fetch(self, url: str) -> FetchResponse:
        self.call_count += 1
        if url in self.pages:
            return FetchResponse(
                url=url,
                status_code=200,
                html=self.pages[url],
                content_type="text/html",
                final_url=url,
            )
        return FetchResponse(
            url=url,
            status_code=404,
            html=None,
            content_type="text/html",
            error="404 Not Found",
        )


MOCK_WEBSITE_GRAPH = {
    "https://hospital.org/": """
        <html>
            <head><title>Hospital Home</title></head>
            <body>
                <a href="/about">About Us</a>
                <a href="/departments">Departments</a>
                <a href="https://external-health.org/news">External Health News</a>
                <a href="/images/banner.jpg">Banner</a>
                <a href="mailto:info@hospital.org">Email Us</a>
            </body>
        </html>
    """,
    "https://hospital.org/about": """
        <html>
            <head><title>About St. Jude</title></head>
            <body>
                <a href="/">Home</a>
                <a href="/doctors">Our Doctors</a>
            </body>
        </html>
    """,
    "https://hospital.org/departments": """
        <html>
            <head><title>Departments</title></head>
            <body>
                <a href="/departments/cardiology">Cardiology</a>
                <a href="/departments/neurology">Neurology</a>
            </body>
        </html>
    """,
    "https://hospital.org/doctors": """
        <html>
            <head><title>Doctors Directory</title></head>
            <body>
                <a href="/doctors/dr-smith">Dr. Smith</a>
            </body>
        </html>
    """,
    "https://hospital.org/departments/cardiology": """
        <html>
            <head><title>Cardiology</title></head>
            <body>
                <p>Cardiology services</p>
                <a href="/departments/cardiology/surgery">Cardio Surgery</a>
            </body>
        </html>
    """,
    "https://hospital.org/departments/neurology": """
        <html>
            <head><title>Neurology</title></head>
            <body><p>Neurology services</p></body>
        </html>
    """,
    "https://hospital.org/doctors/dr-smith": """
        <html>
            <head><title>Dr. Smith Profile</title></head>
            <body><a href="/doctors/dr-smith/publications">Publications</a></body>
        </html>
    """,
    "https://hospital.org/departments/cardiology/surgery": """
        <html>
            <head><title>Cardio Surgery</title></head>
            <body><p>Surgical details</p></body>
        </html>
    """,
    "https://hospital.org/doctors/dr-smith/publications": """
        <html>
            <head><title>Publications</title></head>
            <body><p>Research papers</p></body>
        </html>
    """,
}


def test_crawler_max_pages_limit():
    """Verify that crawler strictly stops when max_pages is reached."""
    mock_fetcher = MockFetcher(MOCK_WEBSITE_GRAPH)
    crawler = HospitalCrawler(fetcher=mock_fetcher, delay_seconds=0.0)

    summary = crawler.crawl(
        start_url="https://hospital.org/",
        max_pages=3,
        max_depth=5,
    )

    assert summary.pages_fetched <= 3
    assert len(summary.pages) == 3
    assert summary.pages_failed == 0


def test_crawler_max_depth_zero():
    """Depth 0 should only crawl the start page itself."""
    mock_fetcher = MockFetcher(MOCK_WEBSITE_GRAPH)
    crawler = HospitalCrawler(fetcher=mock_fetcher, delay_seconds=0.0)

    summary = crawler.crawl(
        start_url="https://hospital.org/",
        max_pages=20,
        max_depth=0,
    )

    # Only the root page is fetched
    assert summary.pages_fetched == 1
    assert summary.pages[0].url == "https://hospital.org/"
    assert summary.pages[0].depth == 0


def test_crawler_max_depth_one():
    """Depth 1 should crawl start page and its immediate same-domain children."""
    mock_fetcher = MockFetcher(MOCK_WEBSITE_GRAPH)
    crawler = HospitalCrawler(fetcher=mock_fetcher, delay_seconds=0.0)

    summary = crawler.crawl(
        start_url="https://hospital.org/",
        max_pages=20,
        max_depth=1,
    )

    urls_fetched = [p.url for p in summary.pages]
    # Expected: root (depth 0), /about (depth 1), /departments (depth 1)
    assert "https://hospital.org/" in urls_fetched
    assert "https://hospital.org/about" in urls_fetched
    assert "https://hospital.org/departments" in urls_fetched
    # Depth 2 pages should NOT be fetched
    assert "https://hospital.org/doctors" not in urls_fetched
    assert "https://hospital.org/departments/cardiology" not in urls_fetched
    assert len(urls_fetched) == 3


def test_crawler_external_and_asset_rejection():
    """Verify external domains, mailto, and assets are never fetched."""
    mock_fetcher = MockFetcher(MOCK_WEBSITE_GRAPH)
    crawler = HospitalCrawler(fetcher=mock_fetcher, delay_seconds=0.0)

    summary = crawler.crawl(
        start_url="https://hospital.org/",
        max_pages=20,
        max_depth=2,
    )

    urls_fetched = [p.url for p in summary.pages]
    for url in urls_fetched:
        assert url.startswith("https://hospital.org")
        assert not url.endswith(".jpg")
        assert "external-health.org" not in url
        assert "mailto:" not in url


def test_crawler_loop_prevention_duplicate_visits():
    """Verify that circular references (e.g. /about -> /) do not trigger infinite loops or double fetching."""
    mock_fetcher = MockFetcher(MOCK_WEBSITE_GRAPH)
    crawler = HospitalCrawler(fetcher=mock_fetcher, delay_seconds=0.0)

    summary = crawler.crawl(
        start_url="https://hospital.org/",
        max_pages=50,
        max_depth=4,
    )

    urls_fetched = [p.url for p in summary.pages]
    # Check uniqueness
    assert len(urls_fetched) == len(set(urls_fetched))
