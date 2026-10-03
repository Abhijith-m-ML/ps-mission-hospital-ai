from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.models.schemas import CrawlResponse, CrawledPageItem, PreviewResponse

client = TestClient(app)


def test_api_health_endpoint():
    """Verify health endpoint continues to operate normally."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_api_crawl_endpoint_mocked():
    """Verify POST /api/crawl endpoint contract and schema matching."""
    mock_response = CrawlResponse(
        start_url="https://stjude-hospital.org",
        pages_discovered=5,
        pages_fetched=4,
        pages_failed=1,
        pages=[
            CrawledPageItem(
                url="https://stjude-hospital.org/",
                title="St. Jude Hospital Home",
                status=200,
                depth=0,
                content="St. Jude Hospital Home\n\nWelcome to our hospital.",
                metadata={"source": "website", "content_type": "webpage"},
            ),
            CrawledPageItem(
                url="https://stjude-hospital.org/departments",
                title="Departments",
                status=200,
                depth=1,
                content="Departments\n\nCardiology and Neurology.",
                metadata={"source": "website", "content_type": "webpage"},
            ),
            CrawledPageItem(
                url="https://stjude-hospital.org/broken",
                title="Failed Page",
                status=404,
                depth=1,
                error="404 Not Found",
            ),
        ],
    )

    with patch("app.services.crawler_service.CrawlerService.run_crawl", return_value=mock_response):
        payload = {
            "start_url": "https://stjude-hospital.org",
            "max_pages": 10,
            "max_depth": 2,
        }
        response = client.post("/api/crawl", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["start_url"] == "https://stjude-hospital.org"
        assert data["pages_discovered"] == 5
        assert data["pages_fetched"] == 4
        assert data["pages_failed"] == 1
        assert len(data["pages"]) == 3
        assert data["pages"][0]["title"] == "St. Jude Hospital Home"
        assert data["pages"][0]["status"] == 200
        assert "Welcome to our hospital" in data["pages"][0]["content"]


def test_api_crawl_preview_endpoint_mocked():
    """Verify POST /api/crawl/preview endpoint contract for single document inspection."""
    mock_preview = PreviewResponse(
        url="https://stjude-hospital.org/departments/cardiology",
        title="St. Jude Hospital - Cardiology",
        content="Cardiology Department\n\nComprehensive cardiovascular treatments.",
        metadata={
            "source": "website",
            "content_type": "webpage",
            "headings": ["Cardiology Department"],
            "char_count": 62,
            "word_count": 7,
        },
    )

    with patch("app.services.crawler_service.CrawlerService.preview_page", return_value=mock_preview):
        payload = {
            "url": "https://stjude-hospital.org/departments/cardiology",
        }
        response = client.post("/api/crawl/preview", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["url"] == "https://stjude-hospital.org/departments/cardiology"
        assert data["title"] == "St. Jude Hospital - Cardiology"
        assert "Cardiology Department" in data["content"]
        assert data["metadata"]["source"] == "website"
        assert data["metadata"]["char_count"] == 62


def test_api_crawl_validation_error():
    """Verify Pydantic validation handles invalid payload."""
    payload = {
        # missing start_url
        "max_pages": -1,
    }
    response = client.post("/api/crawl", json=payload)
    assert response.status_code == 422
