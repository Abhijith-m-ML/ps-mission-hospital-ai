from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.models.schemas import ChunkPreviewResponse, DocumentBrief, ChunkItem

client = TestClient(app)


def test_api_chunks_preview_mocked():
    mock_response = ChunkPreviewResponse(
        document=DocumentBrief(
            url="https://psmissionhospital.org/cardiology",
            title="P S Mission Hospital - Cardiology",
        ),
        chunk_count=2,
        chunks=[
            ChunkItem(
                chunk_id="chk_001",
                document_id="doc_cardio",
                content="Cardiology Department\n\nAdvanced cardiac care at P S Mission Hospital.",
                metadata={
                    "url": "https://psmissionhospital.org/cardiology",
                    "title": "P S Mission Hospital - Cardiology",
                    "section": "Cardiology Department",
                    "source": "website",
                    "chunk_index": 0,
                    "total_chunks": 2,
                },
            ),
            ChunkItem(
                chunk_id="chk_002",
                document_id="doc_cardio",
                content="Visiting Hours\n\nDaily ICU visiting hours: 10 AM to 12 PM.",
                metadata={
                    "url": "https://psmissionhospital.org/cardiology",
                    "title": "P S Mission Hospital - Cardiology",
                    "section": "Visiting Hours",
                    "source": "website",
                    "chunk_index": 1,
                    "total_chunks": 2,
                },
            ),
        ],
    )

    with patch("app.services.chunking_service.ChunkingService.preview_chunks", return_value=mock_response):
        payload = {
            "url": "https://psmissionhospital.org/cardiology",
            "chunk_size": 500,
            "chunk_overlap": 80,
        }
        response = client.post("/api/chunks/preview", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["document"]["url"] == "https://psmissionhospital.org/cardiology"
        assert data["chunk_count"] == 2
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["chunk_id"] == "chk_001"
        assert data["chunks"][0]["metadata"]["section"] == "Cardiology Department"
        assert "Advanced cardiac care" in data["chunks"][0]["content"]


def test_api_chunks_preview_validation_error():
    payload = {}  # Missing required 'url'
    response = client.post("/api/chunks/preview", json=payload)
    assert response.status_code == 422
