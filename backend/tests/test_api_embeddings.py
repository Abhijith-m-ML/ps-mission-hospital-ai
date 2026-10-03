import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_text_embedding_preview_endpoint():
    """Test POST /api/embeddings/preview with raw text."""
    payload = {
        "text": "Cardiology department provides heart care.",
        "preview_dims": 4,
    }
    response = client.post("/api/embeddings/preview", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "model" in data
    assert data["embedding_dimension"] == 384
    assert data["count"] == 1
    assert len(data["preview"]) == 4
    assert len(data["items"]) == 1
    assert data["items"][0]["embedding_dimension"] == 384
    assert len(data["items"][0]["preview"]) == 4


def test_api_text_embedding_batch_preview():
    """Test POST /api/embeddings/preview with batch of texts."""
    payload = {
        "texts": [
            "Emergency services are available 24/7.",
            "Pediatrics ward is on the second floor.",
        ],
        "preview_dims": 5,
    }
    response = client.post("/api/embeddings/preview", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 2
    assert len(data["items"]) == 2
    assert len(data["preview"]) == 5


def test_api_text_embedding_validation_error():
    """Test POST /api/embeddings/preview with empty payload."""
    payload = {}
    response = client.post("/api/embeddings/preview", json=payload)
    assert response.status_code == 400
    assert "No text provided" in response.json()["detail"]


def test_api_chunk_embedding_preview_with_direct_chunks():
    """Test POST /api/embeddings/chunks/preview with direct chunks."""
    payload = {
        "chunks": [
            {
                "chunk_id": "chunk_mock_1",
                "document_id": "doc_mock_1",
                "content": "Cardiology\nThe department provides cardiovascular diagnosis.",
                "metadata": {
                    "url": "https://www.psmissionhospital.org/departments/cardiology",
                    "title": "Departments",
                    "section": "Cardiology",
                    "source": "website",
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
            }
        ],
        "max_preview_chunks": 5,
        "preview_dims": 5,
    }
    response = client.post("/api/embeddings/chunks/preview", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_chunks_processed"] == 1
    assert data["preview_chunks_count"] == 1
    assert data["embedding_dimension"] == 384
    assert len(data["sample_results"]) == 1

    sample = data["sample_results"][0]
    assert sample["chunk_id"] == "chunk_mock_1"
    assert sample["document_id"] == "doc_mock_1"
    assert sample["section"] == "Cardiology"
    assert len(sample["embedding_preview"]) == 5
    assert sample["metadata"]["source"] == "website"


def test_api_chunk_embedding_preview_validation_error():
    """Test POST /api/embeddings/chunks/preview without url or chunks."""
    payload = {}
    response = client.post("/api/embeddings/chunks/preview", json=payload)
    assert response.status_code == 400
    assert "Either 'url' or 'chunks'" in response.json()["detail"]
