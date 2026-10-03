import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_api_search_preview_validation_error():
    """Test POST /api/search/preview with missing or empty query."""
    res_empty = client.post("/api/search/preview", json={"query": "   "})
    assert res_empty.status_code == 400
    assert "Query string cannot be empty" in res_empty.json()["detail"]


@patch("app.services.search_service.SearchService.search_query")
def test_api_search_preview_success(mock_search):
    """Test POST /api/search/preview returns properly structured response."""
    mock_search.return_value = {
        "query": "Which department treats heart problems?",
        "index": "hospital-ai",
        "namespace": "ps_mission_hospital",
        "top_k": 3,
        "total_found": 1,
        "search_duration_seconds": 0.045,
        "results": [
            {
                "id": "chunk_cardio_123",
                "score": 0.845,
                "content": "Cardiology\nThe Cardiology department provides 24-hour heart emergency care.",
                "section": "Cardiology",
                "url": "https://www.psmissionhospital.org/departments/cardiology",
                "title": "Departments",
                "source": "website",
                "metadata": {"chunk_index": 0},
            }
        ],
    }

    payload = {"query": "Which department treats heart problems?", "top_k": 3}
    response = client.post("/api/search/preview", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == "Which department treats heart problems?"
    assert data["index"] == "hospital-ai"
    assert data["namespace"] == "ps_mission_hospital"
    assert data["total_found"] == 1
    assert len(data["results"]) == 1
    result_item = data["results"][0]
    assert result_item["id"] == "chunk_cardio_123"
    assert result_item["section"] == "Cardiology"
    assert result_item["score"] == 0.845


@patch("app.services.indexing_service.IndexingService.index_from_url")
def test_api_index_preview_success(mock_index):
    """Test POST /api/index/preview returns indexing metrics."""
    mock_index.return_value = {
        "index_name": "hospital-ai",
        "namespace": "ps_mission_hospital",
        "url": "https://www.psmissionhospital.org/",
        "title": "P.S. Mission Hospital, Maradu, Cochin",
        "indexed_count": 43,
        "failed_count": 0,
        "dimension": 384,
        "duration_seconds": 1.25,
    }

    payload = {"url": "https://www.psmissionhospital.org/", "max_chunks": 50}
    response = client.post("/api/index/preview", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["index_name"] == "hospital-ai"
    assert data["namespace"] == "ps_mission_hospital"
    assert data["indexed_count"] == 43
    assert data["dimension"] == 384


@patch("app.retrieval.vector_store.PineconeVectorStore.get_index_stats")
def test_api_index_stats_endpoint(mock_stats):
    """Test GET /api/index/stats returns Pinecone vector statistics."""
    mock_stats.return_value = {
        "index": "hospital-ai",
        "namespace": "ps_mission_hospital",
        "dimension": 384,
        "total_vector_count": 43,
        "namespace_vector_count": 43,
    }

    response = client.get("/api/index/stats")
    assert response.status_code == 200
    data = response.json()

    assert data["index"] == "hospital-ai"
    assert data["namespace"] == "ps_mission_hospital"
    assert data["dimension"] == 384
    assert data["total_vector_count"] == 43
    assert data["namespace_vector_count"] == 43
