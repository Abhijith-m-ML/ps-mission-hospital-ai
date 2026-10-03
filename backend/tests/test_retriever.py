from typing import Any, Dict, List, Optional
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import RetrievalChunkItem
from app.retrieval.context_builder import ContextBuilder
from app.retrieval.retriever import Retriever


class MockVectorStore:
    """Mock PineconeVectorStore simulating search, candidates, and namespace routing."""

    def __init__(self, candidates: Optional[List[Dict[str, Any]]] = None):
        self.namespace = "ps_mission_hospital"
        self.index_name = "hospital-ai"
        self.candidates = candidates or []
        self.last_query_vector: Optional[List[float]] = None
        self.last_top_k: Optional[int] = None
        self.last_namespace: Optional[str] = None
        self.last_filter: Optional[Dict[str, Any]] = None

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        index_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.last_query_vector = query_vector
        self.last_top_k = top_k
        self.last_namespace = namespace or self.namespace
        self.last_filter = filter_dict
        return self.candidates[:top_k]


class MockEmbedder:
    """Mock DocumentEmbedder returning predictable 384-dimensional vectors."""

    def __init__(self, dimension: int = 384):
        self.dim = dimension
        self.embed_calls: List[str] = []

    def embedding_dimension(self) -> int:
        return self.dim

    def embed_text(self, text: str) -> List[float]:
        self.embed_calls.append(text)
        # Deterministic 384-d vector based on string length
        base_val = (len(text) % 10) * 0.1
        return [base_val] * self.dim


def sample_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "id": "chunk_cardio_01",
            "score": 0.88,
            "content": "Cardiology Department provides diagnostic ECG and heart surgery.",
            "section": "Cardiology",
            "url": "https://www.psmissionhospital.org/cardiology",
            "title": "Cardiology Care",
            "source": "website",
            "metadata": {
                "chunk_id": "chunk_cardio_01",
                "document_id": "doc_01",
                "content": "Cardiology Department provides diagnostic ECG and heart surgery.",
                "section": "Cardiology",
                "url": "https://www.psmissionhospital.org/cardiology",
                "title": "Cardiology Care",
            },
        },
        {
            "id": "chunk_cardio_02",
            "score": 0.75,
            "content": "Cardiology Outpatient Clinic hours are 9 AM to 1 PM.",
            "section": "Cardiology",
            "url": "https://www.psmissionhospital.org/cardiology",
            "title": "Cardiology Care",
            "source": "website",
            "metadata": {
                "chunk_id": "chunk_cardio_02",
                "document_id": "doc_01",
                "content": "Cardiology Outpatient Clinic hours are 9 AM to 1 PM.",
                "section": "Cardiology",
                "url": "https://www.psmissionhospital.org/cardiology",
                "title": "Cardiology Care",
            },
        },
        {
            "id": "chunk_general_01",
            "score": 0.42,
            "content": "General hospital registration counter opens at 8 AM.",
            "section": "General",
            "url": "https://www.psmissionhospital.org/contact",
            "title": "Hospital Registration",
            "source": "website",
            "metadata": {
                "chunk_id": "chunk_general_01",
                "document_id": "doc_02",
                "content": "General hospital registration counter opens at 8 AM.",
                "section": "General",
                "url": "https://www.psmissionhospital.org/contact",
                "title": "Hospital Registration",
            },
        },
    ]


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_retriever_empty_query():
    retriever = Retriever(vector_store=MockVectorStore(), embedder=MockEmbedder())
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        retriever.retrieve("   ")


def test_retriever_query_too_long():
    retriever = Retriever(vector_store=MockVectorStore(), embedder=MockEmbedder())
    long_q = "heart " * 250  # > 1000 characters
    with pytest.raises(ValueError, match="exceeds maximum allowed limit"):
        retriever.retrieve(long_q)


def test_retriever_invalid_top_k():
    retriever = Retriever(vector_store=MockVectorStore(), embedder=MockEmbedder())
    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 50"):
        retriever.retrieve("heart care", top_k=0)
    with pytest.raises(ValueError, match="top_k must be an integer between 1 and 50"):
        retriever.retrieve("heart care", top_k=51)


def test_retriever_invalid_min_score():
    retriever = Retriever(vector_store=MockVectorStore(), embedder=MockEmbedder())
    with pytest.raises(ValueError, match="min_score must be a float between 0.0 and 1.0"):
        retriever.retrieve("heart care", min_score=1.5)


def test_retriever_valid_query_and_embedding_generation():
    embedder = MockEmbedder()
    store = MockVectorStore(candidates=sample_candidates())
    retriever = Retriever(vector_store=store, embedder=embedder)

    resp = retriever.retrieve("cardiology heart problems", top_k=3)

    assert resp.query == "cardiology heart problems"
    assert len(embedder.embed_calls) == 1
    assert embedder.embed_calls[0] == "cardiology heart problems"
    assert store.last_top_k == 3
    assert store.last_namespace == "ps_mission_hospital"
    assert resp.has_context is True
    assert len(resp.results) == 3


def test_retriever_result_ordering_and_ranks():
    # Candidates deliberately provided out of order
    unordered = [
        {"id": "c1", "score": 0.60, "content": "Text 1", "section": "A", "url": "url1", "metadata": {"chunk_id": "c1"}},
        {"id": "c2", "score": 0.95, "content": "Text 2", "section": "B", "url": "url2", "metadata": {"chunk_id": "c2"}},
        {"id": "c3", "score": 0.78, "content": "Text 3", "section": "C", "url": "url3", "metadata": {"chunk_id": "c3"}},
    ]
    retriever = Retriever(vector_store=MockVectorStore(unordered), embedder=MockEmbedder())
    resp = retriever.retrieve("query", top_k=5)

    scores = [r.score for r in resp.results]
    ranks = [r.rank for r in resp.results]

    assert scores == [0.95, 0.78, 0.60]
    assert ranks == [1, 2, 3]
    assert resp.results[0].chunk_id == "c2"


def test_retriever_minimum_similarity_filtering():
    retriever = Retriever(vector_store=MockVectorStore(sample_candidates()), embedder=MockEmbedder())

    # Set threshold to 0.70; candidate with 0.42 should be filtered out
    resp = retriever.retrieve("heart query", min_score=0.70)

    assert len(resp.results) == 2
    assert all(r.score >= 0.70 for r in resp.results)
    assert resp.debug.removed_by_threshold == 1
    assert resp.has_context is True


def test_retriever_no_result_behavior_when_threshold_exceeded():
    retriever = Retriever(vector_store=MockVectorStore(sample_candidates()), embedder=MockEmbedder())

    # Set threshold to 0.99 (higher than all candidates)
    resp = retriever.retrieve("query with high bar", min_score=0.99)

    assert resp.has_context is False
    assert len(resp.results) == 0
    assert resp.context.has_context is False
    assert resp.context.context_text == ""
    assert resp.context.reason == "No sufficiently relevant information was found."
    assert resp.debug.removed_by_threshold == 3


def test_retriever_empty_pinecone_results():
    retriever = Retriever(vector_store=MockVectorStore([]), embedder=MockEmbedder())
    resp = retriever.retrieve("unfound query")

    assert resp.has_context is False
    assert len(resp.results) == 0
    assert resp.context.context_text == ""
    assert resp.debug.candidates_retrieved == 0


def test_retriever_duplicate_removal():
    duplicates = [
        {"id": "chunk_01", "score": 0.90, "content": "Duplicate content", "metadata": {"chunk_id": "chunk_01", "document_id": "doc_1"}},
        {"id": "chunk_01", "score": 0.85, "content": "Duplicate content", "metadata": {"chunk_id": "chunk_01", "document_id": "doc_1"}},
        {"id": "chunk_02", "score": 0.80, "content": "Distinct chunk same doc", "metadata": {"chunk_id": "chunk_02", "document_id": "doc_1"}},
    ]
    retriever = Retriever(vector_store=MockVectorStore(duplicates), embedder=MockEmbedder())
    resp = retriever.retrieve("query")

    # Should retain 2 chunks: chunk_01 (higher score 0.90 kept) and chunk_02
    assert len(resp.results) == 2
    assert resp.results[0].chunk_id == "chunk_01"
    assert resp.results[0].score == 0.90
    assert resp.results[1].chunk_id == "chunk_02"
    assert resp.debug.removed_duplicates == 1


def test_retriever_metadata_preservation():
    retriever = Retriever(vector_store=MockVectorStore(sample_candidates()), embedder=MockEmbedder())
    resp = retriever.retrieve("heart")

    first = resp.results[0]
    assert first.chunk_id == "chunk_cardio_01"
    assert first.document_id == "doc_01"
    assert first.section == "Cardiology"
    assert first.url == "https://www.psmissionhospital.org/cardiology"
    assert first.title == "Cardiology Care"
    assert first.source == "website"


def test_context_builder_formatting_and_demarcation():
    builder = ContextBuilder()
    items = [
        RetrievalChunkItem(
            rank=1,
            chunk_id="c1",
            document_id="d1",
            score=0.9,
            content="Cardiology provides cardiology care.",
            section="Cardiology",
            url="https://psmissionhospital.org/cardio",
            title="Cardiology Dept",
        ),
        RetrievalChunkItem(
            rank=2,
            chunk_id="c2",
            document_id="d2",
            score=0.8,
            content="Emergency 24x7 ambulance services.",
            section="Emergency",
            url="https://psmissionhospital.org/emergency",
            title="Emergency Care",
        ),
    ]

    res = builder.build_context(items)

    assert res.has_context is True
    assert "--- SOURCE 1 ---" in res.context_text
    assert "Section: Cardiology" in res.context_text
    assert "URL: https://psmissionhospital.org/cardio" in res.context_text
    assert "Cardiology provides cardiology care." in res.context_text

    assert "--- SOURCE 2 ---" in res.context_text
    assert "Section: Emergency" in res.context_text
    assert "Emergency 24x7 ambulance services." in res.context_text
    assert res.total_chunks == 2


def test_context_builder_character_budget_cutoff():
    builder = ContextBuilder()
    # Create two large chunks
    items = [
        RetrievalChunkItem(
            rank=1,
            chunk_id="c1",
            document_id="d1",
            score=0.9,
            content="A" * 200,
            section="Dept 1",
            url="url1",
        ),
        RetrievalChunkItem(
            rank=2,
            chunk_id="c2",
            document_id="d2",
            score=0.8,
            content="B" * 200,
            section="Dept 2",
            url="url2",
        ),
    ]

    # Set character budget to 260 chars: allows first chunk, cuts off second chunk
    res = builder.build_context(items, max_characters=260)

    assert res.has_context is True
    assert res.total_chunks == 1
    assert "--- SOURCE 1 ---" in res.context_text
    assert "--- SOURCE 2 ---" not in res.context_text


def test_api_retrieval_preview_endpoint(monkeypatch):
    # Mock Retriever in API endpoint
    from app.retrieval.retriever import Retriever
    client = TestClient(app)

    def mock_retrieve(self, **kwargs):
        from app.models.schemas import ContextResult, RetrievalChunkItem, RetrievalDebugInfo, RetrievalPreviewResponse
        return RetrievalPreviewResponse(
            query=kwargs.get("query", ""),
            has_context=True,
            results=[
                RetrievalChunkItem(
                    rank=1,
                    chunk_id="chunk_test_01",
                    document_id="doc_test",
                    score=0.85,
                    content="Pediatrics cares for children.",
                    section="Pediatrics",
                    url="https://psmissionhospital.org/pediatrics",
                    title="Pediatrics",
                    source="website",
                )
            ],
            context=ContextResult(
                has_context=True,
                context_text="--- SOURCE 1 ---\nSection: Pediatrics\n\nPediatrics cares for children.",
                total_chunks=1,
                total_characters=68,
            ),
            debug=RetrievalDebugInfo(
                pinecone_top_k=5,
                candidates_retrieved=1,
                removed_by_threshold=0,
                removed_duplicates=0,
                final_chunks_count=1,
                context_character_count=68,
            ),
            retrieval_duration_seconds=0.015,
        )

    monkeypatch.setattr(Retriever, "retrieve", mock_retrieve)

    resp = client.post("/api/retrieval/preview", json={"query": "child doctors", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()

    assert data["query"] == "child doctors"
    assert data["has_context"] is True
    assert len(data["results"]) == 1
    assert data["results"][0]["section"] == "Pediatrics"
    assert "--- SOURCE 1 ---" in data["context"]["context_text"]
    assert data["debug"]["final_chunks_count"] == 1


def test_api_retrieval_preview_validation_error():
    client = TestClient(app)
    # Empty query should return 422 Unprocessable Entity
    resp = client.post("/api/retrieval/preview", json={"query": ""})
    assert resp.status_code in (400, 422)
