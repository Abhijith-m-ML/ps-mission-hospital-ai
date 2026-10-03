import math
import os
import pytest
from unittest.mock import MagicMock

from app.ingestion.chunker import Chunk
from app.ingestion.embedder import DocumentEmbedder
from app.retrieval.vector_store import PineconeVectorStore
from app.services.indexing_service import IndexingService
from app.services.search_service import SearchService


class MockPineconeIndex:
    """In-memory simulator of Pinecone Index with real cosine vector similarity."""

    def __init__(self, name: str, dimension: int = 384, metric: str = "cosine"):
        self.name = name
        self.dimension = dimension
        self.metric = metric
        # Storage mapped by: {namespace: {id: {"values": [...], "metadata": {...}}}}
        self._namespaces = {}

    def upsert(self, vectors, namespace="default"):
        if namespace not in self._namespaces:
            self._namespaces[namespace] = {}
        for item in vectors:
            v_id = item["id"]
            # Idempotent overwrite/upsert
            self._namespaces[namespace][v_id] = {
                "values": list(item["values"]),
                "metadata": dict(item.get("metadata", {})),
            }
        return {"upserted_count": len(vectors)}

    def query(self, vector, top_k=5, include_metadata=True, namespace="default", filter=None):
        ns_records = self._namespaces.get(namespace, {})
        if not ns_records:
            return {"matches": []}

        scored = []
        for v_id, rec in ns_records.items():
            meta = rec["metadata"]
            # Apply metadata filter if present
            if filter:
                matched = True
                for k, v in filter.items():
                    if meta.get(k) != v:
                        matched = False
                        break
                if not matched:
                    continue

            # Compute cosine similarity
            v_stored = rec["values"]
            dot = sum(a * b for a, b in zip(vector, v_stored))
            norm_q = math.sqrt(sum(a * a for a in vector))
            norm_s = math.sqrt(sum(b * b for b in v_stored))
            sim = dot / (norm_q * norm_s) if (norm_q > 0 and norm_s > 0) else 0.0

            match_obj = MagicMock()
            match_obj.id = v_id
            match_obj.score = float(sim)
            match_obj.metadata = dict(meta) if include_metadata else {}
            scored.append(match_obj)

        scored.sort(key=lambda m: m.score, reverse=True)
        return {"matches": scored[:top_k]}

    def describe_index_stats(self):
        total = sum(len(records) for records in self._namespaces.values())
        ns_stats = {
            ns: MagicMock(vector_count=len(records))
            for ns, records in self._namespaces.items()
        }
        mock_stats = MagicMock()
        mock_stats.dimension = self.dimension
        mock_stats.total_vector_count = total
        mock_stats.namespaces = ns_stats
        return mock_stats


class MockIndexSummary:
    def __init__(self, name: str, dimension: int = 384, metric: str = "cosine"):
        self.name = name
        self.dimension = dimension
        self.metric = metric


class MockPineconeClient:
    """In-memory simulator of Pinecone top-level client."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._indexes = {}

    def list_indexes(self):
        return [MockIndexSummary(name=idx.name, dimension=idx.dimension, metric=idx.metric) for idx in self._indexes.values()]

    def describe_index(self, name):
        if name not in self._indexes:
            raise ValueError(f"Index '{name}' not found.")
        idx = self._indexes[name]
        return MockIndexSummary(name=idx.name, dimension=idx.dimension, metric=idx.metric)

    def create_index(self, name, dimension, metric="cosine", spec=None):
        if name in self._indexes:
            raise ValueError(f"Index '{name}' already exists.")
        self._indexes[name] = MockPineconeIndex(name=name, dimension=dimension, metric=metric)

    def Index(self, name):
        if name not in self._indexes:
            self._indexes[name] = MockPineconeIndex(name=name, dimension=384, metric="cosine")
        return self._indexes[name]


@pytest.fixture
def mock_vector_store():
    """Provides a PineconeVectorStore wired to an in-memory MockPineconeClient."""
    mock_client = MockPineconeClient(api_key="pcsk_test_mock_key_12345")
    mock_client.create_index(name="test-hospital-ai", dimension=384, metric="cosine")
    store = PineconeVectorStore(
        api_key="pcsk_test_mock_key_12345",
        index_name="test-hospital-ai",
        namespace="test_hospital",
        client=mock_client,
    )
    return store


# 1. Configuration Validation
def test_pinecone_missing_api_key_validation():
    """Verify that vector store raises ValueError when API key is missing."""
    store = PineconeVectorStore(api_key="", client=None)
    with pytest.raises(ValueError, match="Pinecone API key is missing"):
        _ = store.client


# 2. Index Creation and Discovery
def test_index_creation_and_discovery(mock_vector_store):
    """Verify index creation when absent and verification when present."""
    # Already created in fixture
    created = mock_vector_store.create_index_if_not_exists("test-hospital-ai", dimension=384, metric="cosine")
    assert created is False  # Already exists and verified

    # Create new index
    new_created = mock_vector_store.create_index_if_not_exists("new-hospital-idx", dimension=384, metric="cosine")
    assert new_created is True


# 3 & 4. Dimension and Metric Verification
def test_index_dimension_and_metric_mismatch(mock_vector_store):
    """Verify error is raised if index dimension or metric doesn't match expected settings."""
    # Dimension mismatch
    with pytest.raises(ValueError, match="dimension mismatch"):
        mock_vector_store.create_index_if_not_exists("test-hospital-ai", dimension=1536)

    # Metric mismatch
    with pytest.raises(ValueError, match="metric mismatch"):
        mock_vector_store.create_index_if_not_exists("test-hospital-ai", dimension=384, metric="euclidean")


# 5. Namespace Handling
def test_namespace_routing(mock_vector_store):
    """Verify vectors are isolated by namespace."""
    vec = [0.1] * 384
    mock_vector_store.upsert_vectors(
        vectors=[{"id": "doc1", "values": vec, "metadata": {"section": "Cardio"}}],
        namespace="hospital_a",
    )
    mock_vector_store.upsert_vectors(
        vectors=[{"id": "doc2", "values": vec, "metadata": {"section": "Neuro"}}],
        namespace="hospital_b",
    )

    res_a = mock_vector_store.search(vec, top_k=5, namespace="hospital_a")
    assert len(res_a) == 1
    assert res_a[0]["id"] == "doc1"

    res_b = mock_vector_store.search(vec, top_k=5, namespace="hospital_b")
    assert len(res_b) == 1
    assert res_b[0]["id"] == "doc2"


# 6. Deterministic Vector IDs
def test_deterministic_chunk_ids_used_as_vector_ids(mock_vector_store):
    """Verify the original chunk_id is directly used as the Pinecone vector id."""
    chunk_id = "hospital_page_abc123_chunk_01"
    vec = [0.2] * 384
    mock_vector_store.upsert_vectors(
        vectors=[{"id": chunk_id, "values": vec, "metadata": {"title": "Test"}}],
    )
    matches = mock_vector_store.search(vec, top_k=1)
    assert len(matches) == 1
    assert matches[0]["id"] == chunk_id


# 7. Payload / Metadata Preservation
def test_metadata_preservation(mock_vector_store):
    """Verify that complete content, url, title, and section are preserved in metadata."""
    chunk_id = "chunk_cardio_complete"
    vec = [0.05] * 384
    metadata = {
        "chunk_id": chunk_id,
        "document_id": "doc_cardio_1",
        "content": "Cardiology\nThe department provides comprehensive heart diagnosis.",
        "url": "https://www.psmissionhospital.org/departments/cardiology",
        "title": "Departments",
        "section": "Cardiology",
        "source": "website",
    }
    mock_vector_store.upsert_vectors(
        vectors=[{"id": chunk_id, "values": vec, "metadata": metadata}],
    )

    results = mock_vector_store.search(vec, top_k=1)
    assert len(results) == 1
    match = results[0]
    assert match["id"] == chunk_id
    assert match["section"] == "Cardiology"
    assert match["url"] == "https://www.psmissionhospital.org/departments/cardiology"
    assert match["title"] == "Departments"
    assert "comprehensive heart diagnosis" in match["content"]


# 8 & 9. Batch Upsert & Idempotency
def test_batch_upsert_and_idempotency(mock_vector_store):
    """Verify batch upserts and that re-indexing the same chunk updates rather than duplicates."""
    vectors = [
        {"id": f"chunk_batch_{i}", "values": [0.01 * i] * 384, "metadata": {"count": i}}
        for i in range(15)
    ]
    upserted_1 = mock_vector_store.upsert_vectors(vectors, batch_size=5)
    assert upserted_1 == 15

    stats_1 = mock_vector_store.get_index_stats()
    assert stats_1["namespace_vector_count"] == 15

    # Re-upsert with updated metadata (idempotency check)
    updated_vectors = [
        {"id": f"chunk_batch_{i}", "values": [0.02 * i] * 384, "metadata": {"count": i * 10}}
        for i in range(15)
    ]
    upserted_2 = mock_vector_store.upsert_vectors(updated_vectors, batch_size=5)
    assert upserted_2 == 15

    stats_2 = mock_vector_store.get_index_stats()
    # Count must remain 15, not 30!
    assert stats_2["namespace_vector_count"] == 15


# 10 & 11. Vector Search and Top-K
def test_vector_search_top_k(mock_vector_store):
    """Verify search ranking and top-K limits."""
    target_vec = [1.0, 0.0] + [0.0] * 382
    close_vec = [0.8, 0.6] + [0.0] * 382
    far_vec = [0.0, 1.0] + [0.0] * 382

    mock_vector_store.upsert_vectors(
        vectors=[
            {"id": "c_close", "values": close_vec, "metadata": {"name": "close"}},
            {"id": "c_far", "values": far_vec, "metadata": {"name": "far"}},
        ]
    )

    res_1 = mock_vector_store.search(target_vec, top_k=1)
    assert len(res_1) == 1
    assert res_1[0]["id"] == "c_close"
    assert res_1[0]["score"] > 0.75

    res_all = mock_vector_store.search(target_vec, top_k=5)
    assert len(res_all) == 2
    assert res_all[0]["score"] > res_all[1]["score"]



# 12 & 13. Edge Cases: Empty Query & Empty Search
def test_search_edge_cases(mock_vector_store):
    """Verify empty query error and empty index handling."""
    with pytest.raises(ValueError, match="Query vector cannot be empty"):
        mock_vector_store.search([])

    # Search in non-existent namespace returns empty list
    empty_res = mock_vector_store.search([0.1] * 384, namespace="empty_ns")
    assert empty_res == []


# 17. Metadata Filtering
def test_metadata_filtering(mock_vector_store):
    """Verify optional metadata filter parameter."""
    vec = [0.3] * 384
    mock_vector_store.upsert_vectors(
        vectors=[
            {"id": "c1", "values": vec, "metadata": {"section": "Cardiology"}},
            {"id": "c2", "values": vec, "metadata": {"section": "Pediatrics"}},
        ]
    )

    # Search filtered by Cardiology
    res_cardio = mock_vector_store.search(vec, top_k=5, filter_dict={"section": "Cardiology"})
    assert len(res_cardio) == 1
    assert res_cardio[0]["id"] == "c1"
    assert res_cardio[0]["section"] == "Cardiology"


# 16. Semantic Search Test with Embedder
def test_semantic_search_with_embedder(mock_vector_store):
    """Verify end-to-end semantic retrieval:
    - Query 'What department treats heart problems?' retrieves Cardiology.
    - Query 'What services are available for children?' retrieves Pediatrics.
    """
    embedder = DocumentEmbedder()
    indexing_service = IndexingService(vector_store=mock_vector_store, embedder=embedder)
    search_service = SearchService(vector_store=mock_vector_store, embedder=embedder)

    chunks = [
        Chunk(
            chunk_id="chunk_cardio",
            document_id="doc_hospital",
            content="Cardiology\nThe department of Cardiology provides 24-hour cardiac care, treatment for heart attacks, and heart rhythm management.",
            metadata={
                "url": "https://www.psmissionhospital.org/departments/cardiology",
                "title": "Departments",
                "section": "Cardiology",
                "source": "website",
            },
        ),
        Chunk(
            chunk_id="chunk_pediatrics",
            document_id="doc_hospital",
            content="Pediatrics\nThe department of Pediatrics provides specialized care for children, newborn neonatal ICU, and childhood vaccinations.",
            metadata={
                "url": "https://www.psmissionhospital.org/departments/pediatrics",
                "title": "Departments",
                "section": "Pediatrics",
                "source": "website",
            },
        ),
        Chunk(
            chunk_id="chunk_ortho",
            document_id="doc_hospital",
            content="Orthopaedics\nThe department of Orthopaedics treats bone fractures, joint replacements, and spine conditions.",
            metadata={
                "url": "https://www.psmissionhospital.org/departments/orthopaedics",
                "title": "Departments",
                "section": "Orthopaedics",
                "source": "website",
            },
        ),
    ]

    indexed_count = indexing_service.index_chunks(chunks)
    assert indexed_count == 3

    # Query 1: Heart problems -> Cardiology
    res_heart = search_service.search_query("What department treats heart problems?", top_k=3)
    assert res_heart["total_found"] == 3
    top_hit_heart = res_heart["results"][0]
    assert top_hit_heart["section"] == "Cardiology"
    assert top_hit_heart["id"] == "chunk_cardio"
    assert top_hit_heart["score"] > 0.60

    # Query 2: Children services -> Pediatrics
    res_children = search_service.search_query("What services are available for children?", top_k=3)
    assert res_children["total_found"] == 3
    top_hit_children = res_children["results"][0]
    assert top_hit_children["section"] == "Pediatrics"
    assert top_hit_children["id"] == "chunk_pediatrics"
    assert top_hit_children["score"] > 0.40
