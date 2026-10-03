import pytest
import numpy as np
from app.ingestion.embedder import DocumentEmbedder, EmbeddedChunk
from app.ingestion.chunker import Chunk
from app.ingestion.similarity import cosine_similarity


@pytest.fixture(scope="module")
def embedder():
    """Initializes and reuses the DocumentEmbedder instance across the test module."""
    return DocumentEmbedder()


def test_model_loads_successfully(embedder):
    """1. Test that the embedding model loads and reports dimension."""
    dim = embedder.embedding_dimension()
    assert isinstance(dim, int)
    assert dim > 0
    # For all-MiniLM-L6-v2, dimension is standard 384
    assert dim == 384


def test_single_text_produces_embedding(embedder):
    """2. Test that a single text produces an embedding with matching dimension."""
    text = "Cardiology department provides specialized outpatient consultations."
    vec = embedder.embed_text(text)
    assert isinstance(vec, list)
    assert len(vec) == embedder.embedding_dimension()
    assert all(isinstance(v, float) for v in vec)


def test_multiple_texts_produce_correct_number_of_embeddings(embedder):
    """3. Test that multiple texts produce the exact expected number of embeddings."""
    texts = [
        "Cardiology department provides treatment for heart diseases.",
        "Emergency casualty unit is open 24/7.",
        "Pediatrics OPD is located in Block B.",
    ]
    vecs = embedder.embed_texts(texts)
    assert len(vecs) == 3
    for v in vecs:
        assert len(v) == embedder.embedding_dimension()


def test_embedding_dimension_is_consistent(embedder):
    """4. Test that embedding dimension is always consistent across varied inputs."""
    dim = embedder.embedding_dimension()
    short_vec = embedder.embed_text("Short text.")
    long_vec = embedder.embed_text(
        "P S Mission Hospital has been serving the public with exemplary medical care, "
        "featuring specialized intensive care units, modern diagnostic pathology, and round-the-clock pharmacy."
    )
    assert len(short_vec) == dim
    assert len(long_vec) == dim


def test_empty_text_handling(embedder):
    """5. Test that empty text returns a valid zero-vector of correct dimension."""
    dim = embedder.embedding_dimension()
    empty_vec = embedder.embed_text("")
    whitespace_vec = embedder.embed_text("   \n\t  ")

    assert len(empty_vec) == dim
    assert all(v == 0.0 for v in empty_vec)
    assert len(whitespace_vec) == dim
    assert all(v == 0.0 for v in whitespace_vec)

    # Empty list of texts
    empty_batch = embedder.embed_texts([])
    assert empty_batch == []


def test_batch_processing(embedder):
    """6. Test batch processing with explicit batch_size configuration."""
    custom_embedder = DocumentEmbedder(batch_size=2)
    texts = [
        f"Hospital service description paragraph number {i}"
        for i in range(7)
    ]
    batch_vecs = custom_embedder.embed_texts(texts)
    assert len(batch_vecs) == 7
    for v in batch_vecs:
        assert len(v) == custom_embedder.embedding_dimension()


def test_deterministic_embedding_consistency(embedder):
    """7. Test that the same input produces identical/consistent vector values."""
    text = "General surgery and laparoscopic procedures at P S Mission Hospital."
    vec1 = embedder.embed_text(text)
    vec2 = embedder.embed_text(text)

    assert len(vec1) == len(vec2)
    for v1, v2 in zip(vec1, vec2):
        assert pytest.approx(v1, abs=1e-6) == v2


def test_different_texts_produce_different_vectors(embedder):
    """8. Test that distinct texts produce distinctly different vectors."""
    text_cardio = "Cardiology and heart coronary catheterization."
    text_pediatrics = "Pediatric child immunizations and neonatal care."

    vec_cardio = embedder.embed_text(text_cardio)
    vec_pediatrics = embedder.embed_text(text_pediatrics)

    assert vec_cardio != vec_pediatrics
    similarity = cosine_similarity(vec_cardio, vec_pediatrics)
    assert similarity < 0.95


def test_chunk_metadata_is_preserved(embedder):
    """9. Test that embed_chunks preserves all original chunk identifiers and metadata."""
    sample_chunk = Chunk(
        chunk_id="chunk_test_123",
        document_id="doc_test_456",
        content="Cardiology\nThe department provides cardiovascular diagnosis.",
        metadata={
            "url": "https://www.psmissionhospital.org/departments/cardiology",
            "title": "Departments",
            "section": "Cardiology",
            "source": "website",
            "chunk_index": 0,
            "total_chunks": 1,
        },
    )

    embedded_chunks = embedder.embed_chunks([sample_chunk])
    assert len(embedded_chunks) == 1

    ec = embedded_chunks[0]
    assert ec.chunk_id == "chunk_test_123"
    assert ec.document_id == "doc_test_456"
    assert ec.content == "Cardiology\nThe department provides cardiovascular diagnosis."
    assert ec.metadata["section"] == "Cardiology"
    assert ec.metadata["chunk_index"] == 0
    assert ec.metadata["source"] == "website"
    assert len(ec.embedding) == embedder.embedding_dimension()


def test_embedding_output_numpy_shape(embedder):
    """10. Test that requesting numpy format produces correct array shape."""
    texts = ["Sample A", "Sample B", "Sample C"]
    arr = embedder.embed_texts(texts, as_numpy=True)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, embedder.embedding_dimension())


def test_semantic_sanity_test(embedder):
    """11. SEMANTIC SANITY TEST:
    Verify that semantically related texts have higher cosine similarity than unrelated texts.
    
    Text A: Cardiology heart diseases
    Text B: Heart-related conditions in cardiology
    Text C: Orthopaedics bone and joint problems
    
    Must verify: similarity(A, B) > similarity(A, C)
    """
    text_a = "Cardiology department provides treatment for heart diseases."
    text_b = "Heart-related conditions are treated by the cardiology department."
    text_c = "Orthopaedics provides treatment for bone and joint problems."

    vec_a = embedder.embed_text(text_a)
    vec_b = embedder.embed_text(text_b)
    vec_c = embedder.embed_text(text_c)

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    # Both Text A and Text B discuss cardiology and heart diseases, phrased differently.
    # Text C discusses orthopaedics and bone/joints.
    assert sim_ab > sim_ac, (
        f"Expected similarity(A, B) [{sim_ab:.4f}] > similarity(A, C) [{sim_ac:.4f}]"
    )
    # Typically sim_ab is ~0.80+ while sim_ac is ~0.40-0.55
    assert sim_ab > 0.70
