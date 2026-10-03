import pytest
from app.ingestion.chunker import DocumentChunker, Chunk
from app.ingestion.document_parser import CleanedDocument


def test_empty_document():
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=80)
    empty_doc = CleanedDocument(
        url="https://psmissionhospital.org/empty",
        title="Empty Page",
        content="",
        metadata={"source": "website", "headings": []},
    )

    chunks = chunker.chunk_document(empty_doc)
    assert chunks == []


def test_document_smaller_than_chunk_size():
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=150)
    short_text = "P S Mission Hospital was established to offer compassionate, affordable healthcare."
    doc = CleanedDocument(
        url="https://psmissionhospital.org/about",
        title="About Us",
        content=short_text,
        metadata={"source": "website", "headings": []},
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content == short_text
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["total_chunks"] == 1


def test_heading_and_paragraph_relationship():
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=100)
    content = (
        "Cardiology Department\n\n"
        "The department of Cardiology at P S Mission Hospital provides 24-hour cardiac care, "
        "ECG diagnostics, and post-operative recovery monitoring."
    )
    doc = CleanedDocument(
        url="https://psmissionhospital.org/cardiology",
        title="Cardiology",
        content=content,
        metadata={"source": "website", "headings": ["Cardiology Department"]},
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    # Both the heading and paragraph must be kept together
    assert "Cardiology Department" in chunks[0].content
    assert "24-hour cardiac care" in chunks[0].content
    assert chunks[0].metadata["section"] == "Cardiology Department"


def test_multiple_sections_division():
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=80)
    content = (
        "Cardiology\n\n"
        "Advanced cardiovascular consultation and heart rhythm monitoring.\n\n"
        "Visiting Hours\n\n"
        "General wards: 9 AM - 5 PM daily. Intensive care unit: 10 AM - 12 PM.\n\n"
        "Emergency Services\n\n"
        "24/7 casualty trauma center with ambulance helpline: +91 484 2700000."
    )
    headings = ["Cardiology", "Visiting Hours", "Emergency Services"]
    doc = CleanedDocument(
        url="https://psmissionhospital.org/services",
        title="Hospital Services",
        content=content,
        metadata={"source": "website", "headings": headings},
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 3

    sections_found = [c.metadata["section"] for c in chunks]
    assert "Cardiology" in sections_found
    assert "Visiting Hours" in sections_found
    assert "Emergency Services" in sections_found


def test_long_document_and_overlap():
    # Set small chunk size and generous overlap to enforce splitting and overlap verification
    chunker = DocumentChunker(chunk_size=160, chunk_overlap=50)
    long_content = (
        "P S Mission Hospital has been serving Maradu and Kochi with exemplary healthcare services for decades. "
        "The hospital offers dedicated departments in General Medicine, Pediatrics, Orthopedics, and Gynecology. "
        "Modern diagnostic laboratories and 24-hour pharmacy services are available on the ground floor. "
        "Our qualified medical consultants provide specialized treatments with utmost compassion and diligence."
    )
    doc = CleanedDocument(
        url="https://psmissionhospital.org/overview",
        title="Overview",
        content=long_content,
        metadata={"source": "website", "headings": []},
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1

    # Verify overlap: chunk 1 should share content with chunk 0
    chunk_0_text = chunks[0].content
    chunk_1_text = chunks[1].content

    # Find common words/tokens between the tail of chunk 0 and head of chunk 1
    chunk_0_words = set(chunk_0_text.split()[-6:])
    chunk_1_words = set(chunk_1_text.split()[:8])
    overlap_words = chunk_0_words.intersection(chunk_1_words)
    assert len(overlap_words) > 0, f"Expected overlapping words between chunks, got:\nChunk 0 tail: {chunk_0_text[-60:]}\nChunk 1 head: {chunk_1_text[:60]}"


def test_deterministic_chunk_ids():
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=60)
    content = (
        "Outpatient Department (OPD)\n\n"
        "OPD registrations begin at 8:00 AM every weekday. "
        "Patients are requested to bring their prior medical records and doctor prescriptions. "
        "Consultation tokens are allocated at Counter 1 and Counter 2."
    )
    doc = CleanedDocument(
        url="https://psmissionhospital.org/opd",
        title="OPD Timings",
        content=content,
        metadata={"source": "website", "headings": ["Outpatient Department (OPD)"]},
    )

    # Process first time
    chunks_run1 = chunker.chunk_document(doc)
    # Process second time
    chunks_run2 = chunker.chunk_document(doc)

    assert len(chunks_run1) == len(chunks_run2)
    for c1, c2 in zip(chunks_run1, chunks_run2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.document_id == c2.document_id
        assert c1.content == c2.content
        assert c1.metadata == c2.metadata


def test_long_section_heading_prefix_retention():
    """Verify that when a section is long and split across multiple chunks,
    every sub-chunk retains the section heading prefix and metadata."""
    chunker = DocumentChunker(chunk_size=120, chunk_overlap=30)
    content = (
        "Cardiology\n\n"
        "The Cardiology department provides comprehensive heart care and interventional diagnosis. "
        "The department has advanced catheterization labs and coronary care units. "
        "Post-operative inpatient monitoring is staffed 24/7 by specialized nurses."
    )
    doc = CleanedDocument(
        url="https://www.psmissionhospital.org/departments/cardiology",
        title="Departments",
        content=content,
        metadata={"source": "website", "headings": ["Cardiology"]},
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1

    for idx, c in enumerate(chunks):
        assert c.content.startswith("Cardiology\n")
        assert c.metadata["section"] == "Cardiology"
        assert c.metadata["chunk_index"] == idx
        assert c.metadata["url"] == "https://www.psmissionhospital.org/departments/cardiology"
        assert c.metadata["title"] == "Departments"
        assert c.metadata["source"] == "website"
        assert len(c.chunk_id) > 0
        assert len(c.document_id) > 0
