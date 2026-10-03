"""Unit and integration tests for source citation selection, metadata mapping,
deduplication, and medical safety.
"""
import json
import pytest
from app.models.schemas import ChatMessage, RetrievalChunkItem, SourceItem
from app.rag.llm import BaseLLMProvider
from app.rag.prompts import HOSPITAL_SYSTEM_INSTRUCTION, build_grounded_conversational_prompt
from app.rag.rag_service import (
    RAGService,
    deduplicate_sources,
    format_doctor_display_name,
    get_source_priority,
    parse_llm_response,
)
from app.retrieval.entity_store import HospitalEntityStore
from app.retrieval.intent_analyzer import IntentAnalyzer, QueryIntent
from app.retrieval.structured_retriever import StructuredRetriever


class MockLLM(BaseLLMProvider):
    """Predictable mock LLM provider for testing citation handling."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_system_instruction = None
        self.last_user_message = None

    def generate(self, system_instruction: str, user_message: str) -> str:
        self.last_system_instruction = system_instruction
        self.last_user_message = user_message
        return self.response_text


@pytest.fixture
def entity_store():
    return HospitalEntityStore()


@pytest.fixture
def structured_retriever(entity_store):
    return StructuredRetriever(entity_store=entity_store)


# ==============================================================================
# 1. TEST CASE 1: "I have severe headaches. Which doctor should I see?"
# ==============================================================================
def test_case_1_headache_neurology_citation(structured_retriever):
    """Verifies that asking about severe headaches routes to Neurology / Dr. John J Vaidya
    and only cites the relevant Neurology doctor source."""
    analyzer = IntentAnalyzer(structured_retriever.entity_store)
    intent = analyzer.analyze("I have severe headaches. Which doctor should I see in this hospital?")

    assert intent.intent == QueryIntent.DOCTOR_SEARCH
    assert intent.department == "Neurology"

    # Retrieval should fetch only Dr. John J Vaidya
    retrieval_res = structured_retriever.retrieve("I have severe headaches. Which doctor should I see in this hospital?")
    assert retrieval_res.has_context is True
    assert len(retrieval_res.results) == 1
    assert "vaidya" in retrieval_res.results[0].chunk_id.lower()

    # Mock Gemini returning JSON with that chunk ID
    mock_json = (
        '{\n'
        '  "answer": "For severe headaches, you may consult Dr. John J Vaidya in the Neurology department. OP consultations are on Tuesday and Thursday from 4:30 PM to 6:30 PM. Please note this is general information and not a medical diagnosis. If you experience sudden, severe head pain, please seek emergency care immediately.",\n'
        f'  "source_ids": ["{retrieval_res.results[0].chunk_id}"]\n'
        '}'
    )
    rag = RAGService(retriever=structured_retriever, llm_provider=MockLLM(mock_json))
    response = rag.answer_question("I have severe headaches. Which doctor should I see in this hospital?")

    # Verify only 1 source is cited
    assert len(response.sources) == 1
    src = response.sources[0]
    assert src.doctor_name is not None
    assert "vaidya" in src.doctor_name.lower()
    assert src.department == "Neurology"
    assert "neurology" in src.url.lower()


# ==============================================================================
# 2. TEST CASE 2: "Who are the doctors in Cardiology?"
# ==============================================================================
def test_case_2_cardiology_doctors_citation(structured_retriever):
    """Verifies that asking about Cardiology doctors retrieves and cites ONLY Cardiology doctors."""
    retrieval_res = structured_retriever.retrieve("Who are the doctors in Cardiology?")
    assert retrieval_res.has_context is True

    # Ensure all retrieved records are Cardiology doctors
    for item in retrieval_res.results:
        assert item.metadata.get("content_type") == "doctor"
        assert item.section == "Cardiology"

    doc_ids = [item.chunk_id for item in retrieval_res.results]
    mock_json = (
        '{\n'
        '  "answer": "The Cardiology department has Dr. Sr. Annie Sheela and Dr. Sudheer available for consultations.",\n'
        f'  "source_ids": {json.dumps(doc_ids)}\n'
        '}'
    )
    rag = RAGService(retriever=structured_retriever, llm_provider=MockLLM(mock_json))
    response = rag.answer_question("Who are the doctors in Cardiology?")

    # Should cite only Cardiology doctors (no unrelated departments)
    assert len(response.sources) >= 1
    for src in response.sources:
        assert src.content_type == "doctor"
        assert src.department == "Cardiology"
        assert src.doctor_name is not None


# ==============================================================================
# 3. TEST CASE 3: "What facilities are available in Paediatrics?"
# ==============================================================================
def test_case_3_paediatrics_facilities_citation(structured_retriever):
    """Verifies that asking about Paediatrics facilities returns only Paediatrics facility sources
    and deduplicates duplicate department URLs."""
    retrieval_res = structured_retriever.retrieve("What facilities are available in Paediatrics?")
    assert retrieval_res.has_context is True
    assert len(retrieval_res.results) > 0

    chunk_ids = [c.chunk_id for c in retrieval_res.results]
    mock_json = (
        '{\n'
        '  "answer": "Paediatrics & Neonatology offers NICU facilities, radiant warmers, continuous multi-monitors, and immunization services.",\n'
        f'  "source_ids": {json.dumps(chunk_ids)}\n'
        '}'
    )
    rag = RAGService(retriever=structured_retriever, llm_provider=MockLLM(mock_json))
    response = rag.answer_question("What facilities are available in Paediatrics?")

    # Because all facility chunks share the same department URL, URL deduplication
    # ensures only ONE deduplicated source card is returned
    assert len(response.sources) == 1
    assert "paediatrics" in response.sources[0].url.lower()


# ==============================================================================
# 4. TEST CASE 4: "Tell me about the hospital."
# ==============================================================================
def test_case_4_general_hospital_citation():
    """Verifies that asking general hospital questions returns acceptable general source citations."""
    general_chunk = RetrievalChunkItem(
        rank=1,
        chunk_id="chunk_hospital_overview_01",
        document_id="doc_hospital_overview",
        score=0.92,
        content="P.S. Mission Hospital is a renowned multi-speciality hospital in Maradu, Cochin, offering 24/7 emergency care and specialized clinics.",
        section="General",
        url="https://www.psmissionhospital.org/",
        title="About P.S. Mission Hospital",
        source="website",
        metadata={"content_type": "webpage"},
    )

    class MockGeneralRetriever:
        def retrieve(self, *args, **kwargs):
            from app.models.schemas import ContextResult, RetrievalDebugInfo, RetrievalPreviewResponse
            return RetrievalPreviewResponse(
                query="Tell me about the hospital",
                has_context=True,
                results=[general_chunk],
                context=ContextResult(has_context=True, context_text=general_chunk.content, total_chunks=1, total_characters=len(general_chunk.content)),
                debug=RetrievalDebugInfo(pinecone_top_k=1, candidates_retrieved=1, removed_by_threshold=0, removed_duplicates=0, final_chunks_count=1, context_character_count=100),
            )

    mock_json = (
        '{\n'
        '  "answer": "P.S. Mission Hospital is a multi-speciality healthcare center located in Maradu, Cochin.",\n'
        '  "source_ids": ["chunk_hospital_overview_01"]\n'
        '}'
    )
    rag = RAGService(retriever=MockGeneralRetriever(), llm_provider=MockLLM(mock_json))
    response = rag.answer_question("Tell me about the hospital.")

    assert len(response.sources) == 1
    assert response.sources[0].chunk_id == "chunk_hospital_overview_01"
    assert response.sources[0].url == "https://www.psmissionhospital.org/"


# ==============================================================================
# 5. TEST CASE 5: "Who is Dr. X?"
# ==============================================================================
def test_case_5_specific_doctor_citation(structured_retriever):
    """Verifies that asking about a specific doctor returns only that doctor's source page."""
    retrieval_res = structured_retriever.retrieve("Who is Dr. Denny P Kuttikkat?")
    assert retrieval_res.has_context is True
    assert len(retrieval_res.results) == 1

    doc_chunk = retrieval_res.results[0]
    assert "denny" in doc_chunk.chunk_id.lower()

    mock_json = (
        '{\n'
        '  "answer": "Dr. Denny P Kuttikkat is a consultant in General Surgery with qualifications MBBS, MS, FRCS.",\n'
        f'  "source_ids": ["{doc_chunk.chunk_id}"]\n'
        '}'
    )
    rag = RAGService(retriever=structured_retriever, llm_provider=MockLLM(mock_json))
    response = rag.answer_question("Who is Dr. Denny P Kuttikkat?")

    assert len(response.sources) == 1
    assert response.sources[0].doctor_name is not None
    assert "denny" in response.sources[0].doctor_name.lower()


# ==============================================================================
# 6. TEST CASE 6: Duplicate Retrieval & Specificity Deduplication
# ==============================================================================
def test_case_6_duplicate_retrieval_and_priority_deduplication():
    """Verifies that:
    1. Duplicate URLs are deduplicated to a single entry.
    2. A doctor page takes precedence over a generic department page sharing the same URL.
    3. Multiple distinct doctors sharing the same URL are both preserved.
    """
    shared_url = "https://www.psmissionhospital.org/category/department/cardiology"

    doctor_1 = SourceItem(
        chunk_id="chunk_doc_annie",
        title="Dr. Sr. Annie Sheela",
        url=shared_url,
        section="Cardiology",
        content_type="doctor",
        doctor_name="Dr. Sr. Annie Sheela",
        department="Cardiology",
    )
    doctor_2 = SourceItem(
        chunk_id="chunk_doc_sudheer",
        title="Dr. Sudheer",
        url=shared_url,
        section="Cardiology",
        content_type="doctor",
        doctor_name="Dr. Sudheer",
        department="Cardiology",
    )
    department_page = SourceItem(
        chunk_id="chunk_dept_cardio",
        title="Cardiology Department",
        url=shared_url,
        section="Cardiology",
        content_type="department",
        department="Cardiology",
    )
    duplicate_department = SourceItem(
        chunk_id="chunk_dept_cardio_dup",
        title="Cardiology Overview",
        url=shared_url,
        section="Cardiology",
        content_type="department",
        department="Cardiology",
    )

    # When both doctors and department pages share the URL:
    deduped = deduplicate_sources([department_page, doctor_1, duplicate_department, doctor_2])

    # The department page should be suppressed in favor of the doctors,
    # and both distinct doctors should be preserved
    assert len(deduped) == 2
    names = {s.doctor_name for s in deduped}
    assert "Dr. Sr. Annie Sheela" in names
    assert "Dr. Sudheer" in names


def test_priority_ordering():
    """Verifies priority: Doctor (1) > Department (2) > Facility (3) > Homepage (4)."""
    doc_src = SourceItem(chunk_id="1", url="http://example.com/doc", content_type="doctor", doctor_name="Dr. A")
    dept_src = SourceItem(chunk_id="2", url="http://example.com/category/department/neuro", content_type="department")
    fac_src = SourceItem(chunk_id="3", url="http://example.com/services/nicu", content_type="facility")
    home_src = SourceItem(chunk_id="4", url="https://www.psmissionhospital.org/", content_type="webpage")

    assert get_source_priority(doc_src) == 1
    assert get_source_priority(dept_src) == 2
    assert get_source_priority(fac_src) == 3
    assert get_source_priority(home_src) == 4


# ==============================================================================
# 7. PARSER TESTS: Robust JSON & Fallback Parsing
# ==============================================================================
def test_parse_llm_response_valid_json():
    raw = '{"answer": "Consult Dr. John J Vaidya.", "source_ids": ["id_1", "id_2"]}'
    ans, ids = parse_llm_response(raw, [])
    assert ans == "Consult Dr. John J Vaidya."
    assert ids == ["id_1", "id_2"]


def test_parse_llm_response_markdown_code_fences():
    raw = '```json\n{"answer": "Consult Dr. John J Vaidya.", "source_ids": ["id_1"]}\n```'
    ans, ids = parse_llm_response(raw, [])
    assert ans == "Consult Dr. John J Vaidya."
    assert ids == ["id_1"]


def test_parse_llm_response_fallback_entity_mention():
    chunk = RetrievalChunkItem(
        rank=1,
        chunk_id="doctor_dr_jhon_j_vaidya",
        document_id="doc_1",
        score=0.99,
        content="Dr. JHON J VAIDYA",
        section="Neurology",
        metadata={"content_type": "doctor", "doctor_name": "Dr. JHON J VAIDYA"},
    )
    raw = "For headaches, please visit Dr. John J Vaidya in Neurology."
    ans, ids = parse_llm_response(raw, [chunk])
    assert ans == raw
    assert ids == ["doctor_dr_jhon_j_vaidya"]


# ==============================================================================
# 8. MEDICAL SAFETY INSTRUCTION VERIFICATION
# ==============================================================================
def test_medical_safety_instructions_in_prompt():
    """Ensures prompts explicitly contain medical safety disclaimers."""
    assert "Do NOT provide medical diagnoses" in HOSPITAL_SYSTEM_INSTRUCTION
    assert "emergency" in HOSPITAL_SYSTEM_INSTRUCTION.lower()
    prompt = build_grounded_conversational_prompt("I have severe headaches", "Context")
    assert "Medical Safety" in prompt
