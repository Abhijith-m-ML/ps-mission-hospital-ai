import pytest
from app.models.schemas import Department, Doctor, Facility
from app.rag.llm import MockLLMProvider
from app.rag.rag_service import RAGService
from app.retrieval.entity_store import HospitalEntityStore
from app.retrieval.intent_analyzer import IntentAnalyzer, QueryIntent
from app.retrieval.structured_retriever import StructuredRetriever


@pytest.fixture
def mock_store(tmp_path):
    """Initializes a populated in-memory HospitalEntityStore for testing."""
    store = HospitalEntityStore(data_path=str(tmp_path / "test_entities.json"))
    store.add_department(
        Department(
            department_name="Paediatrics & Neonatology",
            description="Specialized care for infants, children, and adolescents with 8-bed NICU.",
            source_url="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
        )
    )
    store.add_doctor(
        Doctor(
            doctor_name="Dr. Sr. Jaya Joseph",
            qualification="MBBS, MD",
            department="Paediatrics & Neonatology",
            schedule_text="OP: MONDAY-FRIDAY | 9.30AM-1.00PM & 4.00PM-6.00 PM | SATURDAY | (9.30AM-1.00)",
            image_url="https://www.psmissionhospital.org/wp-content/uploads/2023/08/Dr-Sr-Jaya-Joseph-1.jpg",
            source_url="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
            source_title="Paediatrics - PS Mission Hospital",
        )
    )
    store.add_facility(
        Facility(
            facility_name="Immunization",
            description="Vaccination and child immunization clinic",
            department="Paediatrics & Neonatology",
            source_url="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
        )
    )
    store.add_facility(
        Facility(
            facility_name="Neonatology ICU (NICU)",
            description="8-bed intensive care unit for premature babies",
            department="Paediatrics & Neonatology",
            source_url="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
        )
    )
    return store


class TestIntentAnalyzer:
    """Verifies intent classification across doctor, facility, schedule, and general queries."""

    def test_doctor_search_intent(self, mock_store):
        analyzer = IntentAnalyzer(mock_store)
        res = analyzer.analyze("Which are the doctors available for Paediatrics & Neonatology?")
        assert res.intent == QueryIntent.DOCTOR_SEARCH
        assert res.department == "Paediatrics & Neonatology"

    def test_facility_search_intent(self, mock_store):
        analyzer = IntentAnalyzer(mock_store)
        res = analyzer.analyze("What facilities are available in Paediatrics?")
        assert res.intent == QueryIntent.FACILITY_SEARCH
        assert res.department == "Paediatrics & Neonatology"

    def test_schedule_search_intent(self, mock_store):
        analyzer = IntentAnalyzer(mock_store)
        res = analyzer.analyze("When can I visit Dr. Sr. Jaya Joseph?")
        assert res.intent == QueryIntent.SCHEDULE_SEARCH
        assert res.doctor_name == "Dr. Sr. Jaya Joseph"

    def test_schedule_search_short_name(self, mock_store):
        analyzer = IntentAnalyzer(mock_store)
        res = analyzer.analyze("When can I visit Dr. Jaya Joseph?")
        assert res.intent == QueryIntent.SCHEDULE_SEARCH
        assert res.doctor_name == "Dr. Sr. Jaya Joseph"

    def test_general_rag_fallback(self, mock_store):
        analyzer = IntentAnalyzer(mock_store)
        res = analyzer.analyze("What is the hospital contact address and phone number?")
        assert res.intent == QueryIntent.GENERAL_RAG


class TestStructuredRetriever:
    """Tests retrieval abstraction generating grounded context and structured metadata."""

    def test_retrieve_doctors_by_department(self, mock_store):
        retriever = StructuredRetriever(entity_store=mock_store)
        res = retriever.retrieve("Which are the doctors available for Paediatrics & Neonatology?")

        assert res.has_context is True
        assert len(res.results) == 1
        item = res.results[0]
        assert item.metadata["doctor_name"] == "Dr. Sr. Jaya Joseph"
        assert item.metadata["content_type"] == "doctor"
        assert item.metadata["image_url"].endswith(".jpg")
        assert "MBBS, MD" in res.context.context_text
        assert "Dr. Sr. Jaya Joseph" in res.context.context_text

    def test_retrieve_facilities_by_department(self, mock_store):
        retriever = StructuredRetriever(entity_store=mock_store)
        res = retriever.retrieve("What facilities are available in Paediatrics?")

        assert res.has_context is True
        assert len(res.results) == 2
        fac_names = [item.metadata["facility_name"] for item in res.results]
        assert "Immunization" in fac_names
        assert "Neonatology ICU (NICU)" in fac_names
        assert "Immunization" in res.context.context_text

    def test_retrieve_schedule(self, mock_store):
        retriever = StructuredRetriever(entity_store=mock_store)
        res = retriever.retrieve("When can I visit Dr. Sr. Jaya Joseph?")

        assert res.has_context is True
        assert len(res.results) == 1
        assert "9.30AM-1.00PM" in res.context.context_text


class TestRAGServiceIntegration:
    """Tests end-to-end RAG response generation with structured entities."""

    def test_answer_paediatrics_doctors(self, mock_store):
        mock_llm = MockLLMProvider(
            "The doctor available for Paediatrics & Neonatology is Dr. Sr. Jaya Joseph (MBBS, MD). Her OP timing is Monday to Friday from 9:30 AM to 1:00 PM and 4:00 PM to 6:00 PM, and Saturday from 9:30 AM to 1:00 PM."
        )
        retriever = StructuredRetriever(entity_store=mock_store)
        rag_service = RAGService(retriever=retriever, llm_provider=mock_llm)

        chat_response = rag_service.answer_question("Which are the doctors available for Paediatrics & Neonatology?")

        assert "Dr. Sr. Jaya Joseph" in chat_response.answer
        assert len(chat_response.sources) == 1
        source = chat_response.sources[0]
        assert source.doctor_name == "Dr. Sr. Jaya Joseph"
        assert source.content_type == "doctor"
        assert source.image_url == "https://www.psmissionhospital.org/wp-content/uploads/2023/08/Dr-Sr-Jaya-Joseph-1.jpg"

    def test_answer_paediatrics_facilities(self, mock_store):
        mock_llm = MockLLMProvider(
            "The facilities available in Paediatrics & Neonatology include Immunization services and a specialized Neonatology ICU (NICU) with 8 beds."
        )
        retriever = StructuredRetriever(entity_store=mock_store)
        rag_service = RAGService(retriever=retriever, llm_provider=mock_llm)

        chat_response = rag_service.answer_question("What facilities are available in Paediatrics?")

        assert "Immunization" in chat_response.answer or "NICU" in chat_response.answer
        assert len(chat_response.sources) >= 1
        assert all(s.content_type == "facility" for s in chat_response.sources)

