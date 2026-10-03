import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ChatMessage, Department, Doctor, Facility
from app.rag.llm import MockLLMProvider
from app.rag.rag_service import RAGService
from app.retrieval.entity_store import HospitalEntityStore
from app.retrieval.query_resolver import QueryResolver
from app.retrieval.structured_retriever import StructuredRetriever


@pytest.fixture
def mock_store(tmp_path):
    """Initializes a populated HospitalEntityStore for conversation testing."""
    store = HospitalEntityStore(data_path=str(tmp_path / "test_conv_entities.json"))

    # Departments
    store.add_department(
        Department(
            department_name="Paediatrics & Neonatology",
            description="Child health and neonatology care.",
            source_url="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
        )
    )
    store.add_department(
        Department(
            department_name="Cardiology",
            description="Cardiovascular medical and interventional care.",
            source_url="https://www.psmissionhospital.org/category/department/cardiology",
        )
    )

    # Doctors
    store.add_doctor(
        Doctor(
            doctor_name="Dr. Sr. Jaya Joseph",
            qualification="MBBS, MD",
            department="Paediatrics & Neonatology",
            schedule_text="OP: MONDAY-FRIDAY | 9.30AM-1.00PM & 4.00PM-6.00 PM",
            image_url="https://www.psmissionhospital.org/wp-content/uploads/2023/08/Dr-Sr-Jaya-Joseph-1.jpg",
            source_url="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
        )
    )
    store.add_doctor(
        Doctor(
            doctor_name="Dr. Sr. Annie Sheela",
            qualification="MD, DM, FICC, FESC",
            department="Cardiology",
            schedule_text="OP: Monday to Saturday 9:00 AM - 1:00 PM",
            image_url="https://www.psmissionhospital.org/wp-content/uploads/2021/03/i2.jpg",
            source_url="https://www.psmissionhospital.org/category/department/cardiology",
        )
    )
    store.add_doctor(
        Doctor(
            doctor_name="Dr. Sudheer",
            qualification="MBBS, MD, DM",
            department="Cardiology",
            schedule_text="OP: Daily morning",
            image_url="https://www.psmissionhospital.org/wp-content/uploads/2023/08/psmissionhospital-dr-1.jpg",
            source_url="https://www.psmissionhospital.org/category/department/cardiology",
        )
    )

    # Facilities
    store.add_facility(
        Facility(
            facility_name="Immunization",
            description="Routine child vaccines",
            department="Paediatrics & Neonatology",
            source_url="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
        )
    )
    store.add_facility(
        Facility(
            facility_name="Neonatology and Emergency Care",
            description="Emergency newborn intensive care",
            department="Paediatrics & Neonatology",
            source_url="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
        )
    )
    store.add_facility(
        Facility(
            facility_name="Echocardiography",
            description="Cardiac 2D Echo imaging",
            department="Cardiology",
            source_url="https://www.psmissionhospital.org/category/department/cardiology",
        )
    )

    return store


@pytest.fixture
def rag_service(mock_store):
    retriever = StructuredRetriever(entity_store=mock_store)
    mock_llm = MockLLMProvider("Hospital verified response based on retrieved facts.")
    query_resolver = QueryResolver(entity_store=mock_store)
    return RAGService(retriever=retriever, llm_provider=mock_llm, query_resolver=query_resolver)


class TestConversationSessionMemory:
    """Verifies all 6 mandatory conversation memory test cases."""

    def test_case_1_children_services_then_doctor_name_please(self, rag_service):
        """User asks 'Which services are available for children?' followed by 'doctor name please'.
        Expected: Doctors specifically from Paediatrics & Neonatology.
        """
        history = [
            ChatMessage(role="user", content="Which services are available for children?"),
            ChatMessage(
                role="assistant",
                content="Paediatrics & Neonatology provides Immunization and Neonatology and Emergency Care.",
            ),
        ]

        response = rag_service.answer_question(
            message="doctor name please",
            history=history,
            session_id="session-test-1",
        )

        assert response.resolved_query in (
            "Which doctors are available in the Paediatrics & Neonatology department?",
            "Which doctors are available for Paediatrics & Neonatology?",
        )
        assert response.session_id == "session-test-1"
        assert len(response.sources) == 1
        assert response.sources[0].doctor_name == "Dr. Sr. Jaya Joseph"
        assert response.sources[0].section == "Paediatrics & Neonatology"

    def test_case_2_cardiology_then_who_are_the_doctors(self, rag_service):
        """User asks 'Tell me about Cardiology.' followed by 'Who are the doctors?'.
        Expected: Cardiology doctors, not all hospital doctors.
        """
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(
                role="assistant",
                content="The Department of Cardiology provides comprehensive cardiac care.",
            ),
        ]

        response = rag_service.answer_question(
            message="Who are the doctors?",
            history=history,
            session_id="session-test-2",
        )

        assert response.resolved_query in (
            "Which doctors are available in the Cardiology department?",
            "Which doctors are available for Cardiology?",
        )
        # Expecting only Cardiology doctors (Dr. Sr. Annie Sheela and Dr. Sudheer), not Paediatrics
        doc_names = [s.doctor_name for s in response.sources]
        assert "Dr. Sr. Annie Sheela" in doc_names
        assert "Dr. Sudheer" in doc_names
        assert "Dr. Sr. Jaya Joseph" not in doc_names

    def test_case_3_who_is_doctor_x_then_when_can_i_visit(self, rag_service):
        """User asks 'Who is Dr. Sr. Jaya Joseph?' followed by 'When can I visit?'.
        Expected: Dr. Sr. Jaya Joseph's available schedule.
        """
        history = [
            ChatMessage(role="user", content="Who is Dr. Sr. Jaya Joseph?"),
            ChatMessage(
                role="assistant",
                content="Dr. Sr. Jaya Joseph is a consultant in Paediatrics & Neonatology.",
            ),
        ]

        response = rag_service.answer_question(
            message="When can I visit?",
            history=history,
            session_id="session-test-3",
        )

        assert response.resolved_query == "What is Dr. Sr. Jaya Joseph's available schedule?"
        assert len(response.sources) == 1
        assert response.sources[0].doctor_name == "Dr. Sr. Jaya Joseph"
        assert response.sources[0].chunk_id.startswith("schedule_")

    def test_case_4_facilities_then_what_about_doctors(self, rag_service):
        """User asks 'What facilities are available in Paediatrics?' followed by 'What about doctors?'.
        Expected: Paediatrics doctors.
        """
        history = [
            ChatMessage(role="user", content="What facilities are available in Paediatrics?"),
            ChatMessage(
                role="assistant",
                content="Facilities include Immunization and Neonatology ICU.",
            ),
        ]

        response = rag_service.answer_question(
            message="What about doctors?",
            history=history,
            session_id="session-test-4",
        )

        assert response.resolved_query in (
            "Which doctors are available in the Paediatrics & Neonatology department?",
            "Which doctors are available for Paediatrics & Neonatology?",
        )
        assert len(response.sources) == 1
        assert response.sources[0].doctor_name == "Dr. Sr. Jaya Joseph"

    def test_case_5_refresh_browser_empty_history(self, rag_service):
        """Browser refreshed: session starts clean with empty history.
        User asks: 'Who are the doctors?'.
        Expected: No previous department context available, query is not rewritten.
        """
        response = rag_service.answer_question(
            message="Who are the doctors?",
            history=[],  # Empty history simulating fresh browser load
            session_id="session-fresh-after-refresh",
        )

        # No prior department should be assumed
        assert response.resolved_query == "Who are the doctors?"
        assert response.session_id == "session-fresh-after-refresh"

    def test_case_6_new_conversation_independent_cardiology_facilities(self, rag_service):
        """Independent query without prior history: 'What are the cardiology facilities?'.
        Expected: Works cleanly without requiring prior context.
        """
        response = rag_service.answer_question(
            message="What are the cardiology facilities?",
            history=[],
            session_id="session-independent",
        )

        assert response.resolved_query == "What are the cardiology facilities?"
        assert any(s.title == "Facilities: Cardiology" for s in response.sources)


class TestChatAPIWithSession:
    """Verifies HTTP POST /api/chat with session_id and history payload."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_api_chat_returns_session_and_resolved_query(self, client):
        payload = {
            "session_id": "client-uuid-1234",
            "message": "doctor name please",
            "history": [
                {
                    "role": "user",
                    "content": "Which services are available for children?",
                },
                {
                    "role": "assistant",
                    "content": "Paediatrics & Neonatology provides Immunization and Neonatology and Emergency Care.",
                },
            ],
        }

        res = client.post("/api/chat", json=payload)
        assert res.status_code == 200
        data = res.json()

        assert data["session_id"] == "client-uuid-1234"
        assert data["resolved_query"] in (
            "Which doctors are available in the Paediatrics & Neonatology department?",
            "Which doctors are available for Paediatrics & Neonatology?",
        )
        assert "answer" in data
        assert "sources" in data
        # Check that Paediatrics doctor is cited
        assert any(s.get("doctor_name") == "Dr. Sr. Jaya Joseph" for s in data["sources"])
