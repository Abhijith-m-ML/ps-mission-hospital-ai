"""Unit and integration tests for Query Understanding and Query Rewriting layer."""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ChatMessage, RetrievalChunkItem
from app.query.intent import HospitalIntent, IntentClassifier
from app.query.normalizer import Normalizer
from app.query.query_rewriter import QueryRewriter
from app.query.resolver import QueryResolver
from app.retrieval.entity_store import HospitalEntityStore
from app.retrieval.reranker import LightweightReranker


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def resolver():
    return QueryResolver()


class TestNormalizer:
    """Verifies Normalizer behavior, typo correction, and medical name preservation."""

    def test_whitespace_and_punctuation_cleanup(self):
        norm = Normalizer()
        assert norm.normalize("   wat   docters???   ") == "What doctors?"
        assert norm.normalize("hello   world  !!  ") == "Hello world!"

    def test_protect_medical_titles_and_abbreviations(self):
        norm = Normalizer()
        res = norm.normalize("Dr. Sr. Jaya Joseph MBBS MD in OPD")
        assert "Dr. Sr. Jaya Joseph" in res
        assert "MBBS" in res
        assert "OPD" in res

    def test_common_chat_typos_and_abbreviations(self):
        norm = Normalizer()
        res = norm.normalize("wat docters r available in cardiolgy")
        assert "What" in res
        assert "doctors" in res
        assert "are" in res
        assert "Cardiology" in res


class TestIntentClassifier:
    """Verifies recognition across all required hospital intent categories."""

    def test_all_required_intents(self):
        classifier = IntentClassifier()

        cases = [
            ("who are the doctors", HospitalIntent.DOCTOR_SEARCH),
            ("when is dr jaya available", HospitalIntent.DOCTOR_SCHEDULE),
            ("tell me about Cardiology department", HospitalIntent.DEPARTMENT_INFORMATION),
            ("what facilities are in paediatrics", HospitalIntent.FACILITY_INFORMATION),
            ("what are the hospital visiting hours", HospitalIntent.HOSPITAL_HOURS),
            ("what are the op consultation hours", HospitalIntent.OP_HOURS),
            ("when is registration desk open", HospitalIntent.REGISTRATION_HOURS),
            ("emergency ambulance service number", HospitalIntent.EMERGENCY_INFORMATION),
            ("what is the hospital phone number", HospitalIntent.CONTACT_INFORMATION),
            ("where is the hospital located", HospitalIntent.LOCATION),
            ("do you accept star health insurance", HospitalIntent.INSURANCE),
            ("what health checkup packages are offered", HospitalIntent.SERVICES),
            ("tell me about the hospital history", HospitalIntent.GENERAL_HOSPITAL_INFORMATION),
        ]

        for text, expected in cases:
            intent, conf = classifier.classify(text)
            assert intent == expected, f"Failed for '{text}': got {intent}, expected {expected}"
            assert conf >= 0.70

    def test_visiting_hours_is_not_op_hours(self):
        """Crucial requirement: Visiting hours MUST NOT be classified as OP consultation hours."""
        classifier = IntentClassifier()
        intent_visit, _ = classifier.classify("What are the hospital visiting hours?")
        intent_op, _ = classifier.classify("What are the OPD consultation timings?")

        assert intent_visit == HospitalIntent.HOSPITAL_HOURS
        assert intent_op == HospitalIntent.OP_HOURS
        assert intent_visit != intent_op


class TestFuzzyEntityMatching:
    """Verifies fuzzy matching for doctors, departments, and facilities."""

    def test_fuzzy_match_doctor_dr_jaya_jospeh(self, resolver):
        """'Dr Jaya Jospeh' -> 'Dr. Sr. Jaya Joseph'."""
        entities = resolver.query_rewriter.extract_entities("Dr Jaya Jospeh")
        assert entities["doctor"] == "Dr. Sr. Jaya Joseph"

    def test_fuzzy_match_department_cardiolgy(self, resolver):
        """'cardiolgy' -> 'Cardiology'."""
        entities = resolver.query_rewriter.extract_entities("doctors in cardiolgy")
        assert entities["department"] == "Cardiology"

    def test_fuzzy_match_department_neurolgy(self, resolver):
        """'neurolgy' -> 'Neurology'."""
        entities = resolver.query_rewriter.extract_entities("neurolgy doctors")
        assert entities["department"] == "Neurology"

    def test_no_hallucination_on_unknown_doctor(self, resolver):
        """Unknown doctor names should not be matched to random physicians."""
        entities = resolver.query_rewriter.extract_entities("Dr. Nonexistent Unknown")
        assert entities["doctor"] is None or entities["doctor"] != "Dr. Sr. Jaya Joseph"


class TestMandatoryQueries:
    """Verifies all 10 explicit test queries specified in the requirement prompt."""

    def test_query_1_wat_docters_r_available_in_cardiolgy(self, resolver):
        res = resolver.resolve("wat docters r available in cardiolgy")
        assert res.resolved_query == "What doctors are available in Cardiology?"
        assert res.intent == "doctor_search"
        assert res.department == "Cardiology"

    def test_query_2_neurolgy_doctors(self, resolver):
        res = resolver.resolve("neurolgy doctors")
        assert "Neurology" in res.resolved_query
        assert "doctor" in res.resolved_query.lower()
        assert res.intent == "doctor_search"
        assert res.department == "Neurology"

    def test_query_3_dr_jaya_jospeh(self, resolver):
        res = resolver.resolve("Dr Jaya Jospeh")
        assert res.resolved_query == "Dr. Sr. Jaya Joseph"
        assert res.intent == "doctor_search"
        assert res.target_doctor == "Dr. Sr. Jaya Joseph"

    def test_query_4_who_r_the_doctors_no_context(self, resolver):
        res = resolver.resolve("who r the doctors?")
        assert res.resolved_query == "Who are the doctors?"
        assert res.intent == "doctor_search"
        assert res.department is None
        assert res.conversation_context_used is False

    def test_query_5_cardiology_then_who_are_the_doctors(self, resolver):
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Department of Cardiology offers clinical heart care."),
        ]
        res = resolver.resolve("who are the doctors?", history=history)
        assert res.resolved_query == "Which doctors are available in the Cardiology department?"
        assert res.intent == "doctor_search"
        assert res.department == "Cardiology"
        assert res.conversation_context_used is True

    def test_query_6_paediatrics_then_what_are_their_timings(self, resolver):
        history = [
            ChatMessage(role="user", content="Tell me about Paediatrics."),
            ChatMessage(role="assistant", content="Paediatrics & Neonatology provides child health care."),
        ]
        res = resolver.resolve("what are their timings?", history=history)
        assert res.resolved_query == "What is the OP consultation schedule for Paediatrics & Neonatology doctors?"
        assert res.intent == "doctor_schedule"
        assert res.department == "Paediatrics & Neonatology"
        assert res.conversation_context_used is True

    def test_query_7_hospital_visiting_hours(self, resolver):
        res = resolver.resolve("hospital visiting hours")
        assert res.resolved_query == "What are the hospital visiting hours?"
        assert res.intent == "hospital_hours"

    def test_query_8_normal_hospital_hours(self, resolver):
        res = resolver.resolve("normal hospital hours")
        assert res.resolved_query == "What are the hospital working hours?"
        assert res.intent == "hospital_hours"

    def test_query_9_where_is_the_hospital(self, resolver):
        res = resolver.resolve("where is the hospital?")
        assert res.resolved_query == "Where is P.S. Mission Hospital located?"
        assert res.intent == "location"

    def test_query_10_what_doctors_are_available_in_cardiology(self, resolver):
        res = resolver.resolve("what doctors are available in Cardiology?")
        assert res.resolved_query == "What doctors are available in Cardiology?"
        assert res.intent == "doctor_search"
        assert res.department == "Cardiology"


class TestLightweightReranker:
    """Verifies that the reranker boosts relevant entity passages and maintains metadata."""

    def test_reranker_boosts_matching_entity_chunks(self):
        reranker = LightweightReranker()
        chunks = [
            RetrievalChunkItem(
                rank=1,
                chunk_id="chunk_general",
                document_id="doc_general",
                score=0.75,
                content="General hospital guidelines and visitor reception counter.",
                section="General",
                url="https://psmissionhospital.org/about",
                title="About Us",
            ),
            RetrievalChunkItem(
                rank=2,
                chunk_id="chunk_cardio",
                document_id="doc_cardio",
                score=0.70,
                content="Cardiology department diagnostic echocardiography and ECG facilities.",
                section="Cardiology",
                url="https://psmissionhospital.org/cardiology",
                title="Cardiology Department",
                metadata={"department": "Cardiology"},
            ),
        ]

        reranked = reranker.rerank(
            query="Cardiology echocardiography",
            chunks=chunks,
            entities={"department": "Cardiology"},
        )

        assert len(reranked) == 2
        # Cardiology chunk should be boosted to rank 1 due to entity and token match
        assert reranked[0].chunk_id == "chunk_cardio"
        assert reranked[0].rank == 1
        assert reranked[1].chunk_id == "chunk_general"
        assert reranked[1].rank == 2


class TestDevelopmentEndpoint:
    """Verifies POST /api/query/preview development endpoint."""

    def test_query_preview_endpoint_success(self, client):
        payload = {"message": "wat docters r available in cardiolgy"}
        response = client.post("/api/query/preview", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["original_query"] == "wat docters r available in cardiolgy"
        assert data["resolved_query"] == "What doctors are available in Cardiology?"
        assert data["intent"] == "doctor_search"
        assert data["entities"]["department"] == "Cardiology"
        assert data["confidence"] >= 0.90

    def test_query_preview_endpoint_with_history(self, client):
        payload = {
            "message": "who are the doctors?",
            "history": [
                {"role": "user", "content": "Tell me about Cardiology."},
                {"role": "assistant", "content": "Cardiology department information."},
            ],
        }
        response = client.post("/api/query/preview", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["resolved_query"] == "Which doctors are available in the Cardiology department?"
        assert data["intent"] == "doctor_search"
        assert data["entities"]["department"] == "Cardiology"

    def test_query_preview_endpoint_empty_message_validation_error(self, client):
        response = client.post("/api/query/preview", json={"message": ""})
        assert response.status_code == 422


class TestContextResolutionBugRegressions:
    """Verifies regression fixes for general hospital questions with previous department context (Cases A-F)."""

    def test_case_a_cardiology_context_who_are_the_doctors(self, resolver):
        """Case A: Context Cardiology + 'Who are the doctors?' -> Expected department = Cardiology"""
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Department of Cardiology offers clinical heart care."),
        ]
        res = resolver.resolve("Who are the doctors?", history=history)
        assert res.department == "Cardiology"
        assert res.intent == "doctor_search"
        assert res.conversation_context_used is True

    def test_case_b_cardiology_context_what_are_their_timings(self, resolver):
        """Case B: Context Cardiology + 'What are their timings?' -> Expected department = Cardiology"""
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Department of Cardiology offers clinical heart care."),
        ]
        res = resolver.resolve("What are their timings?", history=history)
        assert res.department == "Cardiology"
        assert res.intent == "doctor_schedule"
        assert res.conversation_context_used is True

    def test_case_c_cardiology_context_how_can_i_contact_the_hospital(self, resolver):
        """Case C: Context Cardiology + 'How can I contact the hospital?' -> Expected department = null"""
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Department of Cardiology offers clinical heart care."),
        ]
        res = resolver.resolve("How can I contact the hospital?", history=history)
        assert res.department is None
        assert res.intent == "contact_information"
        assert res.resolved_query == "How can I contact the hospital?"
        assert res.conversation_context_used is False

    def test_case_d_cardiology_context_what_are_the_hospital_visiting_hours(self, resolver):
        """Case D: Context Cardiology + 'What are the hospital visiting hours?' -> Expected department = null"""
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Department of Cardiology offers clinical heart care."),
        ]
        res = resolver.resolve("What are the hospital visiting hours?", history=history)
        assert res.department is None
        assert res.intent == "hospital_hours"
        assert res.resolved_query == "What are the hospital visiting hours?"
        assert res.conversation_context_used is False

    def test_case_e_cardiology_context_where_is_the_hospital(self, resolver):
        """Case E: Context Cardiology + 'Where is the hospital?' -> Expected department = null"""
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Department of Cardiology offers clinical heart care."),
        ]
        res = resolver.resolve("Where is the hospital?", history=history)
        assert res.department is None
        assert res.intent == "location"
        assert res.conversation_context_used is False

    def test_case_f_no_context_how_can_i_contact_the_hospital_retrieval(self, resolver):
        """Case F: No context + 'How can I contact the hospital?' -> Expected contact information retrieval."""
        res = resolver.resolve("How can I contact the hospital?", history=None)
        assert res.department is None
        assert res.intent == "contact_information"
        assert res.resolved_query == "How can I contact the hospital?"
        assert res.conversation_context_used is False


class TestStep15MultilingualQueryUnderstanding:
    """Verifies all mandatory test cases from STEP 15 of user specification."""

    def test_case_1_malayalam_leg_pain_doctor(self, resolver):
        """CASE 1: Pure Malayalam leg pain consultation inquiry."""
        res = resolver.resolve("എനിക്ക് കാലിൽ വേദനയുണ്ട്, ഏത് ഡോക്ടറെ കാണണം?")
        assert res.resolved_query in (
            "Which doctor should I consult for leg pain?",
            "I have leg pain. Which doctor should I consult?",
        )
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"
        assert res.confidence >= 0.85

    def test_case_2_manglish_leg_pain_doctor(self, resolver):
        """CASE 2: Manglish input 'enik kaalil pain aanu doctor aar?'."""
        res = resolver.resolve("enik kaalil pain aanu doctor aar?")
        assert res.resolved_query in (
            "Which doctor should I consult for leg pain?",
            "I have leg pain. Which doctor should I consult?",
        )
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"

    def test_case_3_english_leg_hurting_doctor(self, resolver):
        """CASE 3: Informal English 'my leg is hurting which doctor?'."""
        res = resolver.resolve("my leg is hurting which doctor?")
        assert res.resolved_query in (
            "Which doctor should I consult for leg pain?",
            "I have leg pain. Which doctor should I consult?",
        )
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"

    def test_case_4_mixed_malayalam_english_department(self, resolver):
        """CASE 4: Mixed Malayalam-English 'കാലിനു pain und, ഏത് department?'."""
        res = resolver.resolve("കാലിനു pain und, ഏത് department?")
        assert res.resolved_query == "Which department should I consult for leg pain?"
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"

    def test_case_5_doctors_in_cardiology(self, resolver):
        """CASE 5: Direct doctor query 'Who are the doctors in cardiology?'."""
        res = resolver.resolve("Who are the doctors in cardiology?")
        assert res.intent == "doctor_search"
        assert res.department == "Cardiology"

    def test_case_6_cardiology_followup_who_are_the_doctors(self, resolver):
        """CASE 6: Follow-up query retains previous department."""
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Department of Cardiology offers clinical heart care."),
        ]
        res = resolver.resolve("Who are the doctors?", history=history)
        assert res.department == "Cardiology"
        assert res.intent == "doctor_search"
        assert res.conversation_context_used is True

    def test_case_7_cardiology_followup_contact_hospital_no_dept(self, resolver):
        """CASE 7: General contact query must NOT inherit Cardiology context."""
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Department of Cardiology offers clinical heart care."),
        ]
        res = resolver.resolve("How can I contact the hospital?", history=history)
        assert res.department is None
        assert res.intent == "contact_information"
        assert res.conversation_context_used is False

    def test_case_8_cardiology_followup_visiting_hours_no_dept(self, resolver):
        """CASE 8: Visiting hours query must NOT inherit Cardiology context."""
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Department of Cardiology offers clinical heart care."),
        ]
        res = resolver.resolve("What are the hospital visiting hours?", history=history)
        assert res.department is None
        assert res.intent == "hospital_hours"
        assert res.conversation_context_used is False

    def test_additional_symptom_examples_from_step_3(self, resolver):
        """Verifies Step 3 examples: typos, statements, and calf pain."""
        # 1. "wat docter for leg pain" -> "Which doctor should I consult for leg pain?"
        res1 = resolver.resolve("wat docter for leg pain")
        assert res1.resolved_query == "Which doctor should I consult for leg pain?"
        assert res1.intent == "symptom_to_department"

        # 2. "enik kaalil pain aanu" -> "I have pain in my leg."
        res2 = resolver.resolve("enik kaalil pain aanu")
        assert res2.resolved_query == "I have pain in my leg."

        # 3. "calf pain und, which department?" -> "Which hospital department should I consult for calf pain?"
        res3 = resolver.resolve("calf pain und, which department?")
        assert res3.resolved_query == "Which hospital department should I consult for calf pain?"
        assert res3.intent == "symptom_to_department"

    def test_lay_medical_terms_mapping(self, resolver):
        """Verifies lay medical terms (heart doctor, bone doctor, etc.)."""
        assert resolver.query_rewriter.match_department("heart doctor") == "Cardiology"
        assert resolver.query_rewriter.match_department("brain doctor") == "Neurology"
        assert resolver.query_rewriter.match_department("bone doctor") == "Orthopaedics & Trauma Care"
        assert resolver.query_rewriter.match_department("skin doctor") == "Dermatology"
        assert resolver.query_rewriter.match_department("eye doctor") == "Ophthalmology"

    def test_safety_rule_no_medical_diagnosis(self, resolver):
        """CRITICAL SAFETY RULE: System must never invent diseases or diagnoses."""
        res = resolver.resolve("my chest is hurting which doctor?")
        assert res.entities.get("symptom") == "chest pain"
        assert res.resolved_query in (
            "Which doctor should I consult for chest pain?",
            "I have chest pain. Which doctor should I consult?",
        )


class TestStep10DirectTextUnderstanding:
    """Verifies all direct text test cases A through F from Step 10."""

    def test_case_a_english_leg_pain(self, resolver):
        """A: 'I have leg pain. Which doctor should I see?'"""
        res = resolver.resolve("I have leg pain. Which doctor should I see?")
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"
        assert res.entities.get("body_part") == "leg"
        assert res.entities.get("inferred_department") == "Orthopaedics & Trauma Care"
        assert "leg pain" in res.resolved_query.lower()
        assert "doctor" in res.resolved_query.lower()

    def test_case_b_malayalam_leg_pain(self, resolver):
        """B: 'എനിക്ക് കാലിൽ വേദനയുണ്ട്, ഏത് ഡോക്ടറെ കാണണം?'"""
        res = resolver.resolve("എനിക്ക് കാലിൽ വേദനയുണ്ട്, ഏത് ഡോക്ടറെ കാണണം?")
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"
        assert res.entities.get("body_part") == "leg"
        assert res.entities.get("inferred_department") == "Orthopaedics & Trauma Care"
        assert res.resolved_query in (
            "Which doctor should I consult for leg pain?",
            "I have leg pain. Which doctor should I consult?",
        )

    def test_case_c_manglish_leg_pain(self, resolver):
        """C: 'enik kaalil pain aanu doctor aar?'"""
        res = resolver.resolve("enik kaalil pain aanu doctor aar?")
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"
        assert res.entities.get("body_part") == "leg"
        assert res.entities.get("inferred_department") == "Orthopaedics & Trauma Care"
        assert res.resolved_query in (
            "Which doctor should I consult for leg pain?",
            "I have leg pain. Which doctor should I consult?",
        )

    def test_case_d_concise_leg_pain(self, resolver):
        """D: 'which doctor for leg pain?'"""
        res = resolver.resolve("which doctor for leg pain?")
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"
        assert res.entities.get("body_part") == "leg"
        assert res.entities.get("inferred_department") == "Orthopaedics & Trauma Care"
        assert res.resolved_query in (
            "Which doctor should I consult for leg pain?",
            "I have leg pain. Which doctor should I consult?",
        )

    def test_case_e_malayalam_headache(self, resolver):
        """E: 'എനിക്ക് തലവേദനയുണ്ട് ഏത് ഡോക്ടറെ കാണണം?'"""
        res = resolver.resolve("എനിക്ക് തലവേദനയുണ്ട് ഏത് ഡോക്ടറെ കാണണം?")
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "headache"
        assert res.entities.get("body_part") == "head"
        assert res.entities.get("inferred_department") == "Neurology"
        assert res.resolved_query in (
            "Which doctor should I consult for headache?",
            "I have a headache. Which doctor should I consult?",
            "I have headache. Which doctor should I consult?",
        )

    def test_case_f_malayalam_stomach_pain(self, resolver):
        """F: 'എനിക്ക് വയറുവേദനയാണ് ആരെ കാണണം?'"""
        res = resolver.resolve("എനിക്ക് വയറുവേദനയാണ് ആരെ കാണണം?")
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "stomach pain"
        assert res.entities.get("body_part") == "stomach"
        assert res.entities.get("inferred_department") == "General Medicine"
        assert res.resolved_query in (
            "Which doctor should I consult for stomach pain?",
            "I have stomach pain. Which doctor should I consult?",
        )


class TestStep4NaturalLanguageEquivalence:
    """Verifies that varied natural Malayalam, Manglish, and English phrasing resolve to consistent semantic meaning."""

    def test_all_leg_pain_variations_produce_consistent_intent_and_department(self, resolver):
        variations = [
            "എന്റെ കാലിലൊരു വേദനയുണ്ട്, ഞാൻ ഏത് ഡോക്ടറെയാണ് കാണിക്കേണ്ടത്?",
            "എന്റെ കാലിലൊരു വേദനയുണ്ട്, ഏത് ഡോക്ടറെ കാണണം?",
            "കാലിനു pain ആണ്, ആരെ കാണിക്കണം?",
            "enik kaalil pain aanu, etha doctor?",
            "leg pain und, which doctor?",
            "my leg is hurting, which doctor should I see?",
            "which doctor for pain in my leg?",
            "കാലിനു വേദനയാണ്, ആരെ കാണിക്കണം?",
            "എനിക്ക് കാലിൽ വേദനയുണ്ട്",
        ]
        for query in variations:
            res = resolver.resolve(query)
            assert res.intent == "symptom_to_department", f"Failed intent for: {query}"
            assert res.entities.get("symptom") == "leg pain", f"Failed symptom for: {query}"
            assert res.entities.get("body_part") == "leg", f"Failed body part for: {query}"
            assert res.entities.get("inferred_department") == "Orthopaedics & Trauma Care", f"Failed dept for: {query}"
            assert "leg pain" in res.resolved_query.lower() or "pain in my leg" in res.resolved_query.lower(), f"Failed normalized query for: {query}"


class TestStep8And9ContextIsolation:
    """Verifies that previous department context (e.g. Cardiology or Gastroenterology)
    is NEVER inherited when the user switches to a symptom or independent question.
    """

    def test_previous_cardiology_not_inherited_by_leg_pain(self, resolver):
        history = [
            ChatMessage(role="user", content="Tell me about Cardiology."),
            ChatMessage(role="assistant", content="The Cardiology Department treats heart diseases."),
        ]
        res = resolver.resolve("I have leg pain. Which doctor should I see?", history=history)
        assert res.conversation_context_used is False
        assert res.department == "Orthopaedics & Trauma Care"
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"

    def test_previous_gastroenterology_not_inherited_by_malayalam_leg_pain(self, resolver):
        history = [
            ChatMessage(role="user", content="Tell me about Gastroenterology."),
            ChatMessage(role="assistant", content="The Department of Medical Gastroenterology treats digestive conditions."),
        ]
        res = resolver.resolve("എന്റെ കാലിലൊരു വേദനയുണ്ട്, ഞാൻ ഏത് ഡോക്ടറെയാണ് കാണിക്കേണ്ടത്?", history=history)
        assert res.conversation_context_used is False
        assert res.department == "Orthopaedics & Trauma Care"
        assert res.intent == "symptom_to_department"
        assert res.entities.get("symptom") == "leg pain"


class TestStep12DevelopmentDebugMode:
    """Verifies that debug_pipeline is returned when debug mode is enabled, and omitted otherwise."""

    def test_debug_mode_disabled_by_default(self, monkeypatch, client):
        from app.rag.llm import MockLLMProvider
        import app.rag.rag_service
        monkeypatch.setattr(
            app.rag.rag_service,
            "get_llm_provider",
            lambda: MockLLMProvider('{"answer": "P.S. Mission Hospital is located in Maradu, Cochin.", "source_ids": []}'),
        )
        resp = client.post("/api/chat", json={"message": "Where is the hospital?", "debug": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("raw_transcript") == "Where is the hospital?"
        assert data.get("debug_pipeline") is None

    def test_debug_mode_enabled_returns_pipeline_diagnostics(self, monkeypatch, client):
        from app.rag.llm import MockLLMProvider
        import app.rag.rag_service
        monkeypatch.setattr(
            app.rag.rag_service,
            "get_llm_provider",
            lambda: MockLLMProvider('{"answer": "P.S. Mission Hospital is located in Maradu, Cochin.", "source_ids": []}'),
        )
        resp = client.post("/api/chat", json={"message": "Where is the hospital?", "debug": True})
        assert resp.status_code == 200
        data = resp.json()
        debug_info = data.get("debug_pipeline")
        assert debug_info is not None
        assert debug_info["raw_transcript"] == "Where is the hospital?"
        assert "normalized_query" in debug_info
        assert "intent" in debug_info
        assert "entities" in debug_info
        assert "retrieval_query" in debug_info
        assert "filters" in debug_info
        assert "retrieved_documents" in debug_info
        assert "final_context" in debug_info



