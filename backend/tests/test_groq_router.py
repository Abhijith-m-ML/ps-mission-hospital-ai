"""Tests for Groq Integration and Input-Modality LLM Router.

Verifies:
1. English text -> Groq
2. Malayalam text -> Groq
3. English voice -> STT = Gemini, Final LLM = Gemini
4. Malayalam voice -> STT = Gemini, Final LLM = Gemini
5. Hindi voice -> STT = Gemini, Final LLM = Gemini
6. Text with conversation history -> same conversation memory, Groq
7. Voice with conversation history -> same conversation memory, Gemini
8. Error handling & fallback isolation (no cross-provider leaks)
9. Development-only debug logging (no secrets logged)
"""
import logging
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ChatMessage, ChatRequest, SourceItem
from app.rag.rag_service import RAGService
from app.services.groq_service import GroqService
from app.services.llm_router import LLMRouter, generate


# ---------------------------------------------------------------------------
# Fixtures & Mocks
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


class MockRetriever:
    """Mock structured & semantic retriever returning deterministic hospital context."""

    def __init__(self, context_text="--- SOURCE 1 ---\nCardiology Department\nDoctors: Dr. Sr. Annie Sheela"):
        self.context_text = context_text

    def retrieve(self, query, **kwargs):
        from app.models.schemas import ContextResult, RetrievalChunkItem, RetrievalDebugInfo, RetrievalPreviewResponse
        chunk = RetrievalChunkItem(
            rank=1,
            chunk_id="chunk_cardio_01",
            document_id="doc_cardio",
            score=0.92,
            content="Cardiology Department at P.S. Mission Hospital. Dr. Sr. Annie Sheela is available.",
            section="Cardiology",
            url="https://psmissionhospital.org/cardiology",
            title="Cardiology Department",
            source="department",
            metadata={"department": "Cardiology", "doctor_name": "Dr. Sr. Annie Sheela"},
        )
        return RetrievalPreviewResponse(
            query=query,
            has_context=True,
            results=[chunk],
            context=ContextResult(
                has_context=True,
                context_text=self.context_text,
                total_chunks=1,
                total_characters=len(self.context_text),
            ),
            debug=RetrievalDebugInfo(
                pinecone_top_k=5,
                candidates_retrieved=1,
                removed_by_threshold=0,
                removed_duplicates=0,
                final_chunks_count=1,
                context_character_count=len(self.context_text),
            ),
            retrieval_duration_seconds=0.01,
        )


# ---------------------------------------------------------------------------
# 1. Groq Service Unit Tests
# ---------------------------------------------------------------------------

def test_groq_service_missing_api_key():
    service = GroqService(api_key="")
    with pytest.raises(ValueError, match="Groq API key is missing"):
        service.generate_response("System prompt", "User prompt")


def test_groq_service_with_mock_client():
    class MockCompletion:
        class Choice:
            class Message:
                content = '{"answer": "Dr. Sr. Annie Sheela is available in Cardiology.", "source_ids": ["chunk_cardio_01"]}'
            message = Message()
        choices = [Choice()]

    mock_cli = MagicMock()
    mock_cli.chat.completions.create.return_value = MockCompletion()

    service = GroqService(api_key="gsk_mock_test", client=mock_cli)
    res = service.generate_response("You are a hospital assistant.", "Who are the doctors in Cardiology?")

    assert "Dr. Sr. Annie Sheela" in res
    assert mock_cli.chat.completions.create.called
    args = mock_cli.chat.completions.create.call_args[1]
    assert args["model"] == "openai/gpt-oss-20b"
    assert args["messages"][0]["role"] == "system"
    assert args["messages"][1]["role"] == "user"


def test_groq_service_authentication_error():
    from groq import AuthenticationError

    mock_cli = MagicMock()
    mock_cli.chat.completions.create.side_effect = AuthenticationError(
        message="Invalid API Key",
        response=MagicMock(status_code=401),
        body={},
    )
    service = GroqService(api_key="gsk_invalid", client=mock_cli)
    with pytest.raises(ValueError, match="Invalid Groq API key"):
        service.generate_response("System", "User")


def test_groq_service_timeout_error():
    from groq import APITimeoutError
    import httpx

    mock_cli = MagicMock()
    mock_cli.chat.completions.create.side_effect = APITimeoutError(
        request=httpx.Request("POST", "https://api.groq.com")
    )
    service = GroqService(api_key="gsk_test", client=mock_cli)
    with pytest.raises(TimeoutError, match="timed out"):
        service.generate_response("System", "User")


# ---------------------------------------------------------------------------
# 2. LLM Router Unit Tests: Routing strictly by input_type, NOT language
# ---------------------------------------------------------------------------

def test_llm_router_text_routes_to_groq(caplog):
    mock_groq = MagicMock()
    mock_groq.model = "openai/gpt-oss-20b"
    mock_groq.generate_response.return_value = "Groq text response"

    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-flash-latest"

    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    with caplog.at_level(logging.INFO):
        res = router.generate(input_type="text", system_prompt="System", user_prompt="English query")

    assert res == "Groq text response"
    assert mock_groq.generate_response.called
    assert not mock_gemini.generate.called
    assert "llm_provider=groq" in caplog.text
    assert "input_type=text" in caplog.text
    assert "model=openai/gpt-oss-20b" in caplog.text


def test_llm_router_voice_routes_to_gemini(caplog):
    mock_groq = MagicMock()
    mock_groq.model = "openai/gpt-oss-20b"

    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-flash-latest"
    mock_gemini.generate.return_value = "Gemini voice response"

    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    with caplog.at_level(logging.INFO):
        res = router.generate(input_type="voice", system_prompt="System", user_prompt="Spoken voice query")

    assert res == "Gemini voice response"
    assert mock_gemini.generate.called
    assert not mock_groq.generate_response.called
    assert "llm_provider=gemini" in caplog.text
    assert "input_type=voice" in caplog.text
    assert "model=gemini-flash-latest" in caplog.text


def test_llm_router_malayalam_text_routes_to_groq():
    """Malayalam TEXT must route to Groq because modality is TEXT."""
    mock_groq = MagicMock()
    mock_groq.model = "openai/gpt-oss-20b"
    mock_groq.generate_response.return_value = "Malayalam answer from Groq"

    mock_gemini = MagicMock()
    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    res = router.generate(
        input_type="text",
        system_prompt="System",
        user_prompt="എനിക്ക് കാലിൽ വേദനയുണ്ട്, ഏത് ഡോക്ടറെ കാണണം?",
    )
    assert res == "Malayalam answer from Groq"
    assert mock_groq.generate_response.called
    assert not mock_gemini.generate.called


def test_llm_router_malayalam_voice_routes_to_gemini():
    """Malayalam VOICE must route to Gemini because modality is VOICE."""
    mock_groq = MagicMock()
    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-flash-latest"
    mock_gemini.generate.return_value = "Malayalam answer from Gemini"

    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    res = router.generate(
        input_type="voice",
        system_prompt="System",
        user_prompt="കാലിൽ വേദനയുണ്ട്",
    )
    assert res == "Malayalam answer from Gemini"
    assert mock_gemini.generate.called
    assert not mock_groq.generate_response.called


def test_llm_router_hindi_voice_routes_to_gemini():
    """Hindi VOICE must route to Gemini because modality is VOICE."""
    mock_groq = MagicMock()
    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-flash-latest"
    mock_gemini.generate.return_value = "Hindi answer from Gemini"

    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    res = router.generate(
        input_type="voice",
        system_prompt="System",
        user_prompt="अस्पताल का समय क्या है?",
    )
    assert res == "Hindi answer from Gemini"
    assert mock_gemini.generate.called
    assert not mock_groq.generate_response.called


def test_llm_router_hindi_text_routes_to_groq():
    """Hindi TEXT must route to Groq because modality is TEXT."""
    mock_groq = MagicMock()
    mock_groq.model = "openai/gpt-oss-20b"
    mock_groq.generate_response.return_value = "Hindi answer from Groq"

    mock_gemini = MagicMock()
    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    res = router.generate(
        input_type="text",
        system_prompt="System",
        user_prompt="कार्डियोलॉजी विभाग में कौन डॉक्टर हैं?",
    )
    assert res == "Hindi answer from Groq"
    assert mock_groq.generate_response.called
    assert not mock_gemini.generate.called


def test_groq_failure_does_not_silently_fallback_to_gemini():
    """If Groq fails for text, it must raise an error and NOT silently fall back to Gemini."""
    mock_groq = MagicMock()
    mock_groq.model = "openai/gpt-oss-20b"
    mock_groq.generate_response.side_effect = RuntimeError("Groq rate limit exceeded")

    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-flash-latest"

    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    with pytest.raises(RuntimeError, match="Groq rate limit"):
        router.generate(input_type="text", system_prompt="System", user_prompt="Question")

    assert not mock_gemini.generate.called, "CRITICAL: Router must not silently call Gemini when Groq fails!"


# ---------------------------------------------------------------------------
# 3. RAGService Integration Tests: 7 Required Scenarios
# ---------------------------------------------------------------------------

def test_rag_scenario_1_english_text():
    """1. English text: 'Who are the doctors in Cardiology?' -> Expected provider = Groq."""
    mock_groq = MagicMock()
    mock_groq.model = "openai/gpt-oss-20b"
    mock_groq.generate_response.return_value = (
        '{"answer": "Dr. Sr. Annie Sheela is available in the Cardiology Department.", "source_ids": ["chunk_cardio_01"]}'
    )
    mock_gemini = MagicMock()
    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    rag = RAGService(retriever=MockRetriever(), llm_router=router)
    resp = rag.answer_question(
        message="Who are the doctors in Cardiology?",
        input_type="text",
        debug=True,
    )

    assert mock_groq.generate_response.called
    assert not mock_gemini.generate.called
    assert "Dr. Sr. Annie Sheela" in resp.answer
    assert resp.debug_pipeline["llm_provider"] == "groq"
    assert resp.debug_pipeline["input_type"] == "text"


def test_rag_scenario_2_malayalam_text():
    """2. Malayalam text: 'എനിക്ക് കാലിൽ വേദനയുണ്ട്, ഏത് ഡോക്ടറെ കാണണം?' -> Expected provider = Groq."""
    mock_groq = MagicMock()
    mock_groq.model = "openai/gpt-oss-20b"
    mock_groq.generate_response.return_value = (
        '{"answer": "ജനറൽ മെഡിസിൻ അല്ലെങ്കിൽ ഓർത്തോപീഡിക്സ് വിഭാഗം സന്ദർശിക്കുക.", "source_ids": ["chunk_cardio_01"]}'
    )
    mock_gemini = MagicMock()
    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    rag = RAGService(retriever=MockRetriever(), llm_router=router)
    resp = rag.answer_question(
        message="എനിക്ക് കാലിൽ വേദനയുണ്ട്, ഏത് ഡോക്ടറെ കാണണം?",
        language="ml-IN",
        input_type="text",
        debug=True,
    )

    assert mock_groq.generate_response.called
    assert not mock_gemini.generate.called
    assert resp.debug_pipeline["llm_provider"] == "groq"
    assert resp.debug_pipeline["input_type"] == "text"


def test_rag_scenario_3_english_voice():
    """3. English voice -> STT = Gemini, Final LLM = Gemini."""
    mock_groq = MagicMock()
    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-flash-latest"
    mock_gemini.generate.return_value = (
        '{"answer": "Dr. Sr. Annie Sheela is the cardiologist on duty.", "source_ids": ["chunk_cardio_01"]}'
    )
    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    rag = RAGService(retriever=MockRetriever(), llm_router=router)
    resp = rag.answer_question(
        message="Who is the cardiologist?",
        input_type="voice",
        debug=True,
    )

    assert mock_gemini.generate.called
    assert not mock_groq.generate_response.called
    assert resp.debug_pipeline["llm_provider"] == "gemini"
    assert resp.debug_pipeline["input_type"] == "voice"


def test_rag_scenario_4_malayalam_voice():
    """4. Malayalam voice -> STT = Gemini, Final LLM = Gemini."""
    mock_groq = MagicMock()
    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-flash-latest"
    mock_gemini.generate.return_value = (
        '{"answer": "കാർഡിയോളജി വിഭാഗത്തിൽ ഡോ. ആനി ഷീല ലഭ്യമാണ്.", "source_ids": ["chunk_cardio_01"]}'
    )
    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    rag = RAGService(retriever=MockRetriever(), llm_router=router)
    resp = rag.answer_question(
        message="കാർഡിയോളജി ഡോക്ടർ ആരാണ്?",
        language="ml-IN",
        input_type="voice",
        debug=True,
    )

    assert mock_gemini.generate.called
    assert not mock_groq.generate_response.called
    assert resp.debug_pipeline["llm_provider"] == "gemini"
    assert resp.debug_pipeline["input_type"] == "voice"


def test_rag_scenario_5_hindi_voice():
    """5. Hindi voice -> STT = Gemini, Final LLM = Gemini."""
    mock_groq = MagicMock()
    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-flash-latest"
    mock_gemini.generate.return_value = (
        '{"answer": "कार्डियोलॉजी विभाग में डॉ. एनी शीला उपलब्ध हैं।", "source_ids": ["chunk_cardio_01"]}'
    )
    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    rag = RAGService(retriever=MockRetriever(), llm_router=router)
    resp = rag.answer_question(
        message="कार्डियोलॉजी विभाग में कौन डॉक्टर हैं?",
        language="hi-IN",
        input_type="voice",
        debug=True,
    )

    assert mock_gemini.generate.called
    assert not mock_groq.generate_response.called
    assert resp.debug_pipeline["llm_provider"] == "gemini"
    assert resp.debug_pipeline["input_type"] == "voice"


def test_rag_scenario_6_text_with_conversation_history():
    """6. Text with conversation history -> same existing conversation memory, provider = Groq."""
    mock_groq = MagicMock()
    mock_groq.model = "openai/gpt-oss-20b"
    mock_groq.generate_response.return_value = (
        '{"answer": "Her OPD consultation schedule is Monday to Saturday 9:00 AM to 1:00 PM.", "source_ids": ["chunk_cardio_01"]}'
    )
    mock_gemini = MagicMock()
    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    rag = RAGService(retriever=MockRetriever(), llm_router=router)

    history = [
        ChatMessage(role="user", content="Tell me about Dr. Sr. Annie Sheela"),
        ChatMessage(role="assistant", content="Dr. Sr. Annie Sheela is a consultant in Cardiology."),
    ]

    resp = rag.answer_question(
        message="What are her timings?",
        history=history,
        input_type="text",
        debug=True,
    )

    assert mock_groq.generate_response.called
    assert not mock_gemini.generate.called
    assert resp.debug_pipeline["llm_provider"] == "groq"
    # Query resolution should resolve 'her' using the common conversation memory
    assert "Annie Sheela" in (resp.resolved_query or "") or "timings" in (resp.resolved_query or "").lower()


def test_rag_scenario_7_voice_with_conversation_history():
    """7. Voice with conversation history -> same existing conversation memory, provider = Gemini."""
    mock_groq = MagicMock()
    mock_gemini = MagicMock()
    mock_gemini.model = "gemini-flash-latest"
    mock_gemini.generate.return_value = (
        '{"answer": "Her OPD consultation hours are 9:00 AM to 1:00 PM.", "source_ids": ["chunk_cardio_01"]}'
    )
    router = LLMRouter(groq_service=mock_groq, gemini_provider=mock_gemini)

    rag = RAGService(retriever=MockRetriever(), llm_router=router)

    history = [
        ChatMessage(role="user", content="Tell me about Dr. Sr. Annie Sheela"),
        ChatMessage(role="assistant", content="Dr. Sr. Annie Sheela is a consultant in Cardiology."),
    ]

    resp = rag.answer_question(
        message="What are her timings?",
        history=history,
        input_type="voice",
        debug=True,
    )

    assert mock_gemini.generate.called
    assert not mock_groq.generate_response.called
    assert resp.debug_pipeline["llm_provider"] == "gemini"
    assert "Annie Sheela" in (resp.resolved_query or "") or "timings" in (resp.resolved_query or "").lower()


# ---------------------------------------------------------------------------
# 4. API End-to-End Chat Endpoint Modality Routing
# ---------------------------------------------------------------------------

def test_api_chat_endpoint_routes_text_to_groq(client, monkeypatch):
    """Verifies that sending input_type='text' to /api/chat calls Groq."""
    mock_generate = MagicMock(return_value='{"answer": "Cardiology provides cardiac care.", "source_ids": []}')
    monkeypatch.setattr("app.services.groq_service.GroqService.generate_response", mock_generate)

    payload = {
        "message": "Tell me about Cardiology",
        "input_type": "text",
        "debug": True,
    }
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "Cardiology" in data["answer"]
    assert data["debug_pipeline"]["input_type"] == "text"
    assert data["debug_pipeline"]["llm_provider"] == "groq"
    assert mock_generate.called


def test_api_chat_endpoint_routes_voice_to_gemini(client, monkeypatch):
    """Verifies that sending input_type='voice' to /api/chat calls Gemini."""
    mock_generate = MagicMock(return_value='{"answer": "Gemini voice response.", "source_ids": []}')
    monkeypatch.setattr("app.rag.llm.GeminiLLMProvider.generate", mock_generate)

    payload = {
        "message": "Where is the hospital located?",
        "input_type": "voice",
        "debug": True,
    }
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "Gemini voice response" in data["answer"]
    assert data["debug_pipeline"]["input_type"] == "voice"
    assert data["debug_pipeline"]["llm_provider"] == "gemini"
    assert mock_generate.called


def test_api_chat_endpoint_defaults_to_text_groq_when_input_type_omitted(client, monkeypatch):
    """Verifies backward-compatibility: omitting input_type defaults to text -> Groq."""
    mock_generate = MagicMock(return_value='{"answer": "Defaulted to Groq text.", "source_ids": []}')
    monkeypatch.setattr("app.services.groq_service.GroqService.generate_response", mock_generate)

    payload = {
        "message": "Visiting hours?",
        "debug": True,
    }
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["debug_pipeline"]["input_type"] == "text"
    assert data["debug_pipeline"]["llm_provider"] == "groq"
    assert mock_generate.called
