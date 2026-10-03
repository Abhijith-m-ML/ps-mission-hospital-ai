from typing import Any, Dict, List, Optional
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import (
    ContextResult,
    RetrievalChunkItem,
    RetrievalDebugInfo,
    RetrievalPreviewResponse,
)
from app.rag.llm import (
    BaseLLMProvider,
    MockLLMProvider,
    OpenAILLMProvider,
    get_llm_provider,
)
from app.rag.prompts import HOSPITAL_SYSTEM_INSTRUCTION, build_grounded_user_prompt
from app.rag.rag_service import NO_CONTEXT_DEFAULT_REPLY, RAGService


# ---------------------------------------------------------------------------
# Test Mocks
# ---------------------------------------------------------------------------

class FakeRetriever:
    """Mock Retriever producing controlled RetrievalPreviewResponse objects."""

    def __init__(
        self,
        has_context: bool = True,
        results: Optional[List[RetrievalChunkItem]] = None,
        context_text: str = "--- SOURCE 1 ---\nSection: Cardiology\n\nCardiology care.",
    ):
        self.has_context = has_context
        self.results = results if results is not None else [
            RetrievalChunkItem(
                rank=1,
                chunk_id="chunk_cardio_01",
                document_id="doc_cardio",
                score=0.88,
                content="Cardiology care.",
                section="Cardiology",
                url="https://psmissionhospital.org/cardio",
                title="Cardiology Department",
                source="website",
            )
        ]
        self.context_text = context_text
        self.last_query: Optional[str] = None

    def retrieve(self, query: str, **kwargs) -> RetrievalPreviewResponse:
        self.last_query = query
        return RetrievalPreviewResponse(
            query=query,
            has_context=self.has_context,
            results=self.results if self.has_context else [],
            context=ContextResult(
                has_context=self.has_context,
                context_text=self.context_text if self.has_context else "",
                total_chunks=len(self.results) if self.has_context else 0,
                total_characters=len(self.context_text) if self.has_context else 0,
                reason=None if self.has_context else "No sufficiently relevant information was found.",
            ),
            debug=RetrievalDebugInfo(
                pinecone_top_k=5,
                candidates_retrieved=len(self.results) if self.has_context else 0,
                removed_by_threshold=0,
                removed_duplicates=0,
                final_chunks_count=len(self.results) if self.has_context else 0,
                context_character_count=len(self.context_text) if self.has_context else 0,
            ),
            retrieval_duration_seconds=0.01,
        )


# ---------------------------------------------------------------------------
# 1. LLM Service Tests
# ---------------------------------------------------------------------------

def test_mock_llm_provider_generate():
    provider = MockLLMProvider("Hospital response text")
    answer = provider.generate("Instruction", "User question")

    assert answer == "Hospital response text"
    assert provider.last_system_instruction == "Instruction"
    assert provider.last_user_message == "User question"
    assert provider.call_count == 1


def test_openai_provider_missing_api_key():
    provider = OpenAILLMProvider(api_key="")
    with pytest.raises(ValueError, match="OpenAI API key is missing"):
        provider.generate("Instruction", "Question")


def test_openai_provider_with_mock_client():
    class MockOpenAIClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    class Choice:
                        class Message:
                            content = "Cardiology treats heart disease."
                        message = Message()
                    class Response:
                        choices = [Choice()]
                    return Response()
            completions = Completions()
        chat = Chat()

    provider = OpenAILLMProvider(api_key="sk-test", client=MockOpenAIClient())
    result = provider.generate("Instructions", "Tell me about Cardiology")
    assert result == "Cardiology treats heart disease."


def test_openai_provider_responses_api_success():
    class MockResponsesClient:
        class Responses:
            @staticmethod
            def create(**kwargs):
                class Response:
                    output_text = "Modern Responses API output."
                return Response()
        responses = Responses()

    provider = OpenAILLMProvider(api_key="sk-test", client=MockResponsesClient())
    result = provider.generate("Instructions", "Question")
    assert result == "Modern Responses API output."


def test_openai_provider_timeout_error():
    from openai import APITimeoutError
    import httpx

    class FailingClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    raise APITimeoutError(request=httpx.Request("POST", "https://api.openai.com"))
            completions = Completions()
        chat = Chat()

    provider = OpenAILLMProvider(api_key="sk-test", client=FailingClient())
    with pytest.raises(TimeoutError, match="timed out"):
        provider.generate("Instructions", "Question")


def test_openai_provider_malformed_response():
    class EmptyChoicesClient:
        class Chat:
            class Completions:
                @staticmethod
                def create(**kwargs):
                    class Response:
                        choices = []
                    return Response()
            completions = Completions()
        chat = Chat()

    provider = OpenAILLMProvider(api_key="sk-test", client=EmptyChoicesClient())
    with pytest.raises(ValueError, match="empty choices"):
        provider.generate("Instructions", "Question")


# ---------------------------------------------------------------------------
# 2. Prompt Construction & Context Injection
# ---------------------------------------------------------------------------

def test_prompt_construction_and_context_injection():
    question = "What are the visiting hours?"
    context = "--- SOURCE 1 ---\nGeneral wards: 4 PM to 7 PM."

    prompt = build_grounded_user_prompt(question, context)

    assert "=== VERIFIED HOSPITAL CONTEXT ===" in prompt
    assert "General wards: 4 PM to 7 PM." in prompt
    assert "Patient Question: What are the visiting hours?" in prompt
    assert "P.S. Mission Hospital" in HOSPITAL_SYSTEM_INSTRUCTION
    assert "STRICT GROUNDING RULES" in HOSPITAL_SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# 3. RAG Service: Grounded Flow, No-Context & Source Preservation
# ---------------------------------------------------------------------------

def test_rag_service_complete_flow():
    mock_llm = MockLLMProvider("Cardiology at P.S. Mission Hospital offers 2D Echo and ICCU care.")
    retriever = FakeRetriever(has_context=True)
    rag = RAGService(retriever=retriever, llm_provider=mock_llm)

    response = rag.answer_question("Tell me about cardiology")

    assert response.answer == "Cardiology at P.S. Mission Hospital offers 2D Echo and ICCU care."
    assert len(response.sources) == 1
    assert response.sources[0].section == "Cardiology"
    assert response.sources[0].url == "https://psmissionhospital.org/cardio"
    assert response.sources[0].chunk_id == "chunk_cardio_01"
    assert mock_llm.call_count == 1
    assert "=== VERIFIED HOSPITAL CONTEXT ===" in mock_llm.last_user_message


def test_rag_service_no_context_behavior():
    mock_llm = MockLLMProvider("Hallucinated answer")
    retriever = FakeRetriever(has_context=False, results=[])
    rag = RAGService(retriever=retriever, llm_provider=mock_llm)

    response = rag.answer_question("What is the stock price of Apple?")

    # LLM must NOT be called when no relevant context is found
    assert mock_llm.call_count == 0
    assert response.answer == NO_CONTEXT_DEFAULT_REPLY
    assert response.sources == []


def test_rag_service_empty_message():
    rag = RAGService(retriever=FakeRetriever(), llm_provider=MockLLMProvider())
    with pytest.raises(ValueError, match="Message cannot be empty"):
        rag.answer_question("   ")


def test_rag_service_source_deduplication():
    chunks = [
        RetrievalChunkItem(
            rank=1,
            chunk_id="chunk_01",
            document_id="doc_1",
            score=0.9,
            content="Content 1",
            section="Cardiology",
            url="https://psmissionhospital.org/cardio",
            title="Cardiology",
        ),
        RetrievalChunkItem(
            rank=2,
            chunk_id="chunk_01",  # duplicate ID
            document_id="doc_1",
            score=0.85,
            content="Content 1 repeated",
            section="Cardiology",
            url="https://psmissionhospital.org/cardio",
            title="Cardiology",
        ),
        RetrievalChunkItem(
            rank=3,
            chunk_id="chunk_02",
            document_id="doc_1",
            score=0.80,
            content="Content 2",
            section="Visiting Hours",
            url="https://psmissionhospital.org/visiting",
            title="Visiting Hours",
        ),
    ]
    retriever = FakeRetriever(has_context=True, results=chunks)
    rag = RAGService(retriever=retriever, llm_provider=MockLLMProvider("Grounded text"))

    res = rag.answer_question("Services")
    assert len(res.sources) == 2
    assert res.sources[0].chunk_id == "chunk_01"
    assert res.sources[1].chunk_id == "chunk_02"


# ---------------------------------------------------------------------------
# 4. /api/chat Endpoint Tests
# ---------------------------------------------------------------------------

def test_api_chat_success(monkeypatch):
    client = TestClient(app)

    def mock_answer_question(self, message, **kwargs):
        from app.models.schemas import ChatResponse, SourceItem
        return ChatResponse(
            answer="The cardiology department is located on the 2nd floor.",
            sources=[
                SourceItem(
                    title="Cardiology Department",
                    url="https://psmissionhospital.org/cardio",
                    section="Cardiology",
                    chunk_id="chunk_cardio_01",
                )
            ],
        )

    monkeypatch.setattr(RAGService, "answer_question", mock_answer_question)

    resp = client.post("/api/chat", json={"message": "Where is cardiology?"})
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"] == "The cardiology department is located on the 2nd floor."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["section"] == "Cardiology"
    assert data["sources"][0]["chunk_id"] == "chunk_cardio_01"


def test_api_chat_empty_message():
    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "   "})
    assert resp.status_code == 400


def test_api_chat_timeout_handling(monkeypatch):
    client = TestClient(app)

    def mock_timeout(self, message, **kwargs):
        raise TimeoutError("Simulated timeout")

    monkeypatch.setattr(RAGService, "answer_question", mock_timeout)

    resp = client.post("/api/chat", json={"message": "Query causing timeout"})
    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"].lower()
