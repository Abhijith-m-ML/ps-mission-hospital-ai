# Stage 8: Hosted LLM API & Grounded RAG Generation

This document details the **Retrieval-Augmented Generation (RAG) Layer** implemented in **Stage 8** of the Hospital AI Chatbot project, connecting the Stage 7 retrieval and context assembly pipeline to a hosted cloud LLM API (**OpenAI**) and powering the interactive frontend chat interface.

---

## 1. Architectural Pipeline

```
Patient Question (e.g. "Tell me about Cardiology")
       │
       ▼
Retriever (Stage 7)
       │ Computes 384-d query embedding & searches Pinecone
       ▼
Candidate Results
       │ Deduplication & relevance threshold filtering
       ▼
Context Builder (Stage 7)
       │ Formats demarcated sources (--- SOURCE N ---)
       ▼
Context Check
       ├─ [No Relevant Context] ──► Controlled Response:
       │                            "I couldn't find that information in the available hospital information."
       │                            (Sources: []) [LLM is NOT called]
       │
       ▼ [Verified Context Exists]
Grounded Prompt Assembly (System instructions + Clinical grounding rules)
       │
       ▼
Hosted Cloud LLM Provider (OpenAILLMProvider via OpenAI Responses API)
       │
       ▼
Grounded Factual Answer + Verified Sources
       │
       ▼
Frontend Chat UI (React + Tailwind CSS with Interactive Source Badges)
```

---

## 2. Why a Hosted Cloud LLM API Instead of Ollama
In earlier designs, local LLMs via Ollama were considered. However, this hospital AI system is designed for live deployment across real healthcare environments:
1. **Zero Hardware Burden on Hospital Infrastructure:** Real hospitals rarely have dedicated GPU server racks for local LLM inference. A hosted API runs on enterprise infrastructure with 99.99% availability.
2. **Superior Reasoning & Clinical Grounding:** Modern hosted models (e.g. `gpt-4o-mini`, `gpt-4o`) adhere strictly to negative constraints (refusing to hallucinate missing facts) far more reliably than smaller local quantized models.
3. **Low Latency & High Concurrency:** Handles hundreds of simultaneous patient inquiries without queuing or VRAM out-of-memory crashes.

---

## 3. Provider-Independent Architecture
The LLM integration is abstracted behind the `BaseLLMProvider` interface in `backend/app/rag/llm.py`:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, system_instruction: str, user_message: str) -> str:
        """Generates a completion from the LLM provider."""
        pass
```

### Supported Providers:
- **`OpenAILLMProvider`**: Uses the official `openai` SDK (`openai>=1.50.0`), utilizing the modern **OpenAI Responses API** (`client.responses.create`) with robust fallback to Chat Completions.
- **`MockLLMProvider`**: Deterministic mock provider for automated unit and integration tests without network dependencies or API token expenses.

The rest of the RAG pipeline interacts only with `BaseLLMProvider`, ensuring the underlying provider can be switched (e.g. to Anthropic, Gemini, or Azure OpenAI) without modifying any application code.

---

## 4. Grounding & Hallucination Prevention Rules

The LLM is governed by strict system prompts (`backend/app/rag/prompts.py`):
1. **Source of Truth:** The retrieved context is the *sole* factual basis for answering questions about P.S. Mission Hospital.
2. **No Hallucination or Extrapolation:** If a service, doctor name, fee, or department is not mentioned in the context, the model explicitly states it is not available.
3. **No-Context Guardrail:** If the retriever returns `has_context: false`, the system **bypasses the LLM completely** and directly returns:
   > *"I couldn't find that information in the available hospital information."*
4. **Preservation of Specifics:** Phone numbers (e.g. Emergency: `+91 484 2700543`), visiting hours, and physical locations are preserved verbatim.
5. **No Fabricated Citations:** The model is prohibited from inventing links or external URLs.

---

## 5. Source Citations & Preservation

Every generated response preserves the underlying Pinecone chunk sources:
```json
{
  "answer": "The Department of Cardiology at P.S. Mission Hospital provides clinical cardiology management, 2D echocardiography, treadmill tests (TMT), and intensive cardiac care unit (ICCU) facilities.",
  "sources": [
    {
      "title": "Cardiology Department | P.S. Mission Hospital",
      "url": "https://www.psmissionhospital.org/departments/cardiology",
      "section": "Cardiology",
      "chunk_id": "ps_chunk_cardio_01"
    }
  ]
}
```

In the React frontend, these sources are displayed as clickable pills below each assistant response.

---

## 6. Configuration & Security

Configure the following variables in `backend/.env`:
```env
# Hosted LLM API Configuration
LLM_PROVIDER="openai"
OPENAI_API_KEY="your-openai-api-key-here"
LLM_MODEL="gpt-4o-mini"
LLM_TEMPERATURE=0.1
LLM_MAX_OUTPUT_TOKENS=800
```

### Security Guarantees:
- **Backend Only:** `OPENAI_API_KEY` is loaded exclusively inside FastAPI. The React client never receives or stores the API key.
- **Version Control:** `backend/.env` is ignored by Git in `.gitignore`.
- **Sanitized Errors:** In the event of API timeouts or rate limits, user-friendly messages are returned; raw stack traces and credentials are never leaked.

---

## 7. Chat API Specification

### Endpoint: `POST /api/chat`

#### Request Body
```json
{
  "message": "Tell me about Cardiology"
}
```

#### Response Body
```json
{
  "answer": "The Department of Cardiology at P.S. Mission Hospital, Maradu, Cochin, offers comprehensive clinical cardiology care, diagnostic 2D echocardiography, treadmill tests (TMT), Holter monitoring, electrocardiogram (ECG), and intensive cardiac care unit (ICCU) facilities.",
  "sources": [
    {
      "title": "Cardiology Department | P.S. Mission Hospital",
      "url": "https://www.psmissionhospital.org/departments/cardiology",
      "section": "Cardiology",
      "chunk_id": "ps_chunk_cardio_01"
    }
  ]
}
```

---

## 8. Frontend Integration

The React interface (`frontend/src/App.jsx`) connects directly to `POST /api/chat`:
- Submits `{ message: text }` asynchronously.
- Displays animated typing indicator while retrieval and hosted LLM inference execute.
- Renders the grounded assistant answer.
- Renders verified source badges linking directly to hospital documents.
- The previous simulated Stage 1/2 placeholder reply (*"Thank you for your question..."*) has been completely eliminated.
