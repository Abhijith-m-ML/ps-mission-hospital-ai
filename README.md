# Hospital AI Chatbot

A modern, standalone Hospital AI Chatbot web application designed to assist patients and visitors with hospital inquiries, facility navigation, department services, and visiting hours for **P.S. Mission Hospital**.

---

## Project Status

- **Stage 1 (Foundation):** Completed ✅
  - FastAPI backend foundation with modular architecture, logging, CORS, and `GET /api/health`.
  - React + Vite + Tailwind CSS frontend with hospital branding, chat area, message bubbles, quick suggestions, and microphone UI placeholder.
- **Stage 2 (Website Crawler):** Completed ✅
  - Modular crawler engine in `backend/app/crawler/` (`fetcher.py`, `url_manager.py`, `parser.py`, `crawler.py`).
  - Breadth-first crawl loop bounded by `max_pages` and `max_depth`.
  - Same-domain strict isolation, loop prevention, and rate limiting.
  - REST API endpoint: `POST /api/crawl`.
- **Stage 3 (HTML Extraction & Document Cleaning):** Completed ✅
  - Ingestion engine in `backend/app/ingestion/` (`cleaner.py`, `document_parser.py`).
  - Strips scripts, styles, boilerplate, and ads while preserving headings, paragraphs, lists, and tables.
  - Inspection API endpoint: `POST /api/crawl/preview`.
- **Stage 4 (Structure-Aware Document Chunking):** Completed ✅
  - Reusable chunking engine in `backend/app/ingestion/chunker.py` (`DocumentChunker`).
  - Heading-cohesive segmentation keeping medical departments, visiting hours, and clinical services together.
  - Configurable `CHUNK_SIZE` and `CHUNK_OVERLAP` via `.env`.
  - Sentence-boundary protection and deterministic SHA-256 chunk identifiers.
- **Stage 5 (Vector Embeddings):** Completed ✅
  - Embedding pipeline using `sentence-transformers/all-MiniLM-L6-v2`.
  - 384-dimensional dense vectors with cosine similarity validation.
  - Inspection endpoints: `POST /api/embeddings/preview` and `POST /api/embeddings/chunks/preview`.
- **Stage 6 (Pinecone Serverless Vector Database):** Completed ✅
  - Cloud vector database integration in `backend/app/retrieval/vector_store.py` (`PineconeVectorStore`).
  - Serverless index architecture on AWS (`hospital-ai`), dimension 384, metric cosine.
  - Hospital tenant namespace isolation (`ps_mission_hospital`).
  - Deterministic vector IDs using existing chunk IDs for 100% idempotent upserts.
  - Full metadata preservation (`content`, `url`, `title`, `section`, `document_id`).
  - Vector similarity search service (`SearchService`) and query preview API (`POST /api/search/preview`).
  - Webpage ingestion & Pinecone indexer (`IndexingService`) and stats API (`GET /api/index/stats`).
- **Stage 7 (Retrieval Layer & Context Assembly):** Completed ✅
  - Formalized retrieval pipeline in `backend/app/retrieval/retriever.py` (`Retriever`).
  - Candidate generation from Pinecone using existing 384-d sentence embeddings.
  - Minimum similarity relevance filtering (`min_score` cutoff).
  - Deterministic chunk deduplication preserving distinct document passages.
  - Relevance ordering with 1-indexed ranking.
  - Context assembly engine in `backend/app/retrieval/context_builder.py` (`ContextBuilder`).
  - Strict character budgeting (`max_context_characters`) and chunk limiting (`max_context_chunks`).
  - Explicit source demarcation (`--- SOURCE N ---`) with preserved URLs and titles.
  - Safe no-result reporting (`has_context: false`) without hallucinated fallbacks.
  - Inspection API endpoint: `POST /api/retrieval/preview`.
- **Stage 8 (Grounded RAG with Hosted LLM API):** Completed ✅
  - Provider-independent LLM service in `backend/app/rag/llm.py` (`BaseLLMProvider`).
  - Hosted cloud integration using official OpenAI SDK (`OpenAILLMProvider`) supporting the modern OpenAI Responses API.
  - Strict clinical grounding rules and anti-hallucination guardrails (`backend/app/rag/prompts.py`).
  - Controlled no-context fallback (*"I couldn't find that information in the available hospital information."*) bypassing LLM calls when evidence is absent.
  - Verified source preservation citing official hospital URLs, titles, sections, and chunk IDs.
  - Full conversational chat API: `POST /api/chat`.
  - Frontend integration with live interactive source badges, eliminating old simulated placeholder replies.
  - 104 automated unit and integration tests passing.

---

## Project Structure

```
hospital-ai-chatbot/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── health.py           # GET  /api/health
│   │   │   ├── crawler.py          # POST /api/crawl, POST /api/crawl/preview
│   │   │   ├── chunks.py           # POST /api/chunks/preview
│   │   │   ├── embeddings.py       # POST /api/embeddings/preview, /preview/chunks
│   │   │   ├── search.py           # POST /api/search/preview
│   │   │   ├── index.py            # POST /api/index/preview, GET /stats
│   │   │   ├── retrieval.py        # POST /api/retrieval/preview
│   │   │   ├── chat.py             # POST /api/chat (Conversational RAG)
│   │   │   └── router.py           # Combined API router
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py           # Settings: Pinecone, LLM, retrieval
│   │   │   └── logging_config.py   # Centralized logger
│   │   ├── crawler/
│   │   ├── ingestion/
│   │   ├── retrieval/
│   │   │   ├── vector_store.py     # Pinecone serverless integration
│   │   │   ├── context_builder.py  # Source boundary formatting & budgeting
│   │   │   └── retriever.py        # Query embedding, filtering & dedup
│   │   ├── rag/
│   │   │   ├── llm.py              # Provider interface & OpenAILLMProvider
│   │   │   ├── prompts.py          # Clinical grounding system instruction
│   │   │   └── rag_service.py      # End-to-end RAG orchestrator
│   │   ├── models/
│   │   ├── services/
│   │   └── main.py                 # FastAPI application
│   ├── tests/
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
├── frontend/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── STAGE_6_PINECONE.md
│   ├── STAGE_7_RETRIEVAL.md
│   └── STAGE_8_HOSTED_LLM.md
├── .gitignore
└── README.md
```

---

## 1. Running the Automated Tests

All tests run using in-memory mocks without external network dependencies or API costs:

1. **Activate virtual environment**:
   ```powershell
   cd "D:\Desktop\PS mission\hospital-ai-chatbot\backend"
   .\venv\Scripts\Activate.ps1
   ```

2. **Execute the test suite with pytest**:
   ```powershell
   pytest -v
   ```

All 104 unit and integration tests across Stages 1-8 will execute and pass.

---

## 2. Running the Backend Server

1. **Start the FastAPI server**:
   ```powershell
   cd "D:\Desktop\PS mission\hospital-ai-chatbot\backend"
   .\venv\Scripts\Activate.ps1
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

2. **Access Endpoints**:
   - Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Health check: `GET http://127.0.0.1:8000/api/health`
   - Formalized retrieval & context assembly: `POST http://127.0.0.1:8000/api/retrieval/preview`
   - Conversational RAG Chatbot: `POST http://127.0.0.1:8000/api/chat`

---

## 3. Manually Testing Stage 8 (Conversational RAG Chat Endpoint)

*Note: Requires configuring `PINECONE_API_KEY` and `OPENAI_API_KEY` in `backend/.env`.*

```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Tell me about Cardiology"}'
```

### Example Response:
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

## 4. Running the Frontend (React + Vite)

In a separate terminal window:
```powershell
cd "D:\Desktop\PS mission\hospital-ai-chatbot\frontend"
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.
