# Hospital AI Chatbot - System Architecture

This document details the architectural foundation and modular components of the Hospital AI Chatbot application across its evolutionary stages:
- **Stage 1:** Project Foundation (FastAPI + React Vite)
- **Stage 2:** Single-Domain Website Crawler
- **Stage 3:** HTML Content Extraction & Document Cleaning
- **Stage 4:** Structure-Aware Document Chunking
- **Stage 5:** Text & Chunk Vector Embeddings (Sentence-Transformers)
- **Stage 6:** Pinecone Serverless Vector Database (Upsert & Semantic Search)

---

## 1. High-Level Architecture

```
+-------------------------------------------------------------------+
|                        Client Browser                             |
|    - Desktop & Mobile Web View                                    |
|    - React 19 + Tailwind CSS Interface                            |
|    - Hospital Header, Interactive Chat Stream, Input Bar          |
+-------------------------------------------------------------------+
                                  |
                                  | HTTP REST / JSON
                                  v
+-------------------------------------------------------------------+
|                  FastAPI Backend Server (Port 8000)               |
|                                                                   |
|  +-------------------------------------------------------------+  |
|  | Middleware: CORS Configuration (Allowed Origins)             |  |
|  +-------------------------------------------------------------+  |
|  | Logging & Lifecycle Management (Startup & Shutdown)         |  |
|  +-------------------------------------------------------------+  |
|  | API Router: /api                                            |  |
|  |   ├── GET  /api/health                                      |  |
|  |   ├── POST /api/crawl                                       |  |
|  |   ├── POST /api/crawl/preview                               |  |
|  |   ├── POST /api/chunks/preview                              |  |
|  |   ├── POST /api/embeddings/preview                          |  |
|  |   ├── POST /api/embeddings/chunks/preview                   |  |
|  |   ├── POST /api/search/preview                              |  |
|  |   ├── POST /api/index/preview                               |  |
|  |   ├── GET  /api/index/stats                                 |  |
|  |   ├── POST /api/retrieval/preview                           |  |
|  |   └── POST /api/chat                                        |  |
|  +-------------------------------------------------------------+  |
|  | Service Layer:                                              |  |
|  |   ├── HealthService (Operational status & timestamp)        |  |
|  |   ├── CrawlerService (Crawl coordinator & preview)          |  |
|  |   ├── ChunkingService (Document chunking preview)           |  |
|  |   ├── EmbeddingService (Text & chunk embedding preview)     |  |
|  |   ├── SearchService (Query embed & Pinecone search preview) |  |
|  |   ├── IndexingService (End-to-end web -> Pinecone indexer)  |  |
|  |   ├── Retriever (Stage 7: Filtering, dedup, ordering)       |  |
|  |   ├── ContextBuilder (Stage 7: Demarcation & budgeting)     |  |
|  |   └── RAGService (Stage 8: End-to-end grounded generator)   |  |
|  +-------------------------------------------------------------+  |
|  | Ingestion, Retrieval & Generation Subsystems:               |  |
|  |   ├── HTMLCleaner (Stage 3: Tag stripping & normalization)  |  |
|  |   ├── DocumentParser (Stage 3: Metadata & document builder) |  |
|  |   ├── DocumentChunker (Stage 4: Structure-aware chunking)   |  |
|  |   ├── DocumentEmbedder (Stage 5: Local sentence embeddings) |  |
|  |   ├── PineconeVectorStore (Stage 6: Serverless cloud DB)    |  |
|  |   ├── ContextBuilder (Stage 7: Structured context assembly) |  |
|  |   ├── Retriever (Stage 7: Candidate filter & dedup engine)  |  |
|  |   └── OpenAILLMProvider (Stage 8: Hosted Responses/Chat API)|  |
|  +-------------------------------------------------------------+  |
|  | Crawler Subsystem (Stage 2):                                |  |
|  |   ├── HospitalCrawler (BFS Loop, rate limiting, depth limit)|  |
|  |   ├── URLManager (Normalization, domain isolation, dupes)   |  |
|  |   ├── Fetcher (HTTP GET, redirect handling, headers)        |  |
|  |   └── HTMLParser (BeautifulSoup, link extraction, text)     |  |
|  +-------------------------------------------------------------+  |
|  | Core Settings:                                              |  |
|  |   └── Pydantic BaseSettings (.env / OPENAI_API_KEY, etc.)   |  |
|  +-------------------------------------------------------------+  |
+-------------------------------------------------------------------+
                                  |
              +-------------------+-------------------+
              | Pinecone SDK                          | OpenAI Responses API
              v                                       v
+------------------------------------+   +------------------------------------+
| Pinecone Serverless Vector DB      |   | Hosted Cloud LLM API (OpenAI)      |
| - Index: hospital-ai (384-d cosine)|   | - Model: gpt-4o-mini (temperature 0.1)
| - Namespace: ps_mission_hospital   |   | - Clinical Grounding System Prompt |
+------------------------------------+   +------------------------------------+
```

---

## 2. Decoupled Pipeline Layers

```
+---------------------------------------------------------------+
|                       CRAWLER LAYER                           |
|                      (app/crawler/)                           |
|  Input:  URL (e.g., https://www.psmissionhospital.org/)       |
|  Output: Raw HTML + Response Metadata                         |
+---------------------------------------------------------------+
                               |
                               | Raw HTML String
                               v
+---------------------------------------------------------------+
|                     EXTRACTION LAYER                          |
|                     (app/ingestion/)                          |
|  Components: HTMLCleaner, DocumentParser                      |
|  Output: Clean Document (Structured text + Metadata)          |
+---------------------------------------------------------------+
                               |
                               | Clean Document
                               v
+---------------------------------------------------------------+
|                     CHUNKING LAYER                            |
|                     (app/ingestion/)                          |
|  Component: DocumentChunker                                   |
|  Output: List[Chunk] with semantic headings & metadata        |
+---------------------------------------------------------------+
                               |
                               | Chunks
                               v
+---------------------------------------------------------------+
|                    EMBEDDING LAYER                            |
|                     (app/ingestion/)                          |
|  Component: DocumentEmbedder (all-MiniLM-L6-v2, 384-d)        |
|  Output: Dense 384-dimensional vector float arrays            |
+---------------------------------------------------------------+
                               |
                               | Chunks + Embeddings
                               v
+---------------------------------------------------------------+
|                   VECTOR DATABASE LAYER                       |
|                     (app/retrieval/)                          |
|  Component: PineconeVectorStore                               |
|  - Cloud Serverless Indexing                                  |
|  - Deterministic Vector IDs (idempotency)                     |
|  - Namespaced tenant isolation (ps_mission_hospital)          |
|  - Top-K Nearest-Neighbor Cosine Similarity Retrieval         |
+---------------------------------------------------------------+
                               |
                               | Candidate Matches
                               v
+---------------------------------------------------------------+
|                     RETRIEVAL LAYER                           |
|                      (Stage 7 Formalized)                     |
|  Components: Retriever, ContextBuilder                        |
|  - Minimum Relevance Score Filtering (min_score)              |
|  - Deterministic Chunk Deduplication                          |
|  - Strict Relevance Score Ordering                            |
|  - Explicit Source Demarcation (--- SOURCE N ---)             |
|  - Character Budgeting & Max Chunk Guardrails                 |
+---------------------------------------------------------------+
                               |
                               | Assembled Factual Context
                               v
+---------------------------------------------------------------+
|                     RAG & LLM LAYER                           |
|                      (Stage 8 Formalized)                     |
|  Components: RAGService, OpenAILLMProvider, Prompts           |
|  - Provider-independent BaseLLMProvider Interface             |
|  - Hosted OpenAI Responses / Chat Completions API             |
|  - Strict Clinical Grounding & Hallucination Guardrails       |
|  - No-Context Controlled Fallback without Hallucinations      |
|  - Verifiable Source Citations & Chunk ID Tracing             |
+---------------------------------------------------------------+
                               |
                               | Answer + Sources
                               v
+---------------------------------------------------------------+
|               FUTURE STAGES (Deferred from Stage 8)           |
|  Stage 9: Token Streaming & Automated Evaluation              |
+---------------------------------------------------------------+
```

---

## 3. Structure-Aware Chunking Engine (Stage 4)

### `DocumentChunker` (`app/ingestion/chunker.py`)
Responsible for dividing clean documents into self-contained semantic passages without destroying clinical relationships:

1. **Document Structure Analysis**:
   - Analyzes heading markers (`headings` metadata and section lines).
   - Groups paragraphs under their respective medical departments (e.g., "Cardiology", "Neurology", "Visiting Hours").
2. **Heading-Preserving Cohesion**:
   - If a department description fits within `CHUNK_SIZE`, it remains intact in a single chunk:
     ```text
     Cardiology

     The department of Cardiology at P S Mission Hospital provides medical
     interventional and non interventional management...
     ```
3. **Sentence-Aware Boundary Splitting**:
   - Large sections exceeding `CHUNK_SIZE` are split along sentence terminators (`. `, `? `, `! `, `\n`).
   - Words and numbers (such as phone numbers or visiting hours) are never severed mid-word.
4. **Configurable Size & Overlap**:
   - `CHUNK_SIZE`: Configurable maximum character length per chunk (default `800`).
   - `CHUNK_OVERLAP`: Sliding window overlap between consecutive chunks (default `150`) to preserve transitional context.
5. **Deterministic IDs**:
   - `document_id`: Deterministic 16-character SHA-256 hash of the normalized URL.
   - `chunk_id`: Deterministic 16-character SHA-256 hash computed from `document_id`, index, and content snippet.
   - Guaranteed consistency across repeated ingestions.

---

## 4. Directory Structure

```
hospital-ai-chatbot/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── health.py           # GET  /api/health
│   │   │   ├── crawler.py          # POST /api/crawl, POST /api/crawl/preview
│   │   │   ├── chunks.py           # POST /api/chunks/preview
│   │   │   └── router.py           # Consolidated API router
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py           # Settings: CHUNK_SIZE, CHUNK_OVERLAP (.env)
│   │   │   └── logging_config.py   # Centralized logger
│   │   ├── crawler/
│   │   │   ├── __init__.py
│   │   │   ├── fetcher.py          # HTTP transport (httpx)
│   │   │   ├── url_manager.py      # Normalization & domain isolation
│   │   │   ├── parser.py           # Link discovery parser
│   │   │   └── crawler.py          # BFS crawl loop
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── cleaner.py          # HTML noise removal
│   │   │   ├── document_parser.py  # Structured document extractor
│   │   │   └── chunker.py          # Structure-aware semantic chunker
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py          # Schemas (Health, Crawl, Chunk)
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── health_service.py   # Health logic
│   │   │   ├── crawler_service.py  # Crawl & preview coordinator
│   │   │   └── chunking_service.py # Chunking coordinator
│   │   └── main.py                 # FastAPI application
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── fixtures.py             # HTML test templates & malformed edge cases
│   │   ├── test_chunker.py         # Chunk size, headings, overlap & deterministic IDs
│   │   ├── test_api_chunks.py      # /api/chunks/preview endpoint tests
│   │   ├── test_cleaner.py         # HTML cleaning & structure preservation tests
│   │   ├── test_document_parser.py # Document metadata, empty & malformed tests
│   │   ├── test_url_manager.py     # URL normalization & scheme tests
│   │   ├── test_parser.py          # HTML link discovery tests
│   │   ├── test_crawler.py         # Crawl loop, depth, & max_pages tests
│   │   └── test_api_crawl.py       # API endpoint tests (crawl & preview)
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
├── frontend/
├── docs/
│   └── ARCHITECTURE.md
├── .gitignore
└── README.md
```

---

## 5. Scope Boundary (Stage 4)

- **Included:** Document structure analysis, heading-cohesive chunking, configurable sliding window overlap, deterministic ID hashing, and `POST /api/chunks/preview`.
- **Deferred to Next Phases:** Generating vector embeddings, Qdrant database collection creation, similarity search, and LLM prompting.
