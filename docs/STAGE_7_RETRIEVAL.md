# Stage 7: Formalized Retrieval Layer

This document details the formalized **Retrieval Layer** implemented in **Stage 7** of the Hospital AI Chatbot project, bridging raw vector similarity search with structured context assembly for future LLM consumption.

---

## 1. Architectural Pipeline

```
User Query
    │
    ▼
Query Embedding (384-d with sentence-transformers/all-MiniLM-L6-v2)
    │
    ▼
Pinecone Serverless Candidate Search (Namespace: ps_mission_hospital)
    │
    ▼
Candidate Results (top_k candidates)
    │
    ▼
Relevance Filtering (score >= RETRIEVAL_MIN_SCORE)
    │
    ▼
Deduplication (deterministic chunk_id filtering)
    │
    ▼
Ordering (strict descending similarity score)
    │
    ▼
Context Assembly (ContextBuilder: source demarcation & character budgeting)
    │
    ▼
Structured Context Payload (Ready for Stage 8 LLM)
```

---

## 2. Core Concepts & Educational Guide

### 1. What Retrieval Means in RAG
Retrieval-Augmented Generation (RAG) grounds an AI assistant in verified, real-world factual documents. 
**Retrieval** is the process of translating a user's natural-language inquiry into high-dimensional semantic coordinates, finding the most relevant textual chunks in the vector database, and preparing them as evidence. Rather than relying on the LLM's internal memory (which can hallucinate or become outdated), retrieval ensures the model has the exact hospital facts in front of it.

---

### 2. What Top-K Means in Retrieval
**$K$** represents the maximum number of nearest neighbors retrieved from Pinecone.
- In Stage 7, candidate generation fetches `top_k` (default `5`, configurable up to `50`).
- Because raw vector similarity can return tangential matches, `top_k` is treated as a candidate pool that undergoes filtering and deduplication before being assembled into context.

---

### 3. What Similarity Score Means
When comparing a query embedding vector $\mathbf{q}$ and a chunk embedding vector $\mathbf{c}$, **Cosine Similarity** measures the cosine of the angle between them:
$$\text{Score} = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\| \|\mathbf{c}\|}$$
- **Range:** `[-1.0, 1.0]`, where `1.0` denotes identical semantic orientation.
- In practical sentence-transformers search:
  - `0.65 - 0.85+`: Extremely high direct semantic match.
  - `0.40 - 0.65`: Strong topical relevance (e.g. "heart problems" matching "Cardiology Department").
  - `< 0.30`: Weak or tangential association.

---

### 4. Why Minimum Relevance Threshold Filtering is Critical
A vector search algorithm will *always* return the top-$K$ closest vectors in the database, even if the database has nothing to do with the user's question!
For example, if a user asks:
> *"What is the stock price of Apple?"*

A naive search without a threshold would still return the 5 closest hospital chunks (perhaps general billing or admin). 
By applying `score >= RETRIEVAL_MIN_SCORE`, the system prunes irrelevant chunks.
- **Configurable Nature:** No single threshold is universally correct across all medical domains. In Stage 7, `RETRIEVAL_MIN_SCORE` defaults to `0.0` for safe candidate inspection and can be tuned dynamically per query.

---

### 5. Why Deduplication Matters
When crawling and chunking hospital websites, pages may contain overlapping boilerplate, repeated department headings, or identical announcements across multiple sections.
- The `Retriever` maintains a seen registry based on the deterministic `(document_id, chunk_id)`.
- If identical chunks are returned, only the highest-scoring occurrence is kept.
- Legitimate different chunks from the same document (e.g., `chunk_01` and `chunk_02`) are safely preserved.

---

### 6. What Context Assembly Is
Raw vector search outputs a list of JSON records. An LLM cannot digest disconnected records cleanly without structure.
The `ContextBuilder` transforms ranked chunks into clean, demarcated context:
```text
--- SOURCE 1 ---
Section: Cardiology
Title: Cardiology Department | P.S. Mission Hospital
URL: https://www.psmissionhospital.org/departments/cardiology

The Department of Cardiology at P.S. Mission Hospital offers comprehensive clinical cardiology care...

--- SOURCE 2 ---
Section: Visiting Hours & Guidelines
Title: Patient & Visitor Information | P.S. Mission Hospital
URL: https://www.psmissionhospital.org/patient-guide/visiting-hours

General Wards: Morning: 11:00 AM - 12:00 PM, Evening: 4:30 PM - 7:00 PM...
```
This explicit demarcation allows future LLM prompts to cite specific sources accurately (`[Source 1]`, `[Source 2]`).

---

### 7. Why Context Limits (Budgeting) are Required
LLMs have finite context windows and can suffer from the *"Lost in the Middle"* phenomenon if overwhelmed with excessive text.
- `RETRIEVAL_MAX_CONTEXT_CHUNKS` (default `5`): Restricts total number of passages.
- `RETRIEVAL_MAX_CONTEXT_CHARACTERS` (default `12000` chars): Sets a hard token/character budget.
- **Safe Truncation:** When the budget is reached, lower-ranked chunks are omitted, preserving the highest-ranked evidence intact.

---

### 8. Why Retrieval is Completely Decoupled from the LLM
Keeping retrieval separate from generation provides critical software engineering advantages:
1. **Independent Evaluation:** We can benchmark retrieval accuracy (MRR, Hit Rate@K, Recall) without spending API credits on LLM generation.
2. **Speed & Caching:** Retrieved contexts can be cached, audited, or inspected in real time.
3. **Model Agnostic:** The retrieval layer can feed Claude, GPT-4, Gemini, or local models without modifying vector search logic.

---

### 9. Why "No-Result" Behavior is Essential
If a patient asks a question that the hospital website does not answer:
```json
{
  "has_context": false,
  "results": [],
  "context_text": "",
  "reason": "No sufficiently relevant information was found."
}
```
If the system forced an answer or passed tangential context, the LLM would likely **hallucinate** clinical advice or wrong visiting hours. Reporting `has_context: false` enables the chatbot in Stage 8 to gracefully state:
> *"I apologize, but I do not have verified information regarding this in P.S. Mission Hospital's records. Please contact hospital reception at +91 484 2700541."*

---

### 10. Why Source Metadata Must Be Preserved
Every chunk item preserves:
- `url`: Direct link to the official hospital webpage.
- `title`: Webpage title.
- `section`: Department or topic header (e.g., Cardiology, Pediatrics, Emergency).
- `chunk_id` & `document_id`: Deterministic tracing keys.
- `score`: Mathematical relevance confidence.

In Stages 8 and 9, the frontend UI will display clickable citation badges linking patients directly to official hospital source pages.

---

## 3. REST API Specification

### Endpoint: `POST /api/retrieval/preview`

#### Request Body
```json
{
  "query": "What department treats heart problems?",
  "top_k": 3,
  "min_score": 0.30
}
```

#### Response Body
```json
{
  "query": "What department treats heart problems?",
  "has_context": true,
  "results": [
    {
      "rank": 1,
      "chunk_id": "ps_chunk_cardio_01",
      "document_id": "doc_cardio",
      "score": 0.5617,
      "content": "Cardiology Department\n\nThe Department of Cardiology at P.S. Mission Hospital...",
      "section": "Cardiology",
      "url": "https://www.psmissionhospital.org/departments/cardiology",
      "title": "Cardiology Department | P.S. Mission Hospital",
      "source": "website",
      "metadata": { ... }
    }
  ],
  "context": {
    "has_context": true,
    "context_text": "--- SOURCE 1 ---\nSection: Cardiology\nTitle: Cardiology Department | P.S. Mission Hospital\nURL: https://www.psmissionhospital.org/departments/cardiology\n\nCardiology Department\n\nThe Department of Cardiology at P.S. Mission Hospital...",
    "total_chunks": 1,
    "total_characters": 352,
    "reason": null
  },
  "debug": {
    "pinecone_top_k": 3,
    "candidates_retrieved": 3,
    "removed_by_threshold": 1,
    "removed_duplicates": 0,
    "final_chunks_count": 1,
    "context_character_count": 352
  },
  "retrieval_duration_seconds": 0.014
}
```
