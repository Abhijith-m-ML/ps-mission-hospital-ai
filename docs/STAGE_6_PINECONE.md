# Stage 6: Pinecone Serverless Vector Database

This document details the vector database layer implemented in **Stage 6** of the Hospital AI Chatbot project, replacing local databases with managed **Pinecone Serverless** for cloud production readiness.

---

## 1. Core Concepts & Educational Guide

### 1. What is a Vector Database?
A traditional relational or document database (like PostgreSQL or MongoDB) searches for data using exact matches or text keywords (e.g. `WHERE title LIKE '%heart%'`). However, human patients often ask questions using everyday language, synonyms, or symptoms:
> *"My chest feels tight and my heartbeat is irregular, who should I see?"*

A keyword search might fail if the webpage only mentions *"Cardiology"* and *"arrhythmia"*. 
A **Vector Database** indexes numerical representations (embeddings) of text documents and allows nearest-neighbor search based on **semantic meaning** rather than literal keyword matches.

---

### 2. Why Pinecone is Used Instead of Local Qdrant
In earlier drafts, local Qdrant was considered. However, this hospital AI system is designed for real-world deployment across clinical facilities:
- **Zero Local Maintenance:** Pinecone Serverless handles replication, hardware scaling, storage tiering, and index optimization without managing Docker daemons or local storage disks on hospital servers.
- **High Availability & Durability:** Ensures 99.99% uptime with enterprise cloud SLAs across AWS and GCP.
- **Serverless Cost Efficiency:** Automatically scales down to zero when idle, charging only for read/write operations and active storage.

---

### 3. What is a Pinecone Index?
A **Pinecone Index** is the top-level container that houses vector embeddings and enables similarity queries. Think of an index as a specialized "table" or "database" optimized for geometric vector math.
- **Index Name:** `hospital-ai`
- **Embedding Dimension:** `384`
- **Metric:** `cosine`

---

### 4. What is a Vector?
A vector is an array of floating-point numbers produced by an embedding neural network (`all-MiniLM-L6-v2`).
For example:
```json
[0.0241, -0.0512, 0.0883, ..., -0.0129]  // 384 numbers
```
Each floating-point number represents a coordinate in a high-dimensional concept space. Texts with similar medical meanings end up geographically close to each other in this 384-dimensional space.

---

### 5. What is Metadata?
Pinecone allows each vector record to carry a JSON dictionary of key-value pairs called **metadata**.
While the vector is used to compute mathematical proximity, the metadata stores the real-world textual information:
```json
{
  "id": "doc_a1b2c3_chunk_00",
  "values": [0.0241, -0.0512, ...],
  "metadata": {
    "chunk_id": "doc_a1b2c3_chunk_00",
    "document_id": "doc_a1b2c3",
    "content": "Cardiology\nThe department of Cardiology at P S Mission Hospital provides...",
    "url": "https://www.psmissionhospital.org/departments",
    "title": "Departments",
    "section": "Cardiology",
    "source": "website"
  }
}
```
**Why store content in metadata?**
Preserving the full chunk text inside the metadata enables the application to directly reconstruct the retrieved context for citation and later LLM synthesis without needing a secondary database lookup.

---

### 6. What is a Namespace?
A **Namespace** is a partition within a Pinecone index that segments vector records into isolated logical subsets.
- **Current Single-Hospital Deployment:** All vectors are partitioned under `ps_mission_hospital`.
- **Future Multi-Tenant Roadmap:** When expanding to multiple healthcare networks, the same index can host `hospital_a`, `hospital_b`, and `hospital_c` in strict isolation without data leakage.

Queries executed within a namespace only search vectors in that partition, guaranteeing security and fast retrieval.

---

### 7. Why the Dimension is 384
The vector dimension is strictly dictated by the embedding model chosen in Stage 5:
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Output Dimension:** `384`
Vectors must match the exact dimensionality of the model. If an index expects 384 dimensions and receives a 768 or 1536-dimensional vector, the database will reject it. Our backend inspects the active model dimension dynamically (`embedder.embedding_dimension()`) rather than hard-coding it in multiple files.

---

### 8. Why Cosine Similarity is Used
Cosine similarity measures the **cosine of the angle** between two normalized vectors:
$$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
- **Range:** -1.0 to 1.0 (with 1.0 meaning identical orientation).
- **Advantage:** It evaluates direction rather than magnitude, ensuring that short sentences and longer paragraphs containing the same medical concepts are evaluated fairly.

---

### 9. What Top-K Means
When searching a vector database, **$K$** represents the number of closest neighbors to return:
- If a patient asks: *"What are the visiting hours?"* with `top_k=3`, Pinecone calculates the distance between the query vector and all indexed vectors, returning the **3 most semantically similar chunks** sorted by descending similarity score.

---

### 10. What Upsert Means
**Upsert** is a combination of **"Update"** and **"Insert"**:
- If a vector record with ID `chunk_123` does not exist in the index, Pinecone **inserts** it.
- If a vector record with ID `chunk_123` already exists, Pinecone **overwrites/updates** it.

---

### 11. Why Deterministic IDs Matter
In Stage 4, we implemented deterministic chunk IDs using SHA-256 hashes derived from normalized URL, section title, and chunk sequence:
```text
hospital_page_abc123_chunk_01
```
By reusing this deterministic ID as the Pinecone vector ID:
- Indexing the same website page multiple times is **100% idempotent**.
- It prevents duplicate logical vectors from polluting search results.
- When website content changes in the future, the change detection pipeline can re-index or delete specific outdated chunks by ID.

---

### 12. How the Backend Communicates with Pinecone
```
[Client / Browser]
       │
       │ HTTP POST /api/search/preview
       ▼
[FastAPI Backend]
       │
       ├── 1. Embed Query: all-MiniLM-L6-v2 (Local CPU/GPU, 384 floats)
       │
       ├── 2. Pinecone Python SDK (HTTPS / REST + gRPC)
       │      Sends vector to Pinecone Serverless Index (AWS us-east-1)
       │
       ▼
[Pinecone Cloud]
       │ Computes cosine similarity across namespace 'ps_mission_hospital'
       ▼
[FastAPI Backend]
       │ Formats ranked matches (Score, Content, Section, URL)
       ▼
[Client Response JSON]
```

---

### 13. Why the Frontend Must Never Contain the Pinecone API Key
- The Pinecone API key grants full write, read, and delete permissions to the cloud index.
- If bundled in frontend JavaScript, any visitor could open DevTools, extract the API key, and wipe out or tamper with hospital data.
- **Rule:** The Pinecone API key is kept exclusively in `backend/.env` and accessed only by the secure server environment.

---

### 14. Why We Are Not Using an LLM Yet
In retrieval-augmented generation (RAG), a system can only generate accurate answers if its retrieval layer reliably surfaces the correct facts.
If we connected an LLM immediately:
- Retrieval bugs would be masked or hallucinated by the model.
- Evaluating indexing accuracy, chunk boundaries, and vector similarity would be much harder.
By stopping at Stage 6 and testing pure semantic search first, we verify that the correct hospital department or service chunk is retrieved before introducing text generation in Stage 7/8.

---

## 2. API Endpoints

### 1. `POST /api/search/preview`
Performs natural-language vector similarity search against the Pinecone index.

**Request Body:**
```json
{
  "query": "Which department treats heart problems?",
  "top_k": 3
}
```

**Response Body:**
```json
{
  "query": "Which department treats heart problems?",
  "index": "hospital-ai",
  "namespace": "ps_mission_hospital",
  "top_k": 3,
  "total_found": 3,
  "search_duration_seconds": 0.042,
  "results": [
    {
      "id": "doc_e1f2a3_chunk_00",
      "score": 0.8142,
      "content": "Cardiology\nThe department of Cardiology at P S Mission Hospital provides comprehensive care for heart disorders...",
      "section": "Cardiology",
      "url": "https://www.psmissionhospital.org/departments",
      "title": "Departments",
      "source": "website",
      "metadata": { ... }
    }
  ]
}
```

---

### 2. `POST /api/index/preview`
Controlled ingestion pipeline that fetches, cleans, chunks, embeds, and indexes a hospital webpage.

**Request Body:**
```json
{
  "url": "https://www.psmissionhospital.org/",
  "max_chunks": 100
}
```

**Response Body:**
```json
{
  "index_name": "hospital-ai",
  "namespace": "ps_mission_hospital",
  "url": "https://www.psmissionhospital.org/",
  "title": "P.S. Mission Hospital",
  "indexed_count": 12,
  "failed_count": 0,
  "dimension": 384,
  "duration_seconds": 1.452
}
```

---

### 3. `GET /api/index/stats`
Retrieves vector count and configuration statistics for the active index and namespace.

**Response Body:**
```json
{
  "index": "hospital-ai",
  "namespace": "ps_mission_hospital",
  "dimension": 384,
  "total_vector_count": 12,
  "namespace_vector_count": 12
}
```

---

## 4. Configuration & Setup

Add your Pinecone credentials to `backend/.env`:
```env
# Pinecone Vector Database Configuration
PINECONE_API_KEY="your-pinecone-api-key-here"
PINECONE_INDEX_NAME="hospital-ai"
PINECONE_NAMESPACE="ps_mission_hospital"
PINECONE_CLOUD="aws"
PINECONE_REGION="us-east-1"
PINECONE_UPSERT_BATCH_SIZE=100
```

To create a Pinecone API key:
1. Visit [https://www.pinecone.io/](https://www.pinecone.io/) and create an account.
2. Navigate to **API Keys** in the Pinecone Console.
3. Click **Create API Key**, copy the token, and paste it into `backend/.env`.
