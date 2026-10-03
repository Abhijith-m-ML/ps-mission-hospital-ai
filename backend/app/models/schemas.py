from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for system health check endpoint."""
    status: str = Field(default="healthy", description="Current operational status")
    app_name: str = Field(..., description="Name of the application")
    environment: str = Field(..., description="Current running environment")
    version: str = Field(default="0.1.0", description="API version")
    timestamp: datetime = Field(..., description="Timestamp of the health check")


class CrawlRequest(BaseModel):
    """Payload to initiate a hospital website crawl."""
    start_url: str = Field(
        ...,
        examples=["https://example-hospital.com"],
        description="The starting URL (homepage) of the hospital website to crawl",
    )
    max_pages: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of pages to fetch and parse",
    )
    max_depth: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum link traversal depth relative to start_url (0 = start page only)",
    )


class CrawledPageItem(BaseModel):
    """Summary of an individual crawled page including extracted document content."""
    url: str
    title: str
    status: int
    depth: Optional[int] = None
    error: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CrawlResponse(BaseModel):
    """Summary metrics and page listings from a completed crawl."""
    start_url: str
    pages_discovered: int
    pages_fetched: int
    pages_failed: int
    pages: List[CrawledPageItem]


class PreviewRequest(BaseModel):
    """Request to fetch a single page and inspect its cleaned document representation."""
    url: str = Field(..., examples=["https://example.com"], description="Webpage URL to fetch and preview")


class PreviewResponse(BaseModel):
    """Structured cleaned document returned by the preview inspection endpoint."""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentBrief(BaseModel):
    """Brief metadata identifying the parent document of chunks."""
    url: str
    title: str


class ChunkItem(BaseModel):
    """An individual chunk with deterministic identifiers and metadata."""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]


class ChunkPreviewRequest(BaseModel):
    """Request to fetch, clean, and chunk a target webpage for inspection."""
    url: str = Field(
        ...,
        examples=["https://www.psmissionhospital.org/"],
        description="Target webpage URL to process through extraction and chunking",
    )
    chunk_size: Optional[int] = Field(default=None, description="Optional override for chunk size")
    chunk_overlap: Optional[int] = Field(default=None, description="Optional override for chunk overlap")


class ChunkPreviewResponse(BaseModel):
    """Result of document chunking inspection."""
    document: DocumentBrief
    chunk_count: int
    chunks: List[ChunkItem]


class EmbeddingTextPreviewRequest(BaseModel):
    """Request to generate embeddings for raw text input."""
    text: Optional[str] = Field(default=None, description="Single text string to embed")
    texts: Optional[List[str]] = Field(default=None, description="List of texts to batch embed")
    preview_dims: int = Field(default=5, ge=1, le=50, description="Number of vector dimensions to preview")


class EmbeddingTextItem(BaseModel):
    """Preview of an individual text embedding."""
    text_preview: str
    embedding_dimension: int
    preview: List[float]


class EmbeddingTextPreviewResponse(BaseModel):
    """Response containing embedding preview information."""
    model: str
    embedding_dimension: int
    count: int
    preview: List[float]
    items: List[EmbeddingTextItem]


class EmbeddingChunkPreviewRequest(BaseModel):
    """Request to chunk and embed a webpage or a set of chunks."""
    url: Optional[str] = Field(
        default=None,
        examples=["https://www.psmissionhospital.org/"],
        description="Webpage URL to fetch, clean, chunk, and embed",
    )
    chunks: Optional[List[ChunkItem]] = Field(
        default=None,
        description="Optional pre-existing chunks to embed directly",
    )
    max_preview_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of sample chunk embeddings to return",
    )
    preview_dims: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of vector dimensions to preview in results",
    )
    chunk_size: Optional[int] = Field(default=None, description="Optional override for chunk size")
    chunk_overlap: Optional[int] = Field(default=None, description="Optional override for chunk overlap")


class ChunkEmbeddingItem(BaseModel):
    """Preserved chunk data bundled with its embedding preview."""
    chunk_id: str
    document_id: str
    section: str
    content_preview: str
    embedding_dimension: int
    embedding_preview: List[float]
    metadata: Dict[str, Any]


class ChunkEmbeddingPreviewResponse(BaseModel):
    """Response containing chunk embedding inspection data."""
    model: str
    total_chunks_processed: int
    preview_chunks_count: int
    embedding_dimension: int
    sample_results: List[ChunkEmbeddingItem]


class SearchPreviewRequest(BaseModel):
    """Request payload for semantic vector search inspection."""
    query: str = Field(
        ...,
        examples=["Which department treats heart problems?"],
        description="Natural-language question or topic to search for",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top nearest neighbors to retrieve")
    namespace: Optional[str] = Field(default=None, description="Optional Pinecone namespace override")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter dictionary")


class SearchResultItem(BaseModel):
    """A single retrieved chunk match with semantic similarity score."""
    id: str
    score: float
    content: str
    section: str
    url: str
    title: Optional[str] = None
    source: Optional[str] = "website"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchPreviewResponse(BaseModel):
    """Response containing vector search query results."""
    query: str
    index: str
    namespace: str
    top_k: int
    total_found: int
    search_duration_seconds: Optional[float] = None
    results: List[SearchResultItem]


class IndexPreviewRequest(BaseModel):
    """Request payload to index webpage chunks into Pinecone."""
    url: str = Field(
        default="https://www.psmissionhospital.org/",
        description="Webpage URL to fetch, clean, chunk, embed, and index into Pinecone",
    )
    namespace: Optional[str] = Field(default=None, description="Optional Pinecone namespace")
    chunk_size: Optional[int] = Field(default=None, description="Optional override for chunk size")
    chunk_overlap: Optional[int] = Field(default=None, description="Optional override for chunk overlap")
    max_chunks: Optional[int] = Field(default=50, ge=1, le=200, description="Maximum number of chunks to index")


class IndexPreviewResponse(BaseModel):
    """Response summarizing indexed collection metrics."""
    index_name: str
    namespace: str
    url: str
    title: Optional[str] = None
    indexed_count: int
    failed_count: int = 0
    dimension: int
    duration_seconds: Optional[float] = None


class IndexStatsResponse(BaseModel):
    """Response providing Pinecone index and namespace statistics."""
    index: str
    namespace: str
    dimension: int
    total_vector_count: int
    namespace_vector_count: int


class RetrievalChunkItem(BaseModel):
    """Structured representation of a retrieved and filtered chunk candidate."""
    rank: int = Field(..., description="Relevance rank ordered by similarity score (1-indexed)")
    chunk_id: str = Field(..., description="Deterministic chunk identifier")
    document_id: str = Field(..., description="Deterministic document identifier")
    score: float = Field(..., description="Cosine similarity score")
    content: str = Field(..., description="Original textual content of the chunk")
    section: str = Field(default="General", description="Clinical department or topic section")
    url: str = Field(default="", description="Source webpage URL")
    title: Optional[str] = Field(default=None, description="Source page title")
    source: Optional[str] = Field(default="website", description="Information origin")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Complete preserved metadata dictionary")


class ContextResult(BaseModel):
    """Structured context assembled from retrieved chunks for future LLM ingestion."""
    has_context: bool = Field(..., description="Indicates if sufficiently relevant context exists")
    context_text: str = Field(..., description="Formatted context string with source boundaries")
    total_chunks: int = Field(default=0, description="Number of chunks included in context_text")
    total_characters: int = Field(default=0, description="Total characters in context_text")
    reason: Optional[str] = Field(default=None, description="Explanation when has_context is False")


class RetrievalDebugInfo(BaseModel):
    """Operational diagnostics for retrieval filtering, deduplication, and budgeting."""
    pinecone_top_k: int = Field(..., description="Top-K requested from vector store")
    candidates_retrieved: int = Field(..., description="Raw candidate count returned by vector store")
    removed_by_threshold: int = Field(..., description="Candidates excluded below minimum score")
    removed_duplicates: int = Field(..., description="Candidates dropped as duplicate chunk IDs")
    final_chunks_count: int = Field(..., description="Count of chunks after filtering and deduplication")
    context_character_count: int = Field(..., description="Character length of assembled context text")


class RetrievalPreviewRequest(BaseModel):
    """Request payload to test the formal retrieval and context assembly pipeline."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["Which department treats heart problems?"],
        description="User question to retrieve relevant hospital context for",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of candidate nearest neighbors to retrieve (defaults to RETRIEVAL_TOP_K)",
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (defaults to RETRIEVAL_MIN_SCORE)",
    )
    namespace: Optional[str] = Field(
        default=None,
        description="Pinecone namespace to query (defaults to PINECONE_NAMESPACE)",
    )
    max_context_chunks: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of chunks to include in assembled context",
    )
    max_context_characters: Optional[int] = Field(
        default=None,
        ge=100,
        le=50000,
        description="Maximum character budget for assembled context string",
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filter dictionary for vector search",
    )


class RetrievalPreviewResponse(BaseModel):
    """Comprehensive response from the retrieval layer inspection endpoint."""
    query: str
    has_context: bool
    results: List[RetrievalChunkItem]
    context: ContextResult
    debug: RetrievalDebugInfo
    retrieval_duration_seconds: Optional[float] = None


class SourceItem(BaseModel):
    """Source reference for factual ground truth cited from hospital documents."""
    chunk_id: str = Field(..., description="Deterministic chunk identifier from Pinecone or entity store")
    document_id: Optional[str] = Field(default=None, description="Parent document identifier")
    title: Optional[str] = Field(default=None, description="Title of the source document or section")
    url: str = Field(default="", description="Webpage URL where the factual information originates")
    section: str = Field(default="General", description="Department or section name")
    content_type: Optional[str] = Field(default="webpage", description="Type of source content: webpage, doctor, department, or facility")
    doctor_name: Optional[str] = Field(default=None, description="Doctor name if source represents a physician record")
    department: Optional[str] = Field(default=None, description="Clinical department name when available")
    image_url: Optional[str] = Field(default=None, description="Image URL associated with the source item")


class ChatMessage(BaseModel):
    """Single turn in a conversation history window."""
    role: str = Field(..., description="Message author role: 'user' or 'assistant'")
    content: str = Field(..., description="Text content of the message")


class ChatRequest(BaseModel):
    """Request payload for the hospital conversational AI endpoint."""
    session_id: Optional[str] = Field(default=None, description="Temporary client session ID (runtime state only)")
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["Tell me about Cardiology"],
        description="Patient or visitor question for the hospital assistant",
    )
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Recent conversation turns in the current session (temporary in-memory only)",
    )
    language: Optional[str] = Field(
        default="en-IN",
        description="Language code: 'en-IN', 'ml-IN', 'hi-IN', or 'auto'",
    )


class VoiceTranscribeResponse(BaseModel):
    """Transcription response from Gemini speech-to-text."""
    text: str = Field(..., description="Transcribed query text")
    language: str = Field(default="en-IN", description="Detected or requested language code (en-IN, ml-IN, hi-IN)")


class VoiceSynthesizeRequest(BaseModel):
    """Payload to synthesize speech from text using Gemini TTS."""
    text: str = Field(..., min_length=1, description="Answer text to convert to audio")
    language: Optional[str] = Field(default="en-IN", description="Language code: 'en-IN', 'ml-IN', 'hi-IN', or 'auto'")


class ChatResponse(BaseModel):
    """Response containing grounded answer and verified hospital sources."""
    answer: str = Field(..., description="Grounded factual answer generated by the hosted LLM")
    sources: List[SourceItem] = Field(default_factory=list, description="Verified hospital sources from Pinecone retrieval")
    source_ids: List[str] = Field(default_factory=list, description="Explicit chunk IDs cited in the answer")
    session_id: Optional[str] = Field(default=None, description="Echoed session identifier")
    resolved_query: Optional[str] = Field(default=None, description="Context-resolved standalone question used for retrieval")



class DoctorItem(BaseModel):
    """Structured representation of a hospital doctor or medical consultant."""
    content_type: str = Field(default="doctor", description="Document/entity type discriminator")
    doctor_name: str = Field(..., description="Full name and title of the doctor")
    qualification: Optional[str] = Field(default="", description="Medical degrees, certifications, and fellowships")
    department: Optional[str] = Field(default="", description="Clinical specialty or department name")
    schedule_text: Optional[str] = Field(default="", description="Outpatient clinic timings and availability days")
    image_url: Optional[str] = Field(default="", description="Absolute URL to the doctor's profile photograph")
    source_url: Optional[str] = Field(default="", description="Webpage URL where the doctor profile was discovered")


class FacilityItem(BaseModel):
    """Structured representation of a hospital department facility or diagnostic service."""
    content_type: str = Field(default="facility", description="Document/entity type discriminator")
    facility_name: str = Field(..., description="Name of the hospital facility or medical service")
    description: Optional[str] = Field(default="", description="Additional details or operational context")
    department: Optional[str] = Field(default="", description="Department providing the facility or service")
    source_url: Optional[str] = Field(default="", description="Source webpage URL")


class ExtractPreviewRequest(BaseModel):
    """Payload to inspect structured entity extraction for a specific hospital URL or HTML."""
    url: str = Field(
        default="https://www.psmissionhospital.org/doctors",
        description="Target webpage URL to fetch and extract structured hospital entities from",
    )
    html: Optional[str] = Field(
        default=None,
        description="Optional raw HTML content to bypass network fetching during testing",
    )


class ExtractPreviewResponse(BaseModel):
    """Structured preview of extracted hospital entities prior to vector indexing."""
    url: str
    title: str
    department: Optional[str] = Field(default="", description="Inferred clinical department name")
    total_doctors: int = Field(default=0, description="Count of doctors detected on the page")
    total_facilities: int = Field(default=0, description="Count of facilities detected on the page")
    total_images: int = Field(default=0, description="Count of content images discovered")
    doctors: List[DoctorItem] = Field(default_factory=list, description="Extracted doctor cards")
    facilities: List[FacilityItem] = Field(default_factory=list, description="Extracted clinical facilities/services")
    images: List[str] = Field(default_factory=list, description="Discovered absolute image URLs")


# ==============================================================================
# STRUCTURED ENTITY MODELS (Conceptual Types: Department, Doctor, Facility)
# ==============================================================================

class Department(BaseModel):
    """Structured representation of a hospital clinical department."""
    type: str = Field(default="department", description="Entity type discriminator")
    department_name: str = Field(..., description="Name of the department")
    description: Optional[str] = Field(default="", description="Department overview or description")
    source_url: Optional[str] = Field(default="", description="Webpage URL where department was found")


class Doctor(BaseModel):
    """Structured representation of a hospital physician or medical consultant."""
    type: str = Field(default="doctor", description="Entity type discriminator")
    doctor_name: str = Field(..., description="Full name and title of the doctor")
    qualification: Optional[str] = Field(default="", description="Medical degrees, certifications, and fellowships")
    department: Optional[str] = Field(default="", description="Clinical specialty or department name")
    schedule_text: Optional[str] = Field(default="", description="Outpatient clinic timings and availability days")
    image_url: Optional[str] = Field(default="", description="Absolute URL to the doctor's profile photograph")
    source_url: Optional[str] = Field(default="", description="Webpage URL where the doctor profile was discovered")
    source_title: Optional[str] = Field(default="", description="Page title of source webpage")


class Facility(BaseModel):
    """Structured representation of a hospital clinical facility or diagnostic service."""
    type: str = Field(default="facility", description="Entity type discriminator")
    facility_name: str = Field(..., description="Name of the hospital facility or medical service")
    description: Optional[str] = Field(default="", description="Additional details or operational context")
    department: Optional[str] = Field(default="", description="Department providing the facility or service")
    source_url: Optional[str] = Field(default="", description="Source webpage URL")
    source_title: Optional[str] = Field(default="", description="Page title of source webpage")


class StructuredPreviewRequest(BaseModel):
    """Payload to inspect structured entity extraction for a specific hospital URL or HTML."""
    url: str = Field(
        default="https://www.psmissionhospital.org/category/department/paediatrics-neonatology",
        description="Target webpage URL to fetch and extract structured hospital entities from",
    )
    html: Optional[str] = Field(
        default=None,
        description="Optional raw HTML content to bypass network fetching during testing",
    )


class StructuredPreviewResponse(BaseModel):
    """Structured preview of extracted hospital entities before re-indexing."""
    url: str
    source_title: str
    department: Optional[Department] = None
    doctors: List[Doctor] = Field(default_factory=list)
    facilities: List[Facility] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    total_doctors: int = 0
    total_facilities: int = 0
    total_images: int = 0







