"""Document ingestion, extraction, HTML cleaning, and chunking package."""
from app.ingestion.cleaner import HTMLCleaner
from app.ingestion.document_parser import (
    DocumentParser,
    CleanedDocument,
    DocumentMetadata,
)
from app.ingestion.chunker import (
    DocumentChunker,
    Chunk,
    ChunkMetadata,
)
from app.ingestion.embedder import (
    DocumentEmbedder,
    EmbeddedChunk,
)
from app.ingestion.similarity import cosine_similarity

__all__ = [
    "HTMLCleaner",
    "DocumentParser",
    "CleanedDocument",
    "DocumentMetadata",
    "DocumentChunker",
    "Chunk",
    "ChunkMetadata",
    "DocumentEmbedder",
    "EmbeddedChunk",
    "cosine_similarity",
]
